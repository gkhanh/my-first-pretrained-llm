import torch
import torch.nn as nn
import bitsandbytes as bnb
import time
import os
import signal
import sys
import csv
import multiprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable TF32 for faster matrix multiplication (New PyTorch 2.x Syntax)
# This setting tells PyTorch to use TF32 precision for matrix multiplications (Linear layers)
torch.set_float32_matmul_precision('high') 

# Enable cuDNN benchmark mode for auto-tuning on your specific GPU
torch.backends.cudnn.benchmark = True

# Note: 'high' = TF32 (Fastest, good precision)
#       'highest' = FP32 (Slowest, max precision)
#       'medium' = BF16 (Fastest, lower precision)
# ---------------------------------------------

from datasets import load_dataset, IterableDataset
# from transformers import AutoTokenizer, default_data_collator
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader
from models.khanh_llm import KhanhLLM, V_SIZE
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# --- 1. SETUP AND CHECKPOINT CONFIG ---
CHECKPOINT_PATH = "checkpoint_latest.pth.tar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training Parameters
#BATCH_SIZE = 1 
BATCH_SIZE = 2 
SEQ_LEN = 2048
GRADIENT_ACCUMULATION_STEPS = 64
TOTAL_TOKENS_REQUIRED = 999_999_999_999_999 # Effective 'unlimited' for runtime control
SAVE_FREQUENCY_ROWS = 100_000 # Save every 100k rows 

# Data Parameters for C4 Subset
C4_DATASET_NAME = "allenai/c4"
C4_CONFIG = "en"
TARGET_ROWS = 1_000_000 # 1 million rows

# Global progress tracking (persisted in checkpoint)
total_tokens_processed = 0
total_rows_processed = 0

# Global variables for model, optimizer, tokenizer (initialized in main block)
model = None
optimizer = None
tokenizer = None
data_collator = None

# --- CHECKPOINT FUNCTIONS ---
# ... (save_checkpoint and load_checkpoint functions remain the same) ...

def save_checkpoint(model, optimizer, final_save=False):
    """Saves the essential training states to a file, including progress tracking."""
    global total_tokens_processed, total_rows_processed

    # Move model to CPU before saving to avoid CUDA conflicts with DataLoader workers
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    # Optimizer state might already be on CPU (bitsandbytes), but be safe
    optimizer_state = optimizer.state_dict()
    
    state = {
        'total_tokens_processed': total_tokens_processed,
        'total_rows_processed': total_rows_processed,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
    }
    
    try:
        torch.save(state, CHECKPOINT_PATH, _use_new_zipfile_serialization=False)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        print("Progress may be lost, but training can continue.")
        return

    status = "FINAL" if final_save else "PERIODIC"
    
    # Calculate average tokens per row for informational purposes
    avg_tokens_per_row = total_tokens_processed / total_rows_processed if total_rows_processed > 0 else 0
    
    print(f"\n--- {status} CHECKPOINT SAVED ---")
    print(f"    Rows: {total_rows_processed:,} / {TARGET_ROWS:,} ({total_rows_processed/TARGET_ROWS*100:.2f}%)")
    print(f"    Tokens: {total_tokens_processed/1e6:.2f}M")
    print(f"    Avg tokens/row: {avg_tokens_per_row:.1f}")

def load_checkpoint():
    """Loads the model and optimizer states if a checkpoint file exists.
    
    Returns:
        tuple: (start_tokens, start_rows) - The progress to resume from
    """
    global total_tokens_processed, total_rows_processed, model, optimizer
    
    start_tokens = 0
    start_rows = 0
    
    if os.path.isfile(CHECKPOINT_PATH):
        print(f"--> Checkpoint found at {CHECKPOINT_PATH}. Loading states...")
        
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load progress counters
            start_tokens = checkpoint.get('total_tokens_processed', 0)
            start_rows = checkpoint.get('total_rows_processed', 0)
            
            # Calculate average tokens per row for display
            avg_tokens = start_tokens / start_rows if start_rows > 0 else 0
            
            print(f"--> Resuming training from:")
            print(f"    Rows: {start_rows:,} / {TARGET_ROWS:,} ({start_rows/TARGET_ROWS*100:.2f}%)")
            print(f"    Tokens: {start_tokens/1e6:.2f}M")
            print(f"    Historical avg tokens/row: {avg_tokens:.1f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
            start_tokens = 0
            start_rows = 0
    else:
        print("--> No checkpoint found. Starting training from scratch.")
    
    # Initialize global counters
    total_tokens_processed = start_tokens
    total_rows_processed = start_rows
    
    return start_tokens, start_rows

# --- SIGNAL HANDLER ---
# ... (signal_handler and signal.signal registration remain the same) ...

def signal_handler(sig, frame):
    """Handles Ctrl+C (SIGINT) signal to save a checkpoint before exiting."""
    global model, optimizer
    print('\nCtrl+C detected! Saving interrupt checkpoint...')

    # old code
    # save_checkpoint(model, optimizer, final_save=False)

    # Try to gracefully stop - the DataLoader will be cleaned up automatically
    # when the process exits, but we can try to save first
    try:
        save_checkpoint(model, optimizer, final_save=False)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        print("Attempting to save with error handling...")
        # Try one more time with more aggressive error handling
        try:
            # Force move everything to CPU
            torch.cuda.synchronize()  # Wait for all CUDA operations to finish
            save_checkpoint(model, optimizer, final_save=False)
        except Exception as e2:
            print(f"Failed to save checkpoint: {e2}")
            print("Progress may be lost.")

    sys.exit(0)

# --- DATA LOADING FUNCTION ---

def _get_tokenizer():
    """Lazy initialization of tokenizer for worker processes."""
    global tokenizer
    if tokenizer is None:
        TOKENIZER_PATH = "./khanh_tokenizer"
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please run 'python scripts/build_tokenizer.py' first.")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def tokenize_and_chunk(examples):
    """Tokenization function that can be pickled for multiprocessing."""
    # Lazy load tokenizer in each worker process
    tok = _get_tokenizer()
    # We perform simple tokenization, ensuring max_length is met.
    # Targets are simply the input tokens shifted by one (for next-token prediction).
    tokenized_data = tok(
        examples["text"], 
        max_length=SEQ_LEN, 
        truncation=True, 
        # padding="max_length"  # Removed - let collator handle it
    )
    # Don't set labels here - we'll set them after padding in the training loop
    return tokenized_data

def load_and_prepare_data_stream(skip_rows=0):
    """
    Loads a specific, limited number of examples from the C4 dataset stream,
    and applies tokenization and chunking on-the-fly.
    
    Args:
        skip_rows: Number of rows to skip (for resuming from checkpoint)
    
    Returns:
        Tokenized data stream ready for training
    """
    remaining_rows = TARGET_ROWS - skip_rows
    
    if skip_rows > 0:
        print(f"\n[PHASE 1] Loading C4 stream...")
        print(f"    Skipping {skip_rows:,} already-processed rows...")
        print(f"    Processing remaining {remaining_rows:,} rows to reach {TARGET_ROWS:,} total")
    else:
        print(f"\n[PHASE 1] Loading C4 stream, targeting {TARGET_ROWS:,} rows...")
    
    # 1. Load the dataset in streaming mode
    full_c4_stream = load_dataset(C4_DATASET_NAME, C4_CONFIG, split="train", streaming=True)
    
    # 2. Skip already-processed rows and take remaining
    if skip_rows > 0:
        # Skip the rows we've already trained on
        subset_c4 = full_c4_stream.skip(skip_rows).take(remaining_rows)
    else:
        subset_c4 = full_c4_stream.take(TARGET_ROWS)
    
    # 3. Map tokenization/chunking over the stream (tokenize_and_chunk is now at module level)
    tokenized_stream = subset_c4.map(
        tokenize_and_chunk,
        remove_columns=['text', 'timestamp', 'url'], # Remove original columns
        batched=True,
        batch_size=1000 # Use a large batch size for tokenization speed
    )
    
    # 5. Convert to PyTorch format
    # tokenized_stream = tokenized_stream.with_format("torch")
    
    return tokenized_stream


# --- LEARNING RATE SCHEDULER SETUP ---
def create_lr_scheduler(optimizer, start_rows, base_lr=5e-5, min_lr=5e-6):
    """
    Creates and configures a learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        start_rows: Number of rows already processed (for resuming training)
        base_lr: Base learning rate (default: 5e-5)
        min_lr: Minimum learning rate for cosine decay (default: 5e-6)
    
    Returns:
        Configured scheduler, fast-forwarded to the current training step
    """
    # Calculate steps already completed
    completed_steps = int((start_rows / BATCH_SIZE) / GRADIENT_ACCUMULATION_STEPS)
    
    # Total steps for full dataset
    total_steps = int((TARGET_ROWS / BATCH_SIZE) / GRADIENT_ACCUMULATION_STEPS)
    remaining_steps = total_steps - completed_steps
    
    # Warmup: 3% of total training or minimum 1000 steps
    warmup_steps = max(1000, int(0.03 * total_steps))
    warmup_remaining = max(0, warmup_steps - completed_steps)
    
    print(f"\n--- Learning Rate Schedule Configuration ---")
    print(f"  Completed steps: {completed_steps:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Remaining steps: {remaining_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,} (remaining: {warmup_remaining:,})")
    
    if warmup_remaining > 0:
        # Still in warmup phase
        print(f"  Current phase: Warmup (LR increasing)")
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of base LR
            end_factor=1.0,    # End at 100% of base LR
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr  # Minimum LR = 10% of base LR
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        # Fast-forward scheduler to current step
        for _ in range(completed_steps):
            scheduler.step()
    else:
        # Past warmup, use cosine decay
        print(f"  Current phase: Cosine Decay (LR decreasing)")
        cosine_steps = total_steps - warmup_steps
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=min_lr  # Minimum LR = 10% of base LR
        )
        # Fast-forward to correct position in cosine schedule
        cosine_completed = completed_steps - warmup_steps
        for _ in range(cosine_completed):
            scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Current learning rate: {current_lr:.2e}")
    print(f"  Final learning rate will be: {min_lr:.2e}\n")
    
    return scheduler


# --- 4. MAIN TRAINING LOOP ---
def train_model():
    global total_tokens_processed, total_rows_processed, optimizer
    
    # Load checkpoint before starting (returns both tokens and rows)
    start_tokens, start_rows = load_checkpoint()
    
    # --- LEARNING RATE SCHEDULER SETUP ---
    scheduler = create_lr_scheduler(optimizer, start_rows)
    
    # --- DATA SETUP ---
    # Skip already-processed rows when resuming
    tokenized_stream = load_and_prepare_data_stream(skip_rows=start_rows)
    
    # Create the DataLoader using the default collator for standard tensor conversion
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_stream,
        batch_size=BATCH_SIZE, 
        # collate_fn=default_data_collator, # Handles batching/padding
        collate_fn=data_collator, # Handles batching/padding
        # CRITICAL: 4 workers allows CPU to prep data while GPU trains
        num_workers=4, 
        # CRITICAL: Fast-tracks data transfer to GPU memory
        pin_memory=True, 
        # Optional: Pre-fetches batches to smooth out spikes
        prefetch_factor=2
    )

    # Initialize data iterator. This will reset when the stream is exhausted.
    data_iterator = iter(train_dataloader)
    
    model.train()
    
    # Initialize interval counters for speed calculation
    speed_start_time = time.time()
    tokens_at_last_log = total_tokens_processed
    
    # Track last save point (based on total rows, not session rows)
    rows_at_last_save = total_rows_processed - (total_rows_processed % SAVE_FREQUENCY_ROWS)

    # Track session start time for this training run
    session_start_time = time.time()
    session_start_tokens = total_tokens_processed
    session_start_rows = total_rows_processed
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # Ignore loss on padding
    
    print(f"\n--- Starting Training Session ---")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Max tokens per batch (truncated to SEQ_LEN): {SEQ_LEN}")
    
    # Initialize log file with updated columns
    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rows", "tokens", "loss", "speed", "avg_tokens_per_row"])

    try:
        while total_tokens_processed < TOTAL_TOKENS_REQUIRED:
            
            # --- GRADIENT ACCUMULATION LOOP ---
            for step in range(GRADIENT_ACCUMULATION_STEPS):
                
                # Fetch next batch from the data stream
                try:
                    batch = next(data_iterator)
                    batch["labels"] = batch["input_ids"].clone()
                    total_rows_processed += BATCH_SIZE  # Increment global row counter
                except StopIteration:
                    # Dataset exhausted! All targeted rows are done.
                    print("\n*** C4 Subset Stream Exhausted. Phase 1 Training Complete! ***")
                    raise StopIteration  # Exit the training loop gracefully

                inputs = batch['input_ids'].to(DEVICE)
                targets = batch['labels'].to(DEVICE)
                
                # Count ACTUAL tokens in this batch (not assuming fixed length)
                # Each row may have different lengths due to truncation
                actual_tokens_in_batch = inputs.numel()  # Total tokens in input tensor
                
                # Get output and the balancing loss
                outputs, aux_loss = model(inputs[:, :-1])
                
                # Calculate standard next-token prediction loss
                main_loss = criterion(outputs.view(-1, V_SIZE), targets[:, 1:].reshape(-1))
                
                # Combine them. 
                # Weight 0.01 ensures we don't distract too much from learning language,
                # but it's enough to nudge the router to be fair.
                AUX_LOSS_WEIGHT = 0.01
                total_loss = main_loss + (AUX_LOSS_WEIGHT * aux_loss)
                
                loss = total_loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                # Update token counter with ACTUAL tokens (not assumed fixed value)
                total_tokens_processed += actual_tokens_in_batch
            
            # Clip gradients to prevent exploding gradients (before optimizer step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            # --- OPTIMIZER STEP (Runs after GRADIENT_ACCUMULATION_STEPS) ---
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()
            
            # --- CHECKPOINT SAVING ---
            # Save if we've crossed a save frequency boundary
            if (total_rows_processed - rows_at_last_save) >= SAVE_FREQUENCY_ROWS:
                save_checkpoint(model, optimizer, final_save=False)
                rows_at_last_save = total_rows_processed

            # --- LOGGING (approximately every 100k tokens) ---
            tokens_since_last_log = total_tokens_processed - tokens_at_last_log
            if tokens_since_last_log >= 100_000:
                
                # Calculate speed for this specific interval
                now = time.time()
                interval_seconds = now - speed_start_time
                
                if interval_seconds > 0:
                    current_speed = tokens_since_last_log / interval_seconds
                else:
                    current_speed = 0  # Prevent div by zero
                
                # Reset counters for next interval
                speed_start_time = now
                tokens_at_last_log = total_tokens_processed
                
                # Calculate progress metrics
                percent_complete = (total_rows_processed / TARGET_ROWS) * 100
                avg_tokens_per_row = total_tokens_processed / total_rows_processed if total_rows_processed > 0 else 0
                
                # We multiply loss by GRADIENT_ACCUMULATION_STEPS to show the 'true' average loss,
                # because we divided it earlier for gradient scaling.
                current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # Calculate ETA based on current speed
                remaining_rows = TARGET_ROWS - total_rows_processed
                estimated_remaining_tokens = remaining_rows * avg_tokens_per_row
                if current_speed > 0:
                    eta_seconds = estimated_remaining_tokens / current_speed
                    eta_hours = eta_seconds / 3600
                    eta_str = f"{eta_hours:.1f}h"
                else:
                    eta_str = "N/A"
                
                print(f"Rows: {total_rows_processed:,}/{TARGET_ROWS:,} ({percent_complete:.2f}%) | "
                      f"Tokens: {total_tokens_processed/1e6:.2f}M | "
                      f"Avg tok/row: {avg_tokens_per_row:.0f} | "
                      f"Loss: {current_loss:.4f} | "
                      f"Speed: {current_speed:.0f} t/s | "
                      f"ETA: {eta_str}")
                        
                # Save to CSV with all metrics
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([total_rows_processed, total_tokens_processed, current_loss, current_speed, avg_tokens_per_row])
                        
    except StopIteration:
        pass  # End of dataset, proceed to final save

    # Final save after loop finishes (either by token limit or StopIteration)
    end_time = time.time()
    session_hours = (end_time - session_start_time) / 3600
    session_tokens = total_tokens_processed - session_start_tokens
    session_rows = total_rows_processed - session_start_rows
    
    save_checkpoint(model, optimizer, final_save=True)
    
    print(f"\n--- TRAINING SESSION COMPLETE (Phase 1) ---")
    print(f"This session:")
    print(f"    Duration: {session_hours:.2f} hours")
    print(f"    Rows processed: {session_rows:,}")
    print(f"    Tokens processed: {session_tokens/1e6:.2f}M")
    print(f"\nTotal progress:")
    print(f"    Rows: {total_rows_processed:,} / {TARGET_ROWS:,} ({total_rows_processed/TARGET_ROWS*100:.2f}%)")
    print(f"    Tokens: {total_tokens_processed/1e6:.2f}M")
    if total_rows_processed > 0:
        print(f"    Average tokens per row: {total_tokens_processed/total_rows_processed:.1f}")
    

if __name__ == '__main__':
    # Set multiprocessing start method FIRST - before any CUDA operations
    # This prevents CUDA conflicts with DataLoader workers
    multiprocessing.set_start_method('spawn', force=True)
    
    # --- TOKENIZER SETUP (Using Custom Tokenizer) ---
    # Load the custom tokenizer trained on C4
    TOKENIZER_PATH = "./khanh_tokenizer"
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please run 'python scripts/build_tokenizer.py' first.")
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH) 
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        # max_length removed - sequences already truncated in tokenizer to SEQ_LEN
        return_tensors="pt"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    
    # Initialize Model and Optimizer (CUDA operations - must be after multiprocessing setup)
    print(f"Initializing model on {DEVICE}...")
    model = KhanhLLM().to(DEVICE)
    
    # Compile the model for 10-20% speedup (PyTorch 2.0+)
    print("Compiling model...")
    model = torch.compile(model)
    
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), 
        lr=5e-5, 
        betas=(0.9, 0.95), 
        eps=1e-8
    )
    
    # Register signal handler (needs model and optimizer to be defined)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start training
    train_model()
