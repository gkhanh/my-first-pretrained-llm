import torch
import torch.nn as nn
import bitsandbytes as bnb
import time
import os
import signal
import sys
import csv


# Enable TF32 for faster matrix multiplication (High Speed, Low Impact on Quality)
# torch.backends.cuda.matmul.allow_tf32 = True 
# torch.backends.cudnn.allow_tf32 = True
# --- REPLACE THE OLD TF32 LINES WITH THESE ---
# Enable TF32 for faster matrix multiplication (New PyTorch 2.x Syntax)
# This setting tells PyTorch to use TF32 precision for matrix multiplications (Linear layers)
torch.set_float32_matmul_precision('high') 

# Note: 'high' = TF32 (Fastest, good precision)
#       'highest' = FP32 (Slowest, max precision)
#       'medium' = BF16 (Fastest, lower precision)
# ---------------------------------------------

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- NEW IMPORTS for Data Handling ---
from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer, default_data_collator
# --- Custom Model Import ---
from models.khanh_llm import KhanhLLM, V_SIZE

# --- 1. SETUP AND CHECKPOINT CONFIG ---
CHECKPOINT_PATH = "checkpoint_latest.pth.tar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training Parameters
BATCH_SIZE = 1 
SEQ_LEN = 2048
GRADIENT_ACCUMULATION_STEPS = 64
TOTAL_TOKENS_REQUIRED = 999_999_999_999_999 # Effective 'unlimited' for runtime control
# SAVE_FREQUENCY_TOKENS = 10_000_000 
SAVE_FREQUENCY_ROWS = 100_000 # Save every 100k rows 

# Data Parameters for C4 Subset
C4_DATASET_NAME = "allenai/c4"
C4_CONFIG = "en"
TARGET_ROWS = 1_000_000 # 1 million rows

# --- TOKENIZER SETUP (Using Custom Tokenizer) ---
# Load the custom tokenizer trained on C4
TOKENIZER_PATH = "./khanh_tokenizer"
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please run 'python scripts/build_tokenizer.py' first.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH) 
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

# Initialize Model and Optimizer
model = KhanhLLM().to(DEVICE)
# Compile the model for 10-20% speedup (PyTorch 2.0+)
# We use 'default' mode which is the most stable.
model = torch.compile(model)
# CRITICAL: Resize embeddings if tokenizer size != model V_SIZE (should be equal now)
# model.resize_token_embeddings(len(tokenizer)) 
optimizer = bnb.optim.AdamW8bit(
    model.parameters(), 
    lr=5e-5, 
    betas=(0.9, 0.95), 
    eps=1e-8
)

# --- 2. CHECKPOINT FUNCTIONS (No changes needed) ---
# ... (save_checkpoint and load_checkpoint functions remain the same) ...

def save_checkpoint(model, optimizer, total_tokens_processed, final_save=False):
    """Saves the essential training states to a file."""
    state = {
        'total_tokens_processed': total_tokens_processed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, CHECKPOINT_PATH)
    status = "FINAL" if final_save else "INTERRUPT"
    print(f"\n--- {status} CHECKPOINT SAVED at {total_tokens_processed/1e6:.2f}M TOKENS ---")

def load_checkpoint():
    """Loads the model and optimizer states if a checkpoint file exists."""
    start_tokens = 0
    
    if os.path.isfile(CHECKPOINT_PATH):
        print(f"--> Checkpoint found at {CHECKPOINT_PATH}. Loading states...")
        
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_tokens = checkpoint['total_tokens_processed']
            print(f"--> Resuming training from {start_tokens/1e6:.2f}M tokens.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            if os.path.exists(CHECKPOINT_PATH):
                os.remove(CHECKPOINT_PATH)
    else:
        print("--> No checkpoint found. Starting training from scratch.")
        
    return start_tokens

# --- 3. SIGNAL HANDLER (No changes needed) ---
# ... (signal_handler and signal.signal registration remain the same) ...

def signal_handler(sig, frame):
    """Handles Ctrl+C (SIGINT) signal to save a checkpoint before exiting."""
    print('\nCtrl+C detected! Saving interrupt checkpoint...')
    # Use global variables available in train_model() scope
    save_checkpoint(model, optimizer, total_tokens_processed, final_save=False)
    sys.exit(0)

# Register the signal handler: This is what makes Ctrl+C safe!
signal.signal(signal.SIGINT, signal_handler)

# --- NEW: DATA LOADING FUNCTION ---

def load_and_prepare_data_stream():
    """
    Loads a specific, limited number of examples from the C4 dataset stream,
    and applies tokenization and chunking on-the-fly.
    """
    print(f"\n[PHASE 1] Loading C4 stream, targeting first {TARGET_ROWS:,} rows...")
    
    # 1. Load the dataset in streaming mode
    full_c4_stream = load_dataset(C4_DATASET_NAME, C4_CONFIG, split="train", streaming=True)
    
    # 2. Use .take() to grab only the first N examples from the stream
    subset_c4 = full_c4_stream.take(TARGET_ROWS)
    
    # 3. Define the tokenization/chunking function
    def tokenize_and_chunk(examples):
        # We perform simple tokenization, ensuring max_length is met.
        # Targets are simply the input tokens shifted by one (for next-token prediction).
        tokenized_data = tokenizer(
            examples["text"], 
            max_length=SEQ_LEN, 
            truncation=True, 
            # padding="max_length" # <-- DISABLED FOR SPEED (Finish in <10h)
        )
        tokenized_data["labels"] = tokenized_data["input_ids"]
        return tokenized_data

    # 4. Map tokenization/chunking over the stream
    tokenized_stream = subset_c4.map(
        tokenize_and_chunk,
        remove_columns=['text', 'timestamp', 'url'], # Remove original columns
        batched=True,
        batch_size=1000 # Use a large batch size for tokenization speed
    )
    
    # 5. Convert to PyTorch format
    tokenized_stream = tokenized_stream.with_format("torch")
    
    return tokenized_stream

# --- 4. MAIN TRAINING LOOP (Modified for Streaming Data) ---

# REMOVE get_dummy_data - it is no longer used.

def train_model():
    global total_tokens_processed
    
    # Load checkpoint before starting
    total_tokens_processed = load_checkpoint()
    
    # --- DATA SETUP ---
    tokenized_stream = load_and_prepare_data_stream()
    
    # Create the DataLoader using the default collator for standard tensor conversion
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_stream,
        batch_size=BATCH_SIZE, 
        collate_fn=default_data_collator # Handles batching/padding
    )

    # Initialize data iterator. This will reset when the stream is exhausted.
    data_iterator = iter(train_dataloader)
    
    model.train()
    
    # Initialize interval counters
    speed_start_time = time.time()
    tokens_at_last_log = total_tokens_processed
    
    # --- NEW: Row Counter ---
    # We start from 0 for this session because we don't know the exact previous row count.
    rows_processed_count = 0 
    
    # Track last save to ensure we don't miss intervals due to step jumps
    rows_at_last_save = rows_processed_count - (rows_processed_count % SAVE_FREQUENCY_ROWS)

    # Initialize start_time for final total time calculation
    # We estimate start time based on current progress if resuming
    # This is only for the final print statement, not for speed calculation
    start_time_total = time.time() - (total_tokens_processed / 5000) 
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) # Ignore loss on padding
    
    TOKENS_PER_ACCUM_STEP = BATCH_SIZE * (SEQ_LEN - 1) * GRADIENT_ACCUMULATION_STEPS
    print(f"Starting training with Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Tokens per effective step: {TOKENS_PER_ACCUM_STEP:,}")
    
    # Initialize log file
    log_file = "training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token_count", "loss", "speed"])

    try:
        while total_tokens_processed < TOTAL_TOKENS_REQUIRED:
            
            # --- GRADIENT ACCUMULATION LOOP ---
            for step in range(GRADIENT_ACCUMULATION_STEPS):
                
                # Fetch next batch from the data stream
                try:
                    batch = next(data_iterator)
                    rows_processed_count += BATCH_SIZE # <--- Increment Row Counter
                except StopIteration:
                    # Dataset exhausted! 750k rows are done.
                    print("\n*** C4 Subset Stream Exhausted. Phase 1 Training Complete! ***")
                    raise StopIteration # Exit the training loop gracefully

                inputs = batch['input_ids'].to(DEVICE)
                targets = batch['labels'].to(DEVICE)
                
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
                
                # Update token counter inside the accumulation loop (for accurate logging)
                token_count_in_step = (BATCH_SIZE * (SEQ_LEN - 1))
                total_tokens_processed += token_count_in_step
                
            # --- OPTIMIZER STEP (Runs after GRADIENT_ACCUMULATION_STEPS) ---
            optimizer.step()
            optimizer.zero_grad()
            
            # --- FIX 2: Calculate Instantaneous Speed ---
            
            # Checkpoint Saving and Logging
            # Robust check: Save if we've processed enough rows since last save
            if (rows_processed_count - rows_at_last_save) >= SAVE_FREQUENCY_ROWS:
                save_checkpoint(model, optimizer, total_tokens_processed, rows_processed_count)
                rows_at_last_save = rows_processed_count

            # --- LOGGING ---
            if total_tokens_processed % 100_000 < TOKENS_PER_ACCUM_STEP:
                
                # Calculate speed for this specific interval
                now = time.time()
                interval_seconds = now - speed_start_time
                tokens_in_interval = total_tokens_processed - tokens_at_last_log
                
                if interval_seconds > 0:
                    current_speed = tokens_in_interval / interval_seconds
                else:
                    current_speed = 0 # Prevent div by zero
                
                # Reset counters
                speed_start_time = now
                tokens_at_last_log = total_tokens_processed
                
                # --- NEW: Row-Based Progress ---
                percent_complete = (rows_processed_count / TARGET_ROWS) * 100
                
                # We multiply loss by GRADIENT_ACCUMULATION_STEPS to show the 'true' average loss,
                # because we divided it earlier for gradient scaling.
                current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                print(f"Rows: {rows_processed_count:,} / {TARGET_ROWS:,} ({percent_complete:.2f}%) | "
                      f"Tokens: {total_tokens_processed/1e6:.2f}M | "
                      f"Loss: {current_loss:.4f} | "
                      f"Speed: {current_speed:.0f} t/s")
                        
                # --- NEW: Save to CSV ---
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([total_tokens_processed, current_loss, current_speed])
                        
    except StopIteration:
        pass # End of dataset, proceed to final save

    # Final save after loop finishes (either by token limit or StopIteration)
    end_time = time.time()
    total_hours = (end_time - start_time_total) / 3600
    save_checkpoint(model, optimizer, total_tokens_processed, final_save=True)
    
    print(f"\n--- TRAINING COMPLETE (Phase 1) ---")
    print(f"Total time taken: {total_hours:.2f} hours.")
    
if __name__ == '__main__':
    train_model()
