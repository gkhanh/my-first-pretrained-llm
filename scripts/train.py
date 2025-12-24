import torch
import torch.nn as nn
import bitsandbytes as bnb
import time
import os
import signal
import sys
import csv
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader
from models.khanh_llm import KhanhLLM, V_SIZE
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


@dataclass
class Config:
    checkpoint_path: str = "checkpoint_latest.pth.tar"
    tokenizer_path: str = "./khanh_tokenizer"
    log_file: str = "training_log.csv"
    
    batch_size: int = 2
    seq_len: int = 2048
    gradient_accumulation_steps: int = 64
    total_tokens_required: int = 999_999_999_999_999
    save_frequency_rows: int = 100_000
    log_frequency_tokens: int = 100_000
    
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    target_rows: int = 10_000_000
    
    base_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_factor: float = 0.03
    min_warmup_steps: int = 1000
    warmup_start_factor: float = 0.1
    
    aux_loss_weight: float = 0.01
    max_grad_norm: float = 1.0
    
    num_workers: int = 4
    prefetch_factor: int = 2
    tokenization_batch_size: int = 1000
    
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataManager:
    _tokenizer_cache = None
    _tokenizer_path = None
    _seq_len = None

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.data_collator = None
        self._initialize_tokenizer()
        self._setup_global_tokenizer()

    def _initialize_tokenizer(self):
        if not os.path.exists(self.config.tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {self.config.tokenizer_path}. "
                "Please run 'python scripts/build_tokenizer.py' first."
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

    def _setup_global_tokenizer(self):
        DataManager._tokenizer_path = self.config.tokenizer_path
        DataManager._seq_len = self.config.seq_len

    @staticmethod
    def _get_tokenizer():
        if DataManager._tokenizer_cache is None:
            # Use default path if not set (for multiprocessing workers)
            tokenizer_path = DataManager._tokenizer_path or "./khanh_tokenizer"
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(
                    f"Tokenizer not found at {tokenizer_path}. "
                    "Please run 'python scripts/build_tokenizer.py' first."
                )
            DataManager._tokenizer_cache = AutoTokenizer.from_pretrained(tokenizer_path)
            if DataManager._tokenizer_cache.pad_token is None:
                DataManager._tokenizer_cache.add_special_tokens({'pad_token': '[PAD]'})
        return DataManager._tokenizer_cache

    @staticmethod
    def tokenize_and_chunk(examples):
        tok = DataManager._get_tokenizer()
        # Use default seq_len if not set (for multiprocessing workers)
        seq_len = DataManager._seq_len or 2048
        return tok(
            examples["text"],
            max_length=seq_len,
            truncation=True,
        )

    def load_data_stream(self, skip_rows: int = 0):
        remaining_rows = self.config.target_rows - skip_rows
        
        if skip_rows > 0:
            print(f"\n[PHASE 1] Loading C4 stream...")
            print(f"    Skipping {skip_rows:,} already-processed rows...")
            print(f"    Processing remaining {remaining_rows:,} rows to reach {self.config.target_rows:,} total")
        else:
            print(f"\n[PHASE 1] Loading C4 stream, targeting {self.config.target_rows:,} rows...")
        
        full_stream = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split="train",
            streaming=True
        )
        
        if skip_rows > 0:
            subset = full_stream.skip(skip_rows).take(remaining_rows)
        else:
            subset = full_stream.take(self.config.target_rows)
        
        tokenized_stream = subset.map(
            self.tokenize_and_chunk,
            remove_columns=['text', 'timestamp', 'url'],
            batched=True,
            batch_size=self.config.tokenization_batch_size
        )
        
        return tokenized_stream

    def create_dataloader(self, tokenized_stream):
        return DataLoader(
            tokenized_stream,
            batch_size=self.config.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=self.config.prefetch_factor
        )


class CheckpointManager:
    def __init__(self, config: Config, model: nn.Module, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.total_tokens_processed = 0
        self.total_rows_processed = 0

    def save(self, final_save: bool = False):
        model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        optimizer_state = self.optimizer.state_dict()
        
        state = {
            'total_tokens_processed': self.total_tokens_processed,
            'total_rows_processed': self.total_rows_processed,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
        }
        
        try:
            torch.save(state, self.config.checkpoint_path, _use_new_zipfile_serialization=False)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
            print("Progress may be lost, but training can continue.")
            return

        status = "FINAL" if final_save else "PERIODIC"
        avg_tokens_per_row = self.total_tokens_processed / self.total_rows_processed if self.total_rows_processed > 0 else 0
        
        print(f"\n--- {status} CHECKPOINT SAVED ---")
        print(f"    Rows: {self.total_rows_processed:,} / {self.config.target_rows:,} ({self.total_rows_processed/self.config.target_rows*100:.2f}%)")
        print(f"    Tokens: {self.total_tokens_processed/1e6:.2f}M")
        print(f"    Avg tokens/row: {avg_tokens_per_row:.1f}")

    def load(self) -> Tuple[int, int]:
        start_tokens = 0
        start_rows = 0
        
        if os.path.isfile(self.config.checkpoint_path):
            print(f"--> Checkpoint found at {self.config.checkpoint_path}. Loading states...")
            
            try:
                checkpoint = torch.load(self.config.checkpoint_path, map_location=self.config.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                start_tokens = checkpoint.get('total_tokens_processed', 0)
                start_rows = checkpoint.get('total_rows_processed', 0)
                
                avg_tokens = start_tokens / start_rows if start_rows > 0 else 0
                
                print(f"--> Resuming training from:")
                print(f"    Rows: {start_rows:,} / {self.config.target_rows:,} ({start_rows/self.config.target_rows*100:.2f}%)")
                print(f"    Tokens: {start_tokens/1e6:.2f}M")
                print(f"    Historical avg tokens/row: {avg_tokens:.1f}")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                if os.path.exists(self.config.checkpoint_path):
                    os.remove(self.config.checkpoint_path)
                start_tokens = 0
                start_rows = 0
        else:
            print("--> No checkpoint found. Starting training from scratch.")
        
        self.total_tokens_processed = start_tokens
        self.total_rows_processed = start_rows
        
        return start_tokens, start_rows


class LRSchedulerManager:
    def __init__(self, config: Config):
        self.config = config

    def create_scheduler(self, optimizer, start_rows: int):
        completed_steps = int((start_rows / self.config.batch_size) / self.config.gradient_accumulation_steps)
        total_steps = int((self.config.target_rows / self.config.batch_size) / self.config.gradient_accumulation_steps)
        remaining_steps = total_steps - completed_steps
        
        warmup_steps = max(self.config.min_warmup_steps, int(self.config.warmup_factor * total_steps))
        warmup_remaining = max(0, warmup_steps - completed_steps)
        
        print(f"\n--- Learning Rate Schedule Configuration ---")
        print(f"  Completed steps: {completed_steps:,}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Remaining steps: {remaining_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,} (remaining: {warmup_remaining:,})")
        
        if warmup_remaining > 0:
            print(f"  Current phase: Warmup (LR increasing)")
            scheduler = self._create_warmup_scheduler(optimizer, warmup_steps, total_steps, completed_steps)
        else:
            print(f"  Current phase: Cosine Decay (LR decreasing)")
            scheduler = self._create_cosine_scheduler(optimizer, warmup_steps, total_steps, completed_steps)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current learning rate: {current_lr:.2e}")
        print(f"  Final learning rate will be: {self.config.min_lr:.2e}\n")
        
        return scheduler

    def _create_warmup_scheduler(self, optimizer, warmup_steps, total_steps, completed_steps):
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.config.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.min_lr
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        for _ in range(completed_steps):
            scheduler.step()
        return scheduler

    def _create_cosine_scheduler(self, optimizer, warmup_steps, total_steps, completed_steps):
        cosine_steps = total_steps - warmup_steps
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.config.min_lr
        )
        cosine_completed = completed_steps - warmup_steps
        for _ in range(cosine_completed):
            scheduler.step()
        return scheduler


class ProgressTracker:
    def __init__(self, config: Config):
        self.config = config
        self._initialize_log_file()

    def _initialize_log_file(self):
        if not os.path.exists(self.config.log_file):
            with open(self.config.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rows", "tokens", "loss", "speed", "avg_tokens_per_row"])

    def log_progress(
        self,
        rows: int,
        tokens: int,
        loss: float,
        speed: float,
        avg_tokens_per_row: float,
        percent_complete: float,
        eta_str: str
    ):
        self._print_progress(rows, tokens, loss, speed, avg_tokens_per_row, percent_complete, eta_str)
        self._save_to_csv(rows, tokens, loss, speed, avg_tokens_per_row)

    def _print_progress(
        self,
        rows: int,
        tokens: int,
        loss: float,
        speed: float,
        avg_tokens_per_row: float,
        percent_complete: float,
        eta_str: str
    ):
        print(f"Rows: {rows:,}/{self.config.target_rows:,} ({percent_complete:.2f}%) | "
              f"Tokens: {tokens/1e6:.2f}M | "
              f"Avg tok/row: {avg_tokens_per_row:.0f} | "
              f"Loss: {loss:.4f} | "
              f"Speed: {speed:.0f} t/s | "
              f"ETA: {eta_str}")

    def _save_to_csv(self, rows: int, tokens: int, loss: float, speed: float, avg_tokens_per_row: float):
        with open(self.config.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rows, tokens, loss, speed, avg_tokens_per_row])

    def print_session_start(self, effective_batch_size: int, seq_len: int):
        print(f"\n--- Starting Training Session ---")
        print(f"Effective Batch Size: {effective_batch_size}")
        print(f"Max tokens per batch (truncated to SEQ_LEN): {seq_len}")

    def print_session_complete(
        self,
        session_hours: float,
        session_rows: int,
        session_tokens: int,
        total_rows: int,
        total_tokens: int,
        avg_tokens_per_row: float
    ):
        print(f"\n--- TRAINING SESSION COMPLETE (Phase 1) ---")
        print(f"This session:")
        print(f"    Duration: {session_hours:.2f} hours")
        print(f"    Rows processed: {session_rows:,}")
        print(f"    Tokens processed: {session_tokens/1e6:.2f}M")
        print(f"\nTotal progress:")
        print(f"    Rows: {total_rows:,} / {self.config.target_rows:,} ({total_rows/self.config.target_rows*100:.2f}%)")
        print(f"    Tokens: {total_tokens/1e6:.2f}M")
        if total_rows > 0:
            print(f"    Average tokens per row: {avg_tokens_per_row:.1f}")


class TrainingLoop:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        optimizer,
        scheduler,
        checkpoint_manager: CheckpointManager,
        data_manager: DataManager,
        progress_tracker: ProgressTracker
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.data_manager = data_manager
        self.progress_tracker = progress_tracker
        self.criterion = nn.CrossEntropyLoss(ignore_index=data_manager.tokenizer.pad_token_id)

    def run(self):
        start_tokens, start_rows = self.checkpoint_manager.load()
        
        tokenized_stream = self.data_manager.load_data_stream(skip_rows=start_rows)
        train_dataloader = self.data_manager.create_dataloader(tokenized_stream)
        data_iterator = iter(train_dataloader)
        
        self.model.train()
        
        self._initialize_tracking_variables()
        
        self.progress_tracker.print_session_start(
            self.config.batch_size * self.config.gradient_accumulation_steps,
            self.config.seq_len
        )

        try:
            self._training_loop(data_iterator)
        except StopIteration:
            pass

        self._finalize_training()

    def _initialize_tracking_variables(self):
        self.speed_start_time = time.time()
        self.tokens_at_last_log = self.checkpoint_manager.total_tokens_processed
        self.rows_at_last_save = (
            self.checkpoint_manager.total_rows_processed - 
            (self.checkpoint_manager.total_rows_processed % self.config.save_frequency_rows)
        )
        self.session_start_time = time.time()
        self.session_start_tokens = self.checkpoint_manager.total_tokens_processed
        self.session_start_rows = self.checkpoint_manager.total_rows_processed

    def _training_loop(self, data_iterator):
        while self.checkpoint_manager.total_tokens_processed < self.config.total_tokens_required:
            self.current_loss = self._gradient_accumulation_step(data_iterator)
            self._optimizer_step()
            self._checkpoint_if_needed()
            self._log_if_needed()

    def _gradient_accumulation_step(self, data_iterator):
        for step in range(self.config.gradient_accumulation_steps):
            batch = self._get_next_batch(data_iterator)
            loss = self._forward_pass(batch)
            loss.backward()
            self._update_counters(batch)
            if step == 0:
                self.current_loss = loss.detach()
            else:
                self.current_loss = (self.current_loss * step + loss.detach()) / (step + 1)
        return self.current_loss

    def _get_next_batch(self, data_iterator):
        try:
            batch = next(data_iterator)
            batch["labels"] = batch["input_ids"].clone()
            self.checkpoint_manager.total_rows_processed += self.config.batch_size
            return batch
        except StopIteration:
            print("\n*** C4 Subset Stream Exhausted. Phase 1 Training Complete! ***")
            raise StopIteration

    def _forward_pass(self, batch):
        inputs = batch['input_ids'].to(self.config.device)
        targets = batch['labels'].to(self.config.device)
        
        outputs, aux_loss = self.model(inputs[:, :-1])
        main_loss = self.criterion(outputs.view(-1, V_SIZE), targets[:, 1:].reshape(-1))
        total_loss = main_loss + (self.config.aux_loss_weight * aux_loss)
        
        return total_loss / self.config.gradient_accumulation_steps

    def _update_counters(self, batch):
        inputs = batch['input_ids'].to(self.config.device)
        actual_tokens_in_batch = inputs.numel()
        self.checkpoint_manager.total_tokens_processed += actual_tokens_in_batch

    def _optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def _checkpoint_if_needed(self):
        if (self.checkpoint_manager.total_rows_processed - self.rows_at_last_save) >= self.config.save_frequency_rows:
            self.checkpoint_manager.save(final_save=False)
            self.rows_at_last_save = self.checkpoint_manager.total_rows_processed

    def _log_if_needed(self):
        tokens_since_last_log = self.checkpoint_manager.total_tokens_processed - self.tokens_at_last_log
        if tokens_since_last_log >= self.config.log_frequency_tokens:
            self._perform_logging(tokens_since_last_log)

    def _perform_logging(self, tokens_since_last_log):
        now = time.time()
        interval_seconds = now - self.speed_start_time
        
        if interval_seconds > 0:
            current_speed = tokens_since_last_log / interval_seconds
        else:
            current_speed = 0
        
        self.speed_start_time = now
        self.tokens_at_last_log = self.checkpoint_manager.total_tokens_processed
        
        percent_complete = (self.checkpoint_manager.total_rows_processed / self.config.target_rows) * 100
        avg_tokens_per_row = (
            self.checkpoint_manager.total_tokens_processed / self.checkpoint_manager.total_rows_processed
            if self.checkpoint_manager.total_rows_processed > 0 else 0
        )
        
        if self.current_loss is None:
            return  # Skip logging if loss hasn't been computed yet
        current_loss = self.current_loss.item() * self.config.gradient_accumulation_steps
        eta_str = self._calculate_eta(current_speed, avg_tokens_per_row)
        
        self.progress_tracker.log_progress(
            self.checkpoint_manager.total_rows_processed,
            self.checkpoint_manager.total_tokens_processed,
            current_loss,
            current_speed,
            avg_tokens_per_row,
            percent_complete,
            eta_str
        )

    def _calculate_eta(self, current_speed, avg_tokens_per_row):
        if current_speed > 0:
            remaining_rows = self.config.target_rows - self.checkpoint_manager.total_rows_processed
            estimated_remaining_tokens = remaining_rows * avg_tokens_per_row
            eta_seconds = estimated_remaining_tokens / current_speed
            eta_hours = eta_seconds / 3600
            return f"{eta_hours:.1f}h"
        return "N/A"

    def _finalize_training(self):
        end_time = time.time()
        session_hours = (end_time - self.session_start_time) / 3600
        session_tokens = self.checkpoint_manager.total_tokens_processed - self.session_start_tokens
        session_rows = self.checkpoint_manager.total_rows_processed - self.session_start_rows
        
        self.checkpoint_manager.save(final_save=True)
        
        self.progress_tracker.print_session_complete(
            session_hours,
            session_rows,
            session_tokens,
            self.checkpoint_manager.total_rows_processed,
            self.checkpoint_manager.total_tokens_processed,
            self.checkpoint_manager.total_tokens_processed / self.checkpoint_manager.total_rows_processed
            if self.checkpoint_manager.total_rows_processed > 0 else 0
        )


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.checkpoint_manager = None
        self.data_manager = None
        self.scheduler_manager = None
        self.progress_tracker = None

    def setup(self):
        self._setup_multiprocessing()
        self._setup_data_manager()
        self._setup_model()
        self._setup_optimizer()
        self._setup_checkpoint_manager()
        self._setup_scheduler_manager()
        self._setup_progress_tracker()
        self._setup_signal_handler()

    def _setup_multiprocessing(self):
        multiprocessing.set_start_method('spawn', force=True)

    def _setup_data_manager(self):
        self.data_manager = DataManager(self.config)

    def _setup_model(self):
        print(f"Initializing model on {self.config.device}...")
        self.model = KhanhLLM().to(self.config.device)
        print("Compiling model...")
        self.model = torch.compile(self.model)

    def _setup_optimizer(self):
        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=self.config.base_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps
        )

    def _setup_checkpoint_manager(self):
        self.checkpoint_manager = CheckpointManager(self.config, self.model, self.optimizer)

    def _setup_scheduler_manager(self):
        self.scheduler_manager = LRSchedulerManager(self.config)

    def _setup_progress_tracker(self):
        self.progress_tracker = ProgressTracker(self.config)

    def _setup_signal_handler(self):
        def signal_handler(sig, frame):
            print('\nCtrl+C detected! Saving interrupt checkpoint...')
            try:
                self.checkpoint_manager.save(final_save=False)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                try:
                    torch.cuda.synchronize()
                    self.checkpoint_manager.save(final_save=False)
                except Exception as e2:
                    print(f"Failed to save checkpoint: {e2}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)

    def train(self):
        start_tokens, start_rows = self.checkpoint_manager.load()
        scheduler = self.scheduler_manager.create_scheduler(self.optimizer, start_rows)
        
        training_loop = TrainingLoop(
            self.config,
            self.model,
            self.optimizer,
            scheduler,
            self.checkpoint_manager,
            self.data_manager,
            self.progress_tracker
        )
        
        training_loop.run()


def main():
    config = Config()
    print(f"Using device: {config.device}")
    
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    main()
