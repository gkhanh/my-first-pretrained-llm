import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

# Configuration
VOCAB_SIZE = 50000  # Must match models/khanh_llm.py V_SIZE
BATCH_SIZE = 1000
TRAIN_SIZE = 2_000_000 # Number of examples to use for training the tokenizer (adjust as needed)
SAVE_PATH = "khanh_tokenizer"

def batch_iterator(dataset, batch_size=BATCH_SIZE):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        yield [example["text"] for example in batch]

def build_tokenizer():
    print(f"Loading C4 dataset (first {TRAIN_SIZE} examples)...")
    # Load a small subset of C4 for tokenizer training
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True).take(TRAIN_SIZE)
    
    # We need to materialize the dataset for the tokenizer trainer if we want to batch easily,
    # or we can iterate directly. Converting to list for simplicity with the iterator.
    print("Materializing dataset iterator...")
    data_list = list(dataset) 
    
    # Initialize a BPE tokenizer
    print("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    
    # Use standard pre-tokenization (whitespace split, punctuation, etc.)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder for verifying reconstruction
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train
    print(f"Training tokenizer on {len(data_list)} documents...")
    tokenizer.train_from_iterator(batch_iterator(data_list), trainer=trainer)
    
    # Post-processing (add template for special tokens if needed, usually mostly for BERT/RoBERTa)
    # For a GPT-style model, we just need basic encoding.
    
    # Save simply as a JSON or wrap in Transformers
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    tokenizer.save(os.path.join(SAVE_PATH, "tokenizer.json"))
    print(f"Tokenizer saved to {SAVE_PATH}/tokenizer.json")

    # Save as HuggingFace tokenizer for easy loading in train.py
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    fast_tokenizer.save_pretrained(SAVE_PATH)
    print(f"HuggingFace-compatible tokenizer saved to {SAVE_PATH}/")

if __name__ == "__main__":
    build_tokenizer()

