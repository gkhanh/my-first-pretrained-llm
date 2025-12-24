import os
from dataclasses import dataclass
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast


@dataclass
class TokenizerConfig:
    vocab_size: int = 50000
    batch_size: int = 1000
    train_size: int = 2_000_000
    save_path: str = "khanh_tokenizer"
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    
    special_tokens: list = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


class DataLoader:
    def __init__(self, config: TokenizerConfig):
        self.config = config
    
    def load_dataset(self):
        print(f"Loading {self.config.dataset_name} dataset (first {self.config.train_size} examples)...")
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split="train",
            streaming=True
        ).take(self.config.train_size)
        
        print("Materializing dataset iterator...")
        return list(dataset)
    
    @staticmethod
    def batch_iterator(dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield [example["text"] for example in batch]


class TokenizerBuilder:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizer = None
    
    def initialize(self):
        print("Initializing BPE Tokenizer...")
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
    
    def train(self, data_list):
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=self.config.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        print(f"Training tokenizer on {len(data_list)} documents...")
        self.tokenizer.train_from_iterator(
            DataLoader.batch_iterator(data_list, self.config.batch_size),
            trainer=trainer
        )
    
    def save(self):
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        
        self.tokenizer.save(os.path.join(self.config.save_path, "tokenizer.json"))
        print(f"Tokenizer saved to {self.config.save_path}/tokenizer.json")
    
    def save_huggingface_format(self):
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
        fast_tokenizer.save_pretrained(self.config.save_path)
        print(f"HuggingFace-compatible tokenizer saved to {self.config.save_path}/")


class TokenizerTrainer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.tokenizer_builder = TokenizerBuilder(config)
    
    def build(self):
        data_list = self.data_loader.load_dataset()
        
        self.tokenizer_builder.initialize()
        self.tokenizer_builder.train(data_list)
        self.tokenizer_builder.save()
        self.tokenizer_builder.save_huggingface_format()
        
        print("\n✓ Tokenizer build complete!")


def main():
    config = TokenizerConfig()
    
    trainer = TokenizerTrainer(config)
    trainer.build()


if __name__ == "__main__":
    main()