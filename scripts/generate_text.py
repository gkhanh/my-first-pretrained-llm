import torch
import sys
import os
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.khanh_llm import KhanhLLM
from transformers import AutoTokenizer


@dataclass
class GenerationConfig:
    checkpoint_path: str = "checkpoint_latest.pth.tar"
    tokenizer_path: str = "khanh_tokenizer"
    max_seq_length: int = 2048
    
    default_max_tokens: int = 200
    default_min_tokens: int = 10
    default_temperature: float = 0.6
    default_top_k: int = 40
    default_repetition_penalty: float = 1.2
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelLoader:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load(self):
        self._load_tokenizer()
        self._load_model()
        return self.model, self.tokenizer
    
    def _load_tokenizer(self):
        print(f"Loading tokenizer from {self.config.tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
    
    def _load_model(self):
        print("Initializing model structure...")
        self.model = KhanhLLM().to(self.config.device)
        
        if os.path.exists(self.config.checkpoint_path):
            print(f"Loading weights from {self.config.checkpoint_path}...")
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.config.device, weights_only=False)
            
            state_dict = self._process_state_dict(checkpoint['model_state_dict'])
            self.model.load_state_dict(state_dict)
            
            self._print_training_info(checkpoint)
            print("Model loaded successfully!")
        else:
            print("WARNING: No checkpoint found! Using random weights (expect gibberish).")
        
        self.model.eval()
        self._compile_model()
    
    def _process_state_dict(self, state_dict):
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        return state_dict
    
    def _print_training_info(self, checkpoint):
        if 'total_rows_processed' in checkpoint:
            rows = checkpoint.get('total_rows_processed', 0)
            tokens = checkpoint.get('total_tokens_processed', 0)
            print(f"Model trained on: {rows:,} rows, {tokens/1e6:.2f}M tokens")
    
    def _compile_model(self):
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled for faster inference.")
        except Exception as e:
            print(f"Note: Model compilation skipped ({e})")


class TextGenerator:
    def __init__(self, config: GenerationConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.sentence_endings = {'.', '!', '?'}
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        min_new_tokens: int = None,
        temperature: float = None,
        top_k: int = None,
        top_p: float = 0.9,
        repetition_penalty: float = None
    ) -> Tuple[str, str]:
        max_new_tokens = max_new_tokens or self.config.default_max_tokens
        min_new_tokens = min_new_tokens or self.config.default_min_tokens
        temperature = temperature or self.config.default_temperature
        top_k = top_k or self.config.default_top_k
        repetition_penalty = repetition_penalty or self.config.default_repetition_penalty
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)
        original_length = input_ids.shape[1]
        all_tokens = input_ids[0].tolist()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, _ = self.model(input_ids)
                next_token_logits = logits[:, -1, :].clone() / temperature
                
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, all_tokens, repetition_penalty
                )
                next_token_logits = self._apply_top_k_filtering(next_token_logits, top_k)
                next_token_logits = self._apply_top_p_filtering(next_token_logits, top_p)
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                all_tokens.append(next_token_id)
                
                if self._should_stop(all_tokens, next_token_id, input_ids, original_length, step, max_new_tokens, min_new_tokens):
                    break
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated_ids = input_ids[0, original_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return full_text, generated_text
    
    def _apply_repetition_penalty(self, logits, all_tokens, repetition_penalty):
        if repetition_penalty != 1.0:
            recent_tokens = set(all_tokens[-50:]) if len(all_tokens) > 50 else set(all_tokens)
            for token_id in recent_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
        return logits
    
    def _apply_top_k_filtering(self, logits, top_k):
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        return logits
    
    def _apply_top_p_filtering(self, logits, top_p):
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        return logits
    
    def _should_stop(self, all_tokens, next_token_id, input_ids, original_length, step, max_new_tokens, min_new_tokens):
        if len(all_tokens) >= 3 and all_tokens[-1] == all_tokens[-2] == all_tokens[-3]:
            return True
        
        if self.tokenizer.eos_token_id is not None and next_token_id == self.tokenizer.eos_token_id:
            return True
        
        if input_ids.shape[1] >= self.config.max_seq_length:
            return True
        
        generated_count = len(all_tokens) - original_length
        check_frequency = 3 if generated_count < max_new_tokens * 0.8 else 1
        should_check = (generated_count >= min_new_tokens and 
                       (generated_count % check_frequency == 0 or step == max_new_tokens - 1))
        
        if should_check:
            return self._check_sentence_completion(input_ids, original_length)
        
        return False
    
    def _check_sentence_completion(self, input_ids, original_length):
        generated_ids = input_ids[0, original_length:]
        current_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if current_text.strip():
            text_stripped = current_text.rstrip()
            if text_stripped and text_stripped[-1] in self.sentence_endings:
                if len(text_stripped) > 2:
                    chars_before = text_stripped[-4:-1] if len(text_stripped) >= 4 else text_stripped[:-1]
                    chars_before_clean = chars_before.replace(' ', '').replace('.', '')
                    if chars_before_clean and chars_before_clean.isupper() and not any(c.islower() for c in chars_before_clean):
                        return False
                    return True
                return True
        return False


class GenerationSettings:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.max_tokens = config.default_max_tokens
        self.min_tokens = config.default_min_tokens
        self.temperature = config.default_temperature
        self.top_k = config.default_top_k
        self.repetition_penalty = config.default_repetition_penalty
    
    def display(self):
        print(f"\nCurrent settings:")
        print(f"  Max tokens: {self.max_tokens} (hard limit, but will stop at sentence completion)")
        print(f"  Min tokens: {self.min_tokens} (minimum before checking for sentence completion)")
        print(f"  Temperature: {self.temperature} (lower = more focused, higher = more creative)")
        print(f"  Top-k: {self.top_k}")
        print(f"  Repetition penalty: {self.repetition_penalty} (1.0 = no penalty, >1.0 = less repetition)")
    
    def update_interactive(self):
        try:
            max_tokens_input = input("  New max tokens (or Enter to skip): ")
            if max_tokens_input:
                self.max_tokens = int(max_tokens_input)
            
            min_tokens_input = input("  New min tokens (or Enter to skip): ")
            if min_tokens_input:
                self.min_tokens = int(min_tokens_input)
            
            temp_input = input("  New temperature (or Enter to skip): ")
            if temp_input:
                self.temperature = float(temp_input)
            
            top_k_input = input("  New top-k (or Enter to skip): ")
            if top_k_input:
                self.top_k = int(top_k_input)
            
            rep_penalty_input = input("  New repetition penalty (or Enter to skip): ")
            if rep_penalty_input:
                self.repetition_penalty = float(rep_penalty_input)
            
            print("Settings updated!")
        except ValueError:
            print("Invalid input, keeping current settings.")


class InteractiveGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.generator = None
        self.settings = GenerationSettings(config)
    
    def start(self):
        self._print_header()
        model, tokenizer = self.model_loader.load()
        self.generator = TextGenerator(self.config, model, tokenizer)
        self._print_instructions()
        self._run_interactive_loop()
    
    def _print_header(self):
        print("=" * 60)
        print("  KhanhLLM Text Generator")
        print("=" * 60)
    
    def _print_instructions(self):
        print("\n" + "=" * 60)
        print("Ready to generate text!")
        print("Commands:")
        print("  - Type your prompt and press Enter to generate")
        print("  - Type 'q' or 'exit' to quit")
        print("  - Type 'settings' to adjust generation parameters")
        print("=" * 60)
    
    def _run_interactive_loop(self):
        while True:
            try:
                prompt = input("\n> Prompt: ").strip()
                
                if prompt.lower() in ['q', 'exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'settings':
                    self.settings.display()
                    self.settings.update_interactive()
                    continue
                
                if not prompt:
                    continue
                
                print("Generating...", end="", flush=True)
                full_text, generated_text = self.generator.generate(
                    prompt,
                    max_new_tokens=self.settings.max_tokens,
                    min_new_tokens=self.settings.min_tokens,
                    temperature=self.settings.temperature,
                    top_k=self.settings.top_k,
                    repetition_penalty=self.settings.repetition_penalty
                )
                
                print("\r" + " " * 20 + "\r", end="")
                self._print_result(prompt, generated_text)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    def _print_result(self, prompt, generated_text):
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print(f"Generated:\n{generated_text}")
        print(f"{'='*60}")


def main():
    config = GenerationConfig()
    
    interactive_gen = InteractiveGenerator(config)
    interactive_gen.start()


if __name__ == "__main__":
    main()

