import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.khanh_llm import KhanhLLM
from transformers import AutoTokenizer

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoint_latest.pth.tar"
TOKENIZER_PATH = "khanh_tokenizer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the trained model and tokenizer from checkpoint."""
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    print("Initializing model structure...")
    model = KhanhLLM().to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        # Handle torch.compile() prefix: strip '_orig_mod.' if present
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
            # Create new state dict without the prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]  # Remove prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        
        # Print training info if available
        if 'total_rows_processed' in checkpoint:
            rows = checkpoint.get('total_rows_processed', 0)
            tokens = checkpoint.get('total_tokens_processed', 0)
            print(f"Model trained on: {rows:,} rows, {tokens/1e6:.2f}M tokens")
        
        print("Model loaded successfully!")
    else:
        print("WARNING: No checkpoint found! Using random weights (expect gibberish).")
    
    # Set to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    # Optional: Compile model for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled for faster inference.")
    except Exception as e:
        print(f"Note: Model compilation skipped ({e})")
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.2, min_new_tokens=10):
    """
    Generate text from a prompt using the trained model.
    
    Args:
        model: The trained KhanhLLM model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic, higher = more creative)
        top_k: Top-k sampling (keep only top k tokens)
        top_p: Nucleus sampling (keep tokens with cumulative probability <= top_p)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize repeats)
        min_new_tokens: Minimum tokens to generate before checking for sentence completion
    
    Returns:
        Generated text (including the original prompt)
    """
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    original_length = input_ids.shape[1]
    
    # Track all tokens (input + generated) for repetition penalty
    all_tokens = input_ids[0].tolist()
    
    # Sentence ending punctuation
    sentence_endings = {'.', '!', '?'}
    
    # Generate tokens
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :].clone() / temperature
            
            # Apply repetition penalty to ALL previous tokens (including prompt)
            if repetition_penalty != 1.0:
                # Look at recent tokens (last 50 for better coverage)
                recent_tokens = set(all_tokens[-50:]) if len(all_tokens) > 50 else set(all_tokens)
                for token_id in recent_tokens:
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # Softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()
            
            # Add to tracking list
            all_tokens.append(next_token_id)
            
            # Early stopping: if we repeat the same token 3+ times in a row, stop
            if len(all_tokens) >= 3 and all_tokens[-1] == all_tokens[-2] == all_tokens[-3]:
                break
            
            # Append to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                break
            
            # Dynamic sentence completion detection
            # Check after generating minimum tokens, more frequently as we approach max
            generated_count = len(all_tokens) - original_length
            check_frequency = 3 if generated_count < max_new_tokens * 0.8 else 1  # Check every token near the end
            should_check = (generated_count >= min_new_tokens and 
                          (generated_count % check_frequency == 0 or step == max_new_tokens - 1))
            
            if should_check:
                # Decode current generated text
                generated_ids = input_ids[0, original_length:]
                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Check if text ends with sentence-ending punctuation
                if current_text.strip():
                    text_stripped = current_text.rstrip()
                    # Check if ends with sentence punctuation
                    if text_stripped and text_stripped[-1] in sentence_endings:
                        # Heuristic to avoid stopping at abbreviations (e.g., "Dr.", "U.S.A.", "Inc.")
                        # Check if the last few characters before punctuation are all uppercase (likely abbrev)
                        if len(text_stripped) > 2:
                            # Look at last 2-4 chars before punctuation
                            chars_before = text_stripped[-4:-1] if len(text_stripped) >= 4 else text_stripped[:-1]
                            chars_before_clean = chars_before.replace(' ', '').replace('.', '')
                            # If all uppercase letters (and no lowercase), likely abbreviation - continue
                            if chars_before_clean and chars_before_clean.isupper() and not any(c.islower() for c in chars_before_clean):
                                pass  # Likely abbreviation, don't stop
                            else:
                                # Normal sentence ending, stop here
                                break
                        else:
                            # Very short text ending in punctuation, likely complete sentence
                            break
            
            # Stop if sequence gets too long
            if input_ids.shape[1] >= 2048:
                break
    
    # Decode only the newly generated tokens
    generated_ids = input_ids[0, original_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Return full text (prompt + generated)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text, generated_text

if __name__ == "__main__":
    print("=" * 60)
    print("  KhanhLLM Text Generator")
    print("=" * 60)
    
    model, tokenizer = load_model()
    
    print("\n" + "=" * 60)
    print("Ready to generate text!")
    print("Commands:")
    print("  - Type your prompt and press Enter to generate")
    print("  - Type 'q' or 'exit' to quit")
    print("  - Type 'settings' to adjust generation parameters")
    print("=" * 60)
    
    # Generation parameters - Better defaults for quality
    max_tokens = 200  # Increased since we now stop at sentence completion
    min_tokens = 10  # Minimum tokens before checking for sentence completion
    temperature = 0.6  # Lower for more focused responses
    top_k = 40  # Lower for better quality
    repetition_penalty = 1.2  # Stronger penalty to reduce repetition
    
    while True:
        try:
            prompt = input("\n> Prompt: ").strip()
            
            if prompt.lower() in ['q', 'exit', 'quit']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  Max tokens: {max_tokens} (hard limit, but will stop at sentence completion)")
                print(f"  Min tokens: {min_tokens} (minimum before checking for sentence completion)")
                print(f"  Temperature: {temperature} (lower = more focused, higher = more creative)")
                print(f"  Top-k: {top_k}")
                print(f"  Repetition penalty: {repetition_penalty} (1.0 = no penalty, >1.0 = less repetition)")
                try:
                    max_tokens = int(input("  New max tokens (or Enter to skip): ") or max_tokens)
                    min_tokens = int(input("  New min tokens (or Enter to skip): ") or min_tokens)
                    temp = input("  New temperature (or Enter to skip): ")
                    if temp:
                        temperature = float(temp)
                    top_k_val = input("  New top-k (or Enter to skip): ")
                    if top_k_val:
                        top_k = int(top_k_val)
                    rep_penalty = input("  New repetition penalty (or Enter to skip): ")
                    if rep_penalty:
                        repetition_penalty = float(rep_penalty)
                    print("Settings updated!")
                except ValueError:
                    print("Invalid input, keeping current settings.")
                continue
            
            if not prompt:
                continue
                
            print("Generating...", end="", flush=True)
            full_text, generated_text = generate_text(
                model, tokenizer, prompt, 
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            
            # Clear "Generating..." line
            print("\r" + " " * 20 + "\r", end="")
            
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")
            print(f"Generated:\n{generated_text}")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

