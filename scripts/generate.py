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
        model.load_state_dict(checkpoint['model_state_dict'])
        
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

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
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
    
    Returns:
        Generated text (including the original prompt)
    """
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    original_length = input_ids.shape[1]
    
    # Generate tokens
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass
            # Model returns (logits, aux_loss) - we only need logits for generation
            logits, _ = model(input_ids)
            
            # Get logits for the last token
            last_token_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                last_token_logits[indices_to_remove] = -float('Inf')

            # Softmax to get probabilities
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
            
            # Stop if sequence gets too long (safety check)
            if input_ids.shape[1] >= 2048:  # Max sequence length
                break
    
    # Decode only the newly generated tokens (or full text)
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
    
    # Generation parameters
    max_tokens = 100
    temperature = 0.7
    top_k = 50
    
    while True:
        try:
            prompt = input("\n> Prompt: ").strip()
            
            if prompt.lower() in ['q', 'exit', 'quit']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  Max tokens: {max_tokens}")
                print(f"  Temperature: {temperature} (lower = more focused, higher = more creative)")
                print(f"  Top-k: {top_k}")
                try:
                    max_tokens = int(input("  New max tokens (or Enter to skip): ") or max_tokens)
                    temp = input("  New temperature (or Enter to skip): ")
                    if temp:
                        temperature = float(temp)
                    top_k_val = input("  New top-k (or Enter to skip): ")
                    if top_k_val:
                        top_k = int(top_k_val)
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
                temperature=temperature,
                top_k=top_k
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

