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
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    print("Initializing model structure...")
    model = KhanhLLM().to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("WARNING: No checkpoint found! Using random weights (expect gibberish).")
    
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=40):
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate
    # Basic sampling loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            # The model's internal causal mask handles the attention masking
            logits = model(input_ids)
            
            # Get logits for the last token
            last_token_logits = logits[:, -1, :] / temperature
            
            # Optional: Top-k filtering for better quality
            if top_k > 0:
                v, _ = torch.topk(last_token_logits, top_k)
                last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf')

            # Softmax to get probabilities
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token is generated (if defined and produced)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    print("\n--- KhanhLLM Text Generator ---")
    print("Type 'q' or 'exit' to quit.")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ")
            if prompt.lower() in ['q', 'exit']:
                break
            
            if not prompt.strip():
                continue
                
            print(f"Generating...", end="", flush=True)
            response = generate_text(model, tokenizer, prompt, max_new_tokens=100)
            
            # Clear "Generating..." line
            print(f"\r" + " " * 20 + "\r", end="") 
            print(f"Response:\n{response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

