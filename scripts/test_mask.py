import torch
import torch.nn as nn
from models.khanh_llm import KhanhLLM, V_SIZE

def test_masking():
    model = KhanhLLM()
    model.eval()
    
    # Create a dummy input (Batch=1, Seq=10)
    input_ids = torch.randint(0, V_SIZE, (1, 10))
    
    # Run forward pass
    # We need to hook into the attention layer to see attention weights
    # But standard nn.MultiheadAttention doesn't easily expose weights unless we ask for them.
    # Instead, let's look at the gradients. 
    # If we compute loss on token 0, and gradients flow from token 1, masking is BROKEN.
    
    print("Testing Masking via Gradient Flow...")
    
    # Embeddings require grad
    model.token_embedding.weight.requires_grad = True
    
    # Forward pass
    output = model(input_ids)
    
    # Let's say we want to predict token at index 0.
    # It should ONLY depend on input token 0.
    # If it depends on input token 1, 2, 3... then future leakage is happening.
    
    # Loss on position 0
    loss = output[0, 0].sum() 
    loss.backward()
    
    # Check gradients on input embeddings
    # Input embeddings shape: (V_SIZE, D_MODEL)
    # We can check which input tokens have gradients.
    
    # We need to access the embedding gradients corresponding to the input indices
    embed_grad = model.token_embedding.weight.grad
    
    # Check if input_ids[0, 1] (future token) has gradient
    future_token_idx = input_ids[0, 1].item()
    current_token_idx = input_ids[0, 0].item()
    
    grad_at_future = embed_grad[future_token_idx].sum().item()
    grad_at_current = embed_grad[current_token_idx].sum().item()
    
    print(f"Gradient at Current Token (Idx {current_token_idx}): {grad_at_current}")
    print(f"Gradient at Future Token (Idx {future_token_idx}): {grad_at_future}")
    
    if abs(grad_at_future) > 1e-5:
        print("FAIL: Gradient flowed from future token! Mask is BROKEN.")
    else:
        print("PASS: No gradient from future token. Mask is WORKING.")

if __name__ == "__main__":
    test_masking()

