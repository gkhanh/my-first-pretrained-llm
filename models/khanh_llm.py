import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint # For Activation Checkpointing

# --- ARCHITECTURAL CONSTANTS ---
D_MODEL = 1024              # Hidden Dimension
V_SIZE = 50000              # Vocabulary Size
L_TOTAL = 14                # Total Layers (7 Dense, 7 MoE)
NUM_HEADS = 16              # Common for D_MODEL=1024
E_TOTAL = 8                 # Total Experts (E) 
K_ACTIVE = 2                # Active Experts (K)
D_FF_MOE = 4 * D_MODEL      # FFN dimension ratio (Common 4x)

# --- 1. CORE CUSTOM LAYER: MoE FFN (E=8, K=2) ---
class MoEFatehLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # Router: Maps input token to E_TOTAL experts
        self.gate = nn.Linear(D_MODEL, E_TOTAL)

        # Experts: E_TOTAL independent FFNs
        # Note: P_FFN_Expert is 8 * D_MODEL^2 
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_MODEL, D_FF_MOE),
                nn.GELU(),
                nn.Linear(D_FF_MOE, D_MODEL)
            ) for _ in range(E_TOTAL)
        ])
        
    # NOTE: This MoE implementation is simplified for conceptual clarity and
    # does not include full torch.bmm operations for true sparsity or load balancing.
    def forward(self, x):
        # Flatten for routing: (Batch * Seq_len, D_MODEL)
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)
        routing_logits = self.gate(flat_x)
        
        # --- STEP 1: Get probabilities for ALL experts first ---
        # We need this to calculate the 'prob_mass' for the loss below.
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # --- STEP 2: Calculate the Fairness/Balancing Loss ---
        # 'prob_mass': Average probability assigned to each expert across the batch.
        # If this is high for Expert 1, the router "likes" Expert 1 too much.
        prob_mass = torch.mean(routing_probs, dim=0) 
        
        # 'fraction_tokens': How many tokens actually picked each expert?
        #  We define 'picked' as being the #1 choice (argmax).
        expert_indices = torch.argmax(routing_probs, dim=-1)
        
        # Count how many times each expert was picked (0 to 7)
        tokens_per_expert = torch.histc(expert_indices.float(), bins=E_TOTAL, min=0, max=E_TOTAL-1)
        
        #    Convert count to a fraction (e.g., 0.5 means 50% of tokens went here)
        fraction_tokens = tokens_per_expert / (batch_size * seq_len)
        
        # 3. Compute the penalty.
        #    We multiply (probability * fraction) and sum them up.
        #    This is minimized when distributions are uniform.
        aux_loss = E_TOTAL * torch.sum(prob_mass * fraction_tokens)
        
        # --- STEP 3: Select Top-K for the actual computation ---
        # Now we proceed with the standard MoE logic
        routing_weights, selected_experts = torch.topk(routing_probs, K_ACTIVE, dim=-1)
        
        # --- STEP 4: Re-normalize ---
        # Since we ran softmax on all 8, the top 2 might sum to 0.6 + 0.1 = 0.7.
        # We need them to sum to 1.0 for the weighted average later.
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        final_output = torch.zeros_like(flat_x)
        
        # Vectorized implementation: Loop over experts (8) instead of tokens (2048+)
        for expert_idx in range(E_TOTAL):
            # Find where this expert is selected (in any of the K positions)
            # mask shape: (N, K)
            expert_mask = (selected_experts == expert_idx)
            
            # Get indices of tokens that selected this expert
            # batch_indices: which tokens
            # k_indices: which priority (1st or 2nd choice)
            batch_indices, k_indices = expert_mask.nonzero(as_tuple=True)
            
            if len(batch_indices) > 0:
                # Gather the input tokens that need this expert
                input_tokens = flat_x[batch_indices] # Shape: (num_selected, D)
                
                # Compute expert output efficiently for the batch of selected tokens
                expert_out = self.experts[expert_idx](input_tokens)
                
                # Get the corresponding routing weights
                w = routing_weights[batch_indices, k_indices].unsqueeze(-1) # Shape: (num_selected, 1)
                
                # Apply weights
                weighted_expert_out = expert_out * w
                
                # Scatter add the results back to the final output tensor
                # This adds the contribution of this expert to the correct token positions
                final_output.index_add_(0, batch_indices, weighted_expert_out)

        return final_output.view(x.size()), aux_loss


# --- 2. THE 14-LAYER HYBRID TRANSFORMER BLOCK ---
class HybridTransformerBlock(nn.Module):
    def __init__(self, is_moe_block):
        super().__init__()
        
        # Self-Attention Layer (P_Attn = 4 * D_MODEL^2)
        self.attn = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True)
        self.norm1 = nn.LayerNorm(D_MODEL)
        
        # FFN Mechanism (Alternating Dense or MoE)
        self.is_moe_block = is_moe_block # Store this flag
        if is_moe_block:
            self.ffn = MoEFatehLayer() 
        else:
            self.ffn = nn.Sequential(
                nn.Linear(D_MODEL, D_FF_MOE),
                nn.GELU(),
                nn.Linear(D_FF_MOE, D_MODEL)
            ) 

        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask=None):
        # 1. Attention + Residual
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        # 2. FFN/MoE + Residual
        if self.is_moe_block:
            # Catch the aux_loss from the layer
            ffn_output, aux_loss = self.ffn(x)
        else:
            ffn_output = self.ffn(x)
            # If it's a dense layer, loss is 0
            aux_loss = torch.tensor(0.0, device=x.device) 
            
        x = self.norm2(x + ffn_output)
        return x, aux_loss

# --- 3. THE FULL KHANH LLM MODEL ---
class KhanhLLM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding = nn.Embedding(V_SIZE, D_MODEL)
        self.pos_embedding = nn.Embedding(2048, D_MODEL) # Assume max sequence length 2048
        
        # Building the 14-Layer Alternating Stack
        self.layers = nn.ModuleList()
        for i in range(L_TOTAL):
            # i=0 is Dense, i=1 is MoE, i=2 is Dense, etc.
            is_moe = (i % 2 == 1) 
            self.layers.append(HybridTransformerBlock(is_moe_block=is_moe))
            
        self.output_layer = nn.Linear(D_MODEL, V_SIZE)
        
        # Cache for causal mask (avoid recreating every forward pass)
        self._cached_mask = None
        self._cached_seq_len = 0

    def forward(self, tokens):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.pos_embedding(positions)
        
        # --- Create Causal Mask (The "Blindfold") with Caching ---
        # Ensures token[i] can only see tokens[0...i]
        # Cache the mask to avoid recreating it every forward pass
        if seq_len != self._cached_seq_len or self._cached_mask is None or self._cached_mask.device != tokens.device:
            self._cached_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=tokens.device), 
                diagonal=1
            )
            self._cached_seq_len = seq_len
        mask = self._cached_mask
        
        # Keep track of auxiliary losses from all MoE layers
        total_aux_loss = 0.0
        
        # Apply Activation Checkpointing to every layer
        for layer in self.layers:
            # We must use checkpointing carefully. 
            # The checkpoint function will return whatever the layer returns.
            # use_reentrant=False is the newer, more efficient mode
            x, layer_aux_loss = checkpoint(layer, x, mask, use_reentrant=False)
            
            # Accumulate the loss from all 14 layers (7 of which are MoE)
            total_aux_loss += layer_aux_loss
            
        output = self.output_layer(x)
        
        return output, total_aux_loss
