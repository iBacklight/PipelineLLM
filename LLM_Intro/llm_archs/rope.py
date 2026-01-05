import torch
from typing import Tuple

def build_inv_freq(head_dim:int, base:float=10000.0, device=None, dtype=None):
    """
    Generate the frequency table of RoPE: inv_freq[i] = base^{-2i / head_dim}, i=0,1,...,head_dim/2-1
    Require head_dim to be even.
    Return shape: (head_dim/2,)
    
    B = batch size
	H = number of attention heads
	T = sequence length
	D = head dimension
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)  # 0,2,4,...
    inv_freq = base ** (-idx / head_dim) # 
    print("inv_freq: ", inv_freq.shape)
    return inv_freq

def rope_cos_sin(seq_len:int, inv_freq:torch.Tensor, device=None):
    t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)  # [0..seq_len-1]
    freq_shape = inv_freq.shape[0]
    phase = t.view(seq_len, 1) @ inv_freq.view(1, freq_shape) # (T, D/2)
    # phase = torch.einsum("p,f->pf", t, inv_freq)  # (T, D/2)
    # concat phase for computing embedding cos and sin at phase dim
    emb = torch.cat([phase, phase], dim=1) # (T, D)

    cos = emb.cos()
    sin = emb.sin()

    return cos, sin

def rotate_half(x: torch.Tensor): # x => [B, H, T, D]
    head_dim = x.shape[-1]
    device = x.device
    x_f = x[..., :head_dim//2]
    x_s = x[..., head_dim//2:]
    return torch.cat([-x_f, x_s], dim=-1) # [B, H, T, D]

def apply_rope(q: torch.Tensor,
               k: torch.Tensor,
               position_ids: torch.Tensor,
               base: float = 10000.0):
    B, H, T, D = q.shape
    assert D % 2 == 0, "head_dim must be even for RoPE."
    device, dtype = q.device, q.dtype

    # Ensure the max position must be greater than total seq len, here total seq len = 2*D-1
    # since we have two samples. Or it will raise RuntimeError from CUDA
    max_pos = int(position_ids.max().item()) + 1 
    inv_freq = build_inv_freq(D, base=1e4, device=device, dtype=dtype)
    cos, sin = rope_cos_sin(max_pos, inv_freq, device=device) # [T, D]

    # print(cos.shape)
    cos_pos_val = cos[position_ids].unsqueeze(1) # [B, 1, T, D]
    sin_pos_val = sin[position_ids].unsqueeze(1) 

    q_emb = q * cos_pos_val + rotate_half(q) * sin_pos_val
    k_emb = k * cos_pos_val + rotate_half(k) * sin_pos_val

    print(q_emb.shape)

    return q_emb, k_emb

B, H, T, D = 2, 8, 128, 64  # D must be even 
q = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, H, T, D, dtype=torch.float16, device="cuda")

# two samples ,start from 0 and 128
base_pos = torch.tensor([0, 128], device=q.device).view(B, 1)        # (B,1)
print("Initial pos of each sample: ", base_pos)
position_ids = base_pos + torch.arange(T, device=q.device).view(1, T)  # (B,T)

apply_rope(q, k, position_ids=position_ids)