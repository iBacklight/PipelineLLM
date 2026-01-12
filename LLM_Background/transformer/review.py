import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def pos_embeddings(max_seq_len, d_model, device):
    """
    eq of the position embeddings:
    PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))

    where pos is the position of the token in the sequence, 
    and i is the dimension of the embedding.
    """
    pe = torch.zeros(max_seq_len, d_model, device=device) # [max_seq_len, d_model]
    pos = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1) # [max_seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, step=2, dtype=torch.float, device=device) * (-math.log(1e4) / d_model)) # [d_model//2]

    pe[:, 0::2] = torch.sin(pos*div_term)
    pe[:, 1::2] = torch.cos(pos*div_term)
    return pe

def test_pos_embeddings():
    max_seq_len = 128
    d_model = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe = pos_embeddings(max_seq_len, d_model, device)
    print(pe.shape)
    print(pe)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        self.pe = pos_embeddings(max_seq_len, d_model, device)
        
    def forward(self, x):
        return x + self.pe

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, device):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.device = device


    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        # batch size, sequence length, d_model
        # record each dim for spliting the heads later
        # let's assume self-attention
        B = query.shape[0] # batch size
        T = query.shape[1] # sequence length
        d_model = query.shape[2] # feature dimension
        d_head = d_model // self.nhead # key dimension per head


        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)

        # split heads
        q_proj = q_proj.view(B, T, self.nhead, d_head)
        k_proj = k_proj.view(B, T, self.nhead, d_head)
        v_proj = v_proj.view(B, T, self.nhead, d_head)

        # we compute among the nhead, i.e., eliminate the head dimension
        q_proj = q_proj.transpose(1, 2) # [B, nhead, T, d_head]
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # calculate the attention
        # attn = q_proj @ k_proj.transpose(2,3) / math.sqrt(d_head)
        attn = torch.matmul(q_proj, k_proj.transpose(2,3)) / math.sqrt(d_head) # shape of [B, nhead, T, T]

        # construct causal mask
        if mask is not None:
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attn.device), diagonal=1) # diagonal=1 means we strictly mask all j > i, not mask diagonal(i=j)
            attn = attn.masked_fill(causal_mask, float('-inf'))

        # construct padding mask
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(kpm, float('-inf'))

        # apply the softmax for final attn score
        attn_weights = F.softmax(attn, dim=-1) # shape of [B, nhead, T, T], normalized the attention score
        attn = torch.matmul(attn_weights, v_proj) # shape of [B, nhead, T, d_head]

        # concat the heads
        attn = attn.transpose(1, 2).contiguous().view(B, T, d_model)

        # finally, project the output to the model dimension
        attn = self.out_proj(attn) # shape of [B, T, d_model]
        return attn

def test_multi_head_attention():
    B = 2
    T = 10
    d_model = 512
    nhead = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = torch.randn(B, T, d_model, device=device)
    key = torch.randn(B, T, d_model, device=device)
    value = torch.randn(B, T, d_model, device=device)
    mask = True
    key_padding_mask = torch.randint(0, 2, (B, T), dtype=torch.bool, device=device)
    multi_head_attention = MultiHeadAttention(d_model, nhead, device).to(device)
    attn = multi_head_attention(query, key, value, mask, key_padding_mask)
    print(attn.shape)
    print(attn)


if __name__ == "__main__":
    test_multi_head_attention()