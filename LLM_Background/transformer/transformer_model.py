import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# attention
class MultiHeadAttention(nn.Module):
    """
    A multi-head attention module.
    Attention equation:
    $$
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M)V
    $$
    where Q = W_Q * X, K = W_K * X, V = W_V * X, M is the mask
    d_k is the dimension of the key
    nhead is the number of heads

    Multi-head attention equation:
    $$
    MultiHeadAttention(Q,K,V) = Concat(Head_1, Head_2, ..., Head_n)W_O
    $$
    where Head_i = Attention(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})

    W_O is the output projection matrix
    """
    def __init__(self, d_model, nhead, dropout=0.0, device=None):
        super(MultiHeadAttention, self).__init__()# initialize the parent class
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_head = nhead
        self.d_model = d_model
        self.d_k = d_model // self.n_head # d_k is the feature dimension per head (=key/value dimension)
        self.q_proj = nn.Linear(d_model, d_model, bias=True) # query projection, aka w_q
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        # output projection, aka w_o, processing the concatenated heads
        self.o_proj = nn.Linear(d_model, d_model, bias=True) 
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1) # dim = 2
        
        # Move to device
        self.to(self.device)
        

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        B = query.shape[0] # batch size
        T = query.shape[1] # sequence length
        d_q = query.shape[2] # feature dimension, aka d_model in self-attention

        # if self-attention, query, key, value are the same
        if key is None:
            query = key = value
        d_k = key.shape[2] # k model dim, must be same as d_q, different from self.d_k
        d_v = value.shape[2] # v model dim, can be different from d_q and d_k if not self-attention

        # project the input x to query, key, value
        q = self.q_proj(query) # q, k must have same dimension all the time
        k = self.k_proj(key)
        v = self.v_proj(value)  

        # Multi-head attention, so cut the feature dimension into n_head parts
        q = q.view(B, T, self.n_head, d_q // self.n_head)
        k = k.view(B, T, self.n_head, d_k // self.n_head)
        v = v.view(B, T, self.n_head, d_v // self.n_head)

        # Since the attention computing for MHA is done for each head, 
        # we need to transpose q, k, v to get the shape of [B, n_head, T, d_k]
        q = q.transpose(1, 2) # or we can use q.permute(0, 2, 1, 3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate the attention , in paper it is  Q* K^T / sqrt(d_k), 
        # so we need to transpose k to get the shape of [B, n_head, d_k, T] and then do the matmul
        attn = q @ k.transpose(2, 3) / math.sqrt(self.d_k)
        # also we can use matmul or einsum to calculate the attention
        # attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)
        # attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.d_k)

        # construct causal mask
        if mask is not None:
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attn.device), diagonal=1) # diagonal=1 means we strictly mask all j > i, not mask diagonal(i=j)
            # for pos of q is i, pos of key is j, we need to mask all j > i
            # i\j 0 1 2 3 4
            # 0   · × × × ×
            # 1   · · × × ×
            # 2   · · · × ×
            # 3   · · · · ×
            # 4   · · · · ·
            attn = attn.masked_fill(causal_mask, float('-inf'))

        # construct padding mask
        if key_padding_mask is not None:  # [B,T], True=pad
            kpm = key_padding_mask[:, None, None, :]  # [B,1,1,T]
            attn = attn.masked_fill(kpm, float("-inf"))
        
        # apply the softmax for final attn score
        attn_weights = self.softmax(attn)  # [B, n_head, T, T]
        attn = attn_weights @ v  # [B, n_head, T, d_v]

        # now, ready to concat the heads
        # First permute to [B, T, n_head, d_k] or [B, T, n_head, d_v] then concat to [B, T, d_model]
        # contiguous() is used to make the tensor contiguous in memory, necessary for view()
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, T, self.d_model) 

        # finally, project the output to the model dimension
        attn = self.o_proj(attn) # [B, T, d_model]

        return attn


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer using self-attention.
    """
    def __init__(self, d_model, nhead, dropout=0.0, device=None):
        super(TransformerEncoderLayer, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.n_head = nhead
        self.d_k = d_model // self.n_head
        self.attn = MultiHeadAttention(d_model, nhead, dropout, self.device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), # or nn.SiLU() in LLaMA style
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Move all layers to device
        self.to(self.device)
    
    def forward(self, x, *, key_padding_mask=None):
        # If post-LN, we need to apply the normalization after the attention
        attn = self.attn(x, x, x, None, key_padding_mask) # no mask for encoder
        x = x + self.dropout(attn)
        x = self.norm1(x) # post-LN
        x = x + self.dropout(self.ffn(x)) # FFN
        x = self.norm2(x) # post-LN
        
        # # if pre-LN, we need to apply the normalization before the attention
        # x = self.norm1(x) # pre-LN
        # attn = self.attn(x, x, x, None, key_padding_mask) # no mask for encoder
        # x = x + self.dropout(attn)
        # x = self.norm2(x)
        # x = x + self.dropout(self.ffn(x)) # FFN
        
        return x

class TransformerDecoderLayer(nn.Module):
    """
    A transformer decoder layer with masked self-attention and cross-attention.
    
    Each decoder layer consists of:
    1. Masked self-attention on the target sequence
    2. Cross-attention between decoder and encoder output (memory)
    3. Feed-forward network
    """
    def __init__(self, d_model, nhead, dropout=0.0, device=None):
        super(TransformerDecoderLayer, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.n_head = nhead
        self.d_k = d_model // self.n_head
        
        # Masked self-attention (for target sequence)
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout, self.device)
        
        # Cross-attention (query from decoder, key/value from encoder)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout, self.device)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), # or nn.SiLU() in LLaMA style
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)  # after self-attention
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # after FFN
        
        # Move all layers to device
        self.to(self.device)
    
    def forward(self, x, memory, *, tgt_mask=True, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the decoder layer.
        
        Args:
            x: Target sequence [batch_size, tgt_len, d_model]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Causal mask for target self-attention (default True)
            tgt_key_padding_mask: Padding mask for target [batch_size, tgt_len]
            memory_key_padding_mask: Padding mask for memory/source [batch_size, src_len]
        """
        # For decoder, we always use causal mask (tgt_mask=True)
        if tgt_mask is None:
            tgt_mask = True  # Use causal mask for decoder
        
        # Post-LN architecture:
        # 1. Masked self-attention on target
        self_attn_out = self.self_attn(x, x, x, tgt_mask, tgt_key_padding_mask)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)
        
        # 2. Cross-attention: query from decoder, key/value from encoder
        # No causal mask needed for cross-attention (we can attend to all encoder positions)
        cross_attn_out = self.cross_attn(x, memory, memory, None, memory_key_padding_mask)
        x = x + self.dropout(cross_attn_out)
        x = self.norm2(x)
        
        # 3. Feed-forward network
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)
        
        # # Pre-LN architecture (alternative):
        # # 1. Masked self-attention
        # x_norm = self.norm1(x)
        # self_attn_out = self.self_attn(x_norm, x_norm, x_norm, tgt_mask, tgt_key_padding_mask)
        # x = x + self.dropout(self_attn_out)
        # 
        # # 2. Cross-attention
        # x_norm = self.norm2(x)
        # cross_attn_out = self.cross_attn(x_norm, memory, memory, None, memory_key_padding_mask)
        # x = x + self.dropout(cross_attn_out)
        # 
        # # 3. FFN
        # x_norm = self.norm3(x)
        # x = x + self.dropout(self.ffn(x_norm))
        
        return x


class TransformerEncoder(nn.Module):
    """
    A transformer encoder with multiple encoder layers.
    """
    def __init__(self, d_model, nhead, nlayers, dropout=0.0, device=None):
        super(TransformerEncoder, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout, self.device)
            for _ in range(nlayers)
        ])
        
        self.to(self.device)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the encoder.
        
        Args:
            src: Source sequence [batch_size, src_len, d_model]
            src_mask: Not used in encoder (no causal masking)
            src_key_padding_mask: Mask for source padding [batch_size, src_len]
        """
        output = src
        
        # Pass through each encoder layer
        for layer in self.layers:
            output = layer(output, key_padding_mask=src_key_padding_mask)
        
        return output


class TransformerDecoder(nn.Module):
    """
    A transformer decoder with multiple decoder layers.
    Each layer has masked self-attention, cross-attention, and FFN.
    """
    def __init__(self, d_model, nhead, nlayers, dropout=0.0, device=None):
        super(TransformerDecoder, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dropout, self.device)
            for _ in range(nlayers)
        ])
        
        self.to(self.device)
    
    def forward(self, memory, tgt, tgt_mask=True, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the decoder.
        
        Args:
            memory: Encoder output [batch_size, src_len, d_model]
            tgt: Target sequence [batch_size, tgt_len, d_model]
            tgt_mask: Causal mask for target self-attention
            tgt_key_padding_mask: Mask for target padding [batch_size, tgt_len]
            memory_key_padding_mask: Mask for source/memory padding [batch_size, src_len]
        """
        output = tgt
        
        # Pass through each decoder layer with cross-attention to encoder memory
        for layer in self.layers:
            output = layer(
                output, 
                memory, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        return output


class TransformerModel(nn.Module):
    """
    A complete Transformer model with encoder and decoder.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, device=None):
        super(TransformerModel, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding (sinusoidal)
        self.pos_embedding = self._create_positional_encoding(10000, d_model)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dropout, self.device)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dropout, self.device)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Move to device
        self.to(self.device)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass of the Transformer model.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_key_padding_mask: Mask for source padding [batch_size, src_len]
            tgt_key_padding_mask: Mask for target padding [batch_size, tgt_len]
        """
        batch_size, src_len = src.shape
        tgt_len = tgt.shape[1]
        
        # Source embeddings
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_embedding[:src_len].unsqueeze(0)
        
        # Target embeddings
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_embedding[:tgt_len].unsqueeze(0)
        
        # Generate causal mask for target
        
        # Encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # Decoder (with cross-attention to encoder memory)
        decoder_output = self.decoder(
            memory, 
            tgt_emb, 
            tgt_mask=True, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask  # Use source padding mask for cross-attention
        )
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output


# Integration with SimpleTokenizer
def test_with_simple_tokenizer():
    """
    Test the Transformer model with SimpleTokenizer integration.
    """
    # Import the SimpleTokenizer
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from simple_tokenizer import SimpleTokenizer
    
    print("\n" + "="*60)
    print("=== Testing with SimpleTokenizer ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare corpus and create tokenizer
    corpus = [
        "Transformer is almost the origin of large models",
        "All modern LMs are based on the transformer architecture", 
        "Attention is all you need",
        "The transformer uses self attention mechanism",
        "Multi head attention allows parallel processing",
        "Positional encoding helps understand sequence order"
    ]
    
    print("=== Creating Tokenizer ===")
    tokenizer = SimpleTokenizer(corpus=corpus, max_vocab=1000, d_model=256)
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"Vocabulary size: {vocab_size}")

    # Prepare source and target texts
    src_texts = [
        "Transformer is the origin",
        "attention is all you need"
    ]
    
    tgt_texts = [
        "of large language models",
        "for neural networks"
    ]
    
    # Tokenize texts using SimpleTokenizer
    src_ids, src_mask = tokenizer.collate_fn(src_texts, max_len=10)
    tgt_ids, tgt_mask = tokenizer.collate_fn(tgt_texts, max_len=8)

    return src_ids, tgt_ids, src_mask, tgt_mask
    

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    vocab_size = 1000
    d_model = 256
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    src = torch.randint(0, vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len), device=device)

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # Forward pass
    print("\n=== Transformer Tensor Forward Pass ===")
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Output shape: {output.shape}")

    # 5. Forward pass through Transformer
    src_ids, tgt_ids, src_mask, tgt_mask = test_with_simple_tokenizer()

    print("\n=== Transformer Tokenizer Forward Pass ===")
    with torch.no_grad():
        output = model(src_ids, tgt_ids, src_key_padding_mask=~src_mask.bool(), tgt_key_padding_mask=~tgt_mask.bool())
        print(f"Output shape: {output.shape}")

