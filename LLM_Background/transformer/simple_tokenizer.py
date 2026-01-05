from collections import Counter
import torch
import re
import torch.nn as nn
import math


class SimpleTokenizer:
    """
    A simple tokenizer class for text preprocessing and tokenization.
    Supports vocabulary building, encoding/decoding, and embedding generation.
    """
    
    def __init__(self, corpus=None, max_vocab=10000, d_model=256, max_pos_len=1024):
        """
        Initialize the tokenizer.
        
        Args:
            corpus: List of text strings to build vocabulary from
            max_vocab: Maximum vocabulary size
            d_model: Embedding dimension
            max_pos_len: Maximum position length for position embeddings
        """
        self.max_vocab = max_vocab
        self.d_model = d_model
        self.max_pos_len = max_pos_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Special tokens
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        
        # Initialize vocabulary and mappings
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None
        self.unk_id = None
        
        # Embedding layers
        self.tok_emb = None
        self.pos_emb = None
        
        # Build vocabulary if corpus is provided
        if corpus is not None:
            self.build_vocab(corpus)
    
    def tokenize(self, text):
        """
        Tokenize a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        text = text.lower()
        # Simple cleaning: keep only letters and spaces
        # "[^a-z\s]"means string, "^a-z" means select all EXCEPT for a-z,  "\s" means spaces(so this means keep letters and spaces)
        text = re.sub(r"[^a-z\s]", "", text) # only keep letters and spaces
        return text.split()
    
    def build_vocab(self, corpus):
        """
        Build vocabulary from corpus.
        
        Args:
            corpus: List of text strings
        """
        counter = Counter()
        # use counter to count the frequency of each token
        for line in corpus:
            counter.update(self.tokenize(line))
        
        # Build vocabulary with special tokens first
        # most_common(n) returns a list of the n most common elements and their counts from the most common to the least. 
        # so we get the most common tokens and add them to the vocabulary
        self.vocab = self.special_tokens + [
            w for w, _ in counter.most_common(self.max_vocab - len(self.special_tokens))
        ]

        print(self.vocab)
        
        # Create mappings
        self.stoi = {w: i for i, w in enumerate(self.vocab)} # word to index
        self.itos = {i: w for w, i in self.stoi.items()} # index to word
        
        # Get special token IDs
        self.pad_id = self.stoi["<pad>"] # padding index
        self.bos_id = self.stoi["<bos>"] # beginning of sequence index
        self.eos_id = self.stoi["<eos>"] # end of sequence index
        self.unk_id = self.stoi["<unk>"] # unknown token index
        
        # Initialize embedding layers
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize token and position embedding layers."""
        vocab_size = len(self.vocab)
        self.tok_emb = nn.Embedding(vocab_size, self.d_model, padding_idx=self.pad_id)
        self.tok_emb = self.tok_emb.to(self.device)  # Move to device
        
        # Use sinusoidal position embedding instead of learnable embeddings
        self.use_sinusoidal_pos = True
        if self.use_sinusoidal_pos:
            # Pre-compute sinusoidal position embeddings
            self.pos_embeddings = self._create_sinusoidal_embeddings()
        else:
            # Use learnable position embeddings
            self.pos_emb = nn.Embedding(self.max_pos_len, self.d_model)
            self.pos_emb = self.pos_emb.to(self.device)  # Move to device

    def _create_sinusoidal_embeddings(self):
        """Create sinusoidal position embeddings - much simpler approach."""
        # Create position matrix: [max_pos_len, d_model]
        pos_embeddings = torch.zeros(self.max_pos_len, self.d_model, device=self.device)
        
        # apply the position embedding to max_pos_len positions,and
        # we will do the cut for each postion when we use it
        for pos in range(self.max_pos_len): 

            # For each dimension
            for i in range(0, self.d_model, 2):  # Step by 2 for sin/cos pairs
                # Calculate frequency: 1 / (10000^(2i/d_model))
                freq = 1.0 / (10000 ** (2 * i / self.d_model))
                
                # Apply sin to even dimensions
                pos_embeddings[pos, i] = math.sin(pos * freq)
                
                # Apply cos to odd dimensions (if not the last dimension)
                if i + 1 < self.d_model:
                    pos_embeddings[pos, i + 1] = math.cos(pos * freq)
        # we have another way to do this
        # pe = torch.zeros(self.max_pos_len, self.d_model, device=self.device)
        # pos = torch.arange(0, self.max_pos_len, dtype=torch.float, device=self.device).unsqueeze(1)  # [T,1]
        # div_term = torch.exp(torch.arange(0, self.d_model, 2, device=self.device).float()
        #                     * (-math.log(10000.0) / self.d_model))
        # pe[:, 0::2] = torch.sin(pos * div_term)  # even
        # pe[:, 1::2] = torch.cos(pos * div_term)  # odd
        # return pe  # [T, d_model]
        
        return pos_embeddings

    def _get_position_embeddings(self, seq_len):
        """Get position embeddings for a given sequence length."""
        if self.use_sinusoidal_pos:
            # Return pre-computed sinusoidal embeddings
            return self.pos_embeddings[:seq_len]  # [seq_len, d_model], cut till seq_len from max_pos_len
        else:
            # Use learnable embeddings
            pos_ids = torch.arange(seq_len, device=self.device)
            return self.pos_emb(pos_ids)  # [seq_len, d_model]
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        ids = [self.stoi.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def decode(self, ids):
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = [self.itos.get(id_, "<unk>") for id_ in ids]
        return " ".join(tokens)
    
    def collate_fn(self, texts, max_len=None):
        """
        Batch texts with padding and create attention masks.
        
        Args:
            texts: List of text strings
            max_len: Maximum sequence length (if None, use max length in batch)
            
        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        seqs = [self.encode(t) for t in texts]
        maxT = max_len or max(len(s) for s in seqs)
        
        input_ids = []
        attn_mask = []
        
        for s in seqs:
            s = s[:maxT]  # Truncate if too long
            pad_len = maxT - len(s)
            input_ids.append(s + [self.pad_id] * pad_len)
            attn_mask.append([1] * len(s) + [0] * pad_len)
        
        return torch.tensor(input_ids, device=self.device), torch.tensor(attn_mask, device=self.device)
    
    def get_embeddings(self, input_ids):
        """
        Get token and position embeddings for input IDs.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Combined embeddings tensor of shape [batch_size, seq_len, d_model]
        """
        if self.tok_emb is None:
            raise ValueError("Embeddings not initialized. Call build_vocab() first.")
        
        B, T = input_ids.shape
        
        # Get token embeddings
        token_emb = self.tok_emb(input_ids)  # [B, T, d_model]
        
        # Get position embeddings
        pos_emb = self._get_position_embeddings(T)  # [T, d_model]
        pos_emb = pos_emb.unsqueeze(0).expand(B, T, -1)  # [B, T, d_model]
        
        # Combine token and position embeddings
        X = token_emb + pos_emb
        return X
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_special_token_ids(self):
        """Get special token IDs."""
        return {
            'pad_id': self.pad_id,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
            'unk_id': self.unk_id
        }


# Example usage
if __name__ == "__main__":
    # 1) Prepare corpus
    corpus = [
        "Transformer is almost the origin of large models",
        "All modern LMs are based on the transformer architecture",
        "Attention is all you need",
    ]
    
    # 2) Initialize tokenizer
    tokenizer = SimpleTokenizer(corpus=corpus, max_vocab=10000, d_model=256)
    
    # 3) Encode some texts
    batch_text = [
        "Transformer is the origin",
        "attention is all you need",
    ]
    
    input_ids, attn_mask = tokenizer.collate_fn(batch_text, max_len=12)
    print(f"Input IDs shape: {input_ids.shape}")  # [B, T]
    print(f"Attention mask shape: {attn_mask.shape}")  # [B, T]
    
    # 4) Get embeddings
    embeddings = tokenizer.get_embeddings(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")  # [B, T, d_model]
    
    # 5) Test encode/decode
    test_text = "attention is all you need"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # 6) Print vocabulary info
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special token IDs: {tokenizer.get_special_token_ids()}")