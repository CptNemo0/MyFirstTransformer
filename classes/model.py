import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, bias, dropout):
        super().__init__()
        #settings
        self.embedding_dim = embedding_dim
        self.d_ff          = d_ff
        self.bias          = bias
        self.dropout       = dropout
        
        self.linear_1 = nn.Linear(self.embedding_dim, self.d_ff, self.bias)
        self.linear_2 = nn.Linear(self.d_ff, self.embedding_dim, self.bias)
        self.dropout  = nn.Dropout(self.dropout)
     
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, embedding_dim, n_heads, bias, dropout):
        super().__init__()
        #settings
        self.embedding_dim = embedding_dim
        self.n_heads       = n_heads
        self.bias          = bias
        self.dropout       = dropout
        
        self.matrices = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=self.bias)
        self.linear   = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout  = nn.Dropout(self.dropout)
        
    def forward(self, x):
        #sizes
        B, T, C = x.size()
        
        #queries, keys, values
        Q, K, V = self.matrices(x).split(self.embedding_dim, dim = 2)
        
        #Batch, Number of heads, Time (sequence lenght), Features
        K = K.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        Q = Q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        V = V.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        
        #Causal attention
        y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
        
        #Return to original shape
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.linear(y)
        y = self.dropout(y)
        return y
    
class Normalization(nn.Module):
    def __init__(self, embedding_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias   = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, d_ff, n_heads, bias, dropout):
        super().__init__()
        self.initial_normalization = Normalization(embedding_dim, bias)
        self.attention             = Attention(embedding_dim, n_heads, bias, dropout)
        self.mid_normalization     = Normalization(embedding_dim, bias)
        self.feed_forward          = FeedForward(embedding_dim, d_ff, bias, dropout)
        
    def forward(self, x):
        x = x + self.attention(self.initial_normalization(x))
        x = x + self.feed_forward(self.mid_normalization(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, context_len, n_layers, embedding_dim, d_ff, n_heads, bias, dropout, device):
        super().__init__()
        self.device = device
        self.context_len         = context_len
        self.token_embedding     = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(context_len, embedding_dim)
        self.dropout             = nn.Dropout(dropout)
        self.decoders            = nn.ModuleList([Decoder(embedding_dim, d_ff, n_heads, bias, dropout) for _ in range(n_layers)])
        self.normalization       = Normalization(embedding_dim, bias)
        self.linear              = nn.Linear(embedding_dim, vocab_size)
               
        '''self.token_embedding.to(self.device)     
        self.positional_encoding.to(self.device) 
        self.dropout.to(self.device)             
        self.decoders.to(self.device)            
        self.normalization.to(self.device)       
        self.linear.to(self.device)   '''          

    def forward(self, x):
        assert x.size()[1] == self.context_len
        positions             = torch.arange(0, self.context_len, dtype=torch.long, device=self.device)
        tokens_embedded       = self.token_embedding(x)
        
        positions_embedded    = self.positional_encoding(positions)
        tokens_plus_positions = tokens_embedded + positions_embedded
        x = self.dropout(tokens_plus_positions)
        
        for decoder in self.decoders:
            x = decoder(x)
        x = self.normalization(x)
        logits = self.linear(x)
        
        return logits