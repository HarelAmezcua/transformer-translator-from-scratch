import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        x = self.embedding(x) * (self.d_model ** 0.5) # Scale the embeddings by sqrt(d_model)

        return x
    

class PositionEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len:int,dropout:float):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape(seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len,1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        
        pe[:,0::2] = torch.sin(position * div_term) # sin for even positions
        pe[:,1::2] = torch.cos(position * div_term) # cos for uneven positions

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe',pe)

    def forward (self,x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        x = self.dropout(x)


