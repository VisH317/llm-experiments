import torch
from torch import nn, Tensor, functional as F
from .sparse_mha import SparseMultiHeadAttention

class SparseEncoder(nn.Module):

    def __init__(self, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, dropout: float):
        super(SparseEncoder, self).__init__()

        self.attn = SparseMultiHeadAttention(n_head, n_active, d_model, d_attn)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Linear(d_ff, d_model),
            nn.ReLU()
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        attn_out = self.attn(input)
        y = self.ln1(input + self.dropout(attn_out))
        ff_out = self.ff(y)
        out = self.ln2(y + self.dropout(ff_out))
        return out

class SparseEncoderLayers(nn.Module):

    def __init__(self, n_layers, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, dropout: float = 0.1):
        super(SparseEncoderLayers, self).__init__()

        self.layers = nn.Sequential(
            *[SparseEncoder(d_model, n_head, n_active, d_attn, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)

