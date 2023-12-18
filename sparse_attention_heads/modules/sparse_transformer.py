import torch
from torch import nn, Tensor, functional as F
from modules.sparse_encoder import SparseEncoderLayers
from modules.pos_enc import PositionalEncoding


class Preprocess(nn.Module):

    def __init__(self, n_vocab: int, d_model: int, max_len: int, dropout: float = 0.05):
        self.embed = nn.Embedding(n_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        embed = self.embed(input)
        pos = self.pos_enc(input)
        return self.dropout(embed + pos)
    

class SparseTransformer(nn.Module):

    def __init__(self, n_layers: int, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, vocab_size: int, max_len: int, dropout: float = 0.1, dropout_embed: float = 0.05):
        self.pre = Preprocess(vocab_size, d_model, max_len, dropout_embed)
        self.encoders = SparseEncoderLayers(n_layers, d_model, n_head, n_active, d_attn, d_ff, dropout)

    def forward(self, input: Tensor) -> Tensor:
        pre = self.pre(input)
        return self.encoders(pre)
    
    