import torch
from torch import nn, Tensor, functional as F
from .sparse_mha import SparseMultiHeadAttention, RouteType

class SparseEncoder(nn.Module):

    def __init__(self, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, dropout: float, route_type: str, noise: float, noise_step: float, hidden_act: str):
        super(SparseEncoder, self).__init__()

        self.attn = SparseMultiHeadAttention(n_head, n_active, d_model, d_attn, route_type, noise, noise_step)
        

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
    
    def step_noise(self) -> None:
        self.attn.step_noise()

    def get_last_dist(self) -> Tensor: return self.attn.get_last_dist()

    def get_activation(hidden_act: str):
        if hidden_act == "silu": return nn.SiLU()
        elif hidden_act == "elu": return nn.ELU()
        elif hidden_act == "relu": return nn.ReLU()
        elif hidden_act == "leaky_relu": return nn.LeakyReLU()
        elif hidden_act == "gelu": return nn.GELU()
        else: raise TypeError("Error: activation not detected")

class SparseEncoderLayers(nn.Module):

    def __init__(self, n_layers, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, dropout: float = 0.1, route_type: str = "sum", noise: float = 0.05, noise_step: float = 0.5, hidden_act: str = "relu"):
        super(SparseEncoderLayers, self).__init__()

        self.layers = nn.Sequential(
            *[SparseEncoder(d_model, n_head, n_active, d_attn, d_ff, dropout, route_type, noise, noise_step, hidden_act) for _ in range(n_layers)]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def get_last_dist(self) -> Tensor:
        dists = [layer.get_last_dist() for layer in self.layers]
        return torch.stack(dists)
    
    def step_noise(self) -> None:
        for layer in self.layers:
            layer.step_noise()

