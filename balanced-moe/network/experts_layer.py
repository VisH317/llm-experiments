from typing import Annotated
import torch
from torch import nn, functional as F, Tensor

class TransformerExperts():
    def __init__(self, n_experts: int, topk: int, n_encoders: int, d_model: int, n_head: int, d_ff: int):
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout=0.1)

        self.n_experts = n_experts
        self.experts = nn.ModuleList([nn.TransformerEncoder(encoder_layer, n_encoders) for _ in range(n_experts)])
        self.router = nn.Sequential(
            nn.Linear(d_model, n_experts),
            nn.Softmax(0)
        )
        self.topk = topk

    def forward(self, input: Tensor):
        dist = torch.topk(self.router(input), self.topk)
        return self.forward_dist(input, dist)
    
    def forward_greedy(self, input: Tensor, expert_ix: int):
        if expert_ix >= self.n_experts: raise IndexError("Desired expert is out of range")
        return self.experts[expert_ix](input)
    
    def forward_dist(self, input: Tensor, dist: list[float]):
        if len(dist) != self.n_experts: raise IndexError("Dist length is not equal to the number of experts")
        outputs = torch.empty((self.n_experts, *input.size())) # switch to tensor later
        for ix, expert in enumerate(self.experts):
            outputs[ix] = (torch.zeros_like(input) if dist[ix] == 0 else expert(input) * dist[ix])
        return torch.sum(outputs, 0)
        
