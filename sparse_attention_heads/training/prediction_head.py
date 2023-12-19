import torch
from torch import nn, Tensor

# flatten or sum - idk

class TokenClassifier(nn.Module):

    def __init__(self, d_model: int, max_len: int, vocab_size: int):
        self.seq = nn.Sequential(
            nn.Linear(max_len * d_model, vocab_size),
            nn.Softmax(0)
        )

    def forward(self, input: Tensor) -> Tensor:
        flattened = torch.flatten(input, 1)
        return self.seq(flattened)