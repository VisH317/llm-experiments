import torch
from torch import nn, Tensor, functional as F


class AttentionHead(nn.Module):

    def __init__(self, d_model: int, att_dim: int):
        super(AttentionHead, self).__init__()

        self.d_model = d_model
        self.att_dim = att_dim

        self.query = nn.Linear(d_model, att_dim)
        self.key = nn.Linear(d_model, att_dim)
        self.value = nn.Linear(d_model, att_dim)

    def forward(self, input: Tensor) -> Tensor:
        Q: Tensor = self.query(input)
        K: Tensor = self.key(input)
        V: Tensor = self.value(input)

        QK = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(torch.Tensor(self.att_dim))
        score = F.softmax(QK)

        return torch.matmul(score, V)


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_active: int, d_model: int, d_attn: int):
        super(SparseMultiHeadAttention, self).__init__()

        self.n_head, self.n_active, self.d_model, self.d_attn = n_head, n_active, d_model, d_attn

        self.heads = nn.ModuleList([AttentionHead(d_model, d_attn) for _ in range(n_head)])

        self.router = nn.Sequential(
            nn.Linear(d_model, n_head),
            nn.Softmax(0)
        )
        self.w_O = nn.Linear(d_attn * n_active, d_model)

    def forward(self, input: Tensor) -> Tensor:
        dist = self.router(input)
        sparse_dist = torch.topk(dist, self.n_active)
        # TODO: test if you need to insert an extra softmax for the sparse dist here so it adds up to one value (or layernorm fixes this?)

        outputs = torch.empty((self.n_active, input.size()[-2], input.size()[-1]))

        for ix, head in enumerate(self.heads):
            outputs[ix] = head(input) * sparse_dist[ix] if sparse_dist[ix] != 0 else torch.zeros_like(input)

        return self.w_O(torch.sum(outputs, 0))