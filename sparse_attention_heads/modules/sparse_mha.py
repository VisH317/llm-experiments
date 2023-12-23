import torch
from torch import nn, Tensor, functional as F
from enum import Enum

class RouteType(Enum):
    sum = "sum"
    mean = "mean"
    att = "att"


class AttentionHead(nn.Module):

    def __init__(self, d_model: int, att_dim: int):
        super(AttentionHead, self).__init__()

        self.d_model = d_model
        self.att_dim = att_dim

        # self.query = nn.Linear(d_model, att_dim)
        # self.key = nn.Linear(d_model, att_dim)
        # self.value = nn.Linear(d_model, att_dim)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        # Q: Tensor = self.query(input)
        # K: Tensor = self.key(input)
        # V: Tensor = self.value(input)

        print(Q.size(), ", ", K.size())

        K_T = K.transpose(-2, -1) 

        QK = torch.matmul(Q, K_T)
        print(QK.size())
        QK_scaled = QK / (self.att_dim ** 0.5)
        score = QK_scaled.softmax(-1)

        return torch.matmul(score, V)


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_active: int, d_model: int, d_attn: int, route_type: str):
        super(SparseMultiHeadAttention, self).__init__()

        self.n_head, self.n_active, self.d_model, self.d_attn = n_head, n_active, d_model, d_attn

        self.heads = nn.ModuleList([AttentionHead(d_model, d_attn) for _ in range(n_head)])

        self.route_type: RouteType = RouteType(route_type)

        if self.route_type == RouteType.att:
            self.att = nn.Linear(d_model, 1)

        self.query = nn.Linear(d_model, d_attn * n_head) # normally each head takes a section of the total embedding input, here we're splitting it up more
        self.key = nn.Linear(d_model, d_attn * n_head) # TODO: try with chunking and passing into each head this way, or try clusters
        self.value = nn.Linear(d_model, d_attn * n_head) # TODO: fix this to parallelize

        # self.head_in = [nn.Parameter(torch.zeros(d_model, d_attn * 3)) for _ in range(n_head)]

        self.router = nn.Linear(d_model, n_head) # need to experiment with pooling in this layer here (max pooling, sum pooling, or additive attention)
        self.w_O = nn.Linear(d_attn * n_active, d_model)

    def _reset_parameters(self):
        pass # initialize head_in here

    def forward(self, input: Tensor) -> Tensor:
        # input size: (batch_size, seq_len, d_model)
        dist = self.route(input) # returns dim (batch_size, n_heads)
        sparse_dist_val = torch.topk(dist, self.n_active, -1) # returns dim (batch_size, n_active)
        sparse_dist_idx = sparse_dist_val.indices

        Q = self.query(input).reshape(input.size()[0], input.size()[1], self.n_head, self.d_attn).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, d_attn)
        K = self.key(input).reshape(input.size()[0], input.size()[1], self.n_head, self.d_attn).permute(0, 2, 1, 3)
        V = self.value(input).reshape(input.size()[0], input.size()[1], self.n_head, self.d_attn).permute(0, 2, 1, 3)

        # TODO: test if you need to insert an extra softmax for the sparse dist here so it adds up to one value (or layernorm fixes this?)

        sparse_dist = torch.zeros_like(dist).scatter(-1, sparse_dist_idx, dist)
        sparse_dist = sparse_dist.softmax(1).unsqueeze(-1).unsqueeze(-1) # softmax not confirmed

        Q_dist, K_dist, V_dist = Q * sparse_dist, K * sparse_dist, V * sparse_dist

        print("sizess: ", Q_dist.size())

        Q_dist_filtered, K_dist_filtered, V_dist_filtered = torch.empty((Q.size()[0], self.n_head, Q.size()[2], self.d_attn))

        for b in range(Q.size()[0]): # might need a custom CUDA kernel to do this parallelizedly
            Q_b_filt = Q_dist[b, Q_dist]

        # each batch has its own conditional

        outputs = torch.empty((input.size()[0], self.n_active, input.size()[-2], self.d_attn))

        for ix in range(input.size()[0]):
            outputs[ix] = self.compute_head(input[ix], sparse_dist[ix], sparse_dist_idx[ix])

        return self.w_O(torch.sum(outputs, -2))

    # router functions

    def compute_head(self, input: Tensor, sparse_dist: Tensor, sparse_dist_idx: Tensor) -> Tensor:
        outputs = torch.empty((self.n_active, input.size()[-2], self.d_attn))

        for ix in sparse_dist_idx:
            print(list(sparse_dist_idx))
            outputs[ix] = self.heads[ix](input) * sparse_dist[ix] # if sparse_dist[ix] != 0 else torch.zeros(input.size()[-2], self.d_attn)

        return outputs


    def route(self, input: Tensor) -> Tensor:
        print(self.route_type, ", ", type(self.route_type), ", ", RouteType.sum)
        if self.route_type == RouteType.sum:
            out = self.router(input) # dim (batch, seq, n_head)
            return torch.sum(out, 1).softmax(1)
        elif self.route_type == RouteType.mean:
            out = self.router(input)
            return torch.mean(out, 1).softmax(1)
        elif self.route_type == RouteType.att:
            return self.att_router(input)
        else: raise TypeError("Route pooling type not recognized")

    def att_router(self, input: Tensor) -> Tensor:
        if self.route_type != RouteType.att: raise TypeError("Wrong routing type, attention called when not initialized")
        att: Tensor = self.att(input)
        att_weights = att.softmax(1)
        weighted = att_weights * input
        return torch.sum(weighted, 1).softmax(1) # dim (batch_size, n_head)