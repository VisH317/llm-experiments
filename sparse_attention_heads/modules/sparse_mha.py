import torch
from torch import nn, Tensor, functional as F
from enum import Enum
# from flash_attn.flash_attention import flash_attn_unpadded_qkvpacked_func

class RouteType(Enum):
    sum = "sum"
    mean = "mean"
    att = "att"



class SparseMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_active: int, d_model: int, d_attn: int, route_type: str, noise: float, noise_step: float):
        super(SparseMultiHeadAttention, self).__init__()
        self.n_head, self.n_active, self.d_model, self.d_attn = n_head, n_active, d_model, d_attn

        self.route_type: RouteType = RouteType(route_type)

        if self.route_type == RouteType.att:
            self.att = nn.Linear(d_model, 1)

        self.query = nn.Linear(d_model, d_attn * n_head) # normally each head takes a section of the total embedding input, here we're splitting it up more
        self.key = nn.Linear(d_model, d_attn * n_head) # TODO: try with chunking and passing into each head this way, or try clusters
        self.value = nn.Linear(d_model, d_attn * n_head) # TODO: fix this to parallelize

        self.noise = noise
        self.noise_step = noise_step

        # self.head_in = [nn.Parameter(torch.zeros(d_model, d_attn * 3)) for _ in range(n_head)]

        self.router = nn.Linear(d_model, n_head) # need to experiment with pooling in this layer here (max pooling, sum pooling, or additive attention)
        self.w_O = nn.Linear(d_attn * n_active, d_model)

        self.softmax = nn.Softmax(-1)
        self.router_softmax = nn.Softmax(1)
        self.router_norm = nn.LayerNorm((n_head))
        self.swish = nn.SiLU()

        self.dropout = 0.05

        self.dist_out = []

        self._reset_parameters()

        self.last_dist = None

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.router.weight) # need to find a better weight init for this

    def step_noise(self):
        self.noise *= self.noise_step # TODO: test confidence sampling here later

    def forward(self, input: Tensor) -> Tensor:
        # input size: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = input.size()
        dist = self.route(input) # returns dim (batch_size, n_heads)
        self.dist_out.append(dist)
        # TODO: CONFIDENCE SAMPLING OR RANDOM NOISE TO SIMULATE TRYING OTHER HEADS, OR DIFFERENT INITIALIZATION
        sparse_dist_val = torch.topk(dist, self.n_active, -1) # returns dim (batch_size, n_active)
        sparse_dist_idx = sparse_dist_val.indices

        sparse_dist = torch.full_like(dist, -1).scatter(-1, sparse_dist_idx, dist)

        dist_dense = dist[sparse_dist != -1].reshape(batch_size, self.n_active).unsqueeze(-1).unsqueeze(-1) # size: (batch_size, n_head, 1, 1)

        sparse_dist = sparse_dist.repeat(seq_len, self.d_attn, 1, 1).permute(2, 3, 0, 1)

        Q_unfiltered = self.query(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3) # (batch_size, n_head, seq_len, d_attn)
        K_unfiltered = self.key(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3)
        V_unfiltered = self.value(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3)


        Q = Q_unfiltered[sparse_dist != -1].reshape(batch_size, self.n_active, seq_len, self.d_attn)
        K = K_unfiltered[sparse_dist != -1].reshape(batch_size, self.n_active, seq_len, self.d_attn)
        V = V_unfiltered[sparse_dist != -1].reshape(batch_size, self.n_active, seq_len, self.d_attn)

        # each batch has its own conditional
        # O = flash_attn_func(Q, K, V, dropout_p=self.dropout)
        O = self.scaled_dot_product_attention(Q, K, V)
        O_gated = dist_dense * O
        O_final = O_gated.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_active * self.d_attn) # batch_size, n_active, seq_len, d_attn
        # TODO: experiment with the output layer and see if it can generalize to sparse (prob not, so try separate Os instead later)
        Z = self.w_O(O_final) # batch_size, seq_len, d_attn
        
        self.last_dist = dist

        return Z

    # router functions

    def get_last_dist(self): return self.last_dist


    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:

        batch_size, n_active, seq_len, d_attn = Q.size()

        K_T = K.transpose(-2, -1) 

        QK = torch.matmul(Q, K_T)
        QK_scaled = QK / (d_attn ** 0.5)
        score = self.softmax(QK_scaled)  # TODO: test MQA and GQA

        return torch.matmul(score, V)


    def route(self, input: Tensor) -> Tensor:
        if self.route_type == RouteType.sum:
            out = torch.sum(self.router(input), 1) # dim (batch, seq, n_head)
            return self.router_softmax(self.router_norm(out) + self.create_noise(out))
        
        elif self.route_type == RouteType.mean:
            out = torch.mean(self.router(input), 1) # dim (batch, seq, n_head)
            return  self.router_softmax(self.router_norm(out) + self.create_noise(out))
        
        elif self.route_type == RouteType.att:
            return self.att_router(input)
        else: raise TypeError("Route pooling type not recognized")

    def create_noise(self, out: Tensor) -> Tensor:
        return ((torch.rand_like(out)) * (self.noise))

    def att_router(self, input: Tensor) -> Tensor:
        if self.route_type != RouteType.att: raise TypeError("Wrong routing type, attention called when not initialized")
        att: Tensor = self.att(input)
        att_weights = att.softmax(1)
        weighted = att_weights * input
        summed = torch.sum(weighted, 1)
        return self.router_softmax(self.router_norm(summed) + self.create_noise(summed)) # dim (batch_size, n_head)