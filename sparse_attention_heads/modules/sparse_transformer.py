import os
import torch
import configparser
from torch import nn, Tensor, functional as F
from modules.sparse_encoder import SparseEncoderLayers
from modules.sparse_mha import RouteType
from modules.pos_enc import PositionalEncoding
import logging


class Preprocess(nn.Module):

    def __init__(self, n_vocab: int, d_model: int, max_len: int, dropout: float = 0.05):
        super(Preprocess, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_model).to(dtype=torch.float16)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        embed = self.embed(input)
        pos = self.pos_enc(embed)
        return self.dropout(embed + pos)
    

class SparseTransformer(nn.Module):

    def __init__(self, n_layers: int, d_model: int, n_head: int, n_active: int, d_attn: int, d_ff: int, vocab_size: int, max_len: int, dropout: float = 0.1, dropout_embed: float = 0.05, route_type: str = "sum", noise: float = 0.05, noise_step: float = 0.05, hidden_act: str = "relu"):
        super(SparseTransformer, self).__init__()
        self.pre = Preprocess(vocab_size, d_model, max_len, dropout_embed)
        self.encoders = SparseEncoderLayers(n_layers, d_model, n_head, n_active, d_attn, d_ff, dropout, route_type, noise, noise_step, hidden_act)


    def forward(self, input: Tensor) -> Tensor:
        pre = self.pre(input)
        return self.encoders(pre)
    
    def step_epoch(self) -> None:
        self.encoders.step_noise()
    

    @staticmethod
    def from_config(file_path: str):
        config = configparser.ConfigParser()
        abs_path = os.path.join(os.getcwd(), file_path)
        config.read(abs_path)
        
        # vocab config
        vocab_size = int(config.get("vocab", "vocab_size"))
        max_len = int(config.get("vocab", "max_len"))
        dropout_embed = float(config.get("vocab", "dropout_embed"))

        # transformer config
        n_layers = int(config.get("transformer", "n_layers"))
        d_model = int(config.get("transformer", "d_model"))
        n_head = int(config.get("transformer", "n_head"))
        n_active = int(config.get("transformer", "n_active"))
        d_attn = int(config.get("transformer", "d_attn"))
        d_ff = int(config.get("transformer", "d_ff"))
        dropout = float(config.get("transformer", "dropout"))
        route_type = config.get("transformer", "route_type")
        hidden_act = config.get("transformer", "hidden_act")

        noise = float(config.get("transformer", "noise"))
        noise_step = float(config.get("transformer", "noise_step"))

        logging.getLogger().info("initializing sparse heads transformer")
    
        return SparseTransformer(n_layers, d_model, n_head, n_active, d_attn, d_ff, vocab_size, max_len, dropout, dropout_embed, route_type, noise, noise_step, hidden_act)