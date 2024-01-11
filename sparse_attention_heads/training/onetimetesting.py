import os
import logging
import configparser
import torch
from torch import nn
from tqdm import tqdm
from data import WikipediaData, Vocab, Task
from modules import SparseTransformer
from .prediction_head import TokenClassifier
from transformers import BertTokenizer
import subprocess
# from pytorch_quantization import quant_modules
# import pytorch_quantization.nn as quant_nn
# from .quantization import collect_quant_stats, compute_quant_amax

CFG_FILE = "prelimtest.cfg"
VOCAB_FILE = "../vocab/vocab.txt"

def sim(cfg: str = CFG_FILE, vocab: str = VOCAB_FILE, cuda: bool = True, vocab_stream: bool = True, dtype: torch.dtype = torch.float32) -> tuple[SparseTransformer, list[float], list[float]]:

    # getting train config
    config = configparser.ConfigParser()
    abs_path = os.path.join(os.getcwd(), cfg)
    config.read(abs_path)

    d_model: int = int(config.get("transformer", "d_model"))
    max_len: int = int(config.get("vocab", "max_len"))
    vocab_size: int = int(config.get("vocab", "vocab_size"))

    # initializing modules
    vocab = Vocab.from_config(cfg)
    model = SparseTransformer.from_config(cfg, dtype).to(dtype=dtype)
    if cuda: model.cuda()
    dev_count = torch.cuda.device_count()
    # model = nn.DataParallel(model, device_ids=list(range(dev_count)))

    # classifier
    classifier = TokenClassifier(d_model, max_len, vocab_size).to(dtype=dtype)
    if cuda: classifier.cuda()

    torch.cuda.synchronize()

    print("Starting..")

    losses: list[float] = []
    val_losses: list[float] = []

    last_input = ""

    while last_input != "end":
        last_input = input("Input: ")
        tokens, l = vocab.pad_seq(vocab.tokenizer.tokenize(last_input))
        tokens = torch.tensor([vocab.get_idx(token) for token in tokens]).cuda().unsqueeze(0)

        logits = model(tokens)
        sparse_attn_states = model.get_last_dist()
        output = classifier(logits)
        top: torch.return_types.topk = torch.topk(output, 5)

        print(top.indices.squeeze().cpu().detach().numpy())
        words = [vocab.get_word(word.item()) for word in top.indices.squeeze().cpu().detach().numpy()]

        print("Output: ", list(zip(words, list(top.values.squeeze().cpu().detach().numpy()))))
        print("Attention States: ", sparse_attn_states)
        

        