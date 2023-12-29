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
import deepspeed
import subprocess
import argparse
# from pytorch_quantization import quant_modules
# import pytorch_quantization.nn as quant_nn
# from .quantization import collect_quant_stats, compute_quant_amax

CFG_FILE = "train.cfg"
VOCAB_FILE = "../vocab/vocab.txt"
DS_FILE = "ds_config.json"

ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.0001
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 1
    }
}

args = argparse.Namespace()
args.num_gpus = 2
args.cuda = True

class DeepspeedModel(nn.Module):

    def __init__(self, cfg: str, dtype: torch.dtype, classifier: nn.Sequential, cuda: bool = True):
        super().__init__()

        self.transformer = SparseTransformer.from_config(cfg, dtype).to(dtype=dtype)
        if cuda: self.transformer.cuda()
        self.classifier = classifier
        if cuda: self.classifier.cuda()

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, train_X, train_y):
        logits = self.transformer(train_X)
        out = self.classifier(logits)

        return self.loss(out, train_y)


def train_deepspeed(cfg: str = CFG_FILE, vocab: str = VOCAB_FILE, ds: str = DS_FILE, cuda: bool = True, vocab_stream: bool = True, dtype: torch.dtype = torch.float32) -> tuple[SparseTransformer, list[float], list[float]]:

    # getting train config
    config = configparser.ConfigParser()
    abs_path = os.path.join(os.getcwd(), cfg)
    config.read(abs_path)

    task: Task = config.get("train", "task")
    batch_size: int = int(config.get("train", "batch_size"))
    n_epochs: int = int(config.get("train", "n_epochs"))
    lr: float = float(config.get("train", "lr"))
    val_step: int = int(config.get("train", "val_step"))
    val_size: int = int(config.get("train", "val_size"))
    max_ep_len: int = int(config.get("train", "max_ep_len"))

    d_model: int = int(config.get("transformer", "d_model"))
    max_len: int = int(config.get("vocab", "max_len"))
    vocab_size: int = int(config.get("vocab", "vocab_size"))

    # initializing modules
    dataset = WikipediaData(batch_size + val_size, vocab_stream)
    vocab = Vocab.from_config(cfg)
    # transformer = SparseTransformer.from_config(cfg, dtype).to(dtype=dtype)
    # if cuda: transformer.cuda()

    # classifier
    classifier = TokenClassifier(d_model, max_len, vocab_size).to(dtype=dtype)
    if cuda: classifier.cuda()

    model = DeepspeedModel(cfg, dtype, classifier, cuda)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(torch.cuda.memory_summary())

    process = subprocess.run(["nvidia-smi"])

    print(process.stdout)

    params = list(model.named_parameters())
    params = [p for n, p in params]
    model_engine, optim, _, _ = deepspeed.initialize(args=args, model=nn.DataParallel(model), model_parameters=params, config=ds)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.8)

    loss_func = nn.CrossEntropyLoss()

    # helper function to preprocess a batch (closure for the vocab object)
    def process_batch(data: list[str]):
        train_data, val_data = data[:batch_size], data[batch_size:]
        train_X, train_y = vocab.format_batch(train_data, task)
        val_X, val_y = vocab.format_batch(val_data, task)
        if not cuda: return train_X, train_y, val_X, val_y
        else:
            device = torch.device("cuda")
            return train_X.to(device), train_y.to(device), val_X.to(device), val_y.to(device)

    losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        logging.getLogger().info(f"Loading epoch {epoch}...")

        if epoch != 0: loader = dataset.get_epoch()

        epoch_running_losses = []
        epoch_validation_losses = []

        format_desc = lambda: f"Epoch {epoch}, Loss: {(epoch_running_losses[-1] if len(epoch_running_losses) > 0 else 0) if len(epoch_running_losses) <= 5 else sum(epoch_running_losses)/5}, Val: {(epoch_validation_losses[-1] if len(epoch_validation_losses) > 0 else 0) if len(epoch_validation_losses) <= 5 else sum(epoch_validation_losses)/5}"

        for ix, data in tqdm(enumerate(loader), desc=format_desc(), total=max_ep_len): # use len(dataset for more robust)
            train_X, train_y, val_X, val_y = process_batch(data)

            # optim.zero_grad()

            loss = model_engine(train_X, train_y)
            epoch_running_losses.append(loss.item())
            
            model_engine.backward(loss)
            model_engine.step()

            # validation loop
            if ix % val_step:
                with torch.no_grad():
                    logits = model(val_X)
                    out = classifier(logits)
                    loss = loss_func(out.float(), val_y)
                    epoch_validation_losses.append(loss.item())
            
            if ix >= max_ep_len: break
    

        losses.extend(epoch_running_losses)
        val_losses.extend(epoch_validation_losses)
        scheduler.step()
        model.step_epoch() # step noise value



    return model, losses, val_losses, model.get_route_vals()

        