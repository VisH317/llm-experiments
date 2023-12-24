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

CFG_FILE = "train.cfg"
VOCAB_FILE = "../vocab/vocab.txt"

def train(cfg: str = CFG_FILE, vocab: str = VOCAB_FILE) -> tuple[SparseTransformer, list[float], list[float]]:

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
    dataset = WikipediaData(batch_size + val_size)
    vocab = Vocab.from_config(CFG_FILE)
    model = SparseTransformer.from_config(CFG_FILE).to(dtype=torch.float32)

    # optim config
    optim = torch.optim.AdamW(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.8)

    # loss config
    loss_func = nn.CrossEntropyLoss()

    # classifier
    classifier = TokenClassifier(d_model, max_len, vocab_size)

    # helper function to preprocess a batch (closure for the vocab object)
    def process_batch(data: list[str]):
        train_data, val_data = data[:batch_size], data[batch_size:]
        train_X, train_y = vocab.format_batch(train_data, task)
        val_X, val_y = vocab.format_batch(val_data, task)
        return train_X, train_y, val_X, val_y

    losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        logging.getLogger().info(f"Loading epoch {epoch}...")

        loader = dataset.get_epoch()

        epoch_running_losses = []
        epoch_validation_losses = []

        format_desc = lambda: f"Epoch {epoch}, Loss: {(epoch_running_losses[-1] if len(epoch_running_losses) > 0 else 0) if len(epoch_running_losses) <= 5 else sum(epoch_running_losses)/5}, Val: {(epoch_validation_losses[-1] if len(epoch_validation_losses) > 0 else 0) if len(epoch_validation_losses) <= 5 else sum(epoch_validation_losses)/5}"

        for ix, data in tqdm(enumerate(loader), desc=format_desc(), total=max_ep_len): # use len(dataset for more robust)
            train_X, train_y, val_X, val_y = process_batch(data)

            optim.zero_grad()

            logits = model(train_X)
            out = classifier(logits)

            loss = loss_func(out, train_y)
            epoch_running_losses.append(loss)

            loss.backward()
            optim.step()

            # validation loop
            if ix % val_step:
                with torch.no_grad():
                    logits = model(val_X)
                    out = classifier(logits)
                    loss = loss_func(out, val_y)
                    epoch_validation_losses.append(loss)
            
            if ix >= max_ep_len: break
    

        losses.extend(epoch_running_losses)
        val_losses.extend(epoch_validation_losses)
        scheduler.step()

    return model, losses, val_losses

        