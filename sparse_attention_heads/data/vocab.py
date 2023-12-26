import os
import re
import torch
import configparser
from torch import Tensor
import random
import math
import logging
from enum import Enum
from transformers import BertTokenizer
    
class Task(str, Enum):
    mlm = "mlm"
    clm = "clm"
    clm_rand = "clm_rand"

# THIS NEEDS WORK: special characters not being separated, dates, dashes, missing punctuation in the list, commas

class Vocab:

    def __init__(self, file_path: str, vocab_size: int, max_len: int, num_unk: int, pad: str = "[PAD]", start: str = "[CLS]", end: str = "[SEP]", mask: str = "[MASK]", unk: str = "[UNK]"):
        self.vocab_size, self.max_len, self.num_unk = vocab_size, max_len, num_unk
        self.pad, self.start, self.end, self.mask, self.unk = pad, start, end, mask, unk

        self.current_unused = 0

        self.special = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", ".", "-", "/"]

        self.table = { char:ord(" ") for char in range(1, 32) }

        self.tokenizer = BertTokenizer(file_path, max_len=max_len) # TODO: BYTE-PAIR ENCODING

        with open(file_path) as f:
            self.vocab = f.read().split("\n")
    

    @staticmethod
    def from_config(file_path: str):
        config = configparser.ConfigParser()
        abs_path = os.path.join(os.getcwd(), file_path)
        config.read(abs_path)

        # vocab config
        vocab_size = int(config.get("vocab", "vocab_size"))
        max_len = int(config.get("vocab", "max_len"))
        n_unused = int(config.get("vocab", "n_unused"))
        vocab_path = config.get("vocab", "vocab_path")

        logging.getLogger().info("set up vocab + parser")

        return Vocab(vocab_path, vocab_size, max_len, n_unused)


    def tokenize(self, item: str) -> list[str]: 
        for s in self.special: item.replace(s, f" {s} ")

        # replace escape chars
        item = item.translate(self.table)

        return [tok for tok in item.split(" ") if len(tok) > 0]


    def pad_seq(self, item: list[str], take_last: bool = False) -> tuple[list[str], int]:
        l = len(item)

        if len(item) < self.max_len:
            item += [self.pad] * (self.max_len-l)
        
        return (item[:self.max_len] if not take_last else item[-self.max_len:]), l
    

    def encode(self, item: list[str]) -> Tensor:

        ret = torch.empty((self.max_len))
        toks = []

        for i, tok in enumerate(item):
            ret[i], checked_tok = self.one_hot(tok)
            toks.append(checked_tok)

        return ret


    def one_hot(self, tok: str) -> int:
        checked_tok = tok
        if tok.strip() not in self.vocab:
            if self.current_unused > self.num_unk: checked_tok = self.unk
            else:
                self.vocab[self.current_unused + 1] = tok
                self.current_unused += 1
        
        return self.vocab.index(checked_tok), checked_tok


    def format_mlm(self, item: str) -> tuple[Tensor, Tensor]:
        toks, l = self.pad_seq(self.tokenizer.tokenize(item))
        ix = random.randint(0, l-1)
        y = toks[ix]
        toks[ix] = self.mask
        return self.encode(toks), self.one_hot(y)[0]
    
    def format_clm(self, item: str) -> tuple[Tensor, Tensor]:
        toks = self.tokenizer.tokenize(item)
        y = toks[-1]
        toks[-1] = self.mask
        pad_toks, l = self.pad_seq(toks)
        return self.encode(pad_toks), self.one_hot(y)[0]
    
    def format_clm_rand_length(self, item: str) -> tuple[Tensor, Tensor]:
        toks = self.tokenizer.tokenize(item)
        ix = random.randint(math.floor(len(toks) * 0.5), len(toks))
        new_seq = toks[:ix]
        y = new_seq[-1]
        new_seq.pop()
        pad_toks, l = self.pad_seq(new_seq)
        return self.encode(pad_toks), self.one_hot(y)[0]
    
    def format_batch(self, items: list[str], task: Task) -> tuple[Tensor, Tensor]:
        X, y = torch.empty((len(items), self.max_len)), torch.empty((len(items)))
        # tokens = self.tokenizer(items)
        # return torch.tensor(self.tokenizer(items).input_ids)

        # return tokens

        for ix, item in enumerate(items):
            i_X, i_y = self.format_mlm(item) if task == Task.mlm else self.format_clm(item) if task == Task.clm else self.format_clm_rand_length(item)
            X[ix] = i_X
            y[ix] = i_y
        
        return X.to(dtype=torch.int64), y.to(dtype=torch.int64)
    

    


    