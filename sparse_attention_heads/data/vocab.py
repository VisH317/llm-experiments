import os
import torch
import configparser
from torch import Tensor
import random
import math

    
class Vocab:

    def __init__(self, file_path: str, vocab_size: int, max_len: int, num_unk: int, pad: str = "[PAD]", start: str = "[CLS]", end: str = "[SEP]", mask: str = "[MASK]", unk: str = "[UNK]"):
        self.vocab_size, self.max_len, self.num_unk = vocab_size, max_len, num_unk
        self.pad, self.start, self.end, self.mask, self.unk = pad, start, end, mask, unk

        self.current_unused = 0

        self.special = ["!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", ".", "-", "/"]

        with open(file_path) as f:
            self.vocab = f.read().split("\n")
    

    @staticmethod
    def from_config(file_path: str):
        config = configparser.ConfigParser()
        abs_path = os.path.join(os.getcwd(), file_path)
        config.read(abs_path)

        # vocab config
        vocab_size = config.get("vocab", "vocab_size")
        max_len = config.get("vocab", "max_len")
        n_unused = config.get("vocab", "n_unused")
        vocab_path = config.get("vocab", "vocab_path")

        return Vocab(vocab_path, vocab_size, max_len, n_unused)


    def tokenize(self, item: str) -> list[str]: 
        for s in self.special: item.replace(s, f" {s} ")
        return [tok for tok in item.split(" ") if len(tok) > 0]


    def pad(self, item: list[str], take_last: bool = False) -> tuple[list[str], int]:
        l = len(item)

        if len(item) < 64:
            item += [self.pad] * (64-l)
        
        return (item[:64] if not take_last else item[-64:]), l
    

    def encode(self, item: list[str]) -> Tensor:

        ret = torch.empty((self.max_len, self.vocab_size))

        for i, tok in enumerate(item):
            ret[i] = self.one_hot(tok)
        
        return ret

    def one_hot(self, tok: str) -> Tensor:
        checked_tok = tok
        if tok.strip() not in self.vocab:
            if self.current_unused > self.num_unk: checked_tok = self.unk
            else:
                self.vocab[self.current_unused + 1] = tok
                self.current_unused += 1
        
        ten = torch.zeros((self.vocab_size))
        ten[self.vocab.index(checked_tok)] = 1
        return ten


    def format_mlm(self, item: str) -> Tensor:
        toks, l = self.pad(self.tokenize(item))
        ix = random.randint(0, l-1)
        y = toks[ix]
        toks[ix] = self.mask
        return self.encode(toks), self.one_hot(y)
    
    def format_clm(self, item: str) -> Tensor:
        toks = self.tokenize(item)
        y = toks[-1]
        toks[-1] = self.mask
        pad_toks, l = self.pad(toks)
        return self.encode(pad_toks), self.one_hot(y)
    
    def format_clm_rand_length(self, item: str) -> Tensor:
        toks = self.tokenize(item)
        ix = random.randint(math.floor(len(toks) * 0.5), len(toks))
        new_seq = toks[:ix]
        y = new_seq[-1]
        new_seq[-1] = self.mask
        pad_toks, l = self.pad(new_seq)
        return self.encode(pad_toks), self.one_hot(y)
    

    


    