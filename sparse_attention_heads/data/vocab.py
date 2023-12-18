import torch
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
            checked_tok = tok
            if tok.strip() not in self.vocab:
                if self.current_unused > self.num_unk: checked_tok = self.unk
                else:
                    self.vocab[self.current_unused + 1] = tok
                    self.current_unused += 1
            
            ten = torch.zeros((self.vocab_size))
            ten[self.vocab.index(checked_tok)] = 1
            ret[i] = ten
        
        return ret


    def format_mlm(self, item: str) -> Tensor:
        toks, l = self.pad(self.tokenize(item))
        ix = random.randint(0, l-1)
        toks[ix] = self.mask
        return self.encode(toks)
    
    def format_clm(self, item: str) -> Tensor:
        toks = self.tokenize(item)
        toks.append(self.mask)
        pad_toks, l = self.pad(toks)
        return self.encode(pad_toks)
    
    def format_clm_rand_length(self, item: str) -> Tensor:
        toks = self.tokenize(item)
        ix = random.randint(math.floor(len(toks) * 0.5), len(toks))
        pad_toks, l = self.pad(toks[:ix])
        return self.encode(pad_toks)
    

    


    