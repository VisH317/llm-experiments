from training.train_deepspeed import train_deepspeed, CFG_FILE, VOCAB_FILE, DS_FILE
import logging
import sys
import torch

# imports for kaggle
from modules.pos_enc import PositionalEncoding
from modules.sparse_encoder import SparseEncoder, SparseEncoderLayers, SparseMultiHeadAttention
from modules.sparse_transformer import SparseTransformer

from data.vocab import Vocab
from data.data import WikipediaData

    
logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s (%(asctime)s)")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format("logs", "train"))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

train_deepspeed(CFG_FILE, VOCAB_FILE, DS_FILE, True, False, torch.bfloat16)