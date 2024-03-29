from training.onetimetesting import sim, CFG_FILE, VOCAB_FILE
import logging
import sys
import torch

# imports for kaggle
from modules.pos_enc import PositionalEncoding
from modules.sparse_encoder import SparseEncoder, SparseEncoderLayers, SparseMultiHeadAttention
from modules.sparse_transformer import SparseTransformer

from data.vocab import Vocab
from data.data import WikipediaData

if __name__ == "__main__":
    
    logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s (%(asctime)s)")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format("logs", "train"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

    sim(CFG_FILE, VOCAB_FILE, True, False, torch.float16)