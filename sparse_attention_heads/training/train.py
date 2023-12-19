import os
import configparser
from data import WikipediaData, Vocab, Task
from modules import SparseTransformer

CFG_FILE = "train.cfg"


def train():
    # getting train config
    config = configparser.ConfigParser()
    abs_path = os.path.join(os.getcwd(), CFG_FILE)
    config.read(abs_path)

    task: Task = config.get("train", "task")
    batch_size: int = config.get("train", "batch_size")
    n_epochs: int = config.get("train", "n_epochs")


    # initializing modules
    dataset = WikipediaData(batch_size)
    vocab = Vocab.from_config(CFG_FILE)
    model = SparseTransformer.from_config(CFG_FILE)

    for epoch in range(n_epochs):
        # get shuffled dataloader for the epoch 
        # TODO: add validation data lol

        loader = dataset.get_epoch()

        for()
        