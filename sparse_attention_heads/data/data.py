import os
import configparser
from torch.utils.data import DataLoader
from datasets import load_dataset, IterableDataset
import logging

def collate(items: list[dict]):
    return [item["text"] for item in items]

class WikipediaData:
    def __init__(self, batch_size: int = 64, val_size: int = 4):
        logger = logging.getLogger()
        logger.info("Loading dataset")
        self.dataset: IterableDataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        logger.info("Dataset loaded!")
        self.batch_size = batch_size
        self.val_size = val_size

    def get_epoch(self) -> DataLoader:
        self.dataset.shuffle(buffer_size=5000)
        return DataLoader(self.dataset, num_workers=1, batch_size=self.batch_size+self.val_size, collate_fn=collate)
    
    def __len__(self) -> int:
        return 6458670 # this is just from the dataset for en samples, probably should manually count this but its only for tqdm anyway :)