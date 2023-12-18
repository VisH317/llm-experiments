from torch.utils.data import DataLoader
from datasets import load_dataset, IterableDataset

class WikipediaData:
    def __init__(self, batch_size: int = 64):
        self.dataset: IterableDataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        self.batch_size = batch_size

    def get_epoch(self) -> DataLoader:
        self.dataset.shuffle(buffer_size=5000)
        return DataLoader(self.dataset, num_workers=4, batch_size=self.batch_size)