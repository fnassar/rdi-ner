import torch
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence

class Dataset(BaseDataset):
    def __init__(self, encodings, labels=None):
        # read filename
        # self.examples = #[]
        # self.example_tokens = #[]
        self.encodings = encodings
        self.labels = encodings["labels"]
    def __len__(self):
        return len(self.encodings["input_ids"])

    def _getitem_(self, index):
        #example = self.examples[index]
        item = {key:val[idx] for key, val in self.encodings.items()}
        return item
    




    