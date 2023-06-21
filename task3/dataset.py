import torch
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence

class Dataset(BaseDataset):
    def _init_(self, filename):
        # read filename
        # self.examples = #[]
        # self.example_tokens = #[]
        listt = {'id': [], 'tokens': [], 'sentences': [], 'ner_tags': [], 'num_Words': [], 'labels': []}
        self.all_data = {
            'test': pd.DataFrame(listt),
            'eval': pd.DataFrame(listt),
            'train': pd.DataFrame(listt)
        }
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.all_labels = []
        self.id2label = {}
        self.label2id = {}

    def _len_(self):
        return len(self.examples)

    def _getitem_(self, index):
        #example = self.examples[index]
        example = self.example_tokens[index*window_size:(index+1)*window_size]
        example = preprocess(example)
        return example
        