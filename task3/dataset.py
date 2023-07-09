import torch
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence

from utils import preprocess

class Dataset(BaseDataset):
    def __init__(self, in_name, tokenizer, label2id):
        super().__init__()
        self.tokenizer = tokenizer
        self.example_words = []
        
        data = open(in_name).readlines()
        sentence = []

        for line in data:
            # print(line.strip('\n').split(), len(line), end="___")
            if len(line)>1:
                word, label = line.strip('\n').split(' ')
                # print(word, label)
                sentence.append((word, label2id[label]))
            else:
                word, label = (line.strip('\n'), 'O')
                sentence.append((word, label2id[label]))
                self.example_words.append(sentence)
                sentence = []
        self.label2id = label2id
        
    def __len__(self):
        return len(self.example_words)-1

    def __getitem__(self, index):
      item = self.example_words[index]
      words = [word for word, _ in item]
      labels = [label for _, label in item]
      indices, indices_labels = [], []

      for word_indices, word_label in zip(preprocess(words, self.tokenizer), labels):
        indices.extend(word_indices)
        indices_labels.extend([word_label]*len(word_indices))

      return words, labels , indices, indices_labels
    
    def collate_fn(self, batch):
      """
      batch: list[tuple]
      [example]
      example: [input, output]
      """
      print([torch.stack(item[2]) for item in batch], [torch.tensor(item[3]) for item in batch])
      inputs = [torch.stack(item[2]) for item in batch]
      labels = [torch.tensor(item[3]) for item in batch]

      inputs = pad_sequence(inputs, batch_first=True)
      labels = pad_sequence(labels, batch_first=True)
      return {"inputs":inputs, "labels": labels}
    



# <<<<<<< HEAD
#     def _init_(self,filename ):
#         # read filename
#         # self.examples = #[]
#         # self.example_tokens = #[]
#      def preprocess(self): # add data into dataframe and divide into sentences
#         self.read('/content/drive/MyDrive/rdi-data/rdi-ner/task3/ANERcorp-CamelLabSplits/')
    
#     def read(self, path):
#         data = open(path+'ANERCorp_CamelLab_train.txt').readlines()
#         data_test = open(path+'ANERCorp_CamelLab_test.txt').readlines()
#         data2 = []
#         data_test2 = []
#         for line in data:
#             data2.append(line.strip('\n').split(' '))

#         for line in data_test:
#             data_test2.append(line.strip('\n').split(' '))

#         self.df_train = pd.DataFrame(data2, columns=['text', 'label'])
#         self.df_test = pd.DataFrame(data_test2, columns=['text', 'label'])

#         self.add_data('train')
#         self.add_data('test')
    
#     def add_data(self, name):

#         num_Words = 0
#         labels = []
#         ner_tags = []
#         tokens = []
#         sentence = ""
#         temp = self.df_test
#         if (name == 'test'):
#             temp = self.df_train
#         for column, item in temp.iterrows():

#             if (item['text'] == ''):

#                 self.all_data[name].loc[len(self.all_data[name].index)] = [len(
#                     self.all_data[name].index), tokens, sentence, ner_tags, num_Words, labels]
#                 num_Words = 0
#                 labels = []
#                 ner_tags = []
#                 tokens = []
#                 sentence = ""

#             else:
#                 # Access column name using 'column' and column values using 'values'
#                 if (item['label'] not in labels and item['label'] != None and item['label'] != ''):
#                     labels.append(item['label'])

#                 sentence += item['text'] + " "
#                 tokens.append(item['text'])  # words
#                 ner_tags.append(item['label'])  # tokens
#                 num_Words += 1
#                 if (item['label'] not in self.all_labels):
#                     self.all_labels.append(item['label'])   

#     def _len_(self):
#         return len(self.examples)
# =======
    