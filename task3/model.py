# -*- coding: utf-8 -*-

import torch
from transformers import BertConfig, BertForTokenClassification, AutoTokenizer
# from torch.nn.modules import module
from torch import nn

class Model(nn.Module):
    def __init__(self, flag, config=None):
        super().__init__()  # Call parent class __init__() method
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if flag:
            self.model = BertForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
        else:
            self.model = BertForTokenClassification(BertConfig.from_json_file(config)).to(device)
      # def get_model(self):
      #   return self.model
      # self.base_model = self.model.layers[:11]
      # self.dense1 = nn.Linear(300, 10)

    def forward(self, inputs):
        outputs = self.model(inputs)
        # logits = self.dense1(outputs)
        return outputs

from sklearn.metrics import f1_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.preditions.argmax(-1)
  f1 = f1_score(labels, preds, average="weighted")
  return {"f1":f1}