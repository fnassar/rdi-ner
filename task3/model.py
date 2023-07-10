# -*- coding: utf-8 -*-

import torch
from transformers import BertConfig, BertForTokenClassification, AutoTokenizer
# from torch.nn.modules import module
from torch import nn

class Model(nn.Module):
    def __init__(self, flag, config=None):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if flag:
            self.model = BertForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
        else:
            self.model = BertForTokenClassification(BertConfig.from_json_file(config)).to(device)
    

    def forward(self, inputs, labels=None):
        outputs = self.model(inputs)
        return outputs

from sklearn.metrics import f1_score

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    mask = labels!=-100

    labels, preds = labels[mask], preds[mask]


    f1 = f1_score(labels, preds, average="macro")
    return {"f1":f1}



