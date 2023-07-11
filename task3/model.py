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
        # print(inputs)
        outputs = self.model(inputs)
        return outputs

from sklearn.metrics import f1_score, classification_report

def compute_metrics(pred):
    inputs= pred.inputs
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    mask = labels!=-100
    labels, preds = labels[mask], preds[mask]
    f1 = f1_score(labels, preds, average="macro")
    return {"f1":f1}



"""

    id2label = {0: 'B-LOC',1: 'B-MISC', 2: 'B-ORG', 3: 'B-PERS', 4: 'I-LOC', 5: 'I-MISC', 6: 'I-ORG', 7: 'I-PERS', 8: 'O'}

    classes = [v for k, v in sorted(id2label.items())]

    report = classification_report(labels, preds, target_names=classes)

    print(inputs)

    print(len(labels))

    print(report)
"""