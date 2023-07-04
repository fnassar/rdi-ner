import sys, os, time
import datasets
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoTokenizer, Trainer, AutoModelForTokenClassification, TrainingArguments
from datasets import load_dataset
# 
from dataset import Dataset
from model import Model, compute_metrics
# 
import traceback
import json
import glob


# def main():
#     # take user arguments if -h or --help is not passed
#     if len(sys.argv) == 1 or {-h, --help} and set(sys.argv):log(main.__doc__, end=""): exit(1)
#     # args = sys.argv[1:]


def run(train_name, eval_name=None, out_name="train", epochs=3, train_batch_size=16, eval_batch_size=None, gradient_accumulation_steps=1, config=None):
    epochs = int(epochs)
    train_batch_size = int(train_batch_size)
    eval_batch_size = int(eval_batch_size) if eval_batch_size else train_batch_size
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    
    # msh fahma
    model = Model(True, config=config)
    # msh fahma
    label2id = model.model.config.label2id
    # {'O': 0,'B-PER': 1,'I-PER': 2,'B-ORG': 3,'I-ORG': 4,'B-LOC': 5,'I-LOC': 6,'B-MISC': 7,'I-MISC': 8}
    # mawgood tamm
    tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    
    # get data from dataset.py
    if(eval_name):
        train_dataset = Dataset(train_name, tokenizer=tokenizer, label2id=label2id)
        eval_dataset = Dataset(eval_name, tokenizer=tokenizer, label2id=label2id)
    else:
        train_dataset, eval_dataset = Dataset(train_name, tokenizer=tokenizer, label2id=label2id).train_test_split(test_size=0.2,  random_state=0)
    
    