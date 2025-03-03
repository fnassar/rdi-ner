#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoTokenizer, Trainer, AutoModelForTokenClassification, TrainingArguments
import traceback
import json
import glob

# my files/ classes
from dataset import Dataset
from model import Model, compute_metrics
from utils import split_data_file

def main():
    if len(sys.argv)==1 or {"-h", "--help"} & set(sys.argv): log(main.__doc__, end=""); exit(1)
    args = (a for a in sys.argv[1:] if not (a[:1]=="-" and a[1:]))
    kwargs = {k: next(iter(v), True) for k, *v in (a.lstrip("-").split("=",1) for a in sys.argv[1:] if (a[:1]=="-" and a[1:]))}
    return run(*args, **kwargs)

def run(train_name, eval_name=None, out_name="train", epochs=3, train_batch_size=16, eval_batch_size=None, gradient_accumulation_steps=1, config=None):
    epochs = int(epochs)
    train_batch_size = int(train_batch_size)
    eval_batch_size = int(eval_batch_size) if eval_batch_size else train_batch_size
    gradient_accumulation_steps = int(gradient_accumulation_steps)


    tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    
    config=None
    model = Model(True, config=None)
    
    label2id = model.model.config.label2id
    
    # get data from dataset.py
    if(eval_name):
        train_dataset = Dataset(train_name, tokenizer=tokenizer, label2id=label2id)
        eval_dataset = Dataset(eval_name, tokenizer=tokenizer, label2id=label2id)
    else:
        print('split first')
        train_name, eval_name = split_data_file(train_name, 0.8) #(data_to_be_Split, train_eval_data_ratio)

        train_dataset = Dataset(train_name, tokenizer=tokenizer, label2id=label2id)
        eval_dataset = Dataset(eval_name, tokenizer=tokenizer, label2id=label2id)
    
    logging_steps = len(train_dataset)
    output_dir = "{out_name}/out"
    training_args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs =epochs,
        # learning_rate = 2e-5,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{out_name}/logs",
        save_steps=5000,
        gradient_accumulation_steps = gradient_accumulation_steps,
        label_names=["labels"],
        label_smoothing_factor=0.001,
        # evaluation_strategy="steps",
        evaluation_strategy="epoch",
        logging_steps = logging_steps
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn
    )

    st = time.time();log("Training")
    try: train_outputs = trainer.train(resume_from_checkpoint=True if glob.glob(output_dir+"/*/") else None)
    except Exception as e:
        traceback.print_exception(type(e). e, e.__tracenack__)
        import code; code.interact(local=globals()|locals())
    log(f"\x1b[1K\r]Training: done. time: {time.time()-st:.3f}")

    trainer.save_model(f"{out_name}/model")
    tokenizer.save_pretrained(f"{out_name}/model")
    model.model.config.to_json_file(f"{out_name}/model/config.json")




    # trainer.train()


def log(*args, **kwargs): print(*args, file=sys.sterr, flush=True, **kwargs)

if __name__ == "__main__": main()


# main()
    