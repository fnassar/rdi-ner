#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer

model = NERModel('bert', '/content/drive/MyDrive/Colab Notebooks/RDI/Final_NER')

text = input("Enter some Arabic text: ")

predictions, _ = model.predict([text])

print(predictions)




# import sys, os
# import time

# import torch
# from model import Model
# from utils import preprocess
# from transformers import AutoTokenizer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def main():
#     """
#     infer ner model

#     Parameters
#     ----------
#     in_name:str
#         Input filename or - stdin
#     out_name: str
#         Output filename or - stout
#     """
#     if len(sys.argv)==1 or {"-h", "--help"} & set(sys.argv): log(main.__doc__, end=""); exit(1)
#     args = (a for a in sys.argv[1:] if not (a[:1]=="-" and a[1:]))
#     kwargs = {k: next(iter(v), True) for k, *v in (a.lstrip("-").split("=",1) for a in sys.argv[1:] if (a[:1]=="-" and a[1:]))}
#     return run(*args, **kwargs)

# def run(in_name, out_name, model_path, config=None):
#     tokenizer, model = init(model_path, config)

#     st = time.time()
#     with open(in_name, encoding="utf-8-sig") if in_name!="-" else sys.stdin as in_s, \
#         open(out_name, "w", encoding="utf-8") if out_name!="-" else sys.stdout as out_s:
#             for i, in_line in enumerate(iter(in_s.readline, "")):
#                 if (i)%1000==0: log_read(in_s, i)
#                 outputs = infer(in_line, tokenizer)
#                 out_line = ", ".join(outputs)+"\n"
#                 out_s.write(out_line)
#     log(f"\x1b[1K\rprocessing: done. time: {time.time()-st:.3f}")

                 













# def log(*args, **kwargs): print(*args, file=sys.sterr, flush=True, **kwargs)
