import torch
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.Inn import pad_sequence
devic= torch.device("cuda" if torch.cuda.is_avaliable() else"cpu")
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
def process(examples):
  tokenized = tokenizer(text,truncation=True, add_special_tokens=False,return_tensors="pt")
  input_ids = tokenized["input_id"].to(device)
  tokens = tokenized.tokens()
  return