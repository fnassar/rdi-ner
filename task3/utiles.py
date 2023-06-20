!pip install transformers
import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForTokenClassification
def prepare_data(path):
  my_data=pd.read_csv(path,index_col=0)
  my_data["label"]= my_data["label"].apply(lambda x: literal_eval(x))
  my_data["sentence"]= my_data["sentence"].apply(lambda x: literal_eval(x))
  return list(my_data["label"]),list(my_data["sentence"])
def tokenize_text(data_to_be_tokenized):
  tokenizer= AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
  return tokenizer(data_to_be_tokenized,padding="max_length",truncation=True,max_length=512,is_split_into_words=True)
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs 