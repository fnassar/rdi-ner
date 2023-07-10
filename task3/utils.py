
import random

from torch import nn
from transformers import Trainer

import pandas as pd


# def prep_loss_data(label2id, train_data):
#     id2label = {v:k for k,v in label2id.items()}

#     data = ({'words':train_data[i][0], 'labels':[id2label[label] for label in train_data[i][1]], 'tokens':train_data[i][2], 'token_label':train_data[i][3]} for i in range(len(train_data)))
#     df_train = pd.DataFrame(data)
#     df_train = df_train.explode('labels')
#     class_weights = (1- (df_train['labels'].value_counts(normalize=True).sort_index())).values

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     class_weights = torch.from_numpy(class_weights).float().to(device)
#     return class_weights


def preprocess(data, tokenizer):
  tokens = tokenizer(data) 
  token = [token[1:len(token)-1] for token in tokens['input_ids']]
  return (token)



def split_data_file(data_file, train_ratio):
    with open(data_file, 'r') as file:
      lines = file.readlines()
    lines[len(lines)-1] +='\n'
    random.shuffle(lines)

    total_samples = len(lines)
    train_samples = int(total_samples * train_ratio)

    train_data = lines[:train_samples]
    test_data = lines[train_samples:]

    with open('./ANERcorp-CamelLabSplits/train_data.txt', 'w') as file:
      for line in train_data:
        file.write(line)

    with open('./ANERcorp-CamelLabSplits/eval_data.txt', 'w') as file:
      for line in test_data:
        file.write(line)

    train_name = "./ANERcorp-CamelLabSplits/train_data.txt"
    test_name = "./ANERcorp-CamelLabSplits/train_data.txt"

    return train_name, test_name




# def align_labels_with_tokens(labels, word_ids):
#     new_labels = []
#     current_word = None
#     for word_id in word_ids:
#         if word_id != current_word:
#             # Start of a new word!
#             current_word = word_id
#             label = -100 if word_id is None else labels[word_id]
#             new_labels.append(label)
#         elif word_id is None:
#             # Special token
#             new_labels.append(-100)
#         else:
#             # Same word as previous token
#             label = labels[word_id]
#             # If the label is B-XXX we change it to I-XXX
#             if label <= 3:
#                 label += 4
#             new_labels.append(label)

#     return new_labels

# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(
#         examples["tokens"], truncation=True, is_split_into_words=True
#     )
#     all_labels = examples["ner_tags"]
#     new_labels = []
#     for i, labels in enumerate(all_labels):
#         word_ids = tokenized_inputs.word_ids(i)
#         new_labels.append(align_labels_with_tokens(labels, word_ids))

#     tokenized_inputs["labels"] = new_labels
#     return tokenized_inputs 