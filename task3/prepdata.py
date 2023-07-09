import pandas as pd


def text_label_df(data_arr):
  
  data2 = []
  data_test2 = []
  for line in data_arr:
    line = [line.strip('\n').split(' ')]

  return pd.DataFrame(data_arr, columns=['text', 'label'])

# 'id':[num], 'tokens':[sentence cut], 'ner_tags':[],'sentences':[]



