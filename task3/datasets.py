import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class DataSets_class:
    def __init__(self):
        listt = {'id': [], 'tokens': [], 'sentences': [], 'ner_tags': [], 'num_Words': [], 'labels': []}
        self.all_data = {
            'test': pd.DataFrame(listt),
            'eval': pd.DataFrame(listt),
            'train': pd.DataFrame(listt)
        }
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.all_labels = []
        self.id2label = {}
        self.label2id = {}
    
    def preprocess(self): # add data into dataframe and divide into sentences
        self.read('pathhere')
    
    def read(self, path):
        data = open(path+'ANERCorp_CamelLab_train.txt').readlines()
        data_test = open(path+'ANERCorp_CamelLab_test.txt').readlines()
        data2 = []
        data_test2 = []
        for line in data:
            data2.append(line.strip('\n').split(' '))

        for line in data_test:
            data_test2.append(line.strip('\n').split(' '))

        self.df_train = pd.DataFrame(data2, columns=['text', 'label'])
        self.df_test = pd.DataFrame(data_test2, columns=['text', 'label'])

        self.add_data('train')
        self.add_data('test')
    
    def add_data(self, name):

        num_Words = 0
        labels = []
        ner_tags = []
        tokens = []
        sentence = ""
        temp = self.df_test
        if (name == 'test'):
            temp = self.df_train
        for column, item in temp.iterrows():

            if (item['text'] == ''):

                self.all_data[name].loc[len(self.all_data[name].index)] = [len(
                    self.all_data[name].index), tokens, sentence, ner_tags, num_Words, labels]
                num_Words = 0
                labels = []
                ner_tags = []
                tokens = []
                sentence = ""

            else:
                # Access column name using 'column' and column values using 'values'
                if (item['label'] not in labels and item['label'] != None and item['label'] != ''):
                    labels.append(item['label'])

                sentence += item['text'] + " "
                tokens.append(item['text'])  # words
                ner_tags.append(item['label'])  # tokens
                num_Words += 1
                if (item['label'] not in self.all_labels):
                    self.all_labels.append(item['label'])
                    
    def label_id(self):
        self.id2label = {idx: self.all_labels[idx] for idx in range(len(self.all_labels))}
        self.label2id = {v: k for k, v in self.id2label.items()}
    
    # pass train dataset(df_train), 0.2, random_state(0)
    
    def split_train_eval(self, test_size, random_state):
        train_df, eval_df = train_test_split(self.all_data['train'], test_size=test_size, random_state=random_state)
        self.all_data['train'] = train_df
        self.all_data['eval'] = eval_df
        print(f"Number of examples in the train set: {len(train_df)}")
        print(f"Number of examples in the eval set: {len(eval_df)}")
    
    def train_model(self):
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(self.df_train['text'])
        y_train = self.df_train['label']
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf, vectorizer
    
    def evaluate_model(self,clf, vectorizer, eval_df):
        X_eval = vectorizer.transform(eval_df['text'])
        y_eval = eval_df['label']
        y_pred = clf.predict(X_eval)
        print(classification_report(y_eval, y_pred))
    
    def train_and_evalute_model(self,df_train, test_size, random_state):
        self.split_train_eval(test_size, random_state)
        clf, vectorizer = self.train_model()
        self.evaluate_model(clf, vectorizer, self.all_data['eval'])
  	
    def get_repitition_dt(self, sub):
        tempp = self.all_data['test']
        tempp = tempp.explode('ner_tags')
        return (tempp['ner_tags'].value_counts(normalize=True).sort_index())



# from sklearn.model_selection import train_test_split
# import pandas as pd


# # split the data into train and test set
# train, test = train_test_split(
#     all_data['train'], test_size=0.10, random_state=0)
# # save the data
# all_data['train'] = train
# all_data['dev'] = test


# df_train, df_dev = train_test_split(df_train, test_size=0.10, random_state=0)


# # all_data['test']
# tempp = all_data['test']
# tempp

# # df_train['label'].value_counts(normalize=True).sort_index()

# tempp = tempp.explode('ner_tags')
# tempp['ner_tags'].value_counts(normalize=True).sort_index()
