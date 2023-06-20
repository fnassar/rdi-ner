from transformers import BertForTokenClassification
class Model(BertForTokenClassification):
    def __init__(self, flag):
      if flag:
          self.model = BertForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
      else:
          self.model = BertForTokenClassification.from_json_file()
      # self.base_model = self.model.layers[:11]
      # self.dense1 = nn.Linear(300, 10)
    def forward(self, inputs):
        outputs = self.model(inputs)
        # logits = self.dense1(outputs)
        return outputs