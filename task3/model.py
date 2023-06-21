from transformers import BertForTokenClassification, AutoTokenizer
from torch.nn.modules import module

class Model(nn.module):
    def __init__(self, flag):
      if flag:
          self.model = BertForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
      else:
          self.model = BertForTokenClassification.from_json_file()
      def get_model(self):
        return self.model    
      # self.base_model = self.model.layers[:11]
      # self.dense1 = nn.Linear(300, 10)
    def forward(self, inputs):
        outputs = self.model(inputs)
        # logits = self.dense1(outputs)
        return outputs