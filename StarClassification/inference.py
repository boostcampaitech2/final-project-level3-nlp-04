import sys
from os import path
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))

from core.sql_helper import SqlHelper
from crawling.config import cfg

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

from glob import glob
import pandas as pd
import numpy as np
import os


class Dataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    '''
    label   food    delvice | label   food    delvice
      0       1       1     |   5       3       5
      1       1       3     |   6       5       1
      2       1       5     |   7       5       3
      3       3       1     |   8       5       5
      4       3       3     |   9       0       0
    '''
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# print('DB Dataset Loading')
# if not os.path.isfile('./211217db.csv'):
#     print('Making DB Dataset')
# helper = SqlHelper(**cfg)
# query = "select * from preprocessed_review"
# df = helper.get_df(query)
#     df.to_csv('./211217db.csv', index=False, encoding='utf-8-sig')
# else:
#     df = pd.read_csv('./211217db.csv')
# print('Complete Reading DB Dataset')
# df = df.iloc[:10]
df = pd.read_csv('./211219db.csv')
df = df.dropna(subset=['preprocessed_review_context'])

X = list(df.preprocessed_review_context)
X_tokenized = tokenizer(X, padding=True, truncation=True, max_length=300)
dataset = Dataset(X_tokenized)
args = TrainingArguments(
    output_dir='./',
    per_device_eval_batch_size=512,
)

sum_pred = None
for index in range(1):
    print(f'Inference break{index} Start')
    model = AutoModelForSequenceClassification.from_pretrained(glob(f'./output/star*')[index], config=config)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=args,
    )

    predict = trainer.predict(test_dataset=dataset)
    pred = predict[0]
    if sum_pred is None:
        sum_pred = pred
    else:
        sum_pred += pred

    break

df['predict'] = np.argmax(sum_pred, axis=1)
df.to_csv(f'backup_fold_211219dataset.csv', index=False)

df = df.rename(columns={'predict': 'label', 'restaurant_name': 'restaurant', 'preprocessed_review_context': 'review'})
df = df.dropna(subset=['menu', 'review'])
df['food'] = df.label.apply(lambda x: int(2*(x // 3) + 1))
df['delvice'] = df.label.apply(lambda x: int(2*(x % 3) + 1))

fold_num = 1
assert fold_num is not None

df.to_csv(f'{fold_num}fold_211219dataset.csv', index=False)