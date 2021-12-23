from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from sklearn.metrics import f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
dia_df = pd.read_csv('./data/dialect/1_train_dialect.csv')
std_df = pd.read_csv('./data/dialect/1_train_standard.csv')
dia_vdf = pd.read_csv('./data/dialect/1_valid_dialect.csv')
std_vdf = pd.read_csv('./data/dialect/1_valid_standard.csv')

dia_vdf['label'] = [0] * len(dia_vdf)
std_vdf['label'] = [1] * len(std_vdf)
vdf = pd.concat([dia_vdf, std_vdf])
vdf=vdf.sample(frac=1).reset_index(drop=True)

dia_df['label'] = [0] * len(dia_df)
std_df['label'] = [1] * len(std_df)

df = pd.concat([dia_df, std_df])

df=df.sample(frac=1).reset_index(drop=True)

vdf.label.unique()

args = TrainingArguments(
    output_dir ='./',
    num_train_epochs = 10,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1_score',
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large', fast=True)

model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large').to(device)

train_tok = tokenizer(list(df.iloc[:16000].sentence), padding=True, truncation=True, max_length=50)
valid_tok = tokenizer(list(vdf.iloc[:4000].sentence), padding=True, truncation=True, max_length=50)

class StyleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_tok = tokenizer(list(df.iloc[:16000].sentence), padding=True, truncation=True, max_length=50)
valid_tok = tokenizer(list(vdf.iloc[:4000].sentence), padding=True, truncation=True, max_length=50)
train_dataset = StyleDataset(train_tok, list(df.iloc[:16000].label))
valid_dataset = StyleDataset(valid_tok, list(vdf.iloc[:4000].label))

metric = f1_score

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    pred = np.argmax(pred, axis=-1)
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro', labels=list(range(2))) * 100
    return {'f1_score': f1}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()