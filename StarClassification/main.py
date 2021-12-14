from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
import transformers

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import wandb
import os


class Dataset(Dataset):
    def __init__(self, encodings, labels):
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
        item["labels"] = torch.tensor(int(self.labels[idx]))
        # item["labels"] = F.one_hot(torch.tensor(int(self.labels[idx])), num_classes=9)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average='micro', labels=list(range(9))) * 100

    return {"eval_accuracy": accuracy, "eval_f1": f1}


seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

os.environ['WANDB_LOG_MODEL'] = 'true'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_SILENT'] = "true"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "klue/roberta-large"
# model_name = "monologg/koelectra-base-v3-discriminator"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 9

args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    seed=seed,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    gradient_accumulation_steps=4,
)

df = pd.read_csv('./review_for_tagging.csv')
X, y = np.array(list(df.review)), np.array(list(df.label))
skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

for k, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=150)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=150)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    wandb.init(entity='ssp',
               project='StarClassification',
               name=f'Roberta-large-{k}',
               # name='Electra-v3',
               config=args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, str(k)))

    wandb.join()