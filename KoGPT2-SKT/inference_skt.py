from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_metric

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import wandb
import os
import pickle


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, type):
        path = f'./{type}_dataset_1e5.pickle'
        if os.path.isfile(path):
            print(f"Loading {type} Dataset of {len(df)}")
            with open(path, 'rb') as file:
                self.output = pickle.load(file)
        else:
            print(f"Tokenizing {type} Dataset of {len(df)}")
            self.df = df
            self.output = {'input_ids': [],
                           'attention_mask': [],
                           }

            sents = []
            for _, row in tqdm(self.df.iterrows()):
                row.menu = row.menu[:30]
                sent = tokenizer.bos_token
                sent += f'음식점은 {row.restaurant}, 메뉴는 {row.menu}, 음식 점수는 {int(row.food)}점, 서비스 및 배달 점수는 {int(row.delvice)}점 리뷰는 {row.review}'
                sent += tokenizer.eos_token
                sents.append(sent)

            for sent in tqdm(sents):
                sent = tokenizer(sent,
                                 truncation=True,
                                 max_length=200,
                                 return_tensors='pt',
                                 padding="max_length",
                                 )
                self.output['input_ids'].append(sent['input_ids'])
                self.output['attention_mask'].append(sent['attention_mask'])

            with open(path, 'wb') as file:
                pickle.dump(self.output, file)

    def __len__(self):
        return len(self.output['input_ids'])

    def __getitem__(self, idx):
        return {'input_ids': self.output['input_ids'][idx],
                'attention_mask': self.output['attention_mask'][idx],
                'labels': self.output['input_ids'][idx],
                }


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

df = pd.read_csv('../StarClassification/pre_1fold_211217dataset.csv')
df = df.iloc[:100000]
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df.label, shuffle=True, random_state=seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')

model = GPT2LMHeadModel.from_pretrained('./output/checkpoint-3600', pad_token_id=tokenizer.eos_token_id).to(device)


outputs = []
for _, row in tqdm(test_df.iterrows()):
    row.menu = row.menu[:30]
    sent = tokenizer.bos_token
    sent += f'음식점은 {row.restaurant}, 메뉴는 {row.menu}, 음식 점수는 {int(row.food)}점, 서비스 및 배달 점수는 {int(row.delvice)}점 리뷰는 '

    input = tokenizer(sent, return_tensors='pt')['input_ids'].to(device)
    print(sent, input)
    output = model.generate(input, max_length=150)

    print(output)
    print(tokenizer.decode(output.squeeze(0)))
    break
    outputs.append(output)


# test_df['generated_review'] = outputs
# test_df.to_csv(f'generated.csv', index=False)
