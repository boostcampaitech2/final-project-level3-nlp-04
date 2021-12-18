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

# metric_name = 'glue'
# metric = load_metric(metric_name, keep_in_memory=True)
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return {metric_name: metric.compute(predictions=predictions, references=labels)}

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

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                              bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', pad_token_id=tokenizer.eos_token_id).to(device)
model.train()

df = pd.read_csv('../StarClassification/pre_1fold_211217dataset.csv')
df = df.iloc[:100000]
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df.label, shuffle=True, random_state=seed)
train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df.label, shuffle=True, random_state=seed)


train_dataset = ReviewDataset(train_df, tokenizer, 'train')
valid_dataset = ReviewDataset(valid_df, tokenizer, 'valid')
test_dataset = ReviewDataset(test_df, tokenizer, 'test')

# train_loader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, pin_memory=True)
# valid_loader = DataLoader(valid_dataset, batch_size=args.valid_bs, shuffle=True, pin_memory=True)

args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=30,
    per_device_train_batch_size=56,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    seed=seed,
    save_total_limit=1,
    load_best_model_at_end=True,
    # metric_for_best_model=metric_name,
    # gradient_accumulation_steps=3,
)

wandb.init(entity='ssp',
           project='ReviewGeneration',
           config=args,
           )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
)

trainer.train()
# trainer.save_model()

pred = trainer.predict(test_dataset=test_dataset)

# save
with open('data.pickle', 'wb') as f:
    pickle.dump(pred, f, pickle.HIGHEST_PROTOCOL)

predictions = np.argmax(pred[0], axis=-1)
try:
    print(predictions[:10])
except:
    print(predictions[0])

# learning_rate = 3e-5
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# step = 0
#
# for epoch in range(1, args.epoch + 1):
#     train_pbar = tqdm(train_loader)
#     for idx, data in enumerate(train_pbar):
#         optimizer.zero_grad()
#
#         data = torch.stack(data[0])  # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
#         data = data.transpose(1, 0)
#         data = data.to(device)
#
#         outputs = model(data, labels=data)
#         loss, logits = outputs[:2]
#         loss = loss.to(device)
#         loss.backward()
#         avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
#         optimizer.step()
#         step += 1
#         train_pbar.set_description(f'epoch no.{epoch} loss = {loss:.5f} avg_loss = {avg_loss[0] / avg_loss[1]:.5f}')
#
#         if not step % 100:
#             valid_pbar = tqdm(valid_loader)
#             for jdx, val_data in enumerate(valid_pbar):
#                 model.eval()
#                 model.ge
#
#             model.train()
#
#     # generator 진행
#     sent = sample_sequence(model, tok, vocab, sent=word, text_size=200, temperature=0.7, top_p=0.8, top_k=40)
#     sent = sent.replace("<unused0>", "\n")  # 비효율적이지만 엔터를 위해서 등장
#     sent = auto_enter(sent)
#     sent = sent.replace('<pad>', '')
#     print(sent)
#
#     sents.append(f'{epoch} : {sent}')
#
#     # 모델 저장
#     # try:
#     if not epoch % 25:
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#         }, f'{save_path}/KoGPT2_checkpoint_{str(epoch)}.tar')






# metrics = trainer.evaluate()
# metrics['best_f1'] = metrics.pop('eval_f1')
# wandb.log(metrics)
# wandb.join()
