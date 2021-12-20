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
        self.data = []
        path = f'./{type}_dataset.pickle'

        if os.path.isfile(path):
            print(f"Loading {type} Dataset of {len(df)}")
            with open(path, 'rb') as file:
                self.data = pickle.load(file)

        else:
            print(f"Tokenizing {type} Dataset of {len(df)}")

            for _, row in tqdm(df.iterrows()):
                if len(row.review) < 20:
                    continue
                sent = f'음식점은 {row.restaurant} '
                sent += f'메뉴는 {row.menu} '
                sent += f'음식 점수는 {int(row.food)}점 '
                sent += f'서비스 및 배달 점수는 {int(row.delvice)}점 '
                sent += f'리뷰는 '
                if type != 'test':
                    sent += f'{row.review}'

                self.data.append(tokenizer.encode(sent))

            with open(path, 'wb') as file:
                pickle.dump(self.data, file)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


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

df = pd.read_csv('../StarClassification/combined_211219dataset.csv')
# df = df.iloc[:1000]
# df = pd.read_csv('../StarClassification/star_classification.csv')
train_df, valid_df = train_test_split(df, test_size=0.1, stratify=df.label, shuffle=True, random_state=seed)
valid_df, test_df = train_test_split(valid_df, test_size=0.1, stratify=valid_df.label, shuffle=True, random_state=seed)

if not os.path.isfile('test_combined_211219dataset.csv'):
    test_df.to_csv('test_combined_211219dataset.csv', index=False)

train_dataset = ReviewDataset(train_df, tokenizer, 'train')
valid_dataset = ReviewDataset(valid_df, tokenizer, 'valid')
test_dataset = ReviewDataset(test_df, tokenizer, 'test')

count = 0
epoch = 5
batch_size = 1
save_path = 'checkpoint'

pad = tokenizer.pad_token_id
def collate(batch):
    max_length = max(len(b) for b in batch)
    batch = [b + [pad] * (max_length-len(b)) for b in batch]
    return batch


train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          # num_workers=4,
                          shuffle=True,
                          pin_memory=True,
                          # collate_fn=collate
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=batch_size,
                          # num_workers=4,
                          shuffle=True,
                          pin_memory=True,
                          # collate_fn=collate
                          )

# model.resize_token_embeddings(len(vocab))

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

wandb.init(entity='ssp',
           project='ReviewGeneration',
           )

test_count = 0
avg_loss = (0.0, 0.0)
sents = []

for epoch in range(1, epoch+1):
    pbar = tqdm(train_loader)
    for train_idx, data in enumerate(pbar, 1):
        optimizer.zero_grad()
        data = torch.tensor([data]).to(device)
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]
        loss.backward()
        optimizer.step()

        avg_loss = (avg_loss[0] * 0.99 + loss.item(), avg_loss[1] * 0.99 + 1.0)
        pbar.set_description('epoch no.{0} train no.{1} loss = {2:.5f} avg_loss = {3:.5f}'.format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
        count += batch_size

        if (len(train_dataset) // 10) == count:
            with torch.no_grad():
                model.eval()

                valid_pbar = tqdm(valid_loader)
                valid_loss = (0.0, 0.0)
                for val_idx, valid_data in enumerate(valid_pbar):
                    valid_data = torch.tensor([valid_data]).to(device)
                    outputs = model(valid_data, labels=valid_data)
                    loss = outputs[0].item()
                    valid_loss = (valid_loss[0] * 0.99 + loss, valid_loss[1] * 0.99 + 1.0)
                    pbar.set_description(f'valid no.{val_idx} loss = {loss:.5f} avg_loss = {valid_loss[0] / valid_loss[1]:.5f}')

                wandb.log({
                    'Train Loss': round(avg_loss[0] / avg_loss[1], 5),
                    'Valid Loss': round(valid_loss[0] / valid_loss[1], 5)
                })


            for _ in range(5):
                test_idx = np.random.randint(len(test_dataset))
                # input_ids = tokenizer.encode(text)
                input_ids = test_dataset[test_idx]
                gen_ids = model.generate(torch.tensor([input_ids]).to(device),
                                         max_length=100,
                                         repetition_penalty=2.0,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id,
                                         bos_token_id=tokenizer.bos_token_id,
                                         use_cache=True,
                                         top_p=np.random.randint(80, 100) / 100,
                                         top_k=np.random.randint(20, 50),
                                         temperature=np.random.randint(3, 10) / 10
                                         )
                generated = tokenizer.decode(gen_ids[0, :].tolist())
                print(generated)
                sents.append(f'epoch {epoch} / count {count} : {generated}')
            model.train()


    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Train Loss': round(avg_loss[0] / avg_loss[1], 5),
        'Valid Loss': round(valid_loss[0] / valid_loss[1], 5),
    }, f'{save_path}/checkpoint_ep{epoch}.pth')

sents = '\n'.join(sents)
f = open(f'samples/sample.txt', 'w', encoding="utf-8")
f.write(sents)
f.close()

# if __name__ == "__main__":
#     assert args.save_path
#     args.save_path = os.path.join('checkpoint/', args.save_path)
#     if not os.path.isdir(args.save_path):
#         os.makedirs(args.save_path)
#
#     main(args.epoch, args.save_path, args.load_path, args.samples, args.data_file_path, args.batch_size)