import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class StdDiaDataset(Dataset):
    def __init__(self, is_std:bool, df, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.sentence = []
        self.attribute = 0 if is_std == True else 1
        max_len = 100

        for idx, row in tqdm(self.df.iterrows()):
            sent = row[0]
            tokenized_sent = self.tokenizer(sent, padding="max_length", max_length=50, truncation=True)['input_ids']
            # tokenized_sent = self.tokenizer.tokenize(sent)
            # tokenized_sent = self.tokenizer.encode(sent[:50])
            # tokenized_sent += self.tokenizer.encode(self.tokenizer.pad_token) * (50 - len(tokenized_sent))
            self.sentence.append(tokenized_sent)
            if not idx % 10000:
                print()
                print(f'{idx}th tokenizing....')
                print(f'left {len(self.df.sentence) - idx}')
                print('-'*30)
        print("tokenized finished!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # token_idx = torch.tensor(self.sentence[index]).unsqueeze(0).cuda()
        token_idx = torch.tensor(self.sentence[index]).cuda()
        # token_idx = self.sentence[index]
        attribute = torch.from_numpy(np.asarray(self.attribute)).type(torch.FloatTensor).cuda()

        return token_idx, attribute