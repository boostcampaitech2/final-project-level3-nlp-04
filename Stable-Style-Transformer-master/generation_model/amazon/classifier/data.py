import torch
import numpy as np
from torch.utils.data import Dataset
class StdDiaDataset(Dataset):
    def __init__(self, is_std:bool, df, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.sentence = []
        self.attribute = 0 if is_std == True else 1
        max_len = 100
        for idx, sent in enumerate(self.df.sentence):
            # tokenized_sent = self.tokenizer.tokenize(sent)
            tokenized_sent = self.tokenizer.encode(sent[:30])
            tokenized_sent += self.tokenizer.encode(self.tokenizer.pad_token) * (30 - len(tokenized_sent))
            # tokenized_sent = self.tokenizer.decode(tokenized_sent)
            self.sentence.append(tokenized_sent)
            if not idx % 10000:
                print(f'{idx}th tokenizing....')
                print(f'left {len(self.df.sentence) - idx}')
        print("tokenized finished!")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # input_text = self.tokenizer.encode(self.sentence[index])
        # if len(input_text) < 30:
        #     for i in 30-len(input_text):
        #         input_text.append(self.tokenizer.encode(self.tokenizer.pad_token))
        # elif len(input_text) > 30:
        #     for i in len(input_text) - 30:
        #         input_text.pop()
        # else:
        #     pass
        token_idx = torch.tensor(self.sentence[index]).cuda()
        attribute = torch.from_numpy(np.asarray(self.attribute)).type(torch.FloatTensor).cuda()
        return token_idx, attribute