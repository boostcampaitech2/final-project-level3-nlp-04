from datasets import load_dataset
from transformers import AutoModel, PreTrainedTokenizerFast, GPT2LMHeadModel

from torch.utils.data import Dataset
import torch

from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                              bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', pad_token_id=tokenizer.eos_token_id).to(device)
checkpoint = torch.load('checkpoint/checkpoint_ep5.pth')
model.load_state_dict(checkpoint['model_state_dict'])


class ReviewDataset(Dataset):
    def __init__(self, type):
        self.data = []
        path = f'./{type}_dataset.pickle'

        if os.path.isfile(path):
            print(f"Loading {type} Dataset")
            with open(path, 'rb') as file:
                self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


gens = []
test_dataset = ReviewDataset('test')
for input_ids in [test_dataset[0], test_dataset[1]]:
    gen_ids = model.generate(torch.tensor([input_ids]).to(device),
                             max_length=150,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True,
                             top_p=0.9,#np.random.randint(20, 80) / 100,
                             top_k=100,#np.random.randint(20, 80),
                             temperature=0.5,#np.random.randint(3, 10) / 10,
                             do_sample=True,
                             num_return_sequences=3,
                             )

    for ids in gen_ids:
        generated = tokenizer.decode(ids.tolist())
        print(generated)
        gens.append(generated)

# test_df = pd.read_csv('test_combined_211219dataset.csv')
# test_df['generated'] = gens
# test_df.to_csv('generated.csv', index=False)







