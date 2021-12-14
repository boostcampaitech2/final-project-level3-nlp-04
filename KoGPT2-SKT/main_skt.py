
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

from torch.utils.data import Dataset
import torch

import argparse


class ReviewDataset(Dataset):
    def __init__(self):

    def __len__(self):

    def __getitem__(self, idx):



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100,
                    help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='',
                    help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default='./checkpoint/',
                    help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--samples', type=str, default="samples/",
                    help="생성 결과를 저장할 경로입니다.")
parser.add_argument('--data_file_path', type=str, default='../review_for_tagging.csv',
                    help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=32,
                    help="batch_size 를 지정합니다.")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)
tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                              bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                              pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')

