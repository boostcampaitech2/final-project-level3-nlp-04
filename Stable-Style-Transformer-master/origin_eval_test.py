import torch
import torch.nn as nn
import numpy as np

from transformers import *
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import json
f = open('./amazon_vocab.json')
token2num = json.load(f)

num2token = {}
for key, value in token2num.items():
    num2token[value] = key
import sys
# sys.path.insert(0, "/DATA/joosung/controllable_english/amazon/visual_v1_1/")
from origin_dis_model import *
dismodel = findattribute().cuda()

i='5'
dismodel_name='cls_model_' + str(i)
dismodel.load_state_dict(torch.load('./model_origin/{}'.format(dismodel_name)))
dismodel.eval()

from tqdm import tqdm

### classifier accuracy
import glob

neg_out_files = glob.glob(output_path + '*test.0*')
pos_out_files = glob.glob(output_path + '*test.1*')

for i in range(len(neg_out_files)):
    print("model: ", neg_out_files[i].split('.')[-1])

    neg_data_open = open(neg_out_files[i], "r")
    neg_data_dataset = neg_data_open.readlines()
    neg_len = len(neg_data_dataset)
    neg_data_open.close()

    neg_correct = 0
    for k in range(neg_len):
        out_sen = neg_data_dataset[k].split('\t')[1].strip()

        token_idx = torch.tensor(gpt_tokenizer.encode(out_sen)).unsqueeze(0).cuda()

        """discriminator"""
        if token_idx.shape[1] != 0:
            result = dismodel.discriminator(token_idx=token_idx).argmax(1).cpu().numpy()[0]
            if result == 1:  ## style transfer so result must be 1(positive)
                neg_correct += 1

    pos_out_name = neg_out_files[i].split('.0.')[0] + '.1.' + neg_out_files[i].split('.0.')[-1]
    pos_data_open = open(pos_out_name, "r")
    pos_data_dataset = pos_data_open.readlines()
    pos_len = len(neg_data_dataset)
    pos_data_open.close()

    pos_correct = 0
    for k in range(pos_len):
        out_sen = pos_data_dataset[k].split('\t')[1].strip()

        token_idx = torch.tensor(gpt_tokenizer.encode(out_sen)).unsqueeze(0).cuda()

        """discriminator"""
        if token_idx.shape[1] != 0:
            result = dismodel.discriminator(token_idx=token_idx).argmax(1).cpu().numpy()[0]
            if result == 0:  ## style transfer so result must be 0(negative)
                pos_correct += 1

    Acc = (neg_correct + pos_correct) / (neg_len + pos_len) * 100
    print("Accuracy: {}%".format(Acc))