import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from tqdm import tqdm
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import *
from transformers.training_args import default_logdir

# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

from tqdm import tqdm
from data import StdDiaDataset
# import json

## 초기화
from dis_model import *

dismodel = findattribute(drop_rate=0.4).cuda()
# dismodel.load_state_dict(torch.load('../visual_v1_0/models/cls_model_final'))
dismodel.train()

import torch.optim as optim

# from tensorboardX import SummaryWriter
# summary = SummaryWriter(logdir='./logs')


def main(epoch, batch_size, standard_csv_filename, dialect_csv_filename):

    # f = open(os.paht.join(data_path, csv_path))
    # token2num = json.load(f)

    # num2token = {}
    # for key, value in token2num.items():
    #     num2token[value] = key
    # f.close()

    # data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data"
    # train_amazon_neg_path = data_path + "/amazon/sentiment.train.0"
    # train_amazon_neg_open = open(train_amazon_neg_path, "r")
    # train_amazon_neg_dataset = train_amazon_neg_open.readlines()
    # dev_amazon_neg_path = data_path + "/amazon/sentiment.dev.0"
    # dev_amazon_neg_open = open(dev_amazon_neg_path, "r")
    # dev_amazon_neg_dataset = dev_amazon_neg_open.readlines()
    # amazon_neg_dataset = train_amazon_neg_dataset+dev_amazon_neg_dataset

    # neg_len = len(amazon_neg_dataset)
    # train_amazon_neg_open.close()
    # dev_amazon_neg_open.close()

    # train_amazon_pos_path = data_path + "/amazon/sentiment.train.1"
    # train_amazon_pos_open = open(train_amazon_pos_path, "r")
    # train_amazon_pos_dataset = train_amazon_pos_open.readlines()
    # dev_amazon_pos_path = data_path + "/amazon/sentiment.dev.1"
    # dev_amazon_pos_open = open(dev_amazon_pos_path, "r")
    # dev_amazon_pos_dataset = dev_amazon_pos_open.readlines()
    # amazon_pos_dataset = train_amazon_pos_dataset+dev_amazon_pos_dataset

    # pos_len = len(amazon_pos_dataset)
    # train_amazon_pos_open.close()
    # dev_amazon_pos_open.close()

    data_path = "/opt/ml/SST/data/dialect/"
    standard_df = pd.read_csv(os.path.join(data_path, standard_csv_filename))
    dialect_df = pd.read_csv(os.path.join(data_path, dialect_csv_filename))
    standard_train_data = StdDiaDataset(True, standard_df.iloc[:100000], gpt_tokenizer)
    dialect_train_data = StdDiaDataset(False, dialect_df.iloc[:100000], gpt_tokenizer)
    standard_train_dataloader = DataLoader(standard_train_data, batch_size=batch_size, shuffle=True)
    dialect_train_dataloader = DataLoader(dialect_train_data, batch_size=batch_size, shuffle=True)

    """training parameter"""
    cls_initial_lr = 0.001
    cls_trainer = optim.Adamax(dismodel.cls_params, lr=cls_initial_lr)  # initial 0.001
    max_grad_norm = 25
    batch_size = batch_size
    epoch = epoch
    # stop_point = pos_len*epoch

    pre_epoch = 0
    # for start in tqdm(range(0, stop_point)):
    for epoch in tqdm(range(1, epoch + 1)):
        # """data start point"""
        # neg_start = start%neg_len
        # pos_start = start%pos_len

        # """data setting"""
        # neg_sentence = amazon_neg_dataset[neg_start].strip()
        # pos_sentence = amazon_pos_dataset[pos_start].strip()

        # neg_labels = [] # negative labels
        # neg_labels.append([1,0])
        # neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

        # pos_labels = [] # positive labels
        # pos_labels.append([0,1])
        # pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()

        # sentences = [neg_sentence, pos_sentence]
        # attributes = [neg_attribute, pos_attribute]

        # """data input"""
        # for i in range(2):

        # pbar = tqdm(zip(standard_train_dataloader, dialect_train_dataloader))
        # for std_token_idx, std_attribute, dia_token_idx, dia_attribute in pbar:
        for standard, dialect in tqdm(zip(standard_train_dataloader, dialect_train_dataloader)):

            ## k=0: negative, k=1: positive
            # sentence = sentences[i]
            # attribute = attributes[i] # for generate

            # token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()

            # dis_out = dismodel.discriminator(token_idx)
            dis_out = dismodel.discriminator(standard[0])


            """calculation loss & traning"""
            ## training using discriminator loss
            # cls_loss = dismodel.cls_loss(attribute, dis_out)
            cls_loss = dismodel.cls_loss(standard[1], dis_out)
            ## summary.add_scalar('discriminator loss', cls_loss.item(), start)

            cls_trainer.zero_grad()
            cls_loss.backward()  # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(dismodel.cls_params, max_grad_norm)
            cls_trainer.step()
            # pbar.set_description(f'standard - epoch:{epoch}, loss{cls_loss:.3f}')

            dis_out = dismodel.discriminator(dialect[0])
            cls_loss = dismodel.cls_loss(dialect[1], dis_out)
            cls_trainer.zero_grad()
            cls_loss.backward()
            cls_trainer.step()
            # pbar.set_description(f'dialect - epoch:{epoch}, loss{cls_loss:.3f}')

        """savining point"""
        # if (start+1)%pos_len == 0:
    if not epoch % 10:
        # random.shuffle(amazon_neg_dataset)
        # random.shuffle(amazon_pos_dataset)
        # save_model((start+1)//pos_len)
        save_model(epoch)
    save_model('final')  # final_model

def save_model(iter):
    if not os.path.exists('models_test/'):
        os.makedirs('models_test/')
    torch.save(dismodel.state_dict(), 'models/cls_model_{}'.format(iter))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='num batch_sizes')
    parser.add_argument('--standard_csv_filename', type=str, default='5_valid_standard.csv',
                        help='standard csv file name')
    parser.add_argument('--dialect_csv_filename', type=str, default='5_valid_dialect.csv', help='dialect csv file name')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args.epoch, args.batch_size, args.standard_csv_filename, args.dialect_csv_filename)