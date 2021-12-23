import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from tqdm import tqdm
import os
import random

from transformers import *
from transformers.training_args import default_logdir

# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

from tqdm import tqdm
from data import StdDiaDataset
# import json
import pickle

## 초기화
from dis_model import *

dismodel = findattribute(tokenizer=gpt_tokenizer, drop_rate=0.4).cuda()
# dismodel.load_state_dict(torch.load('../visual_v1_0/models/cls_model_final'))
dismodel.train()

import torch.optim as optim


from tensorboardX import SummaryWriter
summary = SummaryWriter(logdir='./logs')


def main(epoch, batch_size, standard_csv_filename, dialect_csv_filename, val_standard_csv_filename, val_dialect_csv_filename, csv_filename):
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
    # standard_df = pd.read_csv(os.path.join(data_path, standard_csv_filename))
    # dialect_df = pd.read_csv(os.path.join(data_path, dialect_csv_filename))
    # standard_train_data = StdDiaDataset(True, standard_df.iloc[:10000], gpt_tokenizer)
    # dialect_train_data = StdDiaDataset(False, dialect_df.iloc[:10000], gpt_tokenizer)
    # standard_train_dataloader = DataLoader(standard_train_data, batch_size=batch_size, shuffle=True)
    # dialect_train_dataloader = DataLoader(dialect_train_data, batch_size=batch_size, shuffle=True)

    pk_path = '/'.join([data_path, csv_filename.split('_')[0]])

    df_train = pd.read_csv(os.path.join(data_path,csv_filename), encoding='utf-8')
    std_dataset = df_train['std'] # neg
    dia_dataset = df_train['dia'] # pos
    std_len = len(std_dataset)
    dia_len = len(dia_dataset)
    # std_open = open(os.path.join(data_path, standard_csv_filename), 'r')
    # std_dataset = std_open.readlines()
    # std_len = len(std_dataset)
    # std_open.close()
    #
    # dia_open = open(os.path.join(data_path, dialect_csv_filename), 'r')
    # dia_dataset = dia_open.readlines()
    # dia_len = len(dia_dataset)
    # dia_open.close()
    #
    # std_open = open(os.path.join(data_path, val_standard_csv_filename), 'r')
    # std_dataset = std_open.readlines()
    # std_dataset = std_dataset[1:]
    # std_len = len(std_dataset)
    # std_open.close()
    #
    # dia_open = open(os.path.join(data_path, val_dialect_csv_filename), 'r')
    # dia_dataset = dia_open.readlines()
    # dia_dataset = dia_dataset[1:]
    # dia_len = len(dia_dataset)
    # dia_open.close()


    """training parameter"""
    cls_initial_lr = 0.001
    cls_trainer = optim.Adamax(dismodel.cls_params, lr=cls_initial_lr)  # initial 0.001
    max_grad_norm = 25
    batch_size = batch_size
    epoch = epoch
    stop_point = std_len * epoch


    pre_epoch = 0

    for start in tqdm(range(0, stop_point, batch_size)):
        # for epoch in tqdm(range(1, epoch + 1)):
        """data start point"""
        std_start = start % std_len
        dia_start = start % dia_len

        """data setting"""

        std_sentence = std_dataset[std_start:std_start+batch_size]#.strip()
        dia_sentence = dia_dataset[dia_start:dia_start+batch_size]#.strip()

        # std_sentence = [[x.strip()] for x in std_sentence]
        # dia_sentence = [[x.strip()] for x in dia_sentence]

        std_labels = [[1,0] for _ in range(len(std_sentence))] # negative labels
        # std_labels.append([1,0])
        std_attribute = torch.from_numpy(np.asarray(std_labels)).type(torch.FloatTensor).cuda()

        dia_labels = [[0,1] for _ in range(len(dia_sentence))] # positive labels
        # pos_labels.append([0,1])
        dia_attribute = torch.from_numpy(np.asarray(dia_labels)).type(torch.FloatTensor).cuda()

        sentences = [std_sentence, dia_sentence]
        attributes = [std_attribute, dia_attribute]

        """data input"""
        for i in range(2):
            # pbar = tqdm(zip(standard_train_dataloader, dialect_train_dataloader))
            # for standard, dialect in pbar:
            #     std_token_idx, std_attribute = standard
            #     dia_token_idx, dia_attribute = dialect
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i] # for generate


            if batch_size != 1:
                token_idx = []
            else:
                token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()

            # for target in sentence:
            #     token_idx.append(np.array(gpt_tokenizer.encode(target)))
            if batch_size != 1:
                # if os.path.exists(pk_path):
                #     with open(pk_path, 'rb') as f:
                #         token_idx = pickle.load(f)
                # else:
                for sent in sentence:
                    tokenized_sent = gpt_tokenizer(sent.strip(), padding="max_length", max_length=50, truncation=True)['input_ids']
                    token_idx.append(tokenized_sent)
                    # with open(pk_path, 'wb') as f:
                    #     pickle.dump(token_idx, f)

            dis_out = dismodel.discriminator(token_idx)
            # dis_out = dismodel.discriminator(std_token_idx)

            """calculation loss & traning"""
            ## training using discriminator loss
            cls_loss = dismodel.cls_loss(attribute, dis_out)
            # cls_loss = dismodel.cls_loss(std_attribute, dis_out)
            summary.add_scalar('discriminator loss', cls_loss.item(), start)

            cls_trainer.zero_grad()
            cls_loss.backward() # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(dismodel.cls_params, max_grad_norm)
            cls_trainer.step()

        if not start % 10000:
            print(f'standard - epoch:{start//std_len}, loss{cls_loss:.3f}')

        # dis_out = dismodel.discriminator(dia_token_idx)
        # cls_loss = dismodel.cls_loss(dia_attribute, dis_out)
        # cls_trainer.zero_grad()
        # cls_loss.backward()
        # cls_trainer.step()
        # pbar.set_description(f'dialect - epoch:{epoch}, loss{cls_loss:.3f}')

        """savining point"""
        # if (start+1)%pos_len == 0:
        if not (start//std_len) % 1:
            # random.shuffle(amazon_neg_dataset)
            # random.shuffle(amazon_pos_dataset)
            # save_model((start+1)//pos_len)
            save_model(start//std_len)
        save_model('final')  # final_model


def save_model(iter):
    if not os.path.exists('models_new_test_20/'):
        os.makedirs('models_new_test_20/')
    torch.save(dismodel.state_dict(), 'models_new_test_20/cls_model_{}'.format(iter))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='num batch_sizes')
    parser.add_argument('--csv_filename', type=str, default='2_train.csv', help='num batch_sizes')
    parser.add_argument('--standard_csv_filename', type=str, default='5_train_standard.csv',
                        help='standard csv file name')
    parser.add_argument('--dialect_csv_filename', type=str, default='5_train_dialect.csv', help='dialect csv file name')
    parser.add_argument('--val_standard_csv_filename', type=str, default='2_valid_standard.csv',
                        help='standard csv file name')
    parser.add_argument('--val_dialect_csv_filename', type=str, default='2_valid_dialect.csv', help='dialect csv file name')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args.epoch, args.batch_size, args.standard_csv_filename, args.dialect_csv_filename, args.val_standard_csv_filename, args.val_dialect_csv_filename, args.csv_filename)