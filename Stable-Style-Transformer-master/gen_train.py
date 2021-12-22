import logging

logger = logging.getLogger()
logger.setLevel("ERROR")

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import os
import random

from transformers import *

# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
from tqdm import tqdm
import json

## 초기화
from gen_model import *

genmodel = styletransfer(tokenizer=gpt_tokenizer).cuda()
# genmodel.load_state_dict(torch.load('../ST_v2.0/models/gen_model_5'))
genmodel.train()

import sys

sys.path.insert(0, "/DATA/joosung/controllable_english/amazon/classifier/")
from new_dis_model import *

dismodel = findattribute(tokenizer=gpt_tokenizer).cuda()
dismodel_name = 'cls_model_final'
dismodel.load_state_dict(torch.load('./models_test/{}'.format(dismodel_name)))
dismodel.eval()

import torch.optim as optim

from tensorboardX import SummaryWriter
import pickle
import pandas as pd

summary = SummaryWriter(logdir='./logs')


def main(epoch, batch_size, standard_csv_filename, dialect_csv_filename, csv_filename):
    # f = open('amazon_vocab.json')
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
    data_path = './data/dialect/'

    # pk_path = '/'.join([data_path, csv_filename.split('_')[0]])

    df_train = pd.read_csv(os.path.join(data_path,csv_filename), encoding='utf-8')
    std_dataset = df_train['std'][:80000]
    dia_dataset = df_train['dia'][:80000]
    std_len = len(std_dataset)
    dia_len = len(dia_dataset)

    # std_data_name = os.path.join([data_path, standard_csv_filename])
    # dia_data_name = os.path.join([data_path, dialect_csv_filename])
    # std_open = open(std_data_name, 'r')
    # std_dataset = std_open.readlines()
    # std_len = len(std_dataset)
    # std_open.close()
    #
    # dia_open = open(dia_data_name, 'r')
    # dia_dataset = dia_open.readlines()
    # dia_len = len(dia_dataset)
    # dia_open.close()

    """training parameter"""
    aed_initial_lr = 0.00001
    gen_initial_lr = 0.001
    aed_trainer = optim.Adamax(genmodel.aed_params, lr=aed_initial_lr)  # initial 0.0005
    gen_trainer = optim.Adamax(genmodel.aed_params, lr=gen_initial_lr)  # initial 0.0001
    max_grad_norm = 10
    batch = batch_size
    epoch = epoch
    epoch_len = max(std_len, dia_len)
    stop_point = epoch_len * epoch

    pre_epoch = 0
    for start in tqdm(range(0, stop_point, batch_size)):
        ## learing rate decay
        # now_epoch = (start+1)//pos_len

        """data start point"""
        std_start = start % std_len
        dia_start = start % dia_len

        """data setting"""
        std_sentence = std_dataset[std_start:std_start + batch_size]
        dia_sentence = dia_dataset[dia_start:dia_start + batch_size]

        if len(std_sentence) != batch_size:
            continue
        std_labels = [[1, 0] for _ in range(len(std_sentence))]  # negative labels
        # std_labels.append([1,0])
        std_attribute = torch.from_numpy(np.asarray(std_labels)).type(torch.FloatTensor).cuda()

        dia_labels = [[0, 1] for _ in range(len(dia_sentence))]  # positive labels
        # pos_labels.append([0,1])
        dia_attribute = torch.from_numpy(np.asarray(dia_labels)).type(torch.FloatTensor).cuda()

        sentences = [std_sentence, dia_sentence]
        attributes = [std_attribute, dia_attribute]
        sentiments = [0, 1]

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i]  # for decoder
            fake_attribute = attributes[abs(1 - i)]  # for generate
            #             sentiment = sentiments[i] # for delete

            # token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
            token_idx = []
            # pickle_name =
            # if os.path.exists('gen_tokenized.pkl'):
            for sent in sentence:
                tokenized_sent = gpt_tokenizer(sent.strip(), padding="max_length", max_length=50, truncation=True)[
                    'input_ids']
                token_idx.append(tokenized_sent)
            #     with open(pickle_name, 'wb') as f:
            #         pickle.dump(token_idx, f)
            # else:
            #     with open('gen_tokenized.pkl', 'rb') as f:
            #         token_idx = pickle.load(f)

            # delete model
            max_len = int(len(token_idx[1]) / 2)
            dis_out = dismodel.discriminator(token_idx)
            # sentiment = dis_out.argmax(1).cpu().item() ## 변경점 for delete
            sentiment = dis_out.argmax(1).cpu().numpy()
            del_idx = token_idx

            temp_del_idx_list = []
            for b, s in zip(del_idx, sentiment):
                for k in range(max_len):
                    temp_del_idx = dismodel.unbatch_att_prob(b, s)
                    temp_dis_out = dismodel.discriminator(temp_del_idx)
                    temp_sent_prob = F.softmax(temp_dis_out, 1).squeeze(0)[s].cpu().detach().numpy().item()
                    # del_idx = dismodel.att_prob(del_idx, sentiment)
                    # dis_out = dismodel.discriminator(del_idx)
                    # sent_prob = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
                    # sent_prob = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy()
                    # sent_prob = np.array([[x[0]] if i == 0 else [x[1]] for x, i in zip(F.softmax(dis_out, 1).cpu().detach().numpy(), sentiment)])
                    if temp_sent_prob < 0.7 or k == 24:
                        temp_del_idx_list.append(temp_del_idx.cpu().detach().tolist())
                        break
                    # if sent_prob < 0.7:
                    #     break

            temp_del_idx_list = torch.tensor(temp_del_idx_list)  # sㅁp
            del_idx = temp_del_idx_list.squeeze(1).cuda()

            """auto-encoder loss & traning"""
            # training using discriminator loss

            enc_out = genmodel.encoder(del_idx)
            dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

            ## calculation loss
            recon_loss = genmodel.recon_loss(token_idx, vocab_out)
            summary.add_scalar('reconstruction loss', recon_loss.item(), start)

            aed_trainer.zero_grad()
            recon_loss.backward(retain_graph=True)  # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)

            """decoder classification loss & training"""
            ## calculation loss
            gen_cls_out = dismodel.gen_discriminator(vocab_out)

            ## calculation loss
            gen_cls_loss = genmodel.cls_loss(attribute, gen_cls_out)
            summary.add_scalar('generated sentence loss', gen_cls_loss.item(), start)

            gen_trainer.zero_grad()
            gen_cls_loss.backward()  # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)
            aed_trainer.step()
            gen_trainer.step()


        if not start % 10000:
            print(f'epoch:{start // std_len},  recon_loss:{recon_loss:.3f},  gen_cls_loss:{gen_cls_loss:.3f}')

        """savining point"""
        # if (start+1)%epoch_len == 0:
        if not (start // std_len) % 2:
            # random.shuffle(amazon_std_dataset)
            # random.shuffle(amazon_pos_dataset)
            # save_model((start+1)//pos_len)
            save_model(start // std_len)
    save_model('final')  # final_model


def save_model(iter):
    if not os.path.exists('gen_models_test_256/'):
        os.makedirs('gen_models_test_256/')
    torch.save(genmodel.state_dict(), 'gen_models_test_256/gen_model_{}'.format(iter))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='num batch_sizes')
    parser.add_argument('--standard_csv_filename', type=str, default='2_train_standard.csv',
                        help='standard csv file name')
    parser.add_argument('--dialect_csv_filename', type=str, default='2_train_dialect.csv', help='dialect csv file name')
    parser.add_argument('--csv_filename', type=str, default='2_train.csv', help='num batch_sizes')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args.epoch, args.batch_size, args.standard_csv_filename, args.dialect_csv_filename, args.csv_filename)