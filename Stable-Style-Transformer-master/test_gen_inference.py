import torch
from transformers import *
import numpy as np

gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

from tqdm import tqdm
import os
import pandas as pd
import json
# f = open('./amazon_vocab.json')
# token2num = json.load(f)

token2num = gpt_tokenizer.get_vocab()
num2token = {}
for key, value in token2num.items():
    num2token[value] = key
print('패드 토큰',token2num['<pad>'])
print(num2token[3])

import torch
import numpy as np
import torch.nn as nn
import sys

# sys.path.insert(0, "/DATA/joosung/controllable_english/yelp/classifier/")
from dis_model import *
dismodel = findattribute(gpt_tokenizer).cuda()
print('패드 토큰',token2num['<pad>'])
print(num2token[3])
dismodel_name='cls_model_final'
dismodel.load_state_dict(torch.load('./models_test/{}'.format(dismodel_name)))
dismodel.eval()

from gen_model import *
genmodel = styletransfer(gpt_tokenizer).cuda()
print('패드 토큰',token2num['<pad>'])
print(num2token[3])
genmodel_name='new_gen_model_4'
genmodel.load_state_dict(torch.load('./gen_models_test/{}'.format(genmodel_name)))
genmodel.eval()
print('ok')

import tqdm

data_path = "./data/dialect/"
csv_filename = '2_valid.csv'
# yelp_neg_path = data_path + "5_train_standard.csv"
# yelp_neg_open = open(yelp_neg_path, "r")
# yelp_neg_dataset = yelp_neg_open.readlines()
# neg_len = len(yelp_neg_dataset)
# yelp_neg_open.close()
#
# yelp_pos_path = data_path + "5_train_dialect.csv"
# yelp_pos_open = open(yelp_pos_path, "r")
# yelp_pos_dataset = yelp_pos_open.readlines()
# pos_len = len(yelp_pos_dataset)
# yelp_pos_open.close()

df_train = pd.read_csv(os.path.join(data_path, csv_filename), encoding='utf-8')
std_dataset = df_train['std']
dia_dataset = df_train['dia']
neg_len = len(std_dataset)
pos_len = len(dia_dataset)

stop_point = np.random.randint(0, pos_len)
# stop_point = pos_len*epoch+batch


PAD_IDX = token2num['<pad>']

for start in range(stop_point - 1, stop_point):
    """data start point"""
    neg_start = start % neg_len
    pos_start = start % pos_len

    """data setting"""
    neg_sentence = std_dataset[neg_start].strip()
    pos_sentence = dia_dataset[pos_start].strip()
    # neg_sentence = yelp_neg_dataset[neg_start].strip()
    # pos_sentence = yelp_pos_dataset[pos_start].strip()

    neg_labels = []  # negative labels
    neg_labels.append([1, 0])
    neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

    pos_labels = []  # positive labels
    pos_labels.append([0, 1])
    pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()

    sentences = [neg_sentence, pos_sentence]
    attributes = [neg_attribute, pos_attribute]
    fake_attributes = [pos_attribute, neg_attribute]
    sentiments = [0, 1]
    """data input"""
    for i in range(2):
        # k=0: negative, k=1: positive
        sentence = sentences[i]
        for k in range(6):
            fake_attribute = k / 5 * attributes[0] + (1 - k / 5) * attributes[1]
            #             attribute = attributes[i] # for decoder
            #             fake_attribute = attributes[abs(1-i)] # for generate

            token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
            ori_length = token_idx.shape[1]

            # delete model
            max_len = int(token_idx.shape[1] / 2)
            sentiment = sentiments[i]  # for delete
            #             sentiment = dis_out.argmax(1).cpu().item() ## 변경점 for delete
            #             dis_out = dismodel.discriminator(token_idx)

            del_idx = token_idx
            for k in range(max_len):
                del_idx = dismodel.att_prob(del_idx, sentiment)
                dis_out = dismodel.discriminator(del_idx)
                sent_porb = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
                if sent_porb < 0.7:
                    break

            del_list = del_idx.squeeze(0).cpu().tolist()  # list
            del_sen = ''
            for x in range(len(del_list)):
                token = num2token[del_list[x]].strip('Ġ')
                del_sen += token
                del_sen += ' '
            del_sen = del_sen.strip()

            del_percent = 100 - (del_idx.shape[1]) / (token_idx.shape[1]) * 100

            # init_idx = del_idx[0][0]
            # temp_del_idx = []
            # temp_a = ''
            # for i in range(del_idx.shape[1]):
            #     target_idx = del_idx[0][i]
            #     if i == 0:
            #         enc_out = genmodel.encoder(target_idx.unsqueeze(0).unsqueeze(0))
            #         temp_a = genmodel.generated_sentence(enc_out, fake_attribute, i)
            #     else:
            #         temp_del_idx = torch.tensor(gpt_tokenizer.encode(temp_a)).unsqueeze(0).cuda()
            #         list_temp_del_idx = temp_del_idx.squeeze(0).cpu().detach().numpy().tolist()
            #         int_target_idx = target_idx.cpu().detach().numpy().tolist()
            #         list_temp_del_idx.append(int_target_idx)
            #         input_target = torch.tensor(list_temp_del_idx).unsqueeze(0).cuda()
            #         enc_out = genmodel.encoder(input_target)
            #         temp_a = genmodel.generator(enc_out, i, fake_attribute)
                # temp_a = genmodel.generated_sentence(enc_out, fake_attribute, i)

            """
            genmodel.encoder : input(batch, token_idx_len)
                            :  output(token_idx_len, batch, word_embedding)
            """
            enc_out = genmodel.encoder(del_idx)
            #             dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

            #             dec_tokens, dec_sen = genmodel.dec2sen(vocab_out)

            #             gen_sen_1 = genmodel.generated_sentence(enc_out, attribute, ori_length)
            gen_sen_2 = genmodel.generated_sentence(enc_out, fake_attribute, ori_length)
            # gen_sen_2 = genmodel.generator(enc_out, ori_length, fake_attribute)
            # gen_sen_3 = genmodel.dec2sen(gen_sen_2)

            print('Original Attribute: ', sentiment)
            print('Original Sentence: ', sentence)
            print('Delete Sentence: {}, {}%'.format(del_sen, del_percent))
            #             print('Reconstruction(decoder) Sentence: ', dec_sen)
            #             print('Reconstruction(generator) Sentence', sentiment, ': ', gen_sen_1.rstrip('<|endoftext|>'))
            #             print('Style transfer(generator) Sentence', abs(1-sentiment), ': ', gen_sen_2.rstrip('<|endoftext|>'))
            print('Style transfer(generator) Sentence', fake_attribute.cpu().numpy().tolist()[0], ': ',
                  gen_sen_2.rstrip('<|endoftext|>'))
            print('')

