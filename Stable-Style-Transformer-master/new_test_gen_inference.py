import torch
import numpy as np
import torch.nn as nn
import sys
import pandas as pd
from transformers import *
# sys.path.insert(0, "/DATA/joosung/controllable_english/classifier/")
from dis_model import *

gpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

dismodel = findattribute(gpt_tokenizer).cuda()
dismodel_name = 'cls_model_19'
dismodel.load_state_dict(torch.load('./models_new_test_20/{}'.format(dismodel_name)))
dismodel.eval()

from tqdm import tqdm
from gen_model import *
import os

genmodel = styletransfer(gpt_tokenizer).cuda()

# data_path = "/DATA/joosung/sentiment_data/Sentiment-and-Style-Transfer-master/data"
# yelp_neg_path = data_path + "/yelp/sentiment.test.0"
# yelp_neg_open = open(yelp_neg_path, "r")
# yelp_neg_dataset = yelp_neg_open.readlines()
# neg_len = len(yelp_neg_dataset)
# yelp_neg_open.close()
#
# yelp_pos_path = data_path + "/yelp/sentiment.test.1"
# yelp_pos_open = open(yelp_pos_path, "r")
# yelp_pos_dataset = yelp_pos_open.readlines()
# pos_len = len(yelp_pos_dataset)
# yelp_pos_open.close()
data_path = "./data/dialect/"
csv_filename = '2_valid.csv'
df_train = pd.read_csv(os.path.join(data_path, csv_filename), encoding='utf-8')
std_dataset = df_train['std'][:10]
dia_dataset = df_train['dia'][:10]
neg_len = len(std_dataset)
pos_len = len(dia_dataset)

stop_point = pos_len

PAD_IDX = 50258

name_list = [4]
prob = 0.6
save_prob = '06'
for name in tqdm(range(len(name_list))):
    for m in range(4):
        if m == 0:
            per = 0
        elif m == 1:
            per = 50
        elif m == 2:
            per = 60
        else:
            per = 70

        genmodel_name = 'new_gen_model_' + str(name_list[name])
        genmodel.load_state_dict(torch.load('./gen_models_test/{}'.format(genmodel_name)))
        genmodel.eval()
        model0 = 'sentiment.test.0.' + 'joo' + str(name_list[name]) + '_' + str(per) + '_' + str(save_prob)
        model1 = 'sentiment.test.1.' + 'joo' + str(name_list[name]) + '_' + str(per) + '_' + str(save_prob)
        f0 = open(model0, 'w')
        f1 = open(model1, 'w')

        for start in range(stop_point):
            """data start point"""
            neg_start = start
            pos_start = start

            """data setting"""
            neg_sentence = std_dataset[neg_start].strip()
            pos_sentence = dia_dataset[pos_start].strip()

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
                attribute = attributes[i]  # for decoder
                fake_attribute = attributes[abs(1 - i)]  # for generate
                sentiment = sentiments[i]  # for delete

                token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()
                ori_length = token_idx.shape[1]

                # delete model
                if per == 0:
                    max_len = int(token_idx.shape[1] - 1)  # 0%
                elif per == 50:
                    max_len = int(token_idx.shape[1] / 2)  # 50%
                elif per == 60:
                    max_len = int(token_idx.shape[1] / 10 * 4)  # 60%
                else:
                    max_len = int(token_idx.shape[1] / 10 * 3)  # 70%
                #             max_len = 0 # 100%

                #                 dis_out = dismodel.discriminator(token_idx)
                #                 sentiment = dis_out.argmax(1).cpu().item() ## for delete

                del_idx = token_idx
                for k in range(max_len):
                    del_idx = dismodel.att_prob(del_idx, sentiment)
                    dis_out = dismodel.discriminator(del_idx)
                    sent_porb = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item()
                    if sent_porb < prob:  # 0.7
                        break

                del_list = del_idx.squeeze(0).cpu().tolist()  # list
                del_sen = ''
                for x in range(len(del_list)):
                    token = num2token[del_list[x]].strip('Ġ')
                    del_sen += token
                    del_sen += ' '
                del_sen = del_sen.strip()

                del_percent = 100 - (del_idx.shape[1]) / (token_idx.shape[1]) * 100

                enc_out = genmodel.encoder(del_idx)
                dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

                dec_tokens, dec_sen = genmodel.dec2sen(vocab_out)

                gen_sen_2 = genmodel.generated_sentence(enc_out, fake_attribute, ori_length).replace('<|endoftext|>',
                                                                                                     '')

                if i == 0:
                    f0.write(sentence + '\t' + gen_sen_2 + '\t' + str(sentiment) + '\n')
                if i == 1:
                    f1.write(sentence + '\t' + gen_sen_2 + '\t' + str(sentiment) + '\n')
        f0.close()
        f1.close()