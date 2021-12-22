import glob
import torch
from new_dis_model import *
# from origin_dis_model import *
from transformers import *
import pandas as pd
gpt_tokenizer =  PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
import os
from tqdm import tqdm
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
for i in range(0,21):
    # if i != 5:
    #     continue

    print('*'*20)
    print(i)
    # dismodel = findattribute().cuda()
    dismodel = findattribute(gpt_tokenizer).cuda()

    dismodel_name = f'cls_model_{i}' if i != 20 else f'cls_model_final'
    # dismodel.load_state_dict(torch.load('./models_origin/{}'.format(dismodel_name)))
    # dismodel.load_state_dict(torch.load('./models_test/{}'.format(dismodel_name)))
    dismodel.load_state_dict(torch.load('./models_new_test_20/{}'.format(dismodel_name)))
    dismodel.eval()


    # output_path = './data/yelp/'
    # neg_out_files = glob.glob(output_path + 'sentiment.test.1')
    # pos_out_files = glob.glob(output_path + 'sentiment.test.0')

    output_path = './data/dialect/'
    df_valid = pd.read_csv(os.path.join(output_path,'2_valid.csv'), encoding='utf-8')
    std_dataset = df_valid['std']
    dia_dataset = df_valid['dia']
    std_len = len(std_dataset)
    dia_len = len(dia_dataset)
    # neg_out_files = glob.glob(output_path + '5_valid_standard.csv')
    # pos_out_files = glob.glob(output_path + '5_valid_dialect.csv')

    # for i in range(len(neg_out_files)):
    for i in range(1):
        # print("model: ", neg_out_files[i].split('.')[-1])

        # neg_data_open = open(neg_out_files[i], "r")
        # neg_data_dataset = neg_data_open.readlines()
        # neg_len = len(neg_data_dataset)
        # neg_data_open.close()

        neg_correct = 0
        # for k in range(neg_len):
        for k in tqdm(range(std_len)):
            # out_sen = neg_data_dataset[k].split('\t')[0].strip()
            # out_sen = neg_data_dataset[k].split('\n')[0].strip()
            out_sen = std_dataset[k].strip()

            # tokenized_sent = gpt_tokenizer(out_sen.strip(), padding="max_length", max_length=50, truncation=True)['input_ids']
            # # tokenized_sent = gpt_tokenizer(out_sen, padding="max_length", max_length=50, truncation=True)[
            # #     'input_ids']
            #
            # token_idx = torch.tensor(gpt_tokenizer.encode(tokenized_sent)).unsqueeze(0).cuda()
            token_idx = torch.tensor(gpt_tokenizer.encode(out_sen)).unsqueeze(0).cuda()

            """discriminator"""
            if token_idx.shape[1] != 0:
                result = dismodel.discriminator(token_idx=token_idx).argmax(1).cpu().numpy()[0]
                if result == 1:  ## style transfer so result must be 1(positive)
                    neg_correct += 1

        # pos_out_name = pos_out_files[i]
        # pos_data_open = open(pos_out_name, "r")
        # pos_data_dataset = pos_data_open.readlines()
        # pos_len = len(neg_data_dataset)
        # pos_data_open.close()

        pos_correct = 0
        # for k in range(pos_len):
        for k in tqdm(range(dia_len)):
            # out_sen = pos_data_dataset[k].split('\t')[0].strip()
            # out_sen = pos_data_dataset[k].split('\n')[0].strip()
            out_sen = dia_dataset[k].strip()

            # tokenized_sent = gpt_tokenizer(out_sen.strip(), padding="max_length", max_length=50, truncation=True)[
            #     'input_ids']
            # token_idx = torch.tensor(gpt_tokenizer.encode(tokenized_sent)).unsqueeze(0).cuda()
            # token_idx = torch.tensor(gpt_tokenizer.encode(out_sen)).unsqueeze(0).cuda()
            token_idx = torch.tensor(gpt_tokenizer.encode(out_sen)).unsqueeze(0).cuda()

            """discriminator"""
            if token_idx.shape[1] != 0:
                result = dismodel.discriminator(token_idx=token_idx).argmax(1).cpu().numpy()[0]
                if result == 0:  ## style transfer so result must be 0(negative)
                    pos_correct += 1

        # Acc = (neg_correct + pos_correct) / (neg_len + pos_len) * 100
        Acc = (neg_correct + pos_correct) / (std_len + dia_len) * 100
        print(f"Accuracy: {Acc:.3f}%")