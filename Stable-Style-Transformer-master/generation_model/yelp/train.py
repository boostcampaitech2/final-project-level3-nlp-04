import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
import os
import random

from transformers import *
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
from tqdm import tqdm
import json


## 초기화
from gen_model import *
genmodel = styletransfer().cuda()
genmodel.train()

sys.path.insert(0, "./classifier/")
from classifier.dis_model import *
dismodel = findattribute().cuda()
dismodel_name='cls_model_final'
dismodel.load_state_dict(torch.load('./classifier/models/{}'.format(dismodel_name)))
dismodel.eval()


import torch.optim as optim

from tensorboardX import SummaryWriter
summary = SummaryWriter(logdir='./logs')

def main():    
    f = open('gpt_yelp_vocab.json')
    token2num = json.load(f)

    num2token = {} # TODO : 쓰지도 않을거면서 왜 만들어 놓은 걸까?
    for key, value in token2num.items():
        num2token[value] = key
    f.close()

    data_path = '../../data/'
    train_yelp_neg_path = data_path + "/yelp/sentiment.train.0"
    train_yelp_neg_open = open(train_yelp_neg_path, "r")
    train_yelp_neg_dataset = train_yelp_neg_open.readlines()
    yelp_neg_dataset = train_yelp_neg_dataset
    
    neg_len = len(yelp_neg_dataset)
    train_yelp_neg_open.close()

    train_yelp_pos_path = data_path + "/yelp/sentiment.train.1"
    train_yelp_pos_open = open(train_yelp_pos_path, "r")
    train_yelp_pos_dataset = train_yelp_pos_open.readlines()
    yelp_pos_dataset = train_yelp_pos_dataset
    
    pos_len = len(yelp_pos_dataset)
    train_yelp_pos_open.close()

    """training parameter"""
    aed_initial_lr = 0.00001
    gen_initial_lr = 0.001
    aed_trainer = optim.Adamax(genmodel.aed_params, lr=aed_initial_lr) # initial 0.0005
    gen_trainer = optim.Adamax(genmodel.aed_params, lr=gen_initial_lr) # initial 0.0001
    max_grad_norm = 20
    batch = 64
    epoch = 6
    stop_point = pos_len*epoch
    
    pre_epoch = 0

    for start in tqdm(range(0, stop_point)):
        ## learing rate decay
        now_epoch = (start+1)//pos_len

        """data start point"""
        neg_start = start%neg_len
        pos_start = start%pos_len

        """data setting"""
        neg_sentence = yelp_neg_dataset[neg_start].strip()
        pos_sentence = yelp_pos_dataset[pos_start].strip()

        neg_labels = [] # negative labels
        neg_labels.append([1,0])
        neg_attribute = torch.from_numpy(np.asarray(neg_labels)).type(torch.FloatTensor).cuda()

        pos_labels = [] # positive labels
        pos_labels.append([0,1])
        pos_attribute = torch.from_numpy(np.asarray(pos_labels)).type(torch.FloatTensor).cuda()

        sentences = [neg_sentence, pos_sentence]
        attributes = [neg_attribute, pos_attribute]
        sentiments = [0, 1]

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            sentence = sentences[i]
            attribute = attributes[i] # for decoder
            fake_attribute = attributes[abs(1-i)] # for generate
#             sentiment = sentiments[i] # for delete

            token_idx = torch.tensor(gpt_tokenizer.encode(sentence)).unsqueeze(0).cuda()

            # delete model
            max_len = int(token_idx.shape[1]/2)
            dis_out = dismodel.discriminator(token_idx)
            sentiment = dis_out.argmax(1).cpu().item() ## 변경점 for delete

            del_idx = token_idx
            for k in range(max_len): # TODO : 왜 max_len만큼 반복해주는가? -> loop안에 내용은 한번만 해도 될 것 같은데. 이유 : att_prob에서 모든 경우를 고려해 줬다고 생각함.
                # TODO : 그런데 max_len이 token_idx.shape[1]의 절반인데??? 왜 이렇게 해줬을까???
                del_idx = dismodel.att_prob(del_idx, sentiment)  # TODO 여기서 style이 제거되는거 같은데 음....
                dis_out = dismodel.discriminator(del_idx)
                sent_porb = F.softmax(dis_out, 1).squeeze(0)[sentiment].cpu().detach().numpy().item() # TODO 왜 F.softmax(dis_out, 1).squeeze(0)[sentiment] 과 같이 sentiment가 index 요소로 들어갔을까?
                if sent_porb < 0.7: # TODO : sent_prob은 반복되면서 바뀔 수 있는 값인가? 내 생각에는 del_idx에서 항상 같은 값이 나오기 때문에 똑같을 거라고 생각함. <- discriminator는 잘 몰라서 고려해주지 못함
                    break

            """ 위의 부분까지 해서 변경해줘야할 문장의 token 위치를 찾은 것 같음."""
            """auto-encoder loss & traning"""
            # training using discriminator loss
            enc_out = genmodel.encoder(del_idx) # TODO : 분명 encoder의 input값은 변경될 요소의 값이 비어있고 거기에 style과 start 토큰이 추가된 형태여야 하는데 이 부분이 어디있지?
            dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

            ## calculation loss
            recon_loss = genmodel.recon_loss(token_idx, vocab_out) # TODO : recon_loss 재구성 손실함수 인가? 뭐징?
            summary.add_scalar('reconstruction loss', recon_loss.item(), start)

            aed_trainer.zero_grad()
            recon_loss.backward(retain_graph=True) # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)

            """decoder classification loss & training"""
            ## calculation loss
            gen_cls_out = dismodel.gen_discriminator(vocab_out)

            ## calculation loss
            gen_cls_loss = genmodel.cls_loss(attribute, gen_cls_out)  # TODO : 이 부분은 스타일 손실함수를 구해주는 부분인 것 같다.
            summary.add_scalar('generated sentence loss', gen_cls_loss.item(), start)

            gen_trainer.zero_grad()
            gen_cls_loss.backward() # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)
            aed_trainer.step()
            gen_trainer.step()


        """savining point"""
        if (start+1)%pos_len == 0:
            random.shuffle(yelp_neg_dataset)
            random.shuffle(yelp_pos_dataset)
            save_model((start+1)//pos_len)
    save_model('final') # final_model

    
def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(genmodel.state_dict(), 'models/gen_model_{}'.format(iter))  
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
