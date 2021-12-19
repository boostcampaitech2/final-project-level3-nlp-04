import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AdamW

from dense_retrieval import DenseRetrieval
from utils import get_encoders, Config, AverageMeter, set_seed, ReduceLROnPlateauPatch

logger = logging.getLogger(__name__)


def main():
    # get arguments
    retriever_path = Path(os.path.abspath(__file__)).parent  # retriever folder path
    data_path = os.path.join(retriever_path.parent, 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'klue_bert_base_model.yaml'))

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(config.seed)

    # get_tokenizer, model
    tokenizer, p_encoder, q_encoder = get_encoders(config)
    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')

    # set wandb
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=config.project_name,
               entity='ssp',
               name=config.run_name,
               reinit=True,
               )

    train_retrieval(config, tokenizer, p_encoder, q_encoder, data_path)

    wandb.join()


# train step마다 수행되는 과정 : loss구하기
def training_per_step(config, batch, p_encoder, q_encoder, criterion, scaler, optimizer, global_step):
    p_encoder.train()
    q_encoder.train()
    with autocast():
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        if 'roberta' in config.model_name_or_path:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
            q_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
        else:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

        p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
        q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

        # Calculate similarity score & loss
        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, batch_size)

        # target : position of positive samples = diagonal element
        targets = torch.arange(0, config.per_device_train_batch_size).long()
        if torch.cuda.is_available():
            targets = targets.to('cuda')

        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, dim=1)

        loss = criterion(sim_scores, targets)
        acc = torch.sum(preds.cpu() == targets.cpu())

        scaler.scale(loss).backward()
        if global_step % config.gradient_accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return loss.item(), acc.item()


# eval step마다 수행되는 함수 : 몇 개를 맞췄는지 count
def evaluating_per_step(config, eval_dataloader, p_encoder, q_encoder, criterion):
    p_encoder.eval()
    q_encoder.eval()

    # eval phase
    valid_loss = 0
    valid_acc = 0

    epoch_iterator = tqdm(eval_dataloader, desc='valid-Iteration')
    for step, batch in enumerate(epoch_iterator):
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        if 'roberta' in config.model_name_or_path:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
            q_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
        else:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

        p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
        q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

        # Calculate similarity score & loss
        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

        # target : position of positive samples = diagonal element
        targets = torch.arange(0, config.per_device_eval_batch_size).long()
        if torch.cuda.is_available():
            targets = targets.to('cuda')

        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, dim=1)

        loss = criterion(sim_scores, targets)
        acc = torch.sum(preds.cpu() == targets.cpu())

        valid_loss += loss.item() / len(batch[0])
        valid_acc += acc.item() / len(batch[0])

    valid_loss = valid_loss / len(eval_dataloader)
    valid_acc = valid_acc / len(eval_dataloader)

    return valid_loss, valid_acc


'''
retrieval 학습을 위한 메소드 : dataloader, optimizer, scaler, scheduler 등을 설정하고 매 step마다 
training_per_step을 통해서 Loss를 구해주고 gradient_accumulation_steps값에 맞춰 update, backward등을 수행해준다.
학습이 1 epoch 끝나면 validation dataset으로 evaluation을 해주고 나온 결과과 기존의 최고점보다 높으면 새롭게 모델을 저장한다.
'''


def train_retrieval(config, tokenizer, p_encoder, q_encoder, data_path):
    dense_retrieval = DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path)

    train_dataloader, eval_dataloader = dense_retrieval.get_dataloader

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon
    )
    scaler = GradScaler()
    scheduler = ReduceLROnPlateauPatch(optimizer=optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.NLLLoss()

    # model save path
    output_path = os.path.join(os.path.dirname(__file__), config.output_dir)
    p_encoder_path = os.path.join(output_path, 'p_encoder')
    q_encoder_path = os.path.join(output_path, 'q_encoder')
    if not os.path.exists(p_encoder_path):
        os.makedirs(p_encoder_path, exist_ok=True)
    if not os.path.exists(q_encoder_path):
        os.makedirs(q_encoder_path, exist_ok=True)
    p_save_path = os.path.join(p_encoder_path, f'{config.run_name}.pt')
    q_save_path = os.path.join(q_encoder_path, f'{config.run_name}.pt')

    # Start training!
    best_acc = 0.0
    global_step = 0
    patience_cnt = 0
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    for epoch in range(config.num_train_epochs):
        torch.cuda.empty_cache()

        epoch_iterator = tqdm(train_dataloader, desc='train-Iteration')
        for step, batch in enumerate(epoch_iterator):
            loss, acc = training_per_step(config, batch, p_encoder, q_encoder, criterion, scaler, optimizer, global_step)
            train_loss.update(loss / len(batch[0]))
            train_acc.update(acc / len(batch[0]))
            global_step += 1
            description = f"{epoch + 1}epoch {global_step: >5d}step | loss: {train_loss.avg: .4f} | acc: {train_acc.avg: .4f} | best_acc: {best_acc: .4f}"
            epoch_iterator.set_description(description)


            if (global_step + 1) % (config.logging_steps) == 0:
                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/acc': train_acc.avg,
                    'train/learning_rate': scheduler.get_lr()[0] if scheduler is not None else config.learning_rate,
                })

            if (global_step + 1) % (config.logging_steps * config.gradient_accumulation_steps) == 0:
                with torch.no_grad():
                    valid_loss, valid_acc = evaluating_per_step(config, eval_dataloader, p_encoder, q_encoder, criterion)

                    if scheduler is not None:
                        scheduler.step(valid_acc)

                    if valid_acc > best_acc:
                        torch.save(p_encoder.state_dict(), p_save_path)
                        torch.save(q_encoder.state_dict(), q_save_path)
                        best_acc = valid_acc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/acc': train_acc.avg,
                    'train/learning_rate': scheduler.get_lr()[0] if scheduler is not None else config.learning_rate,
                    'eval/best_acc': best_acc,
                    'eval/acc': valid_acc,
                    'eval/loss': valid_loss,
                    'global_step': global_step,
                })
                train_loss.reset()
                train_acc.reset()

                if patience_cnt == config.early_stopping_patience:
                    print('더이상 acc 가 오르지 않아서 학습을 중지합니다!')
                    return
            else:
                wandb.log({'global_step': global_step})


if __name__ == '__main__':
    main()
