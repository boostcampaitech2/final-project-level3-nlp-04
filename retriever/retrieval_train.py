import logging
import os

import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AdamW

from dense_retrieval import DenseRetrieval
from utils import get_encoders, Config
from utils import set_seed

logger = logging.getLogger(__name__)


def main():
    # get arguments
    retriever_path = Path(os.path.abspath(__file__)).parent  # retriever folder path
    data_path = os.path.join(retriever_path.parent, 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'base_model.yaml'))

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

    best_acc = train_retrieval(config, tokenizer, p_encoder, q_encoder, data_path)

    wandb.join()

    return {'best_acc': best_acc}


# train step마다 수행되는 과정 : loss구하기
def training_per_step(training_args, model_args, batch, p_encoder, q_encoder, criterion, scaler):
    with autocast():
        batch_loss, batch_acc = 0, 0
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        if 'roberta' in model_args.retrieval_model_name_or_path:
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
        targets = torch.arange(0, training_args.per_device_retrieval_train_batch_size).long()
        if torch.cuda.is_available():
            targets = targets.to('cuda')

        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, dim=1)

        loss = criterion(sim_scores, targets)
        scaler.scale(loss).backward()

        batch_loss += loss.cpu().item()
        batch_acc += torch.sum(preds.cpu() == targets.cpu())

    return batch_loss, batch_acc


# eval step마다 수행되는 함수 : 몇 개를 맞췄는지 count
def evaluating_per_step(training_args, model_args, batch, p_encoder, q_encoder):
    batch_acc = 0

    if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

    if 'roberta' in model_args.retrieval_model_name_or_path:
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
    sim_scores = F.log_softmax(sim_scores, dim=1)
    _, preds = torch.max(sim_scores, dim=1)

    # target : position of positive samples = diagonal element
    targets = torch.arange(0, training_args.per_device_retrieval_eval_batch_size).long()

    batch_acc += torch.sum(preds.cpu() == targets)

    return batch_acc


'''
retrieval 학습을 위한 메소드 : dataloader, optimizer, scaler, scheduler 등을 설정하고 매 step마다 
training_per_step을 통해서 Loss를 구해주고 gradient_accumulation_steps값에 맞춰 update, backward등을 수행해준다.
학습이 1 epoch 끝나면 validation dataset으로 evaluation을 해주고 나온 결과과 기존의 최고점보다 높으면 새롭게 모델을 저장한다.
'''


def train_retrieval(config, tokenizer, p_encoder, q_encoder, data_path):
    dense_retrieval = DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path)

    train_dataloader, eval_dataloader = dense_retrieval.get_dataloader()

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.retrieval_learning_rate,
        eps=training_args.adam_epsilon
    )
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.NLLLoss()

    # Start training!
    best_acc = 0.0

    train_iterator = tqdm(range(int(training_args.num_train_epochs)), desc='Epoch')
    for epoch in train_iterator:
        optimizer.zero_grad()
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        ## train phase
        running_loss, running_acc, num_cnt = 0, 0, 0

        p_encoder.train()
        q_encoder.train()

        epoch_iterator = tqdm(train_dataloader, desc='train-Iteration')
        for step, batch in enumerate(epoch_iterator):
            batch_loss, batch_acc = training_per_step(training_args, model_args, batch,
                                                      p_encoder, q_encoder, criterion, scaler)
            running_loss += batch_loss / training_args.per_device_retrieval_train_batch_size
            running_acc += batch_acc / training_args.per_device_retrieval_train_batch_size
            num_cnt += 1

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                log_step = epoch * len(epoch_iterator) + step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                p_encoder.zero_grad()
                q_encoder.zero_grad()

        train_epoch_loss = float(running_loss / num_cnt)
        train_epoch_acc = float((running_acc.double() / num_cnt).cpu() * 100)
        print(f'global step-{log_step} | Loss: {train_epoch_loss:.4f} Accuracy: {train_epoch_acc:.2f}')

        # eval phase
        epoch_iterator = tqdm(eval_dataloader, desc='valid-Iteration')
        p_encoder.eval()
        q_encoder.eval()

        running_acc, num_cnt = 0, 0
        for step, batch in enumerate(epoch_iterator):
            with torch.no_grad():
                batch_acc = evaluating_per_step(training_args, model_args, batch, p_encoder, q_encoder)

                running_acc += batch_acc / training_args.per_device_retrieval_eval_batch_size
                num_cnt += 1

        eval_epoch_acc = float((running_acc / num_cnt) * 100)
        print(f'Epoch-{epoch} | Accuracy: {eval_epoch_acc:.2f}')

        # compare current eval & best eval
        if eval_epoch_acc > best_acc:
            best_epoch = epoch
            best_acc = eval_epoch_acc

            p_save_path = os.path.join(training_args.retrieval_output_dir, 'p_encoder')
            q_save_path = os.path.join(training_args.retrieval_output_dir, 'q_encoder')

            p_encoder.encoder.save_pretrained(p_save_path)
            q_encoder.encoder.save_pretrained(q_save_path)
            print(f'\t===> best model saved - {best_epoch} / Accuracy: {best_acc:.2f}')

        wandb.log({
            'train/loss': train_epoch_loss,
            'train/learning_rate': scheduler.get_last_lr()[0],
            'eval/epoch_acc': eval_epoch_acc,
            'epoch': epoch,
        })

    return best_acc


if __name__ == '__main__':
    main()
