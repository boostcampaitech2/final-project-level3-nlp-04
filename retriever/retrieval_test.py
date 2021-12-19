import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pickle
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk, concatenate_datasets

from dense_retrieval import DenseRetrieval, timer
from retriever.utils import Config, get_encoders

'''
retrieval의 성능 확인 
확인 방법 : 주어진 qeustion에 대해서 top k개의 corpus를 open corpus에서 가져오고 가져온 corpus에 정답이 있는지 확인하는 방식
* 주어진 dataset의 지문은 open corpus에 포함되어 있다.
'''
if __name__ == "__main__":
    # get arguments
    retriever_path = Path(os.path.abspath(__file__)).parent  # retriever folder path
    encoder_path = os.path.join(retriever_path, 'output')
    data_path = os.path.join(retriever_path.parent, 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'klue_bert_base_model_test.yaml'))

    tokenizer, p_encoder, q_encoder = get_encoders(config)
    p_encoder.load_state_dict(torch.load(os.path.join(encoder_path, 'p_encoder', f'{config.run_name}.pt')))
    q_encoder.load_state_dict(torch.load(os.path.join(encoder_path, 'q_encoder', f'{config.run_name}.pt')))

    if not os.path.exists(config.dataset_path):
        DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path).make_dataset()
    org_dataset = load_from_disk(config.dataset_path)

    # 전처리가 되어 있는 open corpus를 사용할지에 따라 분기가 나눠진다.
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')

    retriever = DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path)
    retriever.get_dense_embedding()

    # wandb setting
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    wandb.init(project=config.project_name,
               name=config.run_name,
               entity='ssp',
               reinit=True,
               )
    # top k개 주어졌을때 eval 평가하기(k개만큼의 corpus를 가져왔을때 정답 지문이 포함되었있는지 확인)
    if config.use_faiss:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve(full_ds, topk=k)
                df["correct"] = df.apply(lambda x: x["original_context"] in x["context"], axis=1)
                accuracy = round(df['correct'].sum() / len(df) * 100, 2)
                print(
                    f"Top-{k}\n"
                    "correct retrieval result by exhaustive search",
                    accuracy,
                )

                wandb.log({
                    'k': k,
                    'accuracy': accuracy,
                })
