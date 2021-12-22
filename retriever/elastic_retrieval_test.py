import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from pathlib import Path
from elastic_search import ElasticSearchRetrieval, timer
from datasets import load_from_disk, concatenate_datasets

from utils import Config, make_dataset


if __name__ == "__main__":
    # get arguments
    retriever_path = Path(os.path.abspath(__file__)).parent  # retriever folder path
    encoder_path = os.path.join(retriever_path, 'output')
    data_path = os.path.join(retriever_path.parent, 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'klue_bert_base_model.yaml'))

    # Test sparse
    if not os.path.isdir(config.dataset_path):
        make_dataset(config, data_path)
    org_dataset = load_from_disk(config.dataset_path)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = ElasticSearchRetrieval(config, data_path)

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

    query_list = full_ds['query']

    for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        df_list = []
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(query_list, full_ds, topk=k)
            df["correct"] = df.apply(lambda x: x['original_context'] in x["context"], axis=1)
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
