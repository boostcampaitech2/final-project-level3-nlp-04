import os
import pickle
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import sys 
from os import path

from utils import Config

sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from elastic_search import ElasticSearchRetrieval, timer


if __name__ == "__main__":
    # get arguments
    retriever_path = Path(os.path.abspath(__file__)).parent  # retriever folder path
    encoder_path = os.path.join(retriever_path, 'output')
    data_path = os.path.join(retriever_path.parent, 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'elastic_search.yaml'))

    # Test sparse
    org_dataset = load_from_disk(config.dataset_path)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    # print("*" * 40, "query dataset", "*" * 40)
    # print(full_ds)

    retriever = ElasticSearchRetrieval(config)


    query = "육즙이 많아"

    for k in [30]:
        with timer("bulk query by exhaustive search"):

            df, scores = retriever.retrieve(query, full_ds, topk=k)
            df["correct"] = df.apply(lambda x: query in x["review"], axis=1)
            print(df["context"])
            accuracy = round(df['correct'].sum() / len(df) * 100, 2)
            print(
                f"Top-{k}\n"
                "correct retrieval result by exhaustive search",
                accuracy,
            )
