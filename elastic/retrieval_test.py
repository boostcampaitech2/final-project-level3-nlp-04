import os
import pickle
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
import sys 
from os import path 

sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from elastic_search import ElasticSearchRetrieval, timer
# from retrieval import SparseRetrieval
# from dense_retrieval import get_encoders, DenseRetrieval, timer
from utils_qa import get_args

if __name__ == "__main__":
    # get arguments
    model_args, data_args, training_args = get_args()

    # Test sparse

    org_dataset = datasets = load_dataset('samgin/star_tagging')
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            # org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    retriever = ElasticSearchRetrieval(data_args)


    query = "육즙이 많아"

    for k in [30,100]:
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
