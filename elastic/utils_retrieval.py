from typing import Callable, List

import torch.cuda
from datasets import DatasetDict, Features, Value, Sequence, Dataset
from numpy import dot
from numpy.linalg import norm
from transformers import TrainingArguments

from arguments import DataTrainingArguments
from elastic_search import ElasticSearchRetrieval
# from dense_retrieval import DenseRetrieval, get_encoders
# from retrieval import SparseRetrieval


# def run_sparse_retrieval(
#         tokenize_fn: Callable[[str], List[str]],
#         datasets: DatasetDict,
#         training_args: TrainingArguments,
#         data_args: DataTrainingArguments,
#         tokenizer_name,
#         data_path: str = "../data",
#         context_path: str = "wikipedia_documents.json",
# ) -> DatasetDict:
#     # Query에 맞는 Passage들을 Retrieval 합니다.
#     retriever = SparseRetrieval(
#         tokenize_fn=tokenize_fn, tokenizer_name=tokenizer_name, data_path=data_path, context_path=context_path
#     )
#     retriever.get_sparse_embedding(data_args.bm25)

#     if data_args.use_faiss:
#         retriever.build_faiss(num_clusters=data_args.num_clusters)
#         df = retriever.retrieve_faiss(
#             datasets["validation"], topk=data_args.top_k_retrieval
#         )
#     else:
#         df = retriever.retrieve(datasets["validation"], bm25=data_args.bm25, topk=data_args.top_k_retrieval)

#     # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
#     if training_args.do_predict:
#         f = Features(
#             {
#                 "context": Value(dtype="string", id=None),
#                 "id": Value(dtype="string", id=None),
#                 "question": Value(dtype="string", id=None),
#             }
#         )

#     # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
#     elif training_args.do_eval:
#         f = Features(
#             {
#                 "answers": Sequence(
#                     feature={
#                         "text": Value(dtype="string", id=None),
#                         "answer_start": Value(dtype="int32", id=None),
#                     },
#                     length=-1,
#                     id=None,
#                 ),
#                 "context": Value(dtype="string", id=None),
#                 "id": Value(dtype="string", id=None),
#                 "question": Value(dtype="string", id=None),
#             }
#         )
#     datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
#     return datasets


# def run_dense_retrieval(training_args, model_args, data_args, datasets):
#     tokenizer, p_encoder, q_encoder = get_encoders(training_args, model_args)
#     if torch.cuda.is_available():
#         p_encoder.to(training_args.device)
#         q_encoder.to(training_args.device)

#     # Query에 맞는 Passage들을 Retrieval 합니다.
#     retriever = DenseRetrieval(training_args, model_args, data_args, tokenizer, p_encoder, q_encoder)
#     retriever.get_dense_embedding()

#     if data_args.use_faiss:
#         retriever.build_faiss(num_clusters=data_args.num_clusters)
#         df = retriever.retrieve_faiss(
#             datasets["validation"], topk=data_args.top_k_retrieval
#         )
#     else:
#         df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

#     # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
#     if training_args.do_predict:
#         f = Features(
#             {
#                 "context": Value(dtype="string", id=None),
#                 "id": Value(dtype="string", id=None),
#                 "question": Value(dtype="string", id=None),
#             }
#         )

#     # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
#     elif training_args.do_eval:
#         f = Features(
#             {
#                 "answers": Sequence(
#                     feature={
#                         "text": Value(dtype="string", id=None),
#                         "answer_start": Value(dtype="int32", id=None),
#                     },
#                     length=-1,
#                     id=None,
#                 ),
#                 "context": Value(dtype="string", id=None),
#                 "id": Value(dtype="string", id=None),
#                 "question": Value(dtype="string", id=None),
#             }
#         )
#     datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
#     return datasets


def run_elasticsearch(training_args, data_args, datasets):
    # elastic setting & load index
    retriever = ElasticSearchRetrieval(data_args)

    df, scores = retriever.retrieve(datasets['validation'])

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "review": Value(dtype="string", id=None),
                "menu": Value(dtype="string", id=None),
                "resturant": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "review": Value(dtype="string", id=None),
                "label": Value(dtype="string", id=None),
                "menu": Value(dtype="string", id=None),
                "resturant": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets, scores


def cos_sim(A, B):
    '''
    A,B의 cosine similarity
    :return: A,B의 cosine similarity
    '''
    return dot(A, B) / (norm(A) * norm(B))
