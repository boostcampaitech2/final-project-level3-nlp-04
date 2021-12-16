import os
import json
import time
from contextlib import contextmanager
from core.sql_helper import SqlHelper
import config as c

import faiss
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from tqdm.auto import tqdm
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import (
    Dataset,
    load_from_disk,
)


# 학습이 이루어지는 시간을 측정하기 위해 사용되는 함수
from retriever.utils import get_pickle, save_pickle


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:
    def __init__(self, config, tokenizer, p_encoder, q_encoder, num_neg=2, data_path='../data'):
        self.config = config
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.num_neg = num_neg
        self.data_path = data_path

        self.sql_helper = SqlHelper(**c.DB_CONFIG)
        with timer('DB 읽어오는 시간'):
            self.review_df = self.get_review()

        self.contexts = self.review_df.preprocessed_review_context.tolist()
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(self.review_df.index)

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_review(self):
        query = """
            SELECT restaurant_name, preprocessed_review_context FROM preprocessed_review
        """
        review_df = self.sql_helper.get_df(query)
        review_df = review_df.drop_duplicates().reset_index(drop=True)
        return review_df

    def get_dense_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding 을 만들고
            Embedding matrix 와 Embedding 을 pickle 로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle 을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.pkl"
        emb_path = os.path.join(self.data_path, self.config.run_name + '_' + pickle_name)

        if os.path.isfile(emb_path):
            self.p_embedding = get_pickle(emb_path)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            p_seqs = self.tokenizer(
                self.contexts,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False if 'roberta' in self.config.model_name_or_path else True,
            )
            if 'roberta' in self.config.model_name_or_path:
                passage_dataset = TensorDataset(
                    p_seqs['input_ids'],
                    p_seqs['attention_mask'],
                )
            else:
                passage_dataset = TensorDataset(
                    p_seqs['input_ids'],
                    p_seqs['attention_mask'],
                    p_seqs['token_type_ids'],
                )

            passage_dataloader = DataLoader(passage_dataset, batch_size=self.config.per_device_eval_batch_size)

            self.p_encoder.eval()

            p_embedding_list = []
            passage_iterator = tqdm(passage_dataloader, unit='batch')
            for batch in passage_iterator:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)

                    if 'roberta' in self.config.retrieval_model_name_or_path:
                        p_inputs = {'input_ids': batch[0],
                                    'attention_mask': batch[1]}
                    else:
                        p_inputs = {'input_ids': batch[0],
                                    'attention_mask': batch[1],
                                    'token_type_ids': batch[2]}

                    p_outputs = self.p_encoder(**p_inputs)
                    p_embedding_list.append(p_outputs.detach().cpu().numpy())

            self.p_embedding = np.vstack(p_embedding_list)
            print(self.p_embedding.shape)
            save_pickle(emb_path, self.p_embedding)
            print("Embedding pickle saved.")

    def get_dataloader(self):
        '''train, validation, test의 dataloader와 dataset를 반환하는 함수'''
        datasets = load_from_disk(self.data_args.dataset_name)
        print(datasets)

        train_dataset = datasets['train']
        eval_dataset = datasets['validation']

        train_q_seqs = self.tokenizer(
            train_dataset['question'], padding='max_length', truncation=True, return_tensors='pt',
            return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True)
        train_p_seqs = self.tokenizer(
            train_dataset['context'], padding='max_length', truncation=True, return_tensors='pt',
            return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True)
        eval_q_seqs = self.tokenizer(
            eval_dataset['question'], padding='max_length', truncation=True, return_tensors='pt',
            return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True)
        eval_p_seqs = self.tokenizer(
            eval_dataset['context'], padding='max_length', truncation=True, return_tensors='pt',
            return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True)

        if 'roberta' in self.model_args.retrieval_model_name_or_path:
            train_dataset = TensorDataset(train_p_seqs['input_ids'], train_p_seqs['attention_mask'],
                                          train_q_seqs['input_ids'], train_q_seqs['attention_mask'])
            eval_dataset = TensorDataset(eval_p_seqs['input_ids'], eval_p_seqs['attention_mask'],
                                         eval_q_seqs['input_ids'], eval_q_seqs['attention_mask'])
        else:
            train_dataset = TensorDataset(
                train_p_seqs['input_ids'], train_p_seqs['attention_mask'], train_p_seqs['token_type_ids'],
                train_q_seqs['input_ids'], train_q_seqs['attention_mask'], train_q_seqs['token_type_ids'])
            eval_dataset = TensorDataset(
                eval_p_seqs['input_ids'], eval_p_seqs['attention_mask'], eval_p_seqs['token_type_ids'],
                eval_q_seqs['input_ids'], eval_q_seqs['attention_mask'], eval_q_seqs['token_type_ids'])

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=self.training_args.per_device_retrieval_train_batch_size)
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=self.training_args.per_device_retrieval_eval_batch_size)

        return train_dataloader, eval_dataloader

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
            self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i + 1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset["question"], k=topk)

            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        self.q_encoder.eval()

        with timer("query encoding"):
            q_seq = self.tokenizer(
                [query],
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True,
            ).to(self.training_args.device)
            query_vec = self.q_encoder(**q_seq).detach().to('cpu').numpy()  # (num_query=1, emb_dim)

        with timer("query ex search"):
            result = np.matmul(query_vec, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
            self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        self.q_encoder.eval()

        q_seqs = self.tokenizer(
            queries,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=False if 'roberta' in self.model_args.retrieval_model_name_or_path else True,
        ).to(self.training_args.device)

        if 'roberta' in self.model_args.retrieval_model_name_or_path:
            question_dataset = TensorDataset(
                q_seqs['input_ids'],
                q_seqs['attention_mask'],
            )
        else:
            question_dataset = TensorDataset(
                q_seqs['input_ids'],
                q_seqs['attention_mask'],
                q_seqs['token_type_ids'],
            )

        question_dataloader = DataLoader(question_dataset, batch_size=self.training_args.per_device_eval_batch_size)

        q_embedding_list = []
        question_iterator = tqdm(question_dataloader, unit='batch')
        for batch in question_iterator:
            with torch.no_grad():
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                if 'roberta' in self.model_args.retrieval_model_name_or_path:
                    q_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1]}
                else:
                    q_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'token_type_ids': batch[2]}

                q_outputs = self.q_encoder(**q_inputs)
                q_embedding_list.append(q_outputs.detach().cpu().numpy())

        query_embedding = np.vstack(q_embedding_list)

        result = np.matmul(query_embedding, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        # 진명훈님 공유해주신 코드 적용
        doc_scores = np.partition(result, -k)[:, -k:][:, ::-1]
        ind = np.argsort(doc_scores, axis=-1)[:, ::-1]
        doc_scores = np.sort(doc_scores, axis=-1)[:, ::-1].tolist()
        doc_indices = np.argpartition(result, -k)[:, -k:][:, ::-1]
        r, c = ind.shape
        ind = ind + np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
        doc_indices = doc_indices.ravel()[ind].reshape(r, c).tolist()
        return doc_scores, doc_indices

    def retrieve_faiss(
            self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                    tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
            self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
            self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
                np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


