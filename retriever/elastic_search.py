import os
import json
import time
from contextlib import contextmanager

import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

from datasets import load_from_disk, load_dataset, concatenate_datasets


@contextmanager
def timer(name):
    """시간을 작성하는데 필요한 함수입니다."""
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticSearchRetrieval:
    """ElasticSearchRetrieval을 하는데 필요한 함수 입니다."""
    def __init__(self, config):
        self.config = config
        self.index_name = config.elastic_index_name
        self.k = config.top_k_retrieval

        self.es = Elasticsearch() #Elasticsearch 작동

        if not self.es.indices.exists(self.index_name): #wiki-index가 es.indices와 맞지 않을 때 맞춰주기 위한 조건문
            self.qa_records, self.wiki_articles = self.set_datas()
            self.set_index()
            self.populate_index(es_obj=self.es,
                                index_name=self.index_name,
                                evidence_corpus=self.wiki_articles)

    def set_datas(self):
        """elastic search 에 저장하는 데이터 세팅과정"""

        org_dataset = load_from_disk(self.config.dataset_path)
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )


        wiki_contexts = list(dict.fromkeys([full_ds[v]['preprocessed_review_context'] for v in range(len(train_file))]))
        print(wiki_contexts)

        qa_records = [{'resturant': train_file[i]['restaurant'],
                       'menu': train_file[i]['menu'],
                       'review': train_file[i]['review'],
                       'annotator': train_file[i]['annotator'],
                       'label' : train_file[i]['label']}
                      for i in range(len(train_file))]
        wiki_articles = [{'document_text': wiki_contexts[i]} for i in range(len(wiki_contexts))]

        return qa_records, wiki_articles
        # return wiki_articles

    def set_index(self):
        """index 생성 과정"""
        index_config = {
            'settings': {
                'analysis': {
                    'analyzer': {
                        'nori_analyzer': {
                            'type': 'custom',
                            'tokenizer': 'nori_tokenizer',
                            'decompound_mode': 'mixed',
                            'filter': ['shingle'],
                        }
                    }
                }
            },
            'mappings': {
                'dynamic': 'strict',
                'properties': {
                    'document_text': {
                        'type': 'text',
                        'analyzer': 'nori_analyzer',
                    }
                }
            }
        }

        print('elastic search ping:', self.es.ping())
        print(self.es.indices.create(index=self.data_args.elastic_index_name, body=index_config, ignore=400))

    def populate_index(self, es_obj, index_name, evidence_corpus):
        """
        생성된 elastic search 의 index_name 에 context 를 채우는 과정
        populate : 채우다
        """

        for i, rec in enumerate(tqdm(evidence_corpus)):
            try:
                es_obj.index(index=index_name, id=i, document=rec)
            except:
                print(f'Unable to load document {i}.')

        n_records = es_obj.count(index=index_name)['count']
        print(f'Succesfully loaded {n_records} into {index_name}')

    def retrieve(self, query, dataset, topk=None):
        """ retrieve 과정"""
        if topk is not None:
            self.k = topk

        total = []
        scores = []

        with timer("query exhaustive search"):
            pbar = tqdm(dataset, desc='elastic search - query: ')
            for idx, example in enumerate(pbar):
                # top-k 만큼 context 검색
                context_list, score_list = self.elastic_retrieval(query)
                concat_context = []
                for i in range(len(context_list)):
                    concat_context.append(context_list[i])
                tmp = {
                    'restaurant': example['restaurant'],
                    'menu': example['menu'],
                    'review': example['review'],
                    'food': example['food'],
                    'delvice': example['delvice'],
                    'context': ' '.join(concat_context)
                }

                # if "context" in example.keys() and "answers" in example.keys():
                #     # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                #     tmp["original_context"] = example["context"]
                #     tmp["answers"] = example["answers"]

                total.append(tmp)
                scores.append(sum(score_list))

            df = pd.DataFrame(total)

        return df, scores

    def elastic_retrieval(self, query):#_source 원문데이터만 나옴
        result = self.search_es(query)
        # 매칭된 context만 list형태로 만든다.
        context_list = [hit['_source']['document_text'] for hit in result['hits']['hits']]
        score_list = [hit['_score'] for hit in result['hits']['hits']]
        return context_list, score_list

    def search_es(self, query): # match 쿼리로 document_text필드에서 query 검색
        query_1 = {
            'query': {
                'match': {
                    'document_text': query
                }
            }
        }

        result = self.es.search(index=self.index_name, body=query_1, size=self.k)
        return result
