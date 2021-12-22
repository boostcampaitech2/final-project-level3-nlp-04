import os
import sys
from functools import lru_cache

from retriever.utils import save_pickle, get_pickle

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
from contextlib import contextmanager

import pandas as pd
from elasticsearch import Elasticsearch, helpers
from tqdm.auto import tqdm

import config as c
from core.sql_helper import SqlHelper


@contextmanager
def timer(name):
    """시간을 작성하는데 필요한 함수입니다."""
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticSearchRetrieval:
    """ElasticSearchRetrieval을 하는데 필요한 함수 입니다."""
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.index_name = config.elastic_index_name
        self.k = config.top_k_retrieval

        self.es = Elasticsearch(timeout=300, max_retries=10, retry_on_timeout=True) #Elasticsearch 작동
        # self.es.indices.delete(self.index_name)  # index 가 잘못된 경우 주석을 풀고 돌리세요!

        if not self.es.indices.exists(self.index_name): #wiki-index가 es.indices와 맞지 않을 때 맞춰주기 위한 조건문
            self.articles = self.set_datas()
            self.set_index()
            self.populate_index(es_obj=self.es,
                                index_name=self.index_name,
                                evidence_corpus=self.articles)

    def set_datas(self):
        """elastic search 에 저장하는 데이터 세팅과정"""

        self.sql_helper = SqlHelper(**c.DB_CONFIG)
        with timer('DB 읽어오는 시간'):
            review_df = self.get_review()

        self.contexts = review_df.preprocessed_review_context.tolist()
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(review_df.index)

        articles = [{'restaurant_name': row['restaurant_name'],
                     'subway': row['subway'],
                     'review': row['preprocessed_review_context']}
                    for idx, row in tqdm(review_df.iterrows())]

        return articles

    @lru_cache(maxsize=None)
    def get_review(self):
        pickle_path = os.path.join(self.data_path, 'review_df.pkl')
        if not os.path.exists(pickle_path):
            query = """
                SELECT * 
                FROM preprocessed_review
                WHERE insert_time < '2021-12-18'
                    """
            review_df = self.sql_helper.get_df(query)
            review_df = review_df.drop_duplicates().reset_index(drop=True)

            save_pickle(pickle_path, review_df)
        else:
            review_df = get_pickle(pickle_path)
        return review_df

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
                    'review': {
                        'type': 'text',
                        'analyzer': 'nori_analyzer',
                    },
                    'restaurant_name': {
                        'type': 'text',
                    },
                    'subway': {
                        'type': 'text',
                    },
                    'address': {
                        'type': 'text',
                    },
                }
            }
        }

        print('elastic search ping:', self.es.ping())
        print(self.es.indices.create(index=self.config.elastic_index_name, body=index_config, ignore=400))

    def populate_index(self, es_obj, index_name, evidence_corpus):
        """
        생성된 elastic search 의 index_name 에 context 를 채우는 과정
        populate : 채우다
        """

        document_texts = [
            {'_id': i,
             '_index': self.index_name,
             '_source': {'review': review['review'],
                         'subway': review['subway'],
                         'address': review['address'],
                         'restaurant_name': review['restaurant_name'],
                         }}
            for i, review in enumerate(evidence_corpus)
        ]
        helpers.bulk(self.es, document_texts)

        n_records = es_obj.count(index=index_name)['count']
        print(f'Succesfully loaded {n_records} into {index_name}')

    def retrieve(self, query, dataset=None, topk=None):
        """ retrieve 과정"""
        if topk is not None:
            self.k = topk

        with timer("query exhaustive search"):
            # pbar = tqdm(dataset, desc='elastic search - query: ')
            # for idx, example in enumerate(pbar):
            # top-k 만큼 context 검색
            context_list, restaurant_name, address_list, score_list = self.elastic_retrieval(query)
            concat_context = []
            for i in range(len(context_list)):
                concat_context.append(' '.join(context_list[i]))
            tmp = {
                'restaurant_name': restaurant_name,
                'review': context_list,
                'context': concat_context,
                'address': address_list,
                'score': score_list,
            }

            if dataset is not None:
                tmp['original_context'] = dataset['context']

            df = pd.DataFrame(tmp)

        return df

    def elastic_retrieval(self, query):#_source 원문데이터만 나옴
        response = self.search_es(query)
        # 매칭된 context만 list형태로 만든다.
        context_list = [[hit['_source']['review'] for hit in result['hits']['hits']] for result in response]
        restaurant_list = [[hit['_source']['restaurant_name'] for hit in result['hits']['hits']] for result in response]
        address_list = [[hit['_source']['address'] for hit in result['hits']['hits']] for result in response]
        score_list = [[hit['_score'] for hit in result['hits']['hits']] for result in response]
        return context_list, restaurant_list, address_list, score_list

    def make_query(self, query, topk):
        return {'query': {'match': {'review': query}}, 'size': topk}

    def search_es(self, query): # match 쿼리로 document_text필드에서 query 검색
        body = []
        for i in range(len(query) * 2):
            if i % 2 == 0:
                body.append({'index': self.index_name})
            else:
                body.append(self.make_query(query[i//2], self.k))

        result = self.es.msearch(body=body)['responses']
        return result
