import os
import time
from contextlib import contextmanager
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

@contextmanager
def timer(name):
    """시간을 작성하는데 필요한 함수입니다."""
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticSearchRetrieval:
    """ElasticSearchRetrieval을 하는데 필요한 함수 입니다."""
    def __init__(self, data_args):
        self.data_args = data_args
        self.index_name = data_args.elastic_index_name
        self.k = data_args.top_k_retrieval

        self.es = Elasticsearch() #Elasticsearch 작동
        print(type(self.index_name))
        print(self.index_name)
        print("*"*50)
        print(type(self.es.indices))
        print(self.es.indices)
        print("*"*50)

        if not self.es.indices.exists(self.index_name): #wiki-index가 es.indices와 맞지 않을 때 맞춰주기 위한 조건문
            self.qa_records, self.reviews = self.set_datas()
            self.set_index()
            self.populate_index(es_obj=self.es,
                                index_name=self.index_name,
                                evidence_corpus=self.reviews)

    def set_datas(self):
        """elastic search 에 저장하는 데이터 세팅과정"""
        df = pd.read_csv("./elastic_image.csv")
        qa_records = []
        reviews = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            row = row.to_dict()
            qa_records.append(row)
            reviews.append(row['preprocessed_review_context'])
        reviews = list(dict.fromkeys(reviews))  # 리뷰 중복 제거
        reviews = [{'document_text': reviews[i]} for i in range(len(reviews))]

        return qa_records, reviews

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

    def retrieve(self, query_or_dataset, topk=None):
        if topk is not None:
            self.k = topk

        total = []
        scores = []

        with timer("query exhaustive search"):
            pbar = tqdm(query_or_dataset, desc='elastic search - question: ')
            for idx, example in enumerate(pbar):
                # top-k 만큼 context 검색
                context_list, score_list = self.elastic_retrieval(example['preprocessed_review_context'])

                tmp = {
                    'restaurant': example['restaurant'],
                    'menu': example['menu'],
                    'preprocessed_review_context': example['preprocessed_review_context'],
                    'context': ' '.join(context_list)
                }

                total.append(tmp)
                scores.append(sum(score_list))

            df = pd.DataFrame(total)

        return df, scores

    def elastic_retrieval(self, question_text, topk=None):
        result = self.search_es(question_text, topk)
        answer = []
        # 매칭된 context만 list형태로 만든다.
        context_list = [hit['_source']['document_text'] for hit in result['hits']['hits']]
        score_list = [hit['_score'] for hit in result['hits']['hits']]
        id_list = [int(hit['_id']) for hit in result['hits']['hits']]  # 추가하였습니다
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        df = pd.read_csv(os.path.join(data_path, 'elastic_image.csv'))
        for review in context_list:
            answer.append(df[df['preprocessed_review_context']==review]['image_url'].values[0])
        return context_list, score_list, id_list, answer

    def search_es(self, question_text, topk):
        query = {
            'query': {
                'match': {
                    'document_text': question_text
                }
            }
        }
        result = self.es.search(index=self.index_name, body=query, size=self.k if topk is None else topk)
        return result