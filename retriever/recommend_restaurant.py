import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from retriever.dense_retrieval import DenseRetrieval
from retriever.elastic_search import ElasticSearchRetrieval
from retriever.utils import Config, get_encoders


class RecommendRestaurant:
    def __init__(self, config, tokenizer, p_encoder, q_encoder, data_path):
        self.config = config
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.data_path = data_path

        self.es_retriever = ElasticSearchRetrieval(config, data_path)
        self.ds_retriever = DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path)
        self.ds_retriever.get_dense_embedding()

    def get_restaurant(self, query):
        es_df = self.es_retriever.retrieve(query, topk=2000)
        ds_df = self.ds_retriever.retrieve(query, topk=2000)

        es_restaurant_name = es_df.restaurant_name.iloc[0]
        ds_restaurant_name = ds_df.restaurant_name.iloc[0]
        es_score = es_df.score.iloc[0]
        ds_score = ds_df.score.iloc[0]

        es_ds_df = pd.DataFrame()
        es_ds_df['restaurant_name'] = es_restaurant_name + ds_restaurant_name
        es_ds_df['score'] = es_score + ds_score

        top_10_restaurant = es_ds_df.restaurant_name.value_counts().index[:10].tolist()
        top_10_cnt = es_ds_df.restaurant_name.value_counts().head(10).tolist()

        result = {'top_10_restaurant': top_10_restaurant,
                  'top_10_cnt': top_10_cnt}
        return result


if __name__ == '__main__':
    retriever_path = os.path.dirname(os.path.abspath(__file__))  # retriever folder path
    encoder_path = os.path.join(retriever_path, 'output')
    data_path = os.path.join(os.path.dirname(retriever_path), 'data')
    configs_path = os.path.join(retriever_path, 'configs')
    config = Config().get_config(os.path.join(configs_path, 'klue_bert_base_model.yaml'))

    tokenizer, p_encoder, q_encoder = get_encoders(config)
    p_encoder.load_state_dict(torch.load(os.path.join(encoder_path, 'p_encoder', f'{config.run_name}.pt')))
    q_encoder.load_state_dict(torch.load(os.path.join(encoder_path, 'q_encoder', f'{config.run_name}.pt')))

    recommend_restaurant = RecommendRestaurant(config, tokenizer, p_encoder, q_encoder, data_path)

    while True:
        query = input('keyword 를 입력해주세요\n')
        if query == 'exit':
            print('종료합니다!')
            break
        else:
            query = [query]  # 리스트로 바꿔서 get_restaurant() 함수에 들어가야함
            top_10_info = recommend_restaurant.get_restaurant(query)

            print(top_10_info)

