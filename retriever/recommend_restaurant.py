import os

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

        self.es_retriever = ElasticSearchRetrieval(config)
        self.ds_retriever = DenseRetrieval(config, tokenizer, p_encoder, q_encoder, data_path)
        self.ds_retriever.get_dense_embedding()

    def get_restaurant(self, query):
        es_df = self.es_retriever.retrieve(query, topk=2000)
        ds_df = self.ds_retriever.retrieve(query, topk=2000)

        result = {}
        for i, q in enumerate(query):
            es = es_df.iloc[i]
            ds = ds_df.iloc[i]

            es_restaurant_name = es.restaurant_name
            ds_restaurant_name = ds.restaurant_name
            es_score = es.score
            ds_score = ds.score

            es_ds_df = pd.DataFrame()
            es_ds_df['restaurant_name'] = es_restaurant_name + ds_restaurant_name
            es_ds_df['score'] = es_score + ds_score

            top_10_restaurant = es_ds_df.restaurant_name.value_counts().index[:10].tolist()

            result[q] = top_10_restaurant
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

    query = ['육즙이 많은']  # 리스트로 받아서 들어가야 해요! 왜? bluk 로 진행하는걸로만 짜서요
    top_10_restaurant_for_query = recommend_restaurant.get_restaurant(query)

    print(top_10_restaurant_for_query)



