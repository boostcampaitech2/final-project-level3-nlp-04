import os
import sys

import pandas as pd
from datasets import Features, Value, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
import pickle
import random
import yaml

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from transformers import is_torch_available, AutoConfig, AutoTokenizer

from retriever.model.retrieval_encoder import RetrievalEncoder


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# encoder 모델 불러오는 함수
def get_encoders(config):
    model_config = AutoConfig.from_pretrained(config.config_name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    p_encoder = RetrievalEncoder(config.model_name_or_path, model_config)
    q_encoder = RetrievalEncoder(config.model_name_or_path, model_config)
    return tokenizer, p_encoder, q_encoder


def get_pickle(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result


def save_pickle(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


class Config(object):
    def __init__(self, dict_config=None):
        super().__init__()
        self.set_attribute(dict_config)

    @staticmethod
    def get_config(path: str):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return Config(config)

    def set_attribute(self, dict_config):
        if dict_config is None:
            return

        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                self.__dict__[key] = Config(dict_config[key])
            else:
                self.__dict__[key] = dict_config[key]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]


def make_dataset(config, data_path):
    df_path = os.path.join(data_path, 'retriever_df.csv')
    df = pd.read_csv(df_path)
    df['query'] = df['keyword']
    df['context'] = df['preprocessed_review_context']
    df['id'] = list(map(str, range(len(df))))

    train_df, valid_df = train_test_split(df, test_size=.3, shuffle=True, random_state=config.seed)
    f = Features({'context': Value(dtype='string', id=None),
                  'query': Value(dtype='string', id=None),
                  'restaurant_name': Value(dtype='string', id=None),
                  'id': Value(dtype='string', id=None)})

    datasets = DatasetDict({'train': Dataset.from_pandas(train_df, features=f),
                            'validation': Dataset.from_pandas(valid_df, features=f)})

    datasets.save_to_disk(os.path.join(data_path, 'retrieval_dataset'))
