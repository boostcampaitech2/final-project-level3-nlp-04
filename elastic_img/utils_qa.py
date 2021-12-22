from transformers import HfArgumentParser
from elastic_img.arguments import DataTrainingArguments


def get_args():
    '''
    훈련 시 입력한 각종 Argument를 반환하는 함수
    '''
    parser = HfArgumentParser(
        DataTrainingArguments
    )
    data_args = parser.parse_args_into_dataclasses()
    print(type(data_args[0]))
    print(data_args[0])
    return data_args[0]
