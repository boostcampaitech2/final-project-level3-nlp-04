class DataProcessor():
    """ data 전처리를 위한 class """

    def __init__(self, tokenizer, model_args, data_args):
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args

        # Padding에 대한 옵션 설정. (question|context) or (context|question)
        self.pad_on_right = tokenizer.padding_side == "right"

    # train의 전처리 진행
    def prepare_train_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행
        # stride 이용하여 overflow 유지
        # 각 example들은 truncation option 설정으로 이전의 context와 조금씩 겹침
        tokenized_examples = self.tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if 'roberta' in self.model_args.model_name_or_path else True,
            padding="max_length"
        )

        # truncate 진행 과정에서 쉽게 dataset을 찾을 수 있도록 mapping 가능토록 함
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 character 단위 position을 찾기 위해 offset mapping 사용.
        # start_positions와 end_positions 찾기에 도움
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # dataset에 "start position", "end position" label 붙이기
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # offset_mapping 내의 원소들을 기준으로 진행
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]  # input_ids 초기화
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # 문장과 질문을 구분 위한 sequence id 설정
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            answers = examples['answers'][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 end token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인. 벗어났을 경우 cls token의 위치를 정답으로 설정
                if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동
                    while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    # answer가 마지막 단어인 경우 last offset을 따라가도록 함
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                        
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # valid의 전처리 진행
    def prepare_validation_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행
        # stride 이용하여 overflow 유지
        # 각 example들은 truncation option 설정으로 이전의 context와 조금씩 겹침
        tokenized_examples = self.tokenizer(
            examples['question'],
            examples['context'],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if 'roberta' in self.model_args.tokenizer_name else True,
            padding="max_length"
        )

        # truncate 진행 과정에서 쉽게 dataset을 찾을 수 있도록 mapping 가능토록 함
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환
        # corresponding example_id를 유지하고 offset mappings을 저장
        tokenized_examples["example_id"] = []

        # input_ids의 길이를 기준으로 진행
        for i in range(len(tokenized_examples["input_ids"])):
            # 문장과 질문을 구분 위한 sequence id 설정
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # offset_mapping을 None으로 설정함으로써 token의 position이 context의 일부인지 판별 가능
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    # train의 전처리된 dataset return
    def train_tokenizer(self, train_dataset, column_names):
        train_dataset = train_dataset.map(
            self.prepare_train_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        return train_dataset

    # valid의 전처리된 dataset return
    def valid_tokenizer(self, valid_dataset, column_names):
        valid_dataset = valid_dataset.map(
            self.prepare_validation_features,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        return valid_dataset
