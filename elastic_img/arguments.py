from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from transformers import TrainingArguments as OriginTrainingArguments#, IntervalStrategy


@dataclass
class TrainingArguments(OriginTrainingArguments):
    output_dir: str = field(
        default='../output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    retrieval_output_dir: str = field(
        default='../retrieval_output',
        metadata={"help": "The output directory where the retrieval model will be written."},
    )
    with_inference: str = field(
        default=True,
        metadata={"help": "do train then inference"},
    )
    project_name: str = field(
        # PR 하실때는 None 으로 바꿔서 올려주세요! 얘의 목적은 wandb project name 설정을 위함입니다.
        default="dpr_retrieval_test",
        metadata={"help": "wandb project name"},
    )
    run_name: Optional[str] = field(
        default='roberta-small_elastic_sequential_bs16',
        metadata={"help": "wandb run name"},
    )
    retrieval_run_name: Optional[str] = field(
        default='roberta-small_elastic_bs16',  # 이게 제일 성능이 좋았음
        metadata={"help": "retrieval encoder model folder name"},
    )
    # evaluation_strategy: IntervalStrategy = field(
    #     default='steps',
    #     metadata={"help": "The evaluation strategy to use."},
    # )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    per_device_retrieval_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for retrieval_evaluation."}
    )
    per_device_retrieval_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for retrieval_evaluation."}
    )
    num_train_epochs: float = field(default=10.0, metadata={"help": "Total number of training epochs to perform."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    retrieval_learning_rate: float = field(default=1e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default='exact_match', metadata={"help": "The metric to use to compare two different models."}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    )
    save_total_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    eval_steps: int = field(default=100, metadata={"help": "Run an evaluation every X steps."})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    early_stopping_patience: int = field(default=7, metadata={"help": "early_stopping_patience."})
    fold: bool = field(default=False, metadata={"help": "SET 5-Fold or not"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    finetuned_mrc_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    additional_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attach additional layer to end of model"
        },
    )
    retrieval_model_name_or_path: str = field(
        default="klue/roberta-small",  # 이게 성능이 제일 좋았음
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    use_trained_model: bool = field(
        default=False,
        metadata={
            "help": "whether using fine-tuned retrieval model"
        },
    )
    retrieval_type: str = field(
        default='sequential',
        metadata={"help": "Choose run passage retrieval using sparse or dense or both embedding"
                          "ex. sparse, dense, both(예정 BM25 + dense), elastic"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=30,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    elastic_index_name: Optional[str] = field(
        default="wiki-index",
        metadata={
            "help": "Elastic search index name[wiki-index]"}
    )