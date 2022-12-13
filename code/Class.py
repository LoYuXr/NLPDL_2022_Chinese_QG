from dataclasses import dataclass, field
from typing import Optional

'''
model arguments and datatraining arguments

'''

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default = "./qg_outputs/checkpoint-1000",
        metadata = {"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default = "google/mt5-base", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default = "google/mt5-base", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    cache_dir: Optional[str] = field(
        default = None,
        metadata = {"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default = True,
        metadata = {"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default = "main",
        metadata = {"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default = None,
        metadata = {
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    tau: float = field(
        default = 0.1,
        metadata = {
            "help": "Normalizing factor to compute contrastive loss."
        }
    )
    pos_eps: float = field(
        default = 3.0,
        metadata = {
            "help": "pos_eps"
        }
    )
    neg_eps: float = field(
         default = 1.0,
        metadata = {
            "help": "neg_eps"
        }
    )
    adv: bool = field(
        default = False,
        metadata = {
            "help": "whether conduct contrastive or not."
        }
    )
    hidden_size: int = field(
        default = 768,
        metadata = {
            "help": "hidden layer between encoder and decoder."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    project_name: Optional[str] = field(
        default =None, metadata={"help": "wandb init project name"}
    )
    task_name: Optional[str] = field(
        default=None, metadata={"help": "wandb init task name"}
    )
    wandb_name: Optional[str] = field(
        default=None, metadata={"help": "wandb user name"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    dataset_config_name: Optional[str] = field(
        default = None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    #default="../qg_data/squad_zh_train.jsonl,../qg_data/train.jsonl",
    train_file: Optional[str] = field(
        default = None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    evaluation_file: Optional[str] = field(
        default = None,
        metadata = {
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default = None,
        metadata = {
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default = False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default = 1,
        metadata = {"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default = 512,
        metadata = {
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default = 64,
        metadata = {
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default = None,
        metadata = {
            "help": "The maximum total sequence length for evaluation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default = True,
        metadata = {
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default = None,
        metadata = {
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default = True,
        metadata = {
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default = None,
        metadata = {"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.evaluation_file is None:
            raise ValueError("Need either a dataset name or a training/evaluation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json","jsonl"], "`train_file` should be a csv or a json or a jsonl file."
            if self.evaluation_file is not None:
                extension = self.evaluation_file.split(".")[-1]
                assert extension in ["csv", "json","jsonl"], "`evaluation_file` should be a csv or a json or a jsonlfile."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
