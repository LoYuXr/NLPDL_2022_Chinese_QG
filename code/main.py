"""
Fine-tuning mT5 model for chinese QG 
dataset: offered dataset
Author: Luo Yuxuan
"""
import copy
import logging
import os
import sys
from typing import Optional

import datasets
from datasets import load_metric
from Class import *
from load_data import *
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import evaluate
import wandb
from mymodels import AdvContrastiveQG

from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        # wriite a json script to record arguments and pass the path!!
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    ## load checkpoint if there is any saved module
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    #Set Wandb
    wandb.init(project=data_args.project_name, entity = data_args.wandb_name, name=data_args.task_name)
    
    set_seed(training_args.seed)

    raw_datasets = my_load_dataset(data_args.train_file, data_args.evaluation_file, data_args.test_file)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained( model_args.tokenizer_name,
        use_fast=model_args.use_fast_tokenizer,)
    
    tokenizer.add_tokens("[HL]")
    
    model = AdvContrastiveQG(
        config = config,
        model_args = model_args,
        data_args = data_args,
        training_args = data_args,
        device = training_args.device
    )
    
    logger.info(tokenizer.all_special_tokens)
    logger.info(tokenizer.all_special_ids)

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else "问题生成: "


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"].shuffle(seed=training_args.seed)
    elif training_args.do_eval:
        if "evaluation" not in  raw_datasets:
            raise ValueError("--do_train requires a evaluation dataset")
        eval_dataset = raw_datasets["evaluation"]
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
    
    # question generation with max target length! (no more than the amount of the words)
    max_target_length = data_args.max_target_length

    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    
    def preprocess_function(examples):
        '''
        examples: train_dataset / eval_dataset / predict_dataset
        type: Dataset
        elements: 
            Context
            Question
            Answer

        generate:
            inputs
            targets
        '''
        inputs = []
        cont_ = examples["context"]
        ques_ = examples["question"]
        ans_ = examples["answer"]
        Len = len(cont_)
        
        for i in range(Len):
            poss = ans_[i].find('#')
            if poss != -1:
                ans_list = ans_[i].split('#')
                assert type(ans_list) is list
                start_list = []
                for answers in ans_list:
                    start_list.append(cont_[i].index(answers))
                res_inputs = prefix
                for k in range(len(start_list)):
                    res_inputs += cont_[i][:start_list[k]]+'[HL]'+ans_list[k]+'[HL]'
                inputs.append(res_inputs+cont_[i][start_list[-1]+len(ans_list[-1]):])
                #print(i,inputs[-1], sep = ' ')
            else:
                ans_start = cont_[i].index(ans_[i])
                inputs.append(prefix+cont_[i][:ans_start]+'[HL]'+ans_[i]+'[HL]'+cont_[i][ans_start+len(ans_[i]):])

        targets = ques_

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        ## 注意， T5 模型只需要 2 个输入: Input_ids (编码后的输入序列的 Input_ids) 和 labels (编码后的目标序列的 Input_ids)
        ## decoder_input_ids, decoder_attention_mask 应在模型内部会根据 labels 右移得到，无需传入
        return model_inputs


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "evaluation" not in raw_datasets:
            raise ValueError("--do_eval requires a evaluation dataset")
        eval_dataset = raw_datasets["evaluation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="evaluation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on evaluation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    # ## Evaluation metrics
    bleu = evaluate.load('sacrebleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if isinstance(preds, tuple):
            preds = preds[0]
        print(preds[:25])
        print(labels[:25])
        decoded_preds = preds
        decoded_labels = labels

        result_1 = bleu.compute(predictions = decoded_preds, references = decoded_labels, tokenize='zh')['score']
        result_2 = rouge.compute(predictions = decoded_preds, references = decoded_labels, tokenizer=lambda x: tokenizer.tokenize(x))
        result_3 = bertscore.compute(predictions = decoded_preds, references = decoded_labels,lang="zh")
            
        dict_ = {'rouge1': result_2['rouge1']*100, 'rouge2': result_2['rouge2']*100, 'rougeL': result_2['rougeL']*100}
        dict_["bleu"] = result_1
        dict_["bert score"] = np.mean(result_3["f1"])*100
        return dict_


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        predict_results = trainer.predict(
            predict_dataset,  max_length=max_length, num_beams=num_beams, metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True)
                predictions = [pred.strip() for pred in predictions]
                
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.train_file is not None:
        kwargs["dataset_tags"] = data_args.train_file
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.train_file} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.train_file

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()