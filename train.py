import argparse
import os
import logging
import sys
import math

import datasets
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
logger = logging.getLogger(__name__)

import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

#set_global_logging_level()

def main(args):
    
    if args.seed is not None:
        set_seed(args.seed)
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        gradient_checkpointing = args.gradient_checkpointing,
        save_steps = args.save_steps,
        num_train_epochs = args.epochs,
        fp16 = True,
        fp16_full_eval = True,
        log_level = "warning"
    )
    
    ## -- Set up Logger
    if args.debug:
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
    
    ## -- Load dataset
    
    # Dataset needs to be a csv file with each row (column name text/first column) being the entries of text
    
    data_files = {}
    data_files["train"] = args.train
    if args.val is not None:
      data_files["validation"] = args.val
    else:
      args.num_eval_samples = None
    
    data_type = "text" if args.train[-3:] == "txt" else "csv"
    
    raw_datasets = load_dataset(data_type, data_files=data_files, use_auth_token=False, keep_linebreaks=True)
    
    print(f"Loaded training dataset {args.train}, num_rows = {raw_datasets['train'].num_rows}")
    if args.val is not None:
      print(f"Loaded validation dataset {args.val}, num_rows = {raw_datasets['train'].num_rows}")
    
    
    # -- Load Model
    
    tokenizer_kwargs = {
        "use_fast": args.use_fast_tokenizer,
        "padding": "max_length",
    }
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    
    model.resize_token_embeddings(len(tokenizer))
    
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    ## -- Tokenize
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        return output
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    
    ## -- Group texts together
    
    block_size = tokenizer.model_max_length
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    
    train_dataset = lm_datasets["train"]
    if args.val is not None:
      eval_dataset = lm_datasets["validation"]
    else:
      eval_dataset = None

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    
    ## -- Detect Checkpoint
    
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir True to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir True` to train from scratch."
            )
    
    ## -- Train
    
    # Initialize Trainer
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset, 
        eval_dataset  = eval_dataset,
        tokenizer     = tokenizer,
        data_collator = default_data_collator,
        
    )
    
    # Train
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate 
    if args.val is not None:
      metrics = trainer.evaluate()

      max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
      metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
      try:
          perplexity = math.exp(metrics["eval_loss"])
      except OverflowError:
          perplexity = float("inf")
      metrics["perplexity"] = perplexity

      trainer.log_metrics("eval", metrics)
      trainer.save_metrics("eval", metrics)
    
    print(f"Training completed! The model has been saved to the {args.output_dir} directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--seed", type=int, default = 1)
    parser.add_argument("--debug", type=bool, default=False)

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default = "distilgpt2")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--use_fast_tokenizer", type=bool, default = False)
    
    # Data arguments
    parser.add_argument("--train", type=str, default = "train.csv")
    parser.add_argument("--val", type=str, default = None)
    parser.add_argument("--max_train_samples", type=int, default = None)
    parser.add_argument("--max_eval_samples", type=int, default = None)

    # Fitting arguments
    parser.add_argument("--preprocessing_num_workers", type=int, default = torch.cuda.device_count())
    parser.add_argument("--resume_from_checkpoint", type=bool, default = True)
    parser.add_argument("--overwrite_output_dir", type=bool, default = False)
    parser.add_argument("--gradient_checkpointing", type=bool, default = True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    
    # Other arguments
    parser.add_argument("--save_steps", type = int, default = 1000)
    
    args = parser.parse_args()
    
    print("Devices:")
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print("\n\n")
    
    main(args)