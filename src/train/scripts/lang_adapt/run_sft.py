#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from custom_trainer import CustomSFTTrainer, TrainerWithCustomSampler
from data_functions import create_datasets
from datasets import concatenate_datasets, load_dataset
from peft import (
    IA3Config,
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from sklearn.metrics import accuracy_score
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "irish_llm_sft"


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "pt_peft_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "pt_peft_model")
        kwargs["model"].save_pretrained(peft_model_path, safe_serialization=False)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_peft_model")
        kwargs["model"].save_pretrained(peft_model_path, safe_serialization=False)
        # kwargs["tokenizer"].save_pretrained(peft_model_path)


# class SaveModelCallback(transformers.TrainerCallback):
#     def save_model(self, args, state, kwargs):
#         checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

#         model_path = os.path.join(checkpoint_folder)
#         kwargs["model"].save_pretrained(model_path, safe_serialization=False)
#         # kwargs["tokenizer"].save_pretrained(peft_model_path)

#     def on_save(self, args, state, control, **kwargs):
#         self.save_model(args, state, kwargs)
#         return control


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": ("Load the model in 4-bit quantized mode.")},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    data_cache_dir: Optional[str] = field(
        default="./", metadata={"help": "The datasets processed stored"}
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )


@dataclass
class MyTrainingArguments(SFTConfig):
    method: Optional[str] = field(default="lora")
    trainable: Optional[str] = field(default="q_proj,v_proj")
    feedforward_modules: Optional[str] = field(default="down_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    debug_mode: Optional[bool] = field(default=False)
    peft_path: Optional[str] = field(default=None)
    use_peft: Optional[bool] = field(default=True)
    dataset_text_field: Optional[str] = field(default="text")
    packing: Optional[bool] = field(default=False)
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    use_liger: Optional[bool] = field(default=False)
    save_safetensors: Optional[bool] = field(default=False)
    reasoning_collator: Optional[bool] = field(default=False)
    log_extra_losses: Optional[bool] = field(default=False)
    weight_before_think_loss: Optional[float] = field(default=1.0)
    weight_after_think_loss: Optional[float] = field(default=1.0)


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.report_to = ["wandb"]
    training_args.use_cache = False
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.tokenizer_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"

    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        train_dataset, eval_dataset = create_datasets(tokenizer, data_args)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(train_dataset[0]["text"])
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("training example:")
        logger.info(eval_dataset[0]["text"])

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if model_args.load_in_4bit:
            from transformers import BitsAndBytesConfig

            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                quantization_config=q_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params"
        )

    model.resize_token_embeddings(len(tokenizer))
    if training_args.use_peft:
        if training_args.peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(model, training_args.peft_path)
        else:
            logger.info("Init new peft model")
            target_modules = training_args.trainable.split(",")
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(",")
            lora_rank = training_args.lora_rank
            lora_dropout = training_args.lora_dropout
            lora_alpha = training_args.lora_alpha
            logger.info(f"target_modules: {target_modules}")
            logger.info(f"lora_rank: {lora_rank}")
            if training_args.method == "qlora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules="all-linear",
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    modules_to_save=modules_to_save,
                )
            elif training_args.method == "lora":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=target_modules,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    modules_to_save=modules_to_save,
                )
            elif training_args.method == "ia3":
                peft_config = IA3Config(
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=target_modules,
                    feedforward_modules=training_args.feedforward_modules.split(","),
                    inference_mode=False,
                    modules_to_save=modules_to_save,
                )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    from transformers.loss.loss_utils import ForCausalLMLoss

    def custom_loss_func(outputs, labels, num_items_in_batch):
        # NOTE: for deepseek:
        THINK_TOKEN_ID = 128014
        VOCAB_SIZE = 128256

        before_think_labels = []
        after_think_labels = []
        think_indices = []
        for i in range(labels.shape[0]):
            current_label = labels[i, ...]
            think_indice = int(torch.where(current_label == THINK_TOKEN_ID)[0][0])
            before_think_label = current_label.detach().clone()
            before_think_label[think_indice:] = -100
            before_think_labels.append(before_think_label)

            after_think_label = current_label.detach().clone()
            after_think_label[:think_indice] = -100
            after_think_labels.append(after_think_label)

            think_indices.append(think_indice)
        before_think_labels = torch.stack(before_think_labels)
        after_think_labels = torch.stack(after_think_labels)

        before_think_loss = ForCausalLMLoss(
            logits=outputs.logits, labels=before_think_labels, vocab_size=VOCAB_SIZE
        ).detach()
        after_think_loss = ForCausalLMLoss(
            logits=outputs.logits, labels=after_think_labels, vocab_size=VOCAB_SIZE
        ).detach()

        loss = ForCausalLMLoss(
            logits=outputs.logits, labels=labels, vocab_size=VOCAB_SIZE
        )
        return loss, before_think_loss, after_think_loss

    def custom_loss_func_bs1(outputs, labels, num_items_in_batch):
        # controlling the weights for each loss component

        WEIGHT_BEFORE_THINK_LOSS = training_args.weight_before_think_loss
        WEIGHT_AFTER_THINK_LOSS = training_args.weight_after_think_loss

        # NOTE: for deepseek:
        THINK_TOKEN_ID = 128014
        VOCAB_SIZE = 128256

        before_think_labels = []
        after_think_labels = []
        think_indices = []
        for i in range(labels.shape[0]):
            current_label = labels[i, ...]
            think_indice = int(torch.where(current_label == THINK_TOKEN_ID)[0][0])
            before_think_label = current_label.clone()
            before_think_label[think_indice:] = -100
            before_think_labels.append(before_think_label)

            after_think_label = current_label.clone()
            after_think_label[:think_indice] = -100
            after_think_labels.append(after_think_label)

            think_indices.append(think_indice)
        before_think_labels = torch.stack(before_think_labels)
        after_think_labels = torch.stack(after_think_labels)

        before_think_loss = ForCausalLMLoss(
            logits=outputs.logits, labels=before_think_labels, vocab_size=VOCAB_SIZE
        )
        after_think_loss = ForCausalLMLoss(
            logits=outputs.logits, labels=after_think_labels, vocab_size=VOCAB_SIZE
        )

        # loss = ForCausalLMLoss(logits=outputs.logits, labels=labels, vocab_size=VOCAB_SIZE)
        loss = (
            WEIGHT_BEFORE_THINK_LOSS * before_think_loss * think_indices[0]
            + WEIGHT_AFTER_THINK_LOSS
            * after_think_loss
            * (labels.shape[1] - think_indices[0])
        ) / labels.shape[1]

        return loss, before_think_loss, after_think_loss

    if training_args.log_extra_losses:
        # Initialize our Trainer
        trainer = CustomSFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            peft_config=None if not training_args.use_peft else peft_config,
            data_collator=(
                DataCollatorForCompletionOnlyLM("</think>\n\n", tokenizer=tokenizer)
                if training_args.reasoning_collator
                else None
            ),
            extra_losses=["loss_before_think", "loss_after_think"],
            compute_loss_func=custom_loss_func_bs1,
        )
    else:
        # Initialize our Trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            peft_config=None if not training_args.use_peft else peft_config,
            data_collator=(
                DataCollatorForCompletionOnlyLM("</think>\n\n", tokenizer=tokenizer)
                if training_args.reasoning_collator
                else None
            ),
        )
    if training_args.use_peft:
        trainer.add_callback(SavePeftModelCallback)
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
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
