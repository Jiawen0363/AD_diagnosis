# Usage: deepspeed finetune.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import pathlib
import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import (
    TrainingArguments,
    HfArgumentParser, 
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    Trainer, 
    BitsAndBytesConfig,
)
from transformers.integrations import deepspeed

from data_utils import rank0_print, make_supervised_data_module


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="yahma/llama-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    cache_path: str = field(
        default="caches", metadata={"help": "Path to cache all data."}
    )
    padding_side: str = field(
        default="left",  # Set to left for GPT-like models, allowing for batched inference
        metadata={"help": "Padding side (right or left) for padding to max_length"}
    )
    truncation_side: str = field(
        default="left",
        metadata={"help": "Truncation_side (right or left) for input sequences"}
    )
    cutoff_len: int = field(
        default=800,
        metadata={"help": "Sequences will be possibly truncated if longer than cutoff_len."}
    )
    template_name: str = field(
        default="vicuna_v1.1",
        metadata={"help": "Template name for the conversation."}
    )
    mask_dtype: str = field(
        default="bool",
        metadata={"help": "Data type of attention masks."}
    )
    num_proc: int = field(
        default=8, metadata={"help": "Number of processes for data loading."}
    )


@dataclass
class FinetuningArguments(TrainingArguments):
    # The default arguments are defined in transformers.TrainingArguments
    # The following arguments are specified for reference
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model checkpoints will be written."}
    )
    load_in_8bit: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    num_train_epochs: float = 3.0
    evaluation_strategy: str = "no"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch"
    fp16: bool = True
    bf16: bool = False
    lr_scheduler_type: str = "linear"
    logging_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    local_rank: int = field(default=0, metadata={"help": "Local rank of the process."})
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def train():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, FinetuningArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if lora_args.q_lora:
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )
    
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    if "chatglm" in model_args.model_name_or_path.lower():
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=compute_dtype,
            quantization_config=BitsAndBytesConfig(  
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if lora_args.q_lora
            else None,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=["query_key_value"],
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map=device_map,
            torch_dtype=compute_dtype,
            quantization_config=BitsAndBytesConfig(  
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if lora_args.q_lora
            else None,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )


    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    model = get_peft_model(model, lora_config)
    
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=data_args.padding_side,
        truncation_side=data_args.truncation_side,
        trust_remote_code=True,
        use_fast=False,
        legacy=False,
    )
    # Set tokenizer's padding token and padding side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank0_print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    rank0_print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    rank0_print(f"Tokenizer unk_token_id: {tokenizer.unk_token_id}")
    rank0_print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")
    rank0_print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == "__main__":
    train()
