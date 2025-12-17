import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
import os
import sys

# Add parent directory to Python path (for importing modeling_openpangu_dense)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from huggingface_hub import hf_hub_download
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

# Import Pangu model
# Ensure you are running this script from the openPangu directory
try:
    from modeling_openpangu_dense import PanguEmbeddedForCausalLM
except ImportError:
    raise ImportError("Could not import PanguEmbeddedForCausalLM. Make sure you are in the openPangu directory.")

# Try to import fastchat
try:
    from fastchat.conversation import SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
except ImportError:
    print("Warning: fastchat not installed. Some features might not work.")

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# --- Medusa Model Wrapper for Training ---

class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        version="2",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.version = version
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class MedusaModelWrapper(nn.Module):
    """
    A wrapper class for training Medusa heads on top of a base model.
    This structure aligns with the inference model in medusa_model.py (with independent Linear heads).
    """
    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        
        # Initialize Medusa Heads with ResBlocks and a Linear layer
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        with torch.no_grad():
            # Pass input through the base model
            # PanguEmbeddedForCausalLM.model is PanguEmbeddedModel
            # We use the base model to get hidden states
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        
        # Clone the output hidden states to avoid modifying them in place if needed
        hidden_states = outputs[0].clone()
        medusa_logits = []
        
        # Pass hidden states through each Medusa head
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

# --- Trainer Parts ---

class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(model, "module"):
            medusa = model.module.medusa
        else:
            medusa = model.medusa

        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            for k in range(1, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
        self.log(log)
        return (loss, logits) if return_outputs else loss

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=".") # Default to current dir
    load_in_4bit: bool = field(default=False, metadata={"help": "Load in 4 bit."})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load in 8 bit."})

@dataclass
class DataArguments:
    data_path: str = field(default="sharegpt_clean.json", metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    medusa_num_heads: int = field(default=1, metadata={"help": "Number of Medusa heads."})
    medusa_num_layers: int = field(default=1, metadata={"help": "Number of layers for each Medusa head."})

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = []
    for i, source in enumerate(sources):
        if isinstance(source, dict) and "conversations" in source:
            raw_conv = source["conversations"]
        else:
            raw_conv = source

        conversation = []
        for turn in raw_conv:
            if "from" in turn and "value" in turn:
                role = turn["from"]
                content = turn["value"]
                if role == "human": role = "user"
                if role == "gpt": role = "assistant"
                conversation.append({"role": role, "content": content})
            else:
                conversation.append(turn)
        conversations.append(conversation)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for conversation in conversations:
        input_ids = [tokenizer.bos_token_id] if tokenizer.add_bos_token else []
        labels = [IGNORE_TOKEN_ID] if tokenizer.add_bos_token else []
        
        # Check if first message is system, if not, add default system prompt
        if not conversation or conversation[0]['role'] != 'system':
             sys_text = "[unused9]系统：[unused10]"
             sys_ids = tokenizer(sys_text, add_special_tokens=False).input_ids
             input_ids.extend(sys_ids)
             labels.extend([IGNORE_TOKEN_ID] * len(sys_ids))

        for turn in conversation:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                text = f"[unused9]用户：{content}[unused10]"
                ids = tokenizer(text, add_special_tokens=False).input_ids
                input_ids.extend(ids)
                labels.extend([IGNORE_TOKEN_ID] * len(ids))
            elif role == 'assistant':
                # 分开处理前缀和内容，只有内容部分作为 label
                prefix_text = "[unused9]助手："
                prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
                content_ids = tokenizer(content, add_special_tokens=False).input_ids
                suffix_text = "[unused10]"
                suffix_ids = tokenizer(suffix_text, add_special_tokens=False).input_ids
                
                # 前缀标记不作为预测目标
                input_ids.extend(prefix_ids)
                labels.extend([IGNORE_TOKEN_ID] * len(prefix_ids))
                
                # 内容部分作为预测目标
                input_ids.extend(content_ids)
                labels.extend(content_ids)
                
                # 后缀标记不作为预测目标
                input_ids.extend(suffix_ids)
                labels.extend([IGNORE_TOKEN_ID] * len(suffix_ids))
            elif role == 'system':
                text = f"[unused9]系统：{content}[unused10]"
                ids = tokenizer(text, add_special_tokens=False).input_ids
                input_ids.extend(ids)
                labels.extend([IGNORE_TOKEN_ID] * len(ids))
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Truncate
        if len(input_ids) > tokenizer.model_max_length:
            input_ids = input_ids[:tokenizer.model_max_length]
            labels = labels[:tokenizer.model_max_length]
            
        # Pad
        padding_length = tokenizer.model_max_length - len(input_ids)
        if padding_length > 0:
            pad_id = tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), IGNORE_TOKEN_ID, dtype=torch.long)])
            
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(input_ids.ne(tokenizer.pad_token_id))

    return dict(
        input_ids=torch.stack(input_ids_list),
        labels=torch.stack(labels_list),
        attention_mask=torch.stack(attention_mask_list),
    )

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret
        return ret

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    dataset_cls = (LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset)
    rank0_print("Loading data...")
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Pangu Model
    print(f"Loading Pangu model from {model_args.model_name_or_path}...")
    model = PanguEmbeddedForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False

    # Add Medusa heads
    medusa_lm_head = MedusaModelWrapper(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
        version="2"
    )
    medusa_config.save_pretrained(training_args.output_dir)

    trainer = CustomizedTrainer(
        model=medusa_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    model.config.use_cache = True

    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head
    
    # Handle DeepSpeed saving
    try:
        import deepspeed
        with deepspeed.zero.GatheredParameters(lm_head.parameters()):
            state_dict = lm_head.state_dict()
    except ImportError:
        state_dict = lm_head.state_dict()

    if local_rank == 0:
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )

if __name__ == "__main__":
    train()
