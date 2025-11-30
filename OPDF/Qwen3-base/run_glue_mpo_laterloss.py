"""Distilling Qwen3.1 causal language models on CodeSearchNet with MPO compression."""
import logging
import os
import random
import sys
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Sequence

from accelerate import Accelerator
import datasets
import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.nn import MSELoss
import json

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    get_scheduler,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from utils_glue import LGTMTeacher, cal_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from compress_tools.MPOtorch import LinearDecomMPO
from compress_tools.Matrix2MPO import MPO
import re
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

TARGET_LINEAR_KEYWORDS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
]
MPO_NUM_DIMENSIONS = 6


def _factorize_dimension(dim: int, max_dim: int = MPO_NUM_DIMENSIONS) -> List[int]:
    """Split a dimension into a list of factors to build MPO shapes."""
    factors: List[int] = []
    remainder = dim
    primes = [2, 3, 5, 7, 11, 13]
    for prime in primes:
        while remainder % prime == 0 and len(factors) < max_dim:
            factors.append(prime)
            remainder //= prime
    if remainder > 1:
        factors.append(remainder)
    while len(factors) > max_dim:
        factors.sort()
        merged = factors.pop(0) * factors.pop(0)
        factors.append(merged)
    while len(factors) < max_dim:
        factors.append(1)
    return factors


def _get_module_by_name(model, module_name: str):
    module = model
    for attr in module_name.split("."):
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module


def _set_module_by_name(model, module_name: str, new_module):
    parts = module_name.split(".")
    module = model
    for attr in parts[:-1]:
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    last_attr = parts[-1]
    if last_attr.isdigit():
        module[int(last_attr)] = new_module
    else:
        setattr(module, last_attr, new_module)


def _count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _collect_target_linear_modules(model) -> List[str]:
    target_modules: List[str] = []
    for name, _ in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        base_name = name[: -len(".weight")]
        if "model.layers" not in base_name:
            continue
        if any(keyword in base_name for keyword in TARGET_LINEAR_KEYWORDS):
            target_modules.append(base_name)
    return target_modules


def fine_grained_decomposition(model, module_name: str):
    layer_module = _get_module_by_name(model, module_name)
    device = layer_module.weight.device
    bias = layer_module.bias

    input_shape = _factorize_dimension(layer_module.in_features)
    output_shape = _factorize_dimension(layer_module.out_features)
    mpo = MPO(input_shape, output_shape, None)
    mpo_tensor_set, _, _ = mpo.matrix2mpo(layer_module.weight.detach().cpu().numpy())

    mpo_layer = LinearDecomMPO(input_shape, output_shape, None)
    _set_module_by_name(model, module_name, mpo_layer)
    mpo_layer.from_pretrained(None, None, mpo_tensor_set, bias, device=device)


def _convert_model_to_mpo(model) -> List[str]:
    module_names = _collect_target_linear_modules(model)
    for name in module_names:
        fine_grained_decomposition(model, name)
    return module_names


def _build_layer_alignment(student_layers: int, teacher_layers: int) -> List[tuple]:
    mapping: List[tuple] = []
    if student_layers == 0 or teacher_layers == 0:
        return mapping
    for s_idx in range(1, student_layers + 1):
        position = s_idx / float(student_layers)
        teacher_idx = max(1, min(teacher_layers, int(round(position * teacher_layers))))
        mapping.append((s_idx, teacher_idx))
    return mapping


class HiddenStateAligner(nn.Module):
    def __init__(self, student_layers: int, teacher_layers: int, student_hidden: int, teacher_hidden: int):
        super().__init__()
        self.layer_map = _build_layer_alignment(student_layers, teacher_layers)
        self.projector = (
            nn.Linear(teacher_hidden, student_hidden, bias=False)
            if teacher_hidden != student_hidden
            else nn.Identity()
        )
        self.mse = nn.MSELoss()

    def forward(self, student_states, teacher_states):
        if (
            student_states is None
            or teacher_states is None
            or len(student_states) <= 1
            or len(teacher_states) <= 1
            or len(self.layer_map) == 0
        ):
            return None

        total_loss = None
        for student_idx, teacher_idx in self.layer_map:
            if student_idx >= len(student_states) or teacher_idx >= len(teacher_states):
                continue
            student_tensor = student_states[student_idx]
            teacher_tensor = teacher_states[teacher_idx]
            if isinstance(self.projector, nn.Identity):
                aligned_teacher = teacher_tensor
            else:
                aligned_teacher = self.projector(teacher_tensor)
            loss_val = self.mse(student_tensor, aligned_teacher)
            total_loss = loss_val if total_loss is None else total_loss + loss_val

        if total_loss is None:
            return None
        return total_loss / len(self.layer_map)


def _map_teacher_module_name(module_name: str, student_layers: int, teacher_layers: int) -> str:
    if student_layers == 0 or "layers." not in module_name:
        return module_name

    def _replace(match):
        student_idx = int(match.group(2))
        ratio = teacher_layers / float(student_layers)
        teacher_idx = min(teacher_layers - 1, max(0, int(round((student_idx + 0.5) * ratio))))
        return f"{match.group(1)}{teacher_idx}"

    return re.sub(r"(layers\.)(\d+)", _replace, module_name, count=1)


def _ensure_tensor_loss(reference_param):
    return torch.zeros(1, device=reference_param.device, dtype=reference_param.dtype).squeeze()


def loss_layer_fn(model, teacher_model, choose_name: Sequence[str], student_layers: int, teacher_layers: int):
    reference_param = next(model.parameters())
    total_loss = None
    matched = 0
    mse = MSELoss()

    for module_name in choose_name:
        teacher_name = _map_teacher_module_name(module_name, student_layers, teacher_layers)
        student_module = _get_module_by_name(model, module_name)
        teacher_module = _get_module_by_name(teacher_model, teacher_name)
        if not hasattr(student_module, "tensor_set") or not hasattr(teacher_module, "tensor_set"):
            continue

        local_loss = None
        tensor_pairs = zip(student_module.tensor_set, teacher_module.tensor_set)
        for s_tensor, t_tensor in tensor_pairs:
            loss_val = mse(s_tensor, t_tensor)
            local_loss = loss_val if local_loss is None else local_loss + loss_val
        if local_loss is not None:
            matched += 1
            local_loss = local_loss / len(student_module.tensor_set)
            total_loss = local_loss if total_loss is None else total_loss + local_loss

    if total_loss is None or matched == 0:
        return _ensure_tensor_loss(reference_param)
    return total_loss / matched


def _ensure_model_output(outputs):
    if hasattr(outputs, "loss") and hasattr(outputs, "logits"):
        return outputs
    if isinstance(outputs, tuple):
        loss = outputs[0]
        logits = outputs[1] if len(outputs) > 1 else None
        hidden_states = outputs[2] if len(outputs) > 2 else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)
    raise ValueError("Model outputs must contain loss and logits.")






logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments controlling the dataset used for causal language modeling distillation.
    """

    dataset_name: Optional[str] = field(
        default="Nan-Do/code-search-net-python",
        metadata={"help": "Dataset name from the Hub. Defaults to the Python split of CodeSearchNet."},
    )
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Optional dataset config name."})
    dataset_split: str = field(default="train", metadata={"help": "Which split to load from the dataset."})
    text_column: str = field(default="code", metadata={"help": "Column containing raw text/code."})
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length for causal LM training. Longer sequences are truncated."
        },
    )
    validation_split_percentage: float = field(
        default=5.0, metadata={"help": "Percentage of data set aside for evaluation/testing."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional limit on the number of training samples (after split)."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional limit on the number of evaluation samples (shared by dev/test)."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B-Base",
        metadata={"help": "Student model identifier (default: Qwen/Qwen3-0.6B-Base)."},
    )
    teacher_model: str = field(
        default="Qwen/Qwen3-8B-Base",
        metadata={"help": "Teacher model identifier (default: Qwen/Qwen3-8B-Base)."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    teacher_device_map: Optional[str] = field(
        default="auto", metadata={"help": "Device map for teacher (e.g. 'auto' or 'cpu'). Use 'none' for default."}
    )
    student_device_map: Optional[str] = field(
        default=None, metadata={"help": "Device map for student model. Use 'auto' for Accelerate placement."}
    )
    teacher_load_in_8bit: bool = field(
        default=True,
        metadata={"help": "Load teacher in 8-bit for memory efficiency (teacher used for inference only)."},
    )
    student_load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the student in 8-bit (not recommended if you need gradients)."},
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "Torch dtype (float16, bfloat16, float32)."},
    )

    # kd setting
    alpha_kd: float = field(
        default=1.0,
        metadata={
            "help": "The weight of kd loss"
        },
    )
    # mode: str = field(
    #     default="kd",
    #     metadata={"help": "The type of kd loss"},
    # )
    temperature: float = field(
        default=1,
        metadata={
            "help": "The temperature."
        },
    )

    # teacher model setting
    teacher_model: str = field(
        default=None,
        metadata={
            "help": "Path of teacher model."
        },
    )
    train_teacher: bool = field(
        default=False,
        metadata={
            "help": "Train teacher or not."
        },
    )
    t_alpha_kd: float = field(
        default=0.4,
        metadata={
            "help": "The weight of kd loss if train_teacher is True."
        },
    )
    t_learning_rate: float = field(
        default=3e-5,
        metadata={
            "help": "The learning rate of teacher."
        },
    )

    # lgtm setting
    use_lgtm: bool = field(
        default=False,
        metadata={
            "help": "Use LGTM or not."
        },
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator()
    os.makedirs(training_args.output_dir, exist_ok=True)
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer and models following the setup used in distil_prep.ipynb
    tokenizer_name = model_args.tokenizer_name or model_args.teacher_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _string_to_dtype(dtype_str: str):
        if dtype_str is None:
            return None
        dtype_str = dtype_str.lower()
        if dtype_str in ("float16", "fp16"):
            return torch.float16
        if dtype_str in ("bfloat16", "bf16"):
            return torch.bfloat16
        if dtype_str in ("float32", "fp32"):
            return torch.float32
        return None

    def _normalize_device_map(value):
        if value is None:
            return None
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("none", "null", "default"):
                return None
            return value
        return value

    student_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    student_config.output_hidden_states = True
    student_config.pad_token_id = tokenizer.pad_token_id

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_model,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    teacher_config.output_hidden_states = True
    teacher_config.pad_token_id = tokenizer.pad_token_id

    dtype = _string_to_dtype(model_args.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=student_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=_normalize_device_map(model_args.student_device_map),
        load_in_8bit=model_args.student_load_in_8bit,
    )
    if model_args.student_load_in_8bit:
        raise ValueError("Student model cannot be trained in 8-bit. Set student_load_in_8bit=False.")

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.teacher_model,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=_normalize_device_map(model_args.teacher_device_map),
        load_in_8bit=model_args.teacher_load_in_8bit,
    )
    if model_args.train_teacher and model_args.teacher_load_in_8bit:
        raise ValueError("Disable teacher_load_in_8bit when training the teacher.")

    # Dataset preparation (CodeSearchNet Python as default)
    raw_dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.dataset_split,
        cache_dir=model_args.cache_dir,
    )

    split_dataset = raw_dataset.train_test_split(
        test_size=data_args.validation_split_percentage / 100.0,
        seed=training_args.seed,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        texts = examples[data_args.text_column]
        if isinstance(texts, str):
            texts = [texts]
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_attention_mask=True,
        )
        outputs["labels"] = [list(ids) for ids in outputs["input_ids"]]
        return outputs

    with training_args.main_process_first(desc="Tokenize dataset"):
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train split",
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval split",
        )

    eval_split = eval_dataset.train_test_split(test_size=0.5, seed=training_args.seed)
    dev_dataset = eval_split["train"]
    test_dataset = eval_split["test"]

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    dev_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if training_args.do_train and len(train_dataset) > 0:
        sample_count = min(3, len(train_dataset))
        for index in random.sample(range(len(train_dataset)), sample_count):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    #####################################################################################进行MPO#########################################################################################################################
#######################################################################################################################################################################################
    print("#########################################################################################################")
    print(f"Student parameters before MPO: {_count_parameters(model)}")
    print("#########################################################################################################")
    choose_name_student = _convert_model_to_mpo(model)
    print("#########################################################################################################")
    print(f"Student parameters after MPO: {_count_parameters(model)}")
    print("#########################################################################################################")

    if training_args.do_train:
        print("#########################################################################################################")
        print(f"Teacher parameters before MPO: {_count_parameters(teacher_model)}")
        print("#########################################################################################################")
        _convert_model_to_mpo(teacher_model)
        print("#########################################################################################################")
        print(f"Teacher parameters after MPO: {_count_parameters(teacher_model)}")
        print("#########################################################################################################")

#######################################################################################################################################################################################
    student_layer_count = getattr(model.config, "num_hidden_layers", len(choose_name_student))
    teacher_layer_count = getattr(teacher_model.config, "num_hidden_layers", student_layer_count)
    alignment_helper = HiddenStateAligner(
        student_layer_count,
        teacher_layer_count,
        getattr(model.config, "hidden_size"),
        getattr(teacher_model.config, "hidden_size"),
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        dev_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size
    )
    # Optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(alignment_helper.parameters()),
        lr=training_args.learning_rate,
    )

    if model_args.train_teacher:
        t_optimizer = AdamW(teacher_model.parameters(), lr=model_args.t_learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = max_train_steps*training_args.warmup_ratio

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    if model_args.train_teacher:
        t_lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=t_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        model, teacher_model, alignment_helper, optimizer, t_optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            model, teacher_model, alignment_helper, optimizer, t_optimizer, train_dataloader, eval_dataloader, test_dataloader
        )

        if model_args.use_lgtm:
            held_iter = iter(eval_dataloader)
            model_total = LGTMTeacher(teacher_model, model, model_args.alpha_kd, model_args.t_alpha_kd,
                                    t_optimizer, t_lr_scheduler, model_args.temperature)
            
    else:
        model, teacher_model, alignment_helper, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                model, teacher_model, alignment_helper, optimizer, train_dataloader, eval_dataloader, test_dataloader
            )
     
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    def compute_eval_metrics(eval_model, dataloader):
        eval_model.eval()
        total_loss = 0.0
        total_count = 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = eval_model(**batch)
            loss = outputs.loss.detach()
            batch_size = batch["input_ids"].size(0)
            gathered = accelerator.gather(loss.repeat(batch_size))
            total_loss += gathered.sum().item()
            total_count += gathered.numel()
        if total_count == 0:
            return {"loss": float("inf"), "perplexity": float("inf")}
        avg_loss = total_loss / total_count
        perplexity = math.exp(min(avg_loss, 50))
        return {"loss": avg_loss, "perplexity": perplexity}

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {int(training_args.num_train_epochs)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    best_metric = float("inf")
    t_best_metric = float("inf")
    best_dev = {}
    best_test = {}

    for epoch in range(int(training_args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if model_args.train_teacher:
                if model_args.use_lgtm:
                    model.train()
                    teacher_model.train()
                    # fetch the batch for calculating student's feedback 
                    try:
                        held_inputs = next(held_iter)
                    except:
                        held_iter = iter(eval_dataloader)
                        held_inputs = next(held_iter)
                    # update the teacher
                    model_total.step(batch, held_inputs, optimizer)
                else:
                    model.eval()
                    teacher_model.train()
                    with torch.no_grad():
                        outputs = _ensure_model_output(model(**batch))
                        logits = outputs.logits
                    teacher_outputs = _ensure_model_output(teacher_model(**batch))
                    t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
                    t_loss = model_args.t_alpha_kd * cal_loss(t_logits,logits, model_args.temperature) + (1 - model_args.t_alpha_kd) * t_loss
    #####################################################################################对齐#########################################################################################################################
                    loss_layer = 0

                    for i in range(len(outputs.hidden_states)-1):
                        loss_layer = loss_layer + torch.nn.MSELoss()(outputs.hidden_states[i+1],teacher_outputs.hidden_states[2*i+1+1])
                    # t_loss = loss_layer/(len(outputs.hidden_states)-1) + t_loss
    ###############################################################################################################################################################################################################

                    # update the teacher
                    t_loss.backward()
                
                    if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        t_optimizer.step()
                        t_lr_scheduler.step()
                        t_optimizer.zero_grad()

                # use teacher logits as soft labels
                teacher_model.eval()
                model.train()
                alignment_helper.train()
                            
                with torch.no_grad():
                    teacher_outputs = _ensure_model_output(teacher_model(**batch))
                    t_logits = teacher_outputs.logits

                outputs = _ensure_model_output(model(**batch))
                loss, logits = outputs.loss, outputs.logits
                loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1-model_args.alpha_kd) * loss
            else: # no teacher update
                # student loss
                model.train()
                alignment_helper.train()
                outputs = _ensure_model_output(model(**batch))
                loss, logits = outputs.loss, outputs.logits
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = _ensure_model_output(teacher_model(**batch))
                    t_logits = teacher_outputs.logits

                loss = model_args.alpha_kd * cal_loss(logits, t_logits, model_args.temperature) + (1 - model_args.alpha_kd) * loss

            # update the student 
            student_hidden_states = getattr(outputs, "hidden_states", None)
            teacher_hidden_states = getattr(teacher_outputs, "hidden_states", None)
            hidden_alignment_loss = alignment_helper(student_hidden_states, teacher_hidden_states)
            mpo_alignment_loss = loss_layer_fn(
                model,
                teacher_model,
                choose_name_student,
                student_layer_count,
                teacher_layer_count,
            )

            loss = loss / int(training_args.gradient_accumulation_steps)
            if hidden_alignment_loss is not None:
                loss = loss + hidden_alignment_loss
            loss = loss + 0.0001 * mpo_alignment_loss
            print("loss:")
            print(loss)
            loss.backward()

            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % training_args.eval_steps == 0 or completed_steps == max_train_steps:
                model.eval()
                alignment_helper.eval()

                test_metric = compute_eval_metrics(model, test_dataloader)
                logger.info("***** Test Results *****")
                logger.info(f"  Training step = {completed_steps}")
                logger.info(f"  test_loss: {test_metric['loss']:.4f}")
                logger.info(f"  test_perplexity: {test_metric['perplexity']:.4f}")

                eval_metric = compute_eval_metrics(model, eval_dataloader)
                logger.info("***** Validation Results *****")
                logger.info(f"  Training step = {completed_steps}")
                logger.info(f"  eval_loss: {eval_metric['loss']:.4f}")
                logger.info(f"  eval_perplexity: {eval_metric['perplexity']:.4f}")

                if eval_metric["perplexity"] < best_metric:
                    best_metric = eval_metric["perplexity"]
                    tokenizer.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir)
                    path = os.path.join(training_args.output_dir, "eval_results.json")
                    with open(path, "w") as f:
                        json.dump(eval_metric, f, indent=4, sort_keys=True)
                    best_dev = eval_metric
                    best_test = test_metric
                logger.info(f"Best dev so far: {best_dev}")
                logger.info(f"Best test so far: {best_test}")

                data = {'dev': best_dev, 'test': best_test}
                dataset_tag = data_args.dataset_name.replace('/', '_')
                file_path = (
                    f"/mnt/name/data/checkpoint/nlp/lgtm/evalue/"
                    f"{dataset_tag}_{training_args.learning_rate}_{model_args.t_learning_rate}"
                    f"_stu{student_layer_count}_tea{teacher_layer_count}.txt"
                )
                with open(file_path, 'w') as file:
                    for key, value in data.items():
                        file.write(f"{key}: {value}\n")

                teacher_metric = compute_eval_metrics(teacher_model, eval_dataloader)
                logger.info("***** Teacher Validation Results *****")
                logger.info(f"  Training step = {completed_steps}")
                logger.info(f"  teacher_loss: {teacher_metric['loss']:.4f}")
                logger.info(f"  teacher_perplexity: {teacher_metric['perplexity']:.4f}")
                
                if teacher_metric["perplexity"] < t_best_metric:
                    t_best_metric = teacher_metric["perplexity"]
                    path = os.path.join(training_args.output_dir, "teacher_eval_results.json")
                    with open(path, "w") as f:
                        json.dump(teacher_metric, f, indent=4, sort_keys=True) 

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
