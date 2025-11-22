import os
from pathlib import Path
import time
from typing import Literal, cast

from datasets import Dataset
from nlppets.torch import calculate_gradient_accumulation, nested_freeze_tensor
import torch
import torchinfo
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from trl import SFTConfig, SFTTrainer

from src.adapter.qwen import apply_adapter
from src.dataset.cblue import get_dataset

MODEL_NAME = "./data/Qwen2.5-7B-Instruct"

ENHANCEMENTS = ("understanding", "generation", "inferencing")
CURRENT_ENHANCEMENT = os.getenv("ENHANCEMENT", "understanding")
assert CURRENT_ENHANCEMENT in ENHANCEMENTS
CURRENT_ENHANCEMENT = cast(
    Literal["understanding", "generation", "inferencing"], CURRENT_ENHANCEMENT
)
ATT_SIZE = int(os.getenv("ATT_SIZE", "2"))
FFN_SIZE = int(os.getenv("FFN_SIZE", "1024"))

MODEL_DIR_NAME = os.getenv("MODEL_NAME", str(int(time.time())))
OUTPUT_DIR = f"./data/{MODEL_DIR_NAME}/output/"
LOG_DIR = f"./data/{MODEL_DIR_NAME}/log/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = calculate_gradient_accumulation(BATCH_SIZE, 128)
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.01

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    do_train=True,
    logging_first_step=True,
    logging_dir=LOG_DIR,
    logging_steps=100,
    save_strategy=IntervalStrategy.EPOCH,
    save_steps=1,
    num_train_epochs=TRAIN_EPOCHS,
    optim=OptimizerNames.ADAMW_TORCH,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=1024,
)

with args.main_process_first(local=False, desc="loading tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

with args.main_process_first(local=False, desc="loading model"):
    model = apply_adapter(
        Qwen2ForCausalLM,
        ffn_adapter={CURRENT_ENHANCEMENT: FFN_SIZE},
        attn_adapter={CURRENT_ENHANCEMENT: ATT_SIZE},
    ).from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
    )


model = nested_freeze_tensor(
    model,
    exclude={
        *(f"model.layers.*.mlp.{e}_gate.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.mlp.{e}_up.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.mlp.{e}_down.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_q_proj.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_k_proj.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_v_proj.*" for e in ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_o_proj.*" for e in ENHANCEMENTS),
    },
)
torchinfo.summary(model, input_size=(BATCH_SIZE, 32768), dtypes=[torch.int64], depth=5)


def preprocess_dataset(batch: dict[str, list[str]]):
    return {"prompt": batch["input"], "completion": batch["target"]}


with args.main_process_first(local=False, desc="loading dataset"):
    train_dataset = Dataset.from_list(
        get_dataset(task_type=CURRENT_ENHANCEMENT, dataset_type="train")
    )
    train_dataset = train_dataset.map(
        preprocess_dataset, batched=True, remove_columns=train_dataset.column_names
    )

print(train_dataset)  # noqa: T201


trainer = SFTTrainer(
    model, args=args, processing_class=tokenizer, train_dataset=train_dataset
)

if __name__ == "__main__":
    print("Start training...")  # noqa: T201
    trainer.train()
