import os
from pathlib import Path
import time
from typing import cast

from datasets import Dataset
from nlppets.torch import calculate_gradient_accumulation, nested_freeze_tensor
import torch
import torchinfo
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from trl import SFTConfig, SFTTrainer

from src.adapter.qwen import Qwen2DecoderLayerAdapter, apply_adapter
from src.dataset.cblue import get_dataset

MODEL_NAME = "./data/medical-merged-v1"

ADAPTER_TOPK = int(os.getenv("ADAPTER_TOPK", "1"))
ENHANCEMENTS = ("understanding", "generation", "inferencing")
TRAINABLE_ENHANCEMENTS = (
    "understanding",
    "generation",
    "inferencing",
)
# TRAINABLE_ENHANCEMENTS = set()
print(f"{TRAINABLE_ENHANCEMENTS=}")  # noqa: T201
GATE_NORMAL_MEAN = 0.0
print(f"{GATE_NORMAL_MEAN=}")  # noqa: T201
# GATE_ENHANCEMENT = "gate"
# ATT_SIZE = int(os.getenv("ATT_SIZE", "2"))
# FFN_SIZE = int(os.getenv("FFN_SIZE", "1024"))
MODEL_DIR_NAME = os.getenv("MODEL_NAME", str(int(time.time())))
print(f"Output dir: ./data/{MODEL_DIR_NAME}")  # noqa: T201
OUTPUT_DIR = f"./data/{MODEL_DIR_NAME}/output/"
LOG_DIR = f"./data/{MODEL_DIR_NAME}/log/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRADIENT_ACCUMULATION_STEPS = calculate_gradient_accumulation(BATCH_SIZE, 128)
LEARNING_RATE = 0.0006
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
    gradient_checkpointing=True,
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
        # ffn_adapter={GATE_ENHANCEMENT: FFN_SIZE},
        # attn_adapter={GATE_ENHANCEMENT: ATT_SIZE},
        enable_adapter_gate=True,
        # adapter_gate_topk=ADAPTER_TOPK,
    ).from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
    )

# init gate value
for layer in cast(Qwen2ForCausalLM, model).model.layers:
    patched_layer = cast(Qwen2DecoderLayerAdapter, layer)
    assert patched_layer.adapter_gate is not None
    assert patched_layer.shared_gate is not None
    patched_layer.adapter_gate.weight.data.normal_(GATE_NORMAL_MEAN, 0.02)
    patched_layer.shared_gate.weight.data.normal_(0.0, 0.02)

model = nested_freeze_tensor(
    model,
    exclude={
        "model.layers.*.down_gate.*",
        "model.layers.*.adapter_gate.*",
        "model.layers.*.shared_gate.*",
        # f"model.layers.*.mlp.{GATE_ENHANCEMENT}_gate.*",
        # f"model.layers.*.mlp.{GATE_ENHANCEMENT}_up.*",
        # f"model.layers.*.mlp.{GATE_ENHANCEMENT}_down.*",
        # f"model.layers.*.self_attn.{GATE_ENHANCEMENT}_q_proj.*",
        # f"model.layers.*.self_attn.{GATE_ENHANCEMENT}_k_proj.*",
        # f"model.layers.*.self_attn.{GATE_ENHANCEMENT}_v_proj.*",
        # f"model.layers.*.self_attn.{GATE_ENHANCEMENT}_o_proj.*",
        *(f"model.layers.*.mlp.{e}_gate.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.mlp.{e}_up.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.mlp.{e}_down.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_q_proj.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_k_proj.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_v_proj.*" for e in TRAINABLE_ENHANCEMENTS),
        *(f"model.layers.*.self_attn.{e}_o_proj.*" for e in TRAINABLE_ENHANCEMENTS),
    },
)
# used for gradient checkpointing
model.enable_input_require_grads()
torchinfo.summary(model, input_size=(BATCH_SIZE, 32768), dtypes=[torch.int64], depth=5)


def preprocess_dataset(batch: dict[str, list[str]]):
    return {"prompt": batch["input"], "completion": batch["target"]}


with args.main_process_first(local=False, desc="loading dataset"):
    train_dataset = Dataset.from_list(
        [
            row
            for enhancement in ENHANCEMENTS
            for row in get_dataset(task_type=enhancement, dataset_type="train")
        ]
    )
    train_dataset = train_dataset.map(
        preprocess_dataset, batched=True, remove_columns=train_dataset.column_names
    )
    train_dataset = train_dataset.shuffle()

print(train_dataset)  # noqa: T201

trainer = SFTTrainer(
    model,
    args=args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    print("Start training...")  # noqa: T201
    trainer.train(resume_from_checkpoint=None)
