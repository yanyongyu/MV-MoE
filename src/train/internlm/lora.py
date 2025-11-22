import os
from pathlib import Path
import time

from datasets import Dataset
from nlppets.torch import calculate_gradient_accumulation
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torchinfo
from transformers import AutoTokenizer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import OptimizerNames
from trl import SFTConfig, SFTTrainer

from src.dataset.cblue import get_dataset
from src.model.internlm.model import InternLM3ForCausalLM

MODEL_NAME = "./data/internlm3-8b-instruct"

ENHANCEMENTS = ("understanding", "generation", "inferencing")

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
    deepspeed="ds_config.json",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=512,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
)

with args.main_process_first(local=False, desc="loading tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

with args.main_process_first(local=False, desc="loading model"):
    model = InternLM3ForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
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
    trainer.train(resume_from_checkpoint=True)
