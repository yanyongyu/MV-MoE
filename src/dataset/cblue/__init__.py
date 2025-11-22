from collections import defaultdict
from itertools import chain
import json
import os
from pathlib import Path
from typing import Any, Literal

TASK_TYPES = {
    "CHIP-CDEE": "event_extraction",
    "CHIP-CDN": "normalization",
    "CHIP-CTC": "cls",
    "CHIP-MDCFNPC": "attr_cls",
    "CHIP-STS": "nli",
    "CMedCausal": "triple_extraction",
    "CMeEE-V2": "ner",
    "CMeIE-V2": "spo_generation",
    "IMCS-V2-DAC": "cls",
    "IMCS-V2-MRG": "report_generation",
    "IMCS-V2-NER": "ner",
    "IMCS-V2-SR": "attr_cls",
    "KUAKE-IR": "nli",
    "KUAKE-QIC": "cls",
    "KUAKE-QQR": "nli",
    "KUAKE-QTR": "nli",
    "MedDG": "response_generation",
    "Text2DT": "text2dt",
}

TASK_TYPE_CLS = {
    "understanding": [
        "CHIP-CDEE",
        "CHIP-CDN",
        "CHIP-CTC",
        "CHIP-MDCFNPC",
        "CMedCausal",
        "CMeEE-V2",
        "CMeIE-V2",
        "IMCS-V2-DAC",
        "IMCS-V2-NER",
        "IMCS-V2-SR",
        "KUAKE-QIC",
        "Text2DT",
    ],
    "generation": ["IMCS-V2-MRG", "MedDG"],
    "inferencing": ["CHIP-STS", "KUAKE-IR", "KUAKE-QQR", "KUAKE-QTR"],
}


DATASET_DIR = Path(os.getenv("DATASET_DIR", "./data/processed_dataset"))


def get_task_dataset(
    task: str, type: Literal["train", "dev", "test"]
) -> list[dict[str, Any]]:
    if task not in TASK_TYPES:
        raise ValueError(f"Task {task} not found.")

    dataset_file = DATASET_DIR / task / f"{type}.jsonl"
    return [json.loads(line) for line in dataset_file.read_text().splitlines()]


def get_dataset(
    task_type: Literal["understanding", "generation", "inferencing"],
    dataset_type: Literal["train", "dev", "test"],
) -> list[dict[str, Any]]:
    tasks = [k for k in TASK_TYPES.keys() if k in TASK_TYPE_CLS[task_type]]
    return list(
        chain.from_iterable(get_task_dataset(task, dataset_type) for task in tasks)
    )


def process(input_file: Path, type: Literal["train", "dev", "test"], output_dir: Path):
    task_data: dict[str, list[dict]] = defaultdict(list)
    task_type: dict[str, str] = {}

    dataset = input_file.read_text().splitlines()
    for line in dataset:
        data = json.loads(line)
        task_data[data["task_dataset"]].append(data)
        task_type[data["task_dataset"]] = data["task_type"]

    print(task_type)  # noqa: T201

    output_dir.mkdir(parents=True, exist_ok=True)
    for task, task_dataset in task_data.items():
        task_output_dir = output_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        (task_output_dir / f"{type}.jsonl").write_text(
            "\n".join(json.dumps(data, ensure_ascii=False) for data in task_dataset)
        )


if __name__ == "__main__":
    RAW_DATASET = Path("./data/PromptCBLUE")
    OUTPUT_DIR = Path("./data/processed_dataset")
    process(RAW_DATASET / "训练集验证集" / "train.json", "train", OUTPUT_DIR)
    process(RAW_DATASET / "训练集验证集" / "dev.json", "dev", OUTPUT_DIR)
    process(RAW_DATASET / "A榜测试集.json", "test", OUTPUT_DIR)
