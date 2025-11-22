from copy import copy
import json
from pathlib import Path
import sys
from typing import Any, Literal

from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils import PaddingStrategy

from src.adapter.internlm import apply_adapter
from src.dataset.cblue import get_dataset
from src.dataset.cblue.eval import calc_scores, process_generated_results
from src.model.internlm.model import InternLM3ForCausalLM

BATCH_SIZE = 16
BASE_MODEL = "./data/internlm3-8b-instruct"


def _format_inputs(
    tokenizer: PreTrainedTokenizerBase, data: list[dict[str, Any]]
) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["input"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in data
    ]  # type: ignore


def _generate(
    model: InternLM3ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[dict[str, Any]],
):
    results = []

    bar = tqdm(total=len(dataset))
    try:
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i : i + BATCH_SIZE]
            texts = _format_inputs(tokenizer, batch)
            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=PaddingStrategy.LONGEST,
                padding_side="left",  # type: ignore
            ).to(model.device)
            generation_config = copy(model.generation_config) or GenerationConfig()
            generation_config.do_sample = False
            generation_config.max_new_tokens = 4096
            # generation_config.bos_token_id = tokenizer.bos_token_id
            # generation_config.pad_token_id = tokenizer.pad_token_id
            # generation_config.eos_token_id = tokenizer.eos_token_id
            generated_ids = model.generate(
                **model_inputs,  # type: ignore
                generation_config=generation_config,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            for case, text in zip(batch, generated_texts, strict=True):
                results.append({**case, "target": text})
            bar.update(len(batch))
        return results
    finally:
        bar.close()


def evaluate(model_dir: str, dataset_type: Literal["dev", "test"]):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = apply_adapter(InternLM3ForCausalLM).from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    enhancements = list(model.config.ffn_adapter.keys())
    if enhancements == ["medical"]:
        enhancements = ("understanding", "generation", "inferencing")
    elif "gate" in enhancements:
        enhancements.remove("gate")
    eval_dataset = []
    for enhancement in enhancements:
        eval_dataset.extend(get_dataset(enhancement, dataset_type))

    standard_result = process_generated_results(eval_dataset)

    cached_result = Path(model_dir) / f"promptcblue_results_{dataset_type}.jsonl"
    if not cached_result.exists():
        results = _generate(model, tokenizer, eval_dataset)  # type: ignore
        cached_result.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results)
        )
    else:
        results = [json.loads(line) for line in cached_result.read_text().splitlines()]

    result = process_generated_results(results)

    return calc_scores(standard_result, result)


def evaluate_base(model_dir: str, dataset_type: Literal["dev", "test"]):
    eval_dataset = []
    for enhancement in ("understanding", "generation", "inferencing"):
        eval_dataset.extend(get_dataset(enhancement, dataset_type))
    standard_result = process_generated_results(eval_dataset)

    cached_result = Path(model_dir) / f"promptcblue_results_{dataset_type}.jsonl"
    if not cached_result.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = InternLM3ForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        results = []
        for enhancement in ("understanding", "generation", "inferencing"):
            eval_dataset = get_dataset(enhancement, dataset_type)
            results.extend(_generate(model, tokenizer, eval_dataset))  # type: ignore

        cached_result.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results)
        )
    else:
        results = [json.loads(line) for line in cached_result.read_text().splitlines()]

    result = process_generated_results(results)
    return calc_scores(standard_result, result)


def evaluate_lora(model_dir: str, dataset_type: Literal["dev", "test"]):
    eval_dataset = []
    for enhancement in ("understanding", "generation", "inferencing"):
        eval_dataset.extend(get_dataset(enhancement, dataset_type))
    standard_result = process_generated_results(eval_dataset)
    cached_result = Path(model_dir) / f"promptcblue_results_{dataset_type}.jsonl"
    if not cached_result.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        results = []
        for enhancement in ("understanding", "generation", "inferencing"):
            eval_dataset = get_dataset(enhancement, dataset_type)
            results.extend(_generate(model, tokenizer, eval_dataset))  # type: ignore

        cached_result.write_text(
            "\n".join(json.dumps(result, ensure_ascii=False) for result in results)
        )
    else:
        results = [json.loads(line) for line in cached_result.read_text().splitlines()]

    result = process_generated_results(results)
    return calc_scores(standard_result, result)


def evaluate_xlora(model_dir: str, dataset_type: Literal["dev", "test"]):
    eval_dataset = []
    for enhancement in ("understanding", "generation", "inferencing"):
        eval_dataset.extend(get_dataset(enhancement, dataset_type))
    standard_result = process_generated_results(eval_dataset)
    cached_result = Path(model_dir) / f"promptcblue_results_{dataset_type}.jsonl"
    if cached_result.exists():
        results = [json.loads(line) for line in cached_result.read_text().splitlines()]
    else:
        results = []

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )

    order_list = []
    for enhancement in ("understanding", "generation", "inferencing"):
        eval_dataset = get_dataset(enhancement, dataset_type)
        # eval_dataset = eval_dataset[: len(eval_dataset) // 2]
        order_list.extend(
            (case["task_dataset"], case["sample_id"]) for case in eval_dataset
        )
        unfinished_dataset = [
            case
            for case in eval_dataset
            if (case["task_dataset"], case["sample_id"])
            not in [(result["task_dataset"], result["sample_id"]) for result in results]
        ]
        print(f"{enhancement} remain {len(unfinished_dataset)}")  # noqa: T201
        for i in range(0, len(unfinished_dataset), BATCH_SIZE):
            generated = _generate(
                model, tokenizer, unfinished_dataset[i : i + BATCH_SIZE]
            )
            results.extend(generated)  # type: ignore
            with cached_result.open("a") as f:
                for result in generated:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # deduplicate
    dedup_results = {}
    for result in results:
        if (result["task_dataset"], result["sample_id"]) not in dedup_results:
            dedup_results[(result["task_dataset"], result["sample_id"])] = result
    results = [dedup_results[key] for key in order_list]
    result = process_generated_results(results)
    return calc_scores(standard_result, result)


if __name__ == "__main__":
    # print(evaluate_base(sys.argv[1], "dev"))
    print(evaluate(sys.argv[1], "dev"))  # noqa: T201
    # print(evaluate_lora(sys.argv[1], "dev"))
    # print(evaluate_xlora(sys.argv[1], "dev"))
