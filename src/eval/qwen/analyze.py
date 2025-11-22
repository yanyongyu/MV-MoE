from copy import copy
from pathlib import Path
import sys
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
)
from transformers.models.qwen2 import Qwen2ForCausalLM
from transformers.tokenization_utils import PaddingStrategy

from src.adapter.qwen import ANALYSIS_CONTEXT, AnalysisContext, apply_adapter
from src.dataset.cblue import get_dataset

BATCH_SIZE = 1
BASE_MODEL = "./data/Qwen2.5-7B-Instruct"
ENHANCEMENTS = ("understanding", "generation", "inferencing")
ENHANCEMENT_NAME_MAP = {
    "understanding": "NLU",
    "generation": "NLG",
    "inferencing": "NLI",
}
IMAGE_SCALE = 10


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


def _generate_and_collect(
    model: Qwen2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[dict[str, Any]],
) -> AnalysisContext:
    results = []

    enhancements = list(model.config.ffn_adapter.keys())
    if enhancements == ["medical"]:
        enhancements = ENHANCEMENTS
    elif "gate" in enhancements:
        enhancements.remove("gate")

    bar = tqdm(total=len(dataset))
    bar.set_description("Collecting data")
    analysis_context = AnalysisContext(shared_gate={}, adapter_gate={})
    t = ANALYSIS_CONTEXT.set(analysis_context)
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
            if analysis_context["shared_gate"] and analysis_context["adapter_gate"]:
                shared_data_count = analysis_context["shared_gate"][0].size(0)
                adapter_data_count = analysis_context["adapter_gate"][0].size(0)
                bar.set_description(
                    f"Collected {shared_data_count}, {adapter_data_count} gate data"
                )
        return analysis_context
    finally:
        ANALYSIS_CONTEXT.reset(t)
        bar.close()


def _get_cache_path(model_dir: str, enhancement: str, dataset_type: str) -> Path:
    return Path(model_dir) / f"promptcblue_{enhancement}_{dataset_type}_analysis.pt"


def _analyze(model_dir: str, dataset_type: Literal["dev", "test"]):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = apply_adapter(Qwen2ForCausalLM).from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )

    for enhancement in ENHANCEMENTS:
        cached_result = _get_cache_path(model_dir, enhancement, dataset_type)
        if not cached_result.exists():
            eval_dataset = get_dataset(enhancement, dataset_type)
            collected_data = _generate_and_collect(model, tokenizer, eval_dataset)  # type: ignore
            torch.save(collected_data, cached_result)


def _visualize_shared_gate(analysis_data: dict[str, AnalysisContext], save_path: Path):
    shared_figure = plt.figure(
        "shared_gate",
        # figsize=(len(ENHANCEMENTS) * IMAGE_SCALE, 1 * IMAGE_SCALE),
    )
    ax = shared_figure.add_subplot(1, 1, 1)
    colors = sns.color_palette("tab10", n_colors=len(ENHANCEMENTS))
    data = {
        f"{ENHANCEMENT_NAME_MAP[enhancement]} Dataset": (
            torch.concat(
                list(analysis_data[enhancement]["shared_gate"].values()), dim=0
            )
            .reshape(-1)
            .float()
            .numpy()
        )
        for enhancement in ENHANCEMENTS
    }
    sns.kdeplot(
        data,
        clip=(0.5, 1.0),
        bw_adjust=1,
        multiple="layer",
        common_norm=False,
        palette=colors,
        fill=True,
        ax=ax,
    )
    for legend in ax.get_legend().get_texts():  # pyright: ignore[reportOptionalMemberAccess]
        legend.set_fontsize(12)
    # for idx, enhancement in enumerate(ENHANCEMENTS):
    #     gate_data = (
    #         torch.concat(
    #             list(analysis_data[enhancement]["shared_gate"].values()), dim=0
    #         )
    #         .reshape(-1)
    #         .float()
    #         .numpy()
    #     )
    #     sns.kdeplot(
    #         {enhancement: gate_data},
    #         clip=(0.5, 1.0),
    #         bw_adjust=1,
    #         common_norm=True,
    #         color=colors[idx],
    #         label=f"{enhancement.capitalize()} Dataset",
    #         fill=True,
    #         ax=ax,
    #     )
    ax.set_xlabel("Model Gate Value", fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    # for idx, enhancement in enumerate(ENHANCEMENTS):
    #     ax = shared_figure.add_subplot(1, 3, 1 + idx)
    #     ax.set_title(f"{enhancement.title()} Model Gate", y=1.0, fontsize=36)
    #     # gate_data: [B, 1]
    #     gate_data = (
    #         torch.concat(
    #             list(analysis_data[enhancement]["shared_gate"].values()), dim=0
    #         )
    #         .reshape(-1)
    #         .float()
    #         .numpy()
    #     )
    #     sns.kdeplot(
    #         {enhancement: gate_data},
    #         clip=(0.5, 1.0),
    #         bw_adjust=1,
    #         common_norm=True,
    #         legend=False,
    #         ax=ax,
    #     )
    #     # ax.hist(
    #     #     gate_data,
    #     #     bins=100,
    #     #     range=(0.5, 1.0),
    #     #     density=True,
    #     #     label="gate_distribution",
    #     #     histtype="step",
    #     # )
    #     # ax.legend()

    #     # set fontsize
    #     ax.set_xlabel("Model Gate Value", fontsize=28)
    #     ax.set_ylabel(ax.get_ylabel(), fontsize=28)
    #     ax.tick_params(axis="both", which="major", labelsize=28)
    shared_figure.tight_layout()
    print("saving shared gate figure")  # noqa: T201
    shared_figure.savefig(save_path)
    print("saved shared gate figure")  # noqa: T201


def _visualize_adapter_gate(analysis_data: dict[str, AnalysisContext], save_path: Path):
    # draw adapter gate data
    adapter_figure = plt.figure(
        "adapter_gate",
        figsize=(
            len(ENHANCEMENTS) * IMAGE_SCALE,
            # len(analysis_data["understanding"]["adapter_gate"]) * IMAGE_SCALE,
            1 * IMAGE_SCALE,
        ),
    )
    for idx, enhancement in enumerate(ENHANCEMENTS):
        # gate_data: [B, 1] -> [B]
        # shared_gate_data = (
        #     torch.concat(
        #         list(analysis_data[enhancement]["shared_gate"].values()), dim=0
        #     )
        #     .reshape(-1)
        #     .float()
        #     .numpy()
        # )
        # gate_data: [B, 3]
        # gate_data = (
        #     torch.concat(
        #         list(analysis_data[enhancement]["adapter_gate"].values()), dim=0
        #     )
        #     .float()
        #     .numpy()
        # )
        # gate_data = gate_data[
        #     np.logical_and(0.7 < shared_gate_data, shared_gate_data < 0.8), :
        # ]
        ax = cast(
            Axes3D,
            adapter_figure.add_subplot(1, 3, 1 + idx, projection="3d"),
        )
        ax.set_title(
            f"{ENHANCEMENT_NAME_MAP[enhancement]} Dataset",
            y=1.04,
            fontsize=36,
        )
        labels = sorted(ENHANCEMENTS)
        ax.set_xlabel(
            f"{ENHANCEMENT_NAME_MAP[labels[0]]} Expert", labelpad=15, fontsize=30
        )
        ax.set_ylabel(
            f"{ENHANCEMENT_NAME_MAP[labels[1]]} Expert", labelpad=15, fontsize=30
        )
        ax.set_zlabel(
            f"{ENHANCEMENT_NAME_MAP[labels[2]]} Expert", labelpad=15, fontsize=30
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(1.0, 0.0)
        ax.set_zlim(0.0, 1.0)
        # ax.scatter(gate_data[:, 0], gate_data[:, 1], gate_data[:, 2])  # type: ignore
        colors = sns.color_palette(
            "tab10", n_colors=len(analysis_data[enhancement]["adapter_gate"])
        )

        for layer_idx, gate_data in analysis_data[enhancement]["adapter_gate"].items():
            shared_gate = (
                analysis_data[enhancement]["shared_gate"][layer_idx]
                .reshape(-1)
                .float()
                .numpy()
            )
            gate_data = gate_data.float().numpy()[
                np.logical_and(0.7 < shared_gate, shared_gate < 0.8), :
            ]
            ax.scatter(
                gate_data[:, 0],
                gate_data[:, 1],
                gate_data[:, 2],  # type: ignore
                color=colors[layer_idx],
            )

        ax.tick_params(axis="both", which="major", labelsize=22)
    adapter_figure.tight_layout(pad=4)
    print("saving adapter gate figure")  # noqa: T201
    adapter_figure.savefig(save_path)
    print("saved adapter gate figure")  # noqa: T201


def analyze(model_dir: str, dataset_type: Literal["dev", "test"]):
    if not all(
        _get_cache_path(model_dir, enhancement, dataset_type).exists()
        for enhancement in ENHANCEMENTS
    ):
        _analyze(model_dir, dataset_type)
        torch.cuda.empty_cache()

    analysis_data: dict[str, AnalysisContext] = {}

    for enhancement in ENHANCEMENTS:
        analysis_data[enhancement] = torch.load(
            _get_cache_path(model_dir, enhancement, dataset_type), weights_only=True
        )

    _visualize_shared_gate(
        analysis_data, Path(model_dir) / f"promptcblue_{dataset_type}_shared_gate.pdf"
    )
    # _visualize_adapter_gate(
    #     analysis_data,
    #     Path(model_dir) / f"promptcblue_{dataset_type}_adapter_gate.png",
    # )


if __name__ == "__main__":
    analyze(sys.argv[1], "dev")
