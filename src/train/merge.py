from pathlib import Path
import sys

from transformers.models.qwen2 import Qwen2ForCausalLM

from src.adapter.internlm import apply_adapter as internlm_apply_adapter
from src.adapter.qwen import apply_adapter as qwen_apply_adapter
from src.model.internlm.model import InternLM3ForCausalLM

MODEL_TYPE = "qwen"


def get_model_class(model_type: str, **kwargs):
    if model_type == "qwen":
        return qwen_apply_adapter(Qwen2ForCausalLM, **kwargs)
    if model_type == "internlm":
        return internlm_apply_adapter(InternLM3ForCausalLM, **kwargs)
    raise ValueError(f"Unknown model type: {model_type}")


def main(base_model: str, input_models: list[Path], output_dir: Path):
    final_state_dict = {}
    ffn_adapter = {}
    attn_adapter = {}
    for model_dir in input_models:
        model = get_model_class(MODEL_TYPE).from_pretrained(
            model_dir, torch_dtype="auto"
        )
        state_dict = model.get_adapter_state_dict()

        for name, value in getattr(model.config, "ffn_adapter", {}).items():
            if name in ffn_adapter:
                raise ValueError(f"Duplicate adapter name: {name}")
            ffn_adapter[name] = value

        for name, value in getattr(model.config, "attn_adapter", {}).items():
            if name in attn_adapter:
                raise ValueError(f"Duplicate adapter name: {name}")
            attn_adapter[name] = value

        for key, value in state_dict.items():
            if key in final_state_dict:
                raise ValueError(f"Duplicate tensor key: {key}")
            final_state_dict[key] = value

    model = get_model_class(
        MODEL_TYPE,
        ffn_adapter=ffn_adapter,
        attn_adapter=attn_adapter,
        enable_adapter_gate=False,
    ).from_pretrained(base_model, torch_dtype="auto")
    model.load_state_dict(final_state_dict, strict=False)
    model.config.meta = {
        "base_model": base_model,
        "adapter_models": [str(model_dir) for model_dir in input_models],
    }
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    base_model, *input_models, output_dir = sys.argv[1:]
    main(base_model, [Path(arg) for arg in input_models], Path(output_dir))
