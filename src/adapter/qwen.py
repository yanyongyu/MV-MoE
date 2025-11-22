from contextvars import ContextVar
from itertools import chain
import re
from typing import Any, Literal, Protocol, TypedDict, TypeVar, cast

from nlppets.torch import concat_linear, nested_replace_module
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

MT = TypeVar("MT", bound=type[Qwen2PreTrainedModel])

GATE_DOWN_SIZE = 1024


class AnalysisContext(TypedDict):
    shared_gate: dict[int, torch.Tensor]
    adapter_gate: dict[int, torch.Tensor]


ANALYSIS_CONTEXT: ContextVar[AnalysisContext | None] = ContextVar(
    "analysis_context", default=None
)


class Config(Protocol):
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float
    rms_norm_eps: float
    ffn_adapter: dict[str, int]
    attn_adapter: dict[str, int]
    enable_adapter_gate: bool
    # adapter_gate_topk: int


def get_adapter_types(config: Config) -> list[str]:
    # ensure order of adapter types
    return sorted(
        set(getattr(config, "attn_adapter", {}).keys())
        | set(getattr(config, "ffn_adapter", {}).keys())
    )


class Qwen2MLPAdapter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # Added adapter weights
        self.ffn_adapter: dict[str, int] = getattr(config, "ffn_adapter", {})
        self.adapter_types = get_adapter_types(config)
        for name, size in config.ffn_adapter.items():
            setattr(
                self, f"{name}_gate", nn.Linear(config.hidden_size, size, bias=False)
            )
            setattr(self, f"{name}_up", nn.Linear(config.hidden_size, size, bias=False))
            setattr(
                self, f"{name}_down", nn.Linear(size, config.hidden_size, bias=False)
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        shared_gate: torch.Tensor | None = None,
        adapter_gate: torch.Tensor | None = None,
    ):
        # patch for adapters
        # [B, L, H] -> [B, L, I + E]
        gate_value = (
            concat_linear(
                self.gate_proj,
                *(
                    getattr(self, f"{name}_gate")
                    for name in self.adapter_types
                    if name in self.ffn_adapter
                ),
            )
            if self.ffn_adapter
            else self.gate_proj
        )(hidden_state)
        if shared_gate is not None and adapter_gate is not None:
            # B, L, _ = hidden_state.shape
            # [B, L, I + E]
            adapter_mask = torch.concat(
                [
                    # [B, L, I]
                    # torch.ones(
                    #     B,
                    #     L,
                    #     self.intermediate_size,
                    #     device=hidden_state.device,
                    #     dtype=hidden_state.dtype,
                    # ),
                    shared_gate.expand(-1, -1, self.intermediate_size),
                    *(
                        # [B, L, E]
                        (
                            (1 - shared_gate) * (adapter_gate[:, :, idx].unsqueeze(-1))
                        ).expand(-1, -1, self.ffn_adapter[name])
                        for idx, name in enumerate(self.adapter_types)
                        if name in self.ffn_adapter
                    ),
                ],
                dim=-1,
            )
            # [B, L, I + E]
            gate_value = gate_value * adapter_mask

        # [B, L, H] -> [B, L, I + E]
        up_proj = (
            concat_linear(
                self.up_proj,
                *(
                    getattr(self, f"{name}_up")
                    for name in self.adapter_types
                    if name in self.ffn_adapter
                ),
            )
            if self.ffn_adapter
            else self.up_proj
        )

        # [B, L, I + E] -> [B, L, H]
        down_proj = (
            concat_linear(
                self.down_proj,
                *(
                    getattr(self, f"{name}_down")
                    for name in self.adapter_types
                    if name in self.ffn_adapter
                ),
            )
            if self.ffn_adapter
            else self.down_proj
        )
        return down_proj(self.act_fn(gate_value) * up_proj(hidden_state))


class Qwen2AttentionAdapter(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got "
                f"`hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # patch for adapters
        self.attn_adapter: dict[str, int] = getattr(config, "attn_adapter", {})
        self.adapter_types = get_adapter_types(config)
        for name, num in config.attn_adapter.items():
            setattr(
                self,
                f"{name}_q_proj",
                nn.Linear(self.hidden_size, self.head_dim * num, bias=True),
            )
            setattr(
                self,
                f"{name}_k_proj",
                nn.Linear(self.hidden_size, self.head_dim * num, bias=True),
            )
            setattr(
                self,
                f"{name}_v_proj",
                nn.Linear(self.hidden_size, self.head_dim * num, bias=True),
            )
            setattr(
                self,
                f"{name}_o_proj",
                nn.Linear(self.head_dim * num, self.hidden_size, bias=False),
            )

    def _apply_adapter(
        self, hidden_states: torch.Tensor, type: Literal["q", "k", "v", "o"]
    ) -> torch.Tensor:
        proj = (
            concat_linear(
                getattr(self, f"{type}_proj"),
                *(
                    getattr(self, f"{name}_{type}_proj")
                    for name in self.adapter_types
                    if name in self.attn_adapter
                ),
            )
            if self.attn_adapter
            else getattr(self, f"{type}_proj")
        )
        return proj(hidden_states)

    def _repeat_kv_without_adapter(self, kv_states: torch.Tensor) -> torch.Tensor:
        # [B, KVHN + EHN, L, HS] -> [B, KVHN, L, HS], [B, EHN, L, HS]
        original_kv_states, extra_kv_states = (
            kv_states[:, : self.num_key_value_heads],
            kv_states[:, self.num_key_value_heads :],
        )
        # [B, KVHN, L, HS] -> [B, KVHN * G, L, HS] = [B, HN, L, HS]
        repeated = repeat_kv(original_kv_states, self.num_key_value_groups)
        # [B, HN, L, HS], [B, EHN, L, HS] -> [B, HN + EHN, L, HS]
        return torch.cat([repeated, extra_kv_states], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared_gate: torch.Tensor | None = None,
        adapter_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:
            raise NotImplementedError("output_attentions is not supported yet")

        batch_size, sequence_length, _ = hidden_states.size()

        # [B, L, H] -> [B, L, (HN + EHN) * HS]
        query_states = self._apply_adapter(hidden_states, "q")
        # [B, L, H] -> [B, L, (KVHN + EHN) * HS]
        key_states = self._apply_adapter(hidden_states, "k")
        # [B, L, H] -> [B, L, (KVHN + EHN) * HS]
        value_states = self._apply_adapter(hidden_states, "v")

        # [B, L, (HN + EHN) * HS] -> [B, HN + EHN, L, HS]
        query_states = query_states.view(
            batch_size, sequence_length, -1, self.head_dim
        ).transpose(1, 2)
        # [B, L, (KVHN + EHN) * HS] -> [B, KVHN + EHN, L, HS]
        key_states = key_states.view(
            batch_size, sequence_length, -1, self.head_dim
        ).transpose(1, 2)
        # [B, L, (KVHN + EHN) * HS] -> [B, KVHN + EHN, L, HS]
        value_states = value_states.view(
            batch_size, sequence_length, -1, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings is required")
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # [B, KVHN + EHN, L, HS] -> [B, HN + EHN, L, HS]
        key_states = self._repeat_kv_without_adapter(key_states)
        # [B, KVHN + EHN, L, HS] -> [B, HN + EHN, L, HS]
        value_states = self._repeat_kv_without_adapter(value_states)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged
        # with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels
        # via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options.
        # An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d
        # that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and sequence_length > 1 else False

        # [B, HN + EHN, L, HS], [B, HN + EHN, L, HS], [B, HN + EHN, L, HS] -> [B, HN + EHN, L, HS]  # noqa: E501
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # [B, HN + EHN, L, HS] -> [B, L, HN + EHN, HS] -> [B, L, (HN + EHN) * HS]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, -1)

        if shared_gate is not None and adapter_gate is not None:
            # [B, L, H + EHN * HS]
            adapter_mask = torch.concat(
                [
                    # [B, L, H]
                    # torch.ones(
                    #     batch_size,
                    #     sequence_length,
                    #     self.hidden_size,
                    #     device=attn_output.device,
                    #     dtype=attn_output.dtype,
                    # ),
                    shared_gate.expand(-1, -1, self.hidden_size),
                    *(
                        # [B, L, EHN * HS]
                        (
                            (1 - shared_gate) * (adapter_gate[:, :, idx].unsqueeze(-1))
                        ).expand(-1, -1, self.attn_adapter[name] * self.head_dim)
                        for idx, name in enumerate(self.attn_adapter)
                    ),
                ],
                dim=-1,
            )
            attn_output = attn_output * adapter_mask

        # [B, L, (HN + EHN) * HS] -> [B, L, H]
        attn_output = self._apply_adapter(attn_output, "o")

        return attn_output, None, past_key_value  # type: ignore


class Qwen2DecoderLayerAdapter(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen2AttentionAdapter(config, layer_idx)

        self.mlp = Qwen2MLPAdapter(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Added gate for adapters
        if getattr(config, "enable_adapter_gate", False):
            if GATE_DOWN_SIZE:
                self.down_gate = nn.Linear(
                    config.hidden_size, GATE_DOWN_SIZE, bias=False
                )
            else:
                self.down_gate = None
            adapter_types = get_adapter_types(config)
            self.adapter_gate = nn.Linear(
                GATE_DOWN_SIZE or self.hidden_size, len(adapter_types), bias=False
            )
            self.shared_gate = nn.Linear(
                GATE_DOWN_SIZE or self.hidden_size, 1, bias=False
            )
            # self.adapter_gate_topk = getattr(config, "adapter_gate_topk", 1)
        else:
            self.down_gate = None
            self.adapter_gate = None
            self.shared_gate = None
            # self.adapter_gate_topk = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """  # noqa: E501

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Patch for adapter gate
        if self.down_gate is not None:
            # [B, L, H] -> [B, L, GATE_DOWN_SIZE]
            down_states = self.down_gate(hidden_states)
        else:
            down_states = hidden_states

        if (
            self.adapter_gate is not None
            # and self.adapter_gate_topk is not None
            # and self.adapter_gate_topk > 0
        ):
            # [B, L, H] -> [B, L, AT]
            adapter_gate = F.softmax(self.adapter_gate(down_states), dim=-1)
            # # [B, L, TOPK], [B, L, TOPK]
            # selected_gate, selected_index = adapter_gate.topk(
            #     self.adapter_gate_topk, dim=-1
            # )
            adapter_gate = adapter_gate / adapter_gate.sum(dim=-1, keepdim=True)
            # # [B, L, AT]
            # adapter_gate = torch.zeros_like(adapter_gate)
            # # fill in the topk gate
            # adapter_gate.scatter_(-1, selected_index, selected_gate)
        else:
            adapter_gate = None

        # Shared gate
        if self.shared_gate is not None:
            # [B, L, H] -> [B, L, 1]
            shared_gate = F.sigmoid(self.shared_gate(down_states))
        else:
            shared_gate = None

        # store debug analysis info
        if (ctx := ANALYSIS_CONTEXT.get()) is not None:
            # we only need the last token's gate
            # [B, 1]
            if shared_gate is not None:
                if self.layer_idx not in ctx["shared_gate"]:
                    ctx["shared_gate"][self.layer_idx] = shared_gate[:, -1, :].cpu()
                else:
                    ctx["shared_gate"][self.layer_idx] = torch.concat(
                        (
                            ctx["shared_gate"][self.layer_idx],
                            shared_gate[:, -1, :].cpu(),
                        ),
                        dim=0,
                    )

            # [B, AT]
            if adapter_gate is not None:
                if self.layer_idx not in ctx["adapter_gate"]:
                    ctx["adapter_gate"][self.layer_idx] = adapter_gate[:, -1, :].cpu()
                else:
                    ctx["adapter_gate"][self.layer_idx] = torch.concat(
                        (
                            ctx["adapter_gate"][self.layer_idx],
                            adapter_gate[:, -1, :].cpu(),
                        ),
                        dim=0,
                    )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            shared_gate=shared_gate,
            adapter_gate=adapter_gate,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states, shared_gate=shared_gate, adapter_gate=adapter_gate
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


def get_adapter_state_dict(model: Qwen2PreTrainedModel) -> dict[str, Any]:
    adapter_types = get_adapter_types(cast(Config, model.config))
    state_dict = model.state_dict()
    adapter_modules = set(
        chain.from_iterable(
            (
                rf"layers\.[^.]+\.self_attn\.{name}_q_proj",
                rf"layers\.[^.]+\.self_attn\.{name}_k_proj",
                rf"layers\.[^.]+\.self_attn\.{name}_v_proj",
                rf"layers\.[^.]+\.self_attn\.{name}_o_proj",
                rf"layers\.[^.]+\.mlp\.{name}_gate",
                rf"layers\.[^.]+\.mlp\.{name}_up",
                rf"layers\.[^.]+\.mlp\.{name}_down",
                r"layers\.[^.]+\.down_gate",
                r"layers\.[^.]+\.adapter_gate",
                r"layers\.[^.]+\.shared_gate",
            )
            for name in adapter_types
        )
    )
    if not isinstance(model, Qwen2Model):
        adapter_modules = {rf"model\.{name}" for name in adapter_modules}

    return {
        name: tensor
        for name, tensor in state_dict.items()
        if any(re.match(pattern, name) for pattern in adapter_modules)
    }


def apply_adapter(
    model: MT,
    ffn_adapter: dict[str, int] | None = None,
    attn_adapter: dict[str, int] | None = None,
    enable_adapter_gate: bool | None = None,
    # adapter_gate_topk: int | None = None,
) -> MT:
    decoder_module: str = "layers" if issubclass(model, Qwen2Model) else "model.layers"

    origin_init = cast(type[Qwen2PreTrainedModel], model).__init__

    def patched_init(self: Qwen2PreTrainedModel, config: Qwen2Config):
        if ffn_adapter is not None:
            origin_ffn_adapter = getattr(config, "ffn_adapter", {})
            config.ffn_adapter = {**origin_ffn_adapter, **ffn_adapter}
        if attn_adapter is not None:
            origin_attn_adapter = getattr(config, "attn_adapter", {})
            config.attn_adapter = {**origin_attn_adapter, **attn_adapter}
        if enable_adapter_gate is not None:
            config.enable_adapter_gate = enable_adapter_gate
        # if adapter_gate_topk is not None:
        #     config.adapter_gate_topk = adapter_gate_topk

        origin_init(self, config)

        if (
            hasattr(config, "ffn_adapter")
            or hasattr(config, "attn_adapter")
            or hasattr(config, "enable_adapter_gate")
        ):
            nested_replace_module(
                self,
                decoder_module,
                lambda _, old: nn.ModuleList(
                    [
                        Qwen2DecoderLayerAdapter(cast(Config, config), layer_idx)
                        for layer_idx in range(config.num_hidden_layers)
                    ]
                ),
            )

        self.post_init()

    mc = type(
        f"{model.__name__}Adapter",
        (model,),
        {"__init__": patched_init, "get_adapter_state_dict": get_adapter_state_dict},
    )

    return mc  # type: ignore
