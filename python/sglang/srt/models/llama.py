# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import make_layers
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Pipeline parallelism attributes
        self.pipeline_parallel_size = getattr(config, "pipeline_parallel_size", 1)
        self.pipeline_stage_rank = getattr(config, "pipeline_stage_rank", 0)
        self.layer_offset = getattr(config, "layer_offset", 0)
        self.effective_num_hidden_layers = getattr(config, "effective_num_hidden_layers", config.num_hidden_layers)

        if self.pipeline_stage_rank == 0:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = None

        self.layers = make_layers(
            self.effective_num_hidden_layers, # Use effective_num_hidden_layers
            lambda idx, prefix: LlamaDecoderLayer(
                config=config, quant_config=quant_config, layer_id=idx + self.layer_offset, prefix=prefix
            ),
            prefix="model.layers",
        )

        if self.pipeline_stage_rank == self.pipeline_parallel_size - 1:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        hidden_states_from_previous_stage: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pipeline_stage_rank == 0:
            if input_embeds is None:
                assert input_ids is not None
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
        else:
            assert hidden_states_from_previous_stage is not None
            hidden_states = hidden_states_from_previous_stage
            
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        
        if self.pipeline_stage_rank == self.pipeline_parallel_size - 1:
            assert self.norm is not None
            hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class LlamaForCausalLM(nn.Module):

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = LlamaModel(config, quant_config=quant_config)

        # Pipeline parallelism attributes
        self.pipeline_parallel_size = getattr(config, "pipeline_parallel_size", 1)
        self.pipeline_stage_rank = getattr(config, "pipeline_stage_rank", 0)

        if self.pipeline_stage_rank == self.pipeline_parallel_size - 1:
            # Llama 3.2 1B Insturct set tie_word_embeddings to True
            # Llama 3.1 8B Insturct set tie_word_embeddings to False
            if self.config.tie_word_embeddings:
                # This assumes embed_tokens is available on the last stage if tied.
                # For strict pipeline, embed_tokens is only on stage 0.
                # If tied and last stage is not stage 0, this will need adjustment
                # or ensure embed_tokens is passed/replicated.
                # For now, let's assume if tied, the last stage can access it,
                # which might mean pipeline_parallel_size = 1 for tied embeddings,
                # or specific handling for passing embed_tokens.
                # A simple fix for now: only tie if it's a single stage.
                if self.pipeline_parallel_size == 1:
                    assert self.model.embed_tokens is not None
                    self.lm_head = self.model.embed_tokens
                else:
                    # If pipelined and tied, this needs a more complex solution.
                    # For now, default to a separate LM head if pipelined and tied.
                    # This might not be what's intended by "tie_word_embeddings" in a pipeline.
                    logger.warning(
                        "Pipeline parallelism is enabled and tie_word_embeddings is True, "
                        "but the last stage does not have direct access to embed_tokens from stage 0 "
                        "without specific inter-stage communication for it. "
                        "Defaulting to a separate lm_head on the last stage. "
                        "This might deviate from the intended tied weights behavior."
                    )
                    self.lm_head = ParallelLMHead(
                        config.vocab_size, config.hidden_size, quant_config=quant_config
                    )
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size, quant_config=quant_config
                )
            self.logits_processor = LogitsProcessor(config)
        else:
            self.lm_head = None
            self.logits_processor = None
            
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        hidden_states_from_previous_stage: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> Union[LogitsProcessorOutput, torch.Tensor]:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, hidden_states_from_previous_stage
        )

        if self.pipeline_stage_rank == self.pipeline_parallel_size - 1:
            assert self.logits_processor is not None and self.lm_head is not None
            if not get_embedding:
                return self.logits_processor(
                    input_ids, hidden_states, self.lm_head, forward_batch
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            # Return hidden states for the next stage
            return hidden_states

    def get_hidden_dim(self, module_name):
        # return input_dim, output_dim
        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
            return self.config.hidden_size, self.config.hidden_size
        elif module_name in ["kv_proj"]:
            return self.config.hidden_size, self.config.hidden_size // (
                self.config.num_attention_heads // self.config.num_key_value_heads
            )
        elif module_name == "gate_up_proj":
            return self.config.hidden_size, self.config.intermediate_size
        elif module_name == "down_proj":
            return self.config.intermediate_size, self.config.hidden_size
        else:
            raise NotImplementedError()

    def get_module_name(self, name):
        params_mapping = {
            "q_proj": "qkv_proj",
            "k_proj": "qkv_proj",
            "v_proj": "qkv_proj",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
        }
        return params_mapping.get(name, name)

    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # Pipeline parallelism: filter weights based on stage
            original_name = name
            is_layer_weight = name.startswith("model.layers.")
            is_embed_tokens_weight = name.startswith("model.embed_tokens.")
            is_norm_weight = name.startswith("model.norm.")
            is_lm_head_weight = name.startswith("lm_head.")
            
            if is_embed_tokens_weight and self.pipeline_stage_rank != 0:
                continue
            if (is_norm_weight or is_lm_head_weight) and self.pipeline_stage_rank != self.pipeline_parallel_size - 1:
                continue
            
            if is_layer_weight:
                try:
                    layer_idx_str = name.split(".")[2]
                    original_layer_idx = int(layer_idx_str)
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse layer index from {name}: {e}. Skipping weight.")
                    continue

                if not (self.model.layer_offset <= original_layer_idx < self.model.layer_offset + self.model.effective_num_hidden_layers):
                    continue  # Skip this weight, it's for a different stage
                
                local_layer_idx = original_layer_idx - self.model.layer_offset
                parts = name.split(".")
                parts[2] = str(local_layer_idx)
                name = ".".join(parts)

            # Original stacking logic
            processed_by_stacking = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name: # Use original_name for matching stacked patterns
                    continue
                
                # Apply layer remapping to the base name before appending stack-specific parts
                current_base_name = name.split(param_name.split(".")[-1])[0] + param_name.split(".")[-1]
                
                # Reconstruct the correct parameter name for the current stage
                # This is tricky if param_name itself is for a layer, e.g. "model.layers.X.self_attn.qkv_proj"
                # The 'name' variable should already be remapped if it was a layer weight.
                # So, if 'name' was "model.layers.0.self_attn.q_proj" (after remapping),
                # and param_name is ".qkv_proj", weight_name is ".q_proj",
                # it should become "model.layers.0.self_attn.qkv_proj"
                
                # Let's ensure the 'name' used for params_dict lookup is correctly formed
                # If 'name' was remapped: "model.layers.0.some_suffix"
                # And weight_name is ".q_proj", param_name is ".qkv_proj"
                # We need to replace ".q_proj" with ".qkv_proj" in the remapped name.
                # This requires careful handling of how 'name' is modified.
                
                # The 'name' variable has already been remapped for layer index.
                # Now, apply the stacking transformation.
                final_param_name = name.replace(weight_name, param_name)

                if final_param_name.endswith(".bias") and final_param_name not in params_dict:
                    continue
                
                if final_param_name not in params_dict:
                    # This can happen if the module itself is not on this stage (e.g. lm_head on non-last stage)
                    # print(f"Skipping {final_param_name} as it is not in params_dict for stage {self.pipeline_stage_rank}")
                    continue

                param = params_dict[final_param_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                processed_by_stacking = True
                break
            
            if not processed_by_stacking:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                
                if name not in params_dict:
                    # Similar to above, module might not be on this stage
                    # print(f"Skipping {name} as it is not in params_dict for stage {self.pipeline_stage_rank}")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            # Adjust for pipeline parallelism: check if the requested weight belongs to this stage
            is_layer_weight = name.startswith("model.layers.")
            is_embed_tokens_weight = name.startswith("model.embed_tokens.")
            is_norm_weight = name.startswith("model.norm.")
            is_lm_head_weight = name.startswith("lm_head.")

            if is_embed_tokens_weight and self.pipeline_stage_rank != 0:
                logger.info(f"Weight {name} skipped: embed_tokens is on stage 0, current stage is {self.pipeline_stage_rank}")
                return None
            if (is_norm_weight or is_lm_head_weight) and self.pipeline_stage_rank != self.pipeline_parallel_size - 1:
                logger.info(f"Weight {name} skipped: norm/lm_head is on last stage, current stage is {self.pipeline_stage_rank}")
                return None

            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                if self.pipeline_stage_rank == self.pipeline_parallel_size - 1:
                    if self.pipeline_parallel_size == 1 : # Only if single stage
                        logger.info(
                            "word embedding is tied for this model (single stage), return embed_tokens.weight as lm_head.weight."
                        )
                        assert self.model.embed_tokens is not None
                        return (
                            self.model.embed_tokens.weight.cpu()
                            .to(torch.float32)
                            .numpy()
                            .tolist()[:truncate_size]
                        )
                    else: # Pipelined and tied, lm_head is separate or needs special handling
                        logger.info(
                            "word embedding is tied but model is pipelined. lm_head.weight is distinct on the last stage."
                        )
                        # Fall through to load the potentially separate lm_head.weight
                else: # Not last stage, lm_head is None
                     logger.info(f"Weight {name} skipped: lm_head is on last stage, current stage is {self.pipeline_stage_rank}")
                     return None


            mapped_name = name
            mapped_shard_id = None
            
            if is_layer_weight:
                try:
                    original_layer_idx = int(name.split(".")[2])
                    if not (self.model.layer_offset <= original_layer_idx < self.model.layer_offset + self.model.effective_num_hidden_layers):
                        logger.info(f"Weight {name} skipped: layer {original_layer_idx} is not on this stage ({self.pipeline_stage_rank})")
                        return None
                    local_layer_idx = original_layer_idx - self.model.layer_offset
                    parts = name.split(".")
                    parts[2] = str(local_layer_idx)
                    mapped_name = ".".join(parts)
                except (IndexError, ValueError):
                    logger.error(f"Could not parse layer index from {name} for get_weights_by_name.")
                    return None
            else:
                mapped_name = name # Ensure mapped_name is assigned even if not a layer weight


            # Use original name for matching stacked patterns, but remapped name for dict lookup
            original_name_for_stacking = name 
            
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in original_name_for_stacking:
                    # Apply stacking transformation to the potentially layer-remapped name
                    mapped_name = mapped_name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            
            params_dict = dict(self.named_parameters())
            if mapped_name not in params_dict:
                logger.info(f"Weight {mapped_name} (original: {name}) not found in params_dict for stage {self.pipeline_stage_rank}.")
                return None

            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None


class Phi3ForCausalLM(LlamaForCausalLM):
    pass


EntryClass = [LlamaForCausalLM, Phi3ForCausalLM]
