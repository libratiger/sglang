# Copyright 2024 SGLang Team
#
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

"""FlexAttention backend."""
import logging
from typing import Optional

import torch

# Try to import flex_attention, note if not found
try:
    from torch.nn.attention import flex_attention
except ImportError:
    flex_attention = None
    logging.warning("torch.nn.attention.flex_attention not found. FlexAttention backend will not work.")


from sglang.srt.layers.attention.attention_backend import AttentionBackend
from sglang.srt.layers.attention.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_config import ModelConfig

# Make ModelRunner available for type checking without circular import
if False:  # TYPE_CHECKING
    from sglang.srt.model_runner import ModelRunner


class FlexAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_config: ModelConfig,
        name: str = "flex",
        num_heads: int = -1,
        head_dim: int = -1,
        scale: float = -1.0,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
    ):
        super().__init__(
            model_config,
            name,
            num_heads,
            head_dim,
            scale,
            num_qo_heads,
            num_kv_heads,
            sliding_window_size,
        )
        self.device = torch.device(model_config.device)
        if flex_attention is None:
            raise ImportError(
                "torch.nn.attention.flex_attention is not available. "
                "Please install a compatible version of PyTorch."
            )

    def init_forward_metadata(self, batch: ForwardBatch):
        """Can be a pass-through for now."""
        pass

    def init_cuda_graph_state(self, model_runner: "ModelRunner"):
        raise ValueError("FlexAttention backend does not support CUDA graph.")

    def init_forward_metadata_capture_cuda_graph(
        self, batch: ForwardBatch, model_runner: "ModelRunner"
    ):
        raise ValueError("FlexAttention backend does not support CUDA graph.")

    def init_forward_metadata_replay_cuda_graph(
        self, batch: ForwardBatch, model_runner: "ModelRunner"
    ):
        raise ValueError("FlexAttention backend does not support CUDA graph.")

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        raise ValueError("FlexAttention backend does not support CUDA graph.")

    def _run_sdpa_forward_extend(
        self,
        q_input: torch.Tensor, # [total_num_tokens, num_q_heads, head_dim]
        k_new_input: torch.Tensor, # [total_num_tokens, num_kv_heads, head_dim]
        v_new_input: torch.Tensor, # [total_num_tokens, num_kv_heads, head_dim]
        o_output: torch.Tensor, # [total_num_tokens, num_q_heads, head_dim]
        batch: ForwardBatch,
        radix_wrapper: RadixAttention,
    ):
        if flex_attention is None:
            raise RuntimeError("torch.nn.attention.flex_attention is not available.")

        k_cache = batch.token_to_kv_pool.get_key_buffer(radix_wrapper.layer_id)
        v_cache = batch.token_to_kv_pool.get_value_buffer(radix_wrapper.layer_id)
        req_to_token = batch.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices
        extend_seq_lens = batch.extend_seq_lens
        extend_prefix_lens = batch.extend_prefix_lens
        
        scaling = radix_wrapper.scaling
        # Determine if GQA is being used
        is_gqa = radix_wrapper.tp_q_head_num != radix_wrapper.tp_k_head_num
        # Determine causality for extend phase (prefill)
        effective_is_causal = not radix_wrapper.is_cross_attention

        token_start = 0
        for i in range(batch.num_seqs):
            req_idx = req_pool_indices[i]
            current_prefix_len = extend_prefix_lens[i].item() # Used for mask generation
            current_seq_len = extend_seq_lens[i].item()      # Used for mask generation & slicing
            
            cur_q = q_input[token_start : token_start + current_seq_len]
            cur_k_new = k_new_input[token_start : token_start + current_seq_len]
            cur_v_new = v_new_input[token_start : token_start + current_seq_len]

            if current_prefix_len == 0:
                cur_k = cur_k_new
                cur_v = cur_v_new
            else:
                buffer_indices = req_to_token[req_idx, :current_prefix_len]
                cur_k_cache = k_cache[buffer_indices]
                cur_v_cache = v_cache[buffer_indices]
                cur_k = torch.cat([cur_k_cache, cur_k_new], dim=0)
                cur_v = torch.cat([cur_v_cache, cur_v_new], dim=0)
            
            # Reshape for flex_attention: [batch_size, num_heads, seq_len, head_dim]
            # Here, batch_size is 1 as we process per sequence.
            cur_q_flex = cur_q.movedim(0, 1).unsqueeze(0) # [1, num_q_heads, seq_len, head_dim]
            cur_k_flex = cur_k.movedim(0, 1).unsqueeze(0) # [1, num_kv_heads, prefix_len + seq_len, head_dim]
            cur_v_flex = cur_v.movedim(0, 1).unsqueeze(0) # [1, num_kv_heads, prefix_len + seq_len, head_dim]

            # Note: is_causal for extend means causal within the new tokens, and new tokens attend to prefix tokens.
            # flex_attention's is_causal applies to the whole sequence.
            # If prefix_len > 0, this means the attention is not fully causal in the traditional sense for the combined sequence.
            # However, for prefill, we usually want causal masking for the *entire* sequence being processed.
            # If prefill has a prefix, it means we are re-prefilling, which is unusual.
            # Standard prefill: prefix_len = 0, seq_len = num_tokens_in_prompt. is_causal=True.
            # Standard decode: prefix_len = previous_total_len, seq_len = 1. is_causal=False.
            # The `is_causal` flag passed to flex_attention should be True for prefill, False for decode.
            # The current `effective_is_causal` (not radix_wrapper.is_cross_attention) is correct for prefill.

            current_score_mod = None
            if effective_is_causal:
                q_actual_len = cur_q_flex.size(2) # This is current_seq_len
                kv_actual_len = cur_k_flex.size(2) # This is current_prefix_len + current_seq_len
                
                attn_mask = torch.zeros(q_actual_len, kv_actual_len, device=cur_q.device, dtype=cur_q.dtype)
                for r in range(q_actual_len): # Iterate over each query token in the current prefill block
                    # Query token r (0-indexed within cur_q_flex) corresponds to actual token index current_prefix_len + r
                    # It cannot attend to key tokens with index > current_prefix_len + r
                    # So, for key token j, if j > current_prefix_len + r, mask it.
                    # The +1 is because slice upper bound is exclusive.
                    mask_start_idx = current_prefix_len + r + 1
                    if mask_start_idx < kv_actual_len:
                        attn_mask[r, mask_start_idx:] = -float('inf')
                current_score_mod = attn_mask
            
            cur_o_flex = flex_attention(
                cur_q_flex,
                cur_k_flex,
                cur_v_flex,
                score_mod=current_score_mod,
                block_mask=None,
                scale=scaling,
                enable_gqa=is_gqa,
                return_lse=False,
                kernel_options=None
                # is_causal parameter removed
            )
            
            # Reshape output back: [seq_len, num_q_heads, head_dim]
            o_output[token_start : token_start + current_seq_len] = cur_o_flex.squeeze(0).movedim(1, 0)
            token_start += current_seq_len

    def _run_sdpa_forward_decode(
        self,
        q_input: torch.Tensor,  # [num_seqs, num_q_heads, head_dim]
        # k_new_input and v_new_input are not explicitly used here as KV cache is already updated.
        k_unused: torch.Tensor,
        v_unused: torch.Tensor,
        o_output: torch.Tensor,  # [num_seqs, num_q_heads, head_dim]
        batch: ForwardBatch,
        radix_wrapper: RadixAttention,
    ):
        if flex_attention is None:
            raise RuntimeError("torch.nn.attention.flex_attention is not available.")

        k_cache = batch.token_to_kv_pool.get_key_buffer(radix_wrapper.layer_id)
        v_cache = batch.token_to_kv_pool.get_value_buffer(radix_wrapper.layer_id)
        req_to_token = batch.req_to_token_pool.req_to_token
        req_pool_indices = batch.req_pool_indices
        seq_lens = batch.seq_lens # These are full sequence lengths including the new token

        scaling = radix_wrapper.scaling
        # Determine if GQA is being used
        is_gqa = radix_wrapper.tp_q_head_num != radix_wrapper.tp_k_head_num
        # For decode, attention is never causal in the SDPA sense for the whole sequence.
        # Query length is 1, so no causal masking needed via score_mod.
        # effective_is_causal = False (already established, and not needed for score_mod here)

        for i in range(batch.num_seqs):
            req_idx = req_pool_indices[i]
            current_total_seq_len = seq_lens[i].item() # Full sequence length for K/V

            cur_q = q_input[i:i+1] # [1, num_q_heads, head_dim]
            
            # Get KV for the full sequence from cache
            buffer_indices = req_to_token[req_idx, :current_total_seq_len]
            cur_k = k_cache[buffer_indices] # [current_total_seq_len, num_kv_heads, head_dim]
            cur_v = v_cache[buffer_indices] # [current_total_seq_len, num_kv_heads, head_dim]

            # Reshape for flex_attention: [batch_size, num_heads, q_len, head_dim] for Q
            # [batch_size, num_heads, kv_len, head_dim] for K, V
            # Here, batch_size is 1, q_len is 1.
            cur_q_flex = cur_q.movedim(0, 1).unsqueeze(0) # [1, num_q_heads, 1, head_dim]
            cur_k_flex = cur_k.movedim(0, 1).unsqueeze(0) # [1, num_kv_heads, current_total_seq_len, head_dim]
            cur_v_flex = cur_v.movedim(0, 1).unsqueeze(0) # [1, num_kv_heads, current_total_seq_len, head_dim]
            
            cur_o_flex = flex_attention(
                cur_q_flex,
                cur_k_flex,
                cur_v_flex,
                score_mod=None, # No causal mask needed for decode (q_len=1)
                block_mask=None,
                scale=scaling,
                enable_gqa=is_gqa,
                return_lse=False,
                kernel_options=None
                # is_causal parameter removed
            )
            
            # Reshape output back: [1, num_q_heads, head_dim]
            o_output[i:i+1] = cur_o_flex.squeeze(0).movedim(1, 0)


    def forward_extend(
        self,
        q: torch.Tensor, # [total_num_tokens, num_q_heads * head_dim]
        k: torch.Tensor, # [total_num_tokens, num_kv_heads * head_dim]
        v: torch.Tensor, # [total_num_tokens, num_kv_heads * head_dim]
        o: torch.Tensor, # [total_num_tokens, num_q_heads * head_dim]
        batch: ForwardBatch,
        radix_wrapper: RadixAttention,
    ):
        q_ = q.view(-1, radix_wrapper.tp_q_head_num, radix_wrapper.qk_head_dim)
        k_new_ = k.view(-1, radix_wrapper.tp_k_head_num, radix_wrapper.qk_head_dim)
        v_new_ = v.view(-1, radix_wrapper.tp_k_head_num, radix_wrapper.v_head_dim)
        o_ = o.view(-1, radix_wrapper.tp_q_head_num, radix_wrapper.qk_head_dim)

        self._run_sdpa_forward_extend(q_, k_new_, v_new_, o_, batch, radix_wrapper)
        
        # Save new KV to cache
        # Note: k_new_ and v_new_ are already in the correct shape for the cache
        # [total_num_tokens, num_kv_heads, head_dim]
        batch.token_to_kv_pool.set_kv_buffer(radix_wrapper.layer_id, k_new_, v_new_)


    def forward_decode(
        self,
        q: torch.Tensor, # [num_seqs, num_q_heads * head_dim]
        k: torch.Tensor, # [num_seqs, num_kv_heads * head_dim] (new k for the current token)
        v: torch.Tensor, # [num_seqs, num_kv_heads * head_dim] (new v for the current token)
        o: torch.Tensor, # [num_seqs, num_q_heads * head_dim]
        batch: ForwardBatch,
        radix_wrapper: RadixAttention,
    ):
        q_ = q.view(-1, radix_wrapper.tp_q_head_num, radix_wrapper.qk_head_dim)
        # k_new_ and v_new_ are for the single new token per sequence
        k_new_ = k.view(-1, radix_wrapper.tp_k_head_num, radix_wrapper.qk_head_dim)
        v_new_ = v.view(-1, radix_wrapper.tp_k_head_num, radix_wrapper.v_head_dim)
        o_ = o.view(-1, radix_wrapper.tp_q_head_num, radix_wrapper.qk_head_dim)

        # Save new KV to cache first for decode
        # This ensures that when _run_sdpa_forward_decode fetches from cache, it gets the latest KV
        batch.token_to_kv_pool.set_kv_buffer(radix_wrapper.layer_id, k_new_, v_new_)

        # k_new_ and v_new_ are passed to _run_sdpa_forward_decode but not directly used for flex_attention call's K/V arguments,
        # as the full K/V sequence (including new ones) is fetched from cache.
        # They are passed as k_unused, v_unused for signature consistency if needed later.
        self._run_sdpa_forward_decode(q_, k_new_, v_new_, o_, batch, radix_wrapper)

    def get_alibi_bias(self, batch: ForwardBatch) -> Optional[torch.Tensor]:
        # FlexAttention does not inherently support ALiBi,
        # but specific implementations might add it.
        # Returning None as a general case.
        return None

    def close_radix_wrapper(self):
        # If a RadixAttention wrapper is used, this method would handle its cleanup.
        # For now, it's a pass-through.
        pass
