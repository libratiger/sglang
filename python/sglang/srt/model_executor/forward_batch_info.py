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
"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

if TYPE_CHECKING:
    from sglang.srt.layers.attention import AttentionBackend
    from sglang.srt.managers.schedule_batch import ImageInputs, ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None

    # For multimodal
    image_inputs: Optional[List[ImageInputs]] = None

    # Encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For LoRA
    lora_paths: Optional[List[str]] = None

    # For input embeddings
    input_embeds: Optional[torch.tensor] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: AttentionBackend = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    gathered_buffer: Optional[torch.Tensor] = None
    can_run_dp_cuda_graph: bool = False

    def compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        device = model_runner.device
        hf_config = model_runner.model_config.hf_config
        mrope_positions_list = [None] * self.seq_lens.shape[0]
        if self.forward_mode.is_decode():
            for i, _ in enumerate(mrope_positions_list):
                mrope_position_delta = (
                    0
                    if batch.image_inputs[i] is None
                    else batch.image_inputs[i].mrope_position_delta
                )
                mrope_positions_list[i] = MRotaryEmbedding.get_next_input_positions(
                    mrope_position_delta,
                    int(self.seq_lens[i]) - 1,
                    int(self.seq_lens[i]),
                )
        elif self.forward_mode.is_extend():
            extend_start_loc_cpu = self.extend_start_loc.cpu().numpy()
            for i, image_inputs in enumerate(batch.image_inputs):
                extend_start_loc, extend_seq_len, extend_prefix_len = (
                    extend_start_loc_cpu[i],
                    batch.extend_seq_lens[i],
                    batch.extend_prefix_lens[i],
                )
                if image_inputs is None:
                    # text only
                    mrope_positions = [
                        [
                            pos
                            for pos in range(
                                extend_prefix_len, extend_prefix_len + extend_seq_len
                            )
                        ]
                    ] * 3
                else:
                    # TODO: current qwen2-vl do not support radix cache since mrope position calculation
                    mrope_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=self.input_ids[
                                extend_start_loc : extend_start_loc + extend_seq_len
                            ],
                            image_grid_thw=image_inputs.image_grid_thws,
                            vision_start_token_id=hf_config.vision_start_token_id,
                            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                            context_len=0,
                        )
                    )
                    batch.image_inputs[i].mrope_position_delta = mrope_position_delta
                mrope_positions_list[i] = mrope_positions

        self.mrope_positions = torch.concat(
            [torch.tensor(pos, device=device) for pos in mrope_positions_list],
            axis=1,
        )
        self.mrope_positions = self.mrope_positions.to(torch.int64)

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch, # ModelWorkerBatch now includes stage-specific input_ids and other details
        model_runner: ModelRunner, # ModelRunner for the current stage
    ):
        device = model_runner.device
        current_stage_rank = model_runner.model_config.pipeline_stage_rank
        # pipeline_parallel_size = model_runner.model_config.pipeline_parallel_size # Not directly used here but good for context

        # hidden_states_from_previous_stage is set by PipelineStageExecutor directly on ForwardBatch instance
        # It is not part of ModelWorkerBatch.
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.model_worker_reqs) if batch.model_worker_reqs is not None else 0,
            input_ids=None, 
            req_pool_indices=batch.req_pool_indices,
            seq_lens=None, # This will be the GLOBAL seq_len for KV cache, set below.
            out_cache_loc=batch.out_cache_loc, 
            image_inputs=batch.image_inputs if current_stage_rank == 0 else None, # Image inputs only for stage 0
            encoder_cached=batch.encoder_cached, 
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=0, 
            return_logprob=batch.return_logprob, 
            top_logprobs_nums=batch.top_logprobs_nums, 
            global_num_tokens=batch.global_num_tokens, 
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph, 
            lora_paths=batch.lora_paths, 
            sampling_info=batch.sampling_info, 
            input_embeds=batch.input_embeds if current_stage_rank == 0 else None,
        )

        if ret.global_num_tokens is not None: 
            max_len = max(ret.global_num_tokens)
            ret.gathered_buffer = torch.zeros(
                (max_len * model_runner.tp_size, model_runner.model_config.hidden_size),
                dtype=model_runner.dtype,
                device=device,
            )

        if ret.forward_mode.is_idle():
            return ret

        if ret.forward_mode.is_extend():
            input_ids_for_stage_list = []
            positions_for_stage_list = []
            current_extend_seq_lens_list = [] 
            # extend_prefix_lens for attention within this stage's current computation chunk.
            current_extend_prefix_lens_list = [] 
            
            seq_lens_for_kv_cache_list = [] # Global length for KV cache addressing

            assert batch.model_worker_reqs is not None, "model_worker_reqs must be provided for extend mode"

            for req_idx, req in enumerate(batch.model_worker_reqs):
                full_input_ids = req.origin_input_ids + req.output_ids
                global_radix_prefix_len = len(req.prefix_indices) 
                
                tokens_to_prefill_globally = full_input_ids[global_radix_prefix_len:]
                
                # This is how many tokens of the "tokens_to_prefill_globally" part this stage has already processed.
                start_offset_in_global_prefill_chunk = req.processed_token_count_by_pipeline_stages[current_stage_rank]
                
                chunk_size = model_runner.server_args.chunked_prefill_size
                if chunk_size == -1: # Process all remaining for this request globally
                    num_tokens_for_stage_chunk = len(tokens_to_prefill_globally) - start_offset_in_global_prefill_chunk
                else: # Process one chunk
                    num_tokens_for_stage_chunk = min(chunk_size, 
                                               len(tokens_to_prefill_globally) - start_offset_in_global_prefill_chunk)

                if num_tokens_for_stage_chunk <= 0: # This req is done with prefill on this stage or globally
                    current_extend_seq_lens_list.append(0)
                    current_extend_prefix_lens_list.append(0)
                    # KV cache already contains up to global_radix_prefix_len + start_offset_in_global_prefill_chunk
                    seq_lens_for_kv_cache_list.append(global_radix_prefix_len + start_offset_in_global_prefill_chunk)
                    if not positions_for_stage_list and req_idx == 0 : 
                        positions_for_stage_list.append(torch.empty(0, dtype=torch.long, device=device))
                    continue

                actual_input_ids_for_chunk = tokens_to_prefill_globally[
                    start_offset_in_global_prefill_chunk : 
                    start_offset_in_global_prefill_chunk + num_tokens_for_stage_chunk
                ]
                # Only stage 0 uses input_ids for embedding lookup. Other stages use hidden_states.
                # However, model.forward might still expect input_ids for shape or other reasons.
                # For simplicity, we pass the actual chunk of tokens for stage 0,
                # and dummy tokens (or actual tokens if needed by model arch) for other stages.
                # The Llama model in SGLang is modified to accept hidden_states_from_previous_stage.
                if current_stage_rank == 0:
                    input_ids_for_stage_list.extend(actual_input_ids_for_chunk)
                else: 
                    # For stages > 0, if the model uses hidden_states, input_ids might not be strictly needed
                    # for computation but could be for length/shape. Using actual tokens or dummies depends
                    # on specific model implementation. For now, assume actual tokens are passed but might not be used for embedding.
                    input_ids_for_stage_list.extend(actual_input_ids_for_chunk)


                global_start_pos_for_chunk = global_radix_prefix_len + start_offset_in_global_prefill_chunk
                positions_for_chunk = torch.arange(
                    global_start_pos_for_chunk,
                    global_start_pos_for_chunk + num_tokens_for_stage_chunk,
                    device=device, dtype=torch.long
                )
                positions_for_stage_list.append(positions_for_chunk)
                
                current_extend_seq_lens_list.append(num_tokens_for_stage_chunk)
                
                # The "prefix_len" for this stage's attention computation on the current chunk.
                if current_stage_rank == 0:
                    # Stage 0: prefix is the RadixCache hit. Tokens are new.
                    current_extend_prefix_lens_list.append(global_radix_prefix_len)
                else:
                    # Stage > 0: receives hidden states for (Radix prefix + previous chunks).
                    # The current chunk of tokens is new to this stage, so prefix for *this chunk's computation* is 0.
                    current_extend_prefix_lens_list.append(0) 
                                                              
                seq_lens_for_kv_cache_list.append(global_start_pos_for_chunk + num_tokens_for_stage_chunk)

            if input_ids_for_stage_list:
                ret.input_ids = torch.tensor(input_ids_for_stage_list, dtype=torch.int32).to(device)
                if positions_for_stage_list: 
                    ret.positions = torch.cat(positions_for_stage_list)
                else: 
                     ret.positions = torch.empty(0, dtype=torch.long, device=device)
            else: 
                ret.input_ids = torch.empty(0, dtype=torch.int32, device=device)
                ret.positions = torch.empty(0, dtype=torch.long, device=device)

            ret.extend_seq_lens = torch.tensor(current_extend_seq_lens_list, dtype=torch.int32).to(device)
            ret.extend_prefix_lens = torch.tensor(current_extend_prefix_lens_list, dtype=torch.int32).to(device) # Stage-local prefix for attention
            
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens)
            if len(ret.extend_seq_lens) > 1: 
                ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            
            ret.seq_lens = torch.tensor(seq_lens_for_kv_cache_list, dtype=torch.int32).to(device) # Global KV cache length
            ret.extend_num_tokens = sum(current_extend_seq_lens_list) # Number of tokens this stage processes in this batch
            ret.seq_lens_sum = ret.seq_lens.sum().item() if ret.seq_lens.numel() > 0 else 0

            ret.extend_prefix_lens_cpu = current_extend_prefix_lens_list 
            ret.extend_seq_lens_cpu = current_extend_seq_lens_list
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens # Needs stage-specific logic if used beyond stage 0

        elif ret.forward_mode.is_decode():
            ret.positions = (batch.seq_lens - 1).to(torch.int64)
            # input_ids for decode is the single token. For stage 0 or if model needs it.
            # Other stages primarily use hidden_states.
            ret.input_ids = batch.input_ids # This comes from ModelWorkerBatch, should be single token for last stage.
                                            # For intermediate stages, this might be None if model doesn't need it.
            ret.seq_lens = batch.seq_lens # Global seq_len for KV cache
            ret.seq_lens_sum = ret.seq_lens.sum().item() if ret.seq_lens.numel() > 0 else 0

        elif ret.forward_mode.is_mixed():
            # This mode is complex and assumes ModelWorkerBatch is correctly populated by Scheduler
            # This means model_worker_batch.input_ids, .positions etc. are already segmented for this stage.
            ret.input_ids = batch.input_ids 
            ret.positions = batch.positions 
            
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True) 
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True) 
            
            ret.extend_num_tokens = batch.extend_num_tokens 
            
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens) 
            if len(ret.extend_seq_lens) > 1:
                 ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            
            ret.seq_lens = batch.seq_lens 
            ret.seq_lens_sum = ret.seq_lens.sum().item() if ret.seq_lens.numel() > 0 else 0
            
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens


        if model_runner.model_is_mrope:
            if batch.mrope_positions is not None: 
                 ret.mrope_positions = batch.mrope_positions
            elif ret.positions is not None : 
                 # Fallback for mROPE if not explicitly provided, might need scheduler to ensure correctness
                 # For simplicity, this doesn't re-calculate here but relies on batch.mrope_positions
                 pass


        # hidden_states_from_previous_stage will be set by PipelineStageExecutor if not the first stage
        # ret.hidden_states_from_previous_stage = batch.hidden_states_from_previous_stage # This is NOT from ModelWorkerBatch

        ret.req_to_token_pool = model_runner.req_to_token_pool 
        ret.token_to_kv_pool = model_runner.token_to_kv_pool
        ret.attn_backend = model_runner.attn_backend

        # Init lora information
        if model_runner.server_args.lora_paths is not None:
            model_runner.lora_manager.prepare_lora_batch(ret)

        return ret


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    prefix_len = tl.load(extend_prefix_lens + pid)
    seq_len = tl.load(extend_seq_lens + pid)

    # TODO: optimize this?
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.concat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc
