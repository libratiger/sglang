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

from typing import Optional, Tuple, Union

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput


class PipelineStageExecutor:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        tp_rank: int,
        nccl_port: int,
    ):
        self.model_config = model_config
        self.server_args = server_args
        self.tp_rank = tp_rank

        self.model_runner = ModelRunner(
            model_config,
            server_args.mem_fraction_static,
            server_args.base_gpu_id + tp_rank, # TODO: This needs to be global rank eventually for multi-node
            tp_rank, # local tp_rank within this stage's TP group
            server_args.tp_size, # tp_size for this stage
            nccl_port, # This might need to be more nuanced if TP groups have different nccl_ports
            server_args,
        )

        self.pipeline_stage_rank = model_config.pipeline_stage_rank
        self.pipeline_parallel_size = model_config.pipeline_parallel_size
        self.is_first_stage = self.pipeline_stage_rank == 0
        self.is_last_stage = (
            self.pipeline_stage_rank == self.pipeline_parallel_size - 1
        )

    def forward_pass(
        self,
        forward_batch: ForwardBatch,
        received_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor], Optional[LogitsProcessorOutput], Optional[torch.Tensor]
    ]:
        # If not the first stage, the received_hidden_states are the input_embeds for the current stage's model.
        # The model's forward method was modified to accept hidden_states_from_previous_stage,
        # which is internally passed as input_embeds to the underlying HuggingFace model if it's not stage 0.
        # So, we set forward_batch.hidden_states_from_previous_stage here.
        # The LlamaModel.forward will then use this.
        if not self.is_first_stage:
            forward_batch.hidden_states_from_previous_stage = received_hidden_states
            forward_batch.input_ids = None # Ensure input_ids are not used if hidden_states are provided
            forward_batch.input_embeds = None # Ensure input_embeds are not used if hidden_states are provided
        else:
            # For the first stage, input_ids or input_embeds (e.g. for multimodal) are already set in forward_batch
            forward_batch.hidden_states_from_previous_stage = None


        # ModelRunner.forward will call self.model.forward.
        # self.model.forward expects:
        #   - input_ids (if stage 0 and no input_embeds)
        #   - positions
        #   - forward_batch
        #   - input_embeds (if stage 0 and multimodal/pre-embedded)
        #   - hidden_states_from_previous_stage (if not stage 0)
        output = self.model_runner.forward(forward_batch)

        output_hidden_states: Optional[torch.Tensor] = None
        logits_output: Optional[LogitsProcessorOutput] = None
        next_token_ids: Optional[torch.Tensor] = None

        if not self.is_last_stage:
            # If not the last stage, the output of model.forward is the hidden_states tensor
            assert isinstance(output, torch.Tensor)
            output_hidden_states = output
        else:
            # If it's the last stage, the output of model.forward is LogitsProcessorOutput (or hidden_states if get_embedding=True)
            # For generation, we expect LogitsProcessorOutput
            assert isinstance(output, LogitsProcessorOutput)
            logits_output = output
            # Sampling is only done on the last stage
            next_token_ids = self.model_runner.sample(logits_output, forward_batch)

        return output_hidden_states, logits_output, next_token_ids

    def get_memory_pool(self):
        return self.model_runner.mem_manager.get_memory_pool()

    def get_worker_info(self, type: str = "json"):
        """Returns a dictionary containing information about the worker."""
        info = {
            "tp_rank": self.tp_rank,
            "pipeline_stage_rank": self.pipeline_stage_rank,
            "tp_size": self.server_args.tp_size, # This is the TP size per stage
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "gpu_mem_usage": self.model_runner.get_gpu_memory_usage(),
            "max_total_tokens": self.model_runner.max_total_tokens,
            "max_running_requests": self.server_args.max_running_requests,
            "context_len": self.model_config.context_len,
        }
        if type == "json":
            return info
        elif type == "html":
            html_output = "<ul>\n"
            for key, value in info.items():
                html_output += f"<li><b>{key}:</b> {value}</li>\n"
            html_output += "</ul>"
            return html_output
        else:
            return str(info)

    def profile_num_available_blocks(self):
        return self.model_runner.profile_num_available_blocks()

    def set_lock_weights(self, lock_weights):
        self.model_runner.set_lock_weights(lock_weights)
        
    def release_token_blocks(self, item_id):
        self.model_runner.mem_manager.release_token_blocks(item_id)

    def get_cache_block_size(self) -> int:
        return self.model_runner.get_cache_block_size()
        
    # TODO: Add other methods like load_lora, add_lora, remove_lora if needed for pipeline stages.
    # For now, these are typically managed at a higher level or might be broadcasted/coordinated.
    # If LoRA weights are part of the stage-specific model state, ModelRunner already handles them.
    # The main challenge would be coordinating LoRA adapter changes across stages.

    def __del__(self):
        # This might be tricky with CUDA contexts if not handled carefully.
        # For now, rely on Python's GC and ModelRunner's __del__ if any.
        pass
