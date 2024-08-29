import os
import torch
from torch import nn
import torch.distributed as dist
from typing import List, Union
import flux
from flux import pynvshmem
from flux import moe_utils
from contextlib import contextmanager, nullcontext
import random
from random import randint
import time
from datetime import timedelta

import packaging
import packaging.version
import argparse

from megatron.training.arguments import parse_args
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoELayer_wo_gate, MoELayer_wo_gate_v2, MoELayer_wo_gate_v3
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_expert_model_parallel_group,
    get_expert_model_parallel_world_size
)

from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelGroupedLinear
from megatron.test_forward.initialize import set_jit_fusion_options, initialize_megatron
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.global_vars import set_global_variables

def timed__all_gather_base(tensor_list, tensor, group=None, async_op=False,
                            timer_name=None):
    work = torch.distributed._all_gather_base(tensor_list, tensor, group, async_op)
    return work

def timed__reduce_scatter_base(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False,
                                timer_name=None):
    work = torch.distributed._reduce_scatter_base(output, input, op, group, async_op)
    return work

def timed_broadcast(tensor, src, group=None, async_op=False, timer_name=None):
    work = torch.distributed.broadcast(tensor, src, group, async_op)
    return work

def timed_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False,
                        timer_name=None):
    work = dist.all_reduce(tensor, op, group, async_op)
    return work


INP_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8": torch.float8_e4m3fn,
}

OUT_DTYPE_MAP = {
    "bfloat16": torch.bfloat16, 
    "float16": torch.float16, 
    "float8": torch.bfloat16
}

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DIST_ENV = flux.get_dist_env(deterministic=False)
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)

class MoE_layer_megatron(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts, moe_grouped_gemm=True)
        self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, input):
        result, _ = self._moe_layer(input)
        return result


class MoE_layer_megatron_wo_gate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts, moe_grouped_gemm=True)
        # self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)
        self._moe_layer = MoELayer_wo_gate(config, submodules=transformer_layer_spec.submodules.mlp.submodules)
        # self._moe_layer = MoELayer_wo_gate_v2(config, submodules=transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, dispatched_input, tokens_per_expert):
        result = self._moe_layer(dispatched_input, tokens_per_expert)
        return result


class MoE_layer_megatron_wo_gate_v3(torch.nn.Module):
    def __init__(self, config, use_te):
        super().__init__()

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts, moe_grouped_gemm=True)
        self._moe_layer = MoELayer_wo_gate_v3(config, submodules=transformer_layer_spec.submodules.mlp.submodules, use_te=use_te)


    def forward(self):
        result, mlp_bias = self._moe_layer(None, None, None)
        output = result.reshape(-1, args.hidden_size)

        return output, mlp_bias


if __name__ == "__main__":

    # initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # args = get_args()

    args = parse_args(None, False)
    set_global_variables(args, False)
    # if args.rank == 0:
    #     print("> initializing torch distributed ...", flush=True)
    # # Manually set the device ids.
    # torch.cuda.set_device(args.local_rank)
    # device_id = torch.device(f'cuda:{args.local_rank}')

    # # Call the init process
    # init_process_group_kwargs = {
    #     'backend' : args.distributed_backend,
    #     'world_size': args.world_size,
    #     'rank': args.rank,
    #     'timeout': timedelta(minutes=args.distributed_timeout_minutes),
    # }
    # if packaging.version.Version(torch.__version__) >= packaging.version.Version("2.3.0"):
    #     init_process_group_kwargs['device_id'] = device_id

    # torch.distributed.init_process_group(**init_process_group_kwargs)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tensor_model_parallel_size, expert_model_parallel_size=args.expert_model_parallel_size)

    batch_size = args.micro_batch_size
    num_tokens = args.seq_length
    model_dim = args.ffn_hidden_size
    hidden_size = args.hidden_size
    device = torch.cuda.current_device()

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=8,
        num_moe_experts=args.num_experts,
        use_cpu_initialization=True,
        activation_func=torch.nn.functional.gelu,
        gated_linear_unit=False,
        bias_activation_fusion=False,
        moe_router_load_balancing_type="none",
        moe_router_topk=args.moe_router_topk,
        moe_grouped_gemm=True,
        moe_extended_tp=False,
        add_bias_linear=False,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        sequence_parallel=True,
        tp_comm_overlap=True,
        moe_token_dispatcher_type="allgather"
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    warmup_iters = 5
    iters = 100
    megatron_moe_te = MoE_layer_megatron_wo_gate_v3(transformer_config, use_te=True).cuda().to(torch.bfloat16)
    for i in range(warmup_iters):
        output2, mlp_bias = megatron_moe_te()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True
    ) as prof:
        start_event.record()
        for i in range(iters):
            output2, mlp_bias = megatron_moe_te()
            torch.cuda.synchronize()
        end_event.record()
        end_event.synchronize()
        elapsed_time_megatron_te = start_event.elapsed_time(end_event) / iters
        print_rank_0("Megatron_te time: {}".format(elapsed_time_megatron_te))

    # parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=4)
    # print(get_expert_model_parallel_world_size())
    # print(get_tensor_model_parallel_world_size())

    exit()

    for i in range(5):
        output1 = megatron_moe(x)
        # output1 = megatron_moe(dispatched_input, tokens_per_expert)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    iters = 10
    for i in range(iters):
        torch.distributed.barrier()
        output1 = megatron_moe(x)
        # output1 = megatron_moe(dispatched_input, tokens_per_expert)

    end_event.record()
    end_event.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    if RANK == 0:
        print("Elapsed time: ", elapsed_time / iters)
