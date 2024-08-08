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

import argparse

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoELayer_wo_gate, MoELayer_wo_gate_v2
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


parser = argparse.ArgumentParser()

parser.add_argument("--dist", type=str, default="uniform")
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--world_size', type=int, default=8)
parser.add_argument('--ep_world_size', type=int, default=4)
parser.add_argument('--tp_world_size', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_tokens', type=int, default=5120)
parser.add_argument('--model_dim', type=int, default=5120)
parser.add_argument('--hidden_size', type=int, default=5120)
parser.add_argument('--num_local_experts', type=int, default=8) # equals to num_moe_experts//ep_world_size
parser.add_argument('--num_moe_experts', type=int, default=32)
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--mode', type=str, default='single')
parser.add_argument('--expert_shape', type=str, default='abc-abd')
parser.add_argument("--weight_groups", default=1, type=int, help="num of weight groups")
parser.add_argument("--fast_accum", default=False, action="store_true", help="fp8 use fast accum")
args = parser.parse_args()
args.expert_shape = args.expert_shape.replace('-', '->')

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DIST_ENV = flux.get_dist_env()
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)

batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.topk
device = torch.cuda.current_device()


class MoE_layer_megatron(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_moe_experts, moe_grouped_gemm=True)
        self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, input):
        result, _ = self._moe_layer(input)
        return result


class MoE_layer_megatron_wo_gate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_moe_experts, moe_grouped_gemm=True)
        # self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)
        self._moe_layer = MoELayer_wo_gate(config, submodules=transformer_layer_spec.submodules.mlp.submodules)
        # self._moe_layer = MoELayer_wo_gate_v2(config, submodules=transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, dispatched_input, tokens_per_expert):
        result = self._moe_layer(dispatched_input, tokens_per_expert)
        return result


if __name__ == "__main__":

    # x = torch.tensor(torch.zeros([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach(
    # ).numpy(), dtype=torch.bfloat16, requires_grad=False, device=device)
    # ct = model_dim // num_local_experts
    # tmp_cnt = num_tokens // (num_local_experts * DIST_ENV.get_world().size() // args.topk)
    # # print(ct, tmp_cnt, num_tokens)
    # # construct special input
    # for j in range(num_tokens):

    #     if args.topk == 1:
    #         x[:, j, ct * (j % num_local_experts):ct * (j % num_local_experts + 1)] = (j %
    #                                                                                 num_local_experts + 1) / 2
    #     elif args.topk == num_local_experts * DIST_ENV.get_world().size():
    #         x[:, j, :] = 0.1
    #     else:
    #         t_idx = j // tmp_cnt
    #         x[:, j, ct * (t_idx * args.topk):ct * ((t_idx + 1) * args.topk)] = (t_idx + 1) / 5


    x = torch.tensor(torch.zeros([batch_size * num_tokens // args.tp_world_size // args.ep_world_size, model_dim], dtype=torch.float32, device='cpu').detach(
    ).numpy(), dtype=torch.bfloat16, requires_grad=False, device=device)

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=args.hidden_size,
        num_attention_heads=4,
        num_moe_experts=args.num_moe_experts,
        use_cpu_initialization=True,
        activation_func=torch.nn.functional.gelu,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        moe_router_load_balancing_type="sinkhorn",
        moe_router_topk=args.topk,
        moe_grouped_gemm=True,
        moe_extended_tp=False,
        add_bias_linear=False,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        sequence_parallel=True,
    )

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=4)
    # print(get_expert_model_parallel_world_size())
    # print(get_tensor_model_parallel_world_size())

    dispatched_input = torch.rand((args.batch_size * args.num_tokens * args.topk // args.ep_world_size, args.model_dim), dtype=torch.bfloat16).cuda()
    tokens_per_expert = torch.tensor([1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600])

    # megatron_moe = MoE_layer_megatron(transformer_config).cuda().to(torch.bfloat16)
    megatron_moe = MoE_layer_megatron_wo_gate(transformer_config).cuda().to(torch.bfloat16)

    if args.expert_shape in ['ab->ac']:
        x = x.reshape(-1, args.model_dim)

    for i in range(5):
        # output1 = megatron_moe(x)
        output1 = megatron_moe(dispatched_input, tokens_per_expert)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    iters = 10
    for i in range(iters):
        torch.distributed.barrier()
        # output1 = megatron_moe(x)
        output1 = megatron_moe(dispatched_input, tokens_per_expert)

    end_event.record()
    end_event.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    if RANK == 0:
        print("Elapsed time: ", elapsed_time / iters)
