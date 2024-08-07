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

import argparse

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
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

parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
# model_parallel_cuda_manual_seed(123)

def init_ep_group(ep_size: int):
    assert DIST_ENV.WORLD_SIZE % ep_size == 0, f"{DIST_ENV.WORLD_SIZE} % {ep_size} != 0"
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP already initialized"

    assert TP_GROUP.size() % ep_size == 0, f"{TP_GROUP.size()} % {ep_size} != 0"
    ffn_tp_size = TP_GROUP.size() // ep_size

    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)

    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])

    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group

class NopAsyncHandle:
    # unified interfaces with nccl handle
    def wait(self):
        pass


def tp_allgather(input_, group=None, sync=True):
    if group is None:
        group = get_tensor_model_parallel_group()
        world_size = get_tensor_model_parallel_world_size()
    else:
        world_size = torch.distributed.get_world_size(group)

    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_, NopAsyncHandle()

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    if sync:
        timed__all_gather_base(
            output,
            input_.contiguous(),
            group=group,
            timer_name='tp-allgather')
        return output, None
    else:
        handle = timed__all_gather_base(
            output,
            input_.contiguous(),
            group=group,
            async_op=True,
            timer_name='tp-allgather')
        return output, handle


def randomGateFunc(token_num, num_experts, topk):
    seq_len = torch.zeros(num_experts, dtype=torch.int32)
    exp_list = list(range(num_experts))
    gate = []
    exp_tokens = [[] for _ in range(num_experts)]
    top_index = [[] for _ in range(num_experts)]
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        seq_len[top_selected] += 1
        gate.append(top_selected)
        for _rank, eid in enumerate(top_selected):
            exp_tokens[eid].append(tid)
            top_index[eid].append(_rank)
    t_tokens = torch.tensor(sum(exp_tokens, []), dtype=torch.int32).cuda()
    t_topk_index = torch.tensor(sum(top_index, []), dtype=torch.int32).cuda()
    routing_idx = [0] * (token_num * topk)
    for i in range(token_num * topk):
        token_id = t_tokens[i].item()
        topk_id = t_topk_index[i].item()
        pos = token_id * topk + topk_id
        routing_idx[pos] = i
    t_routing_index = torch.tensor(routing_idx, dtype=torch.int32).cuda()
    return seq_len, torch.Tensor(gate).to(torch.int32), t_tokens, t_topk_index, t_routing_index


class MoeMlp1Ctx:
    def __init__(
        self,
        b: int,
        s: int,
        h: int,
        ffn_size: int,
        nexperts: int,
        topk: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        dist: str,
        fast_accum: bool,
        weight_groups: int,
    ) -> None:
        self.b = b
        self.s = s
        self.h = h
        self.ffn_size = ffn_size
        self.nexperts = nexperts
        self.topk = topk
        self.ntokens = b * s
        self.fast_accum = fast_accum
        self.weight_groups = weight_groups
        self.tp_rank = TP_GROUP.rank()
        self.tp_size = TP_GROUP.size()
        self.ep_rank = EP_GROUP.rank()
        self.ep_size = EP_GROUP.size()
        self.ffn_tp_size = self.tp_size // self.ep_size
        self.nexperts_ep = self.nexperts // self.ep_size
        assert self.nexperts % self.ep_size == 0

        assert self.ffn_size % self.ffn_tp_size == 0
        self.ffn_size_shard = ffn_size // self.ffn_tp_size
        assert self.ntokens % self.tp_size == 0
        self.ntokens_shard = self.ntokens // self.tp_size

        if RANK == 0:
            print("self.tp_size: ", self.tp_size)

        device = torch.cuda.current_device()

        init_tensor_ctx = nullcontext
        with init_tensor_ctx():
            # input tensors
            self.inputs_shard = (
                torch.rand((self.ntokens_shard, h), dtype=input_dtype, device=device)
                * 0.01
                * (self.tp_rank + 1)
            )
            self.weights = [
                (
                    torch.rand(
                        (self.nexperts_ep, self.ffn_size_shard, h),
                        dtype=input_dtype,
                        device=device,
                    )
                    * 0.01
                    * (self.tp_rank + 1)
                )
                for _ in range(weight_groups)
            ]

            self.splits_cpu: torch.Tensor = moe_utils.generate_splits(dist, b, s, topk, nexperts)
            self.splits_gpu = self.splits_cpu.to(device)
            torch.distributed.broadcast(self.splits_gpu, src=0, group=TP_GROUP)
            self.splits_cpu = self.splits_gpu.cpu()
            self.nrows_ep = torch.sum(
                self.splits_cpu[
                    self.nexperts_ep * self.ep_rank : self.nexperts_ep * (self.ep_rank + 1)
                ]
            )

            if self.tp_rank == 0:
                print("Splits:", self.splits_cpu.tolist(), "Sum:", sum(self.splits_cpu.tolist()))

            self.scatter_index = moe_utils.generate_scatter_index(
                self.splits_cpu, self.ntokens, self.topk, device
            ).to(torch.int32)

            gate_weight = torch.rand((self.ntokens, topk), dtype=input_dtype, device=device)
            gather_index, _ = moe_utils.calculate_gather_index_weight(
                self.scatter_index, gate_weight
            )
            self.gather_index = gather_index.to(torch.int32)

            # buffers
            self.inputs = torch.zeros((self.ntokens, h), dtype=input_dtype, device=device)
            self.scatter_inputs = torch.zeros(
                (self.ntokens * topk, h), dtype=input_dtype, device=device
            )
            self.output_scale = [
                torch.ones((self.nexperts_ep,), dtype=torch.float, device=device)
                for _ in range(weight_groups)
            ]
            self.outputs = [
                torch.zeros((self.nrows_ep, self.ffn_size_shard), dtype=output_dtype, device=device)
                for _ in range(weight_groups)
            ]

            torch.cuda.synchronize()

    def clear_outputs(self):
        for i in range(self.weight_groups):
            self.outputs[i].fill_(0.0)

    def get_outputs_clone(self):
        return [out.clone() for out in self.outputs]


class MoE_layer_megatron(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # self._moe_layer = FFN(dim=model_dim, expert_shape=args.expert_shape).to(device)
        # self._moe_layer = MoE(hidden_size=model_dim,
        #                       expert=self._moe_layer,
        #                       num_experts=num_local_experts * dist_world_size,
        #                       k=top_value,
        #                       expert_shape=args.expert_shape).to(device)
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_moe_experts, moe_grouped_gemm=True)
        self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, input):
        result, _ = self._moe_layer(input)
        return result




class MoE_layer_flux(torch.nn.Module):
    def __init__(self, config: TransformerConfig, ctx):
        super().__init__()

        self.ctx = ctx
        input_dtype = INP_DTYPE_MAP[args.dtype]
        output_dtype = OUT_DTYPE_MAP[args.dtype]
        tp_group = TP_GROUP
        tp_size = TP_GROUP.size()
        ep_group = EP_GROUP
        ep_size = EP_GROUP.size()

        n_experts_per_rank = args.num_moe_experts // args.ep_world_size
        # n_experts_per_rank = args.num_local_experts
        ep_rank = TP_GROUP.rank() // args.tp_world_size
        tp_rank = TP_GROUP.rank() % args.tp_world_size
        eid_start = ep_rank * n_experts_per_rank
        eid_end = eid_start + n_experts_per_rank

        self.router = TopKRouter(config=config)
        # self.token_dispatcher = MoEAllGatherTokenDispatcher(
        #     args.num_local_experts, [i for i in range(eid_start, eid_end)], config=config
        # )

        # self.splits_cpu, _, _, _, self.routing_idx = randomGateFunc(
        #     args.batch_size * args.num_tokens // args.topk, args.num_moe_experts, args.topk
        # )

        tp_env = flux.DistEnvTPWithEP(tp_group=tp_group, nnodes=1, ep_group=ep_group)
        flux_m_max = args.batch_size * args.num_tokens * args.topk
        bf16_moe_args = flux.MoeArguments(
            max_ntokens=flux_m_max // args.topk,
            hidden=args.hidden_size,
            ffn_hidden=args.model_dim,
            nexperts=args.num_moe_experts,
            topk=args.topk,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        self.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, bf16_moe_args)

        # if RANK == 0:
        #     print("eid_start, eid_end ", eid_start, " ", eid_end)

        M_cur_ep_rank = torch.sum(self.ctx.splits_cpu[eid_start:eid_end]).item()
        local_K = args.model_dim // args.tp_world_size
        self._input = torch.rand((M_cur_ep_rank, local_K), dtype=input_dtype).cuda() - 0.5
        self._weight = torch.rand((n_experts_per_rank, args.model_dim, local_K), dtype=input_dtype).cuda() - 0.5
        n_dim = args.model_dim # Check this!
        self.flux_rs_op = flux.GemmGroupedV3GatherRS(args.num_moe_experts, flux_m_max, n_dim, args.topk, RANK, WORLD_SIZE, 
                                                args.tp_world_size, args.ep_world_size, 1)


    def forward(self, hidden_states):


        # moe gate
        # num_experts = gate_wg.shape[-1]
        # wg_running = 0.5 * (gate_wg_ema + gate_wg.float())
        # logits_fp32 = torch.matmul(ln2_output.reshape(-1, hidden_size).float(), wg_running)
        # logits = logits_fp32.bfloat16()
        # gathered_logits, handle_logits = tp_allgather(logits, sync=False)
        # gates = torch.nn.functional.softmax(logits_fp32, dim=1)
        # values, indices_long = torch.topk(gates, k=args.topk, dim=-1, sorted=False)
        # indices = indices_long.int()
        # gathered_indices, handle_indices = tp_allgather(indices, sync=False)
        # gate_weight_fp32 = (values / torch.sum(values, dim=-1, keepdim=True))
        # gate_weight = gate_weight_fp32.bfloat16()
        # gathered_gate_weight, handle_gate_weight = tp_allgather(gate_weight, sync=False)
        # handle_logits.wait()
        # handle_indices.wait()
        # splits = torch.ops.FasterMoe.expert_histogram(gathered_indices, args.num_moe_experts)
        # scatter_index = torch.ops.FasterMoe.index_compute(gathered_indices, splits)
        # splits_cpu = torch.empty(splits.size(), dtype=splits.dtype, pin_memory=True)
        # splits_cpu.copy_(splits, non_blocking=True)
        # handle_gate_weight.wait()
        # reshaped_gate_weight = gathered_gate_weight.reshape(-1, 1)
        # scattered_gate_weight = torch.empty_like(reshaped_gate_weight)

        # probs, indices = self.router(hidden_states)
        # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
        #     hidden_states, probs, indices
        # )

        # if RANK == 0:
        #     print("probs: ", probs.size())
        #     print("indices: ", indices.size())

        self.flux_ag_op.forward_multiple_weights(
            inputs_shard=self.ctx.inputs_shard,
            weights=self.ctx.weights,
            splits_gpu=self.ctx.splits_gpu,
            scatter_index=self.ctx.scatter_index,
            output_scale=None,
            outputs_buf=self.ctx.outputs,
            fast_accum=False,
        )

        # weighted_gelu_output = torch.ops.FasterTransformer.GatedLinearUnit_forward_swiglu(
        #     ctx.outputs[0], ctx.outputs[1], scattered_gate_weight
        # )

        # if RANK == 0:
        #     print("_input shape: ", self._input.size())
        #     print("_weight shape: ", self._weight.size())
        #     print("self.ctx.scatter_index: ", self.ctx.scatter_index.size())

        mlp_output = self.flux_rs_op.forward_gather_rs(
            self._input,
            self._weight,
            self.ctx.splits_cpu,
            self.ctx.scatter_index.view(-1),
            None,
            None,
            None,
            False,
        )

        return mlp_output



if __name__ == "__main__":

    flux.init_flux_shm(TP_GROUP)
    init_ep_group(args.num_local_experts)

    x = torch.tensor(torch.zeros([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach(
    ).numpy(), dtype=torch.bfloat16, requires_grad=False, device=device)
    ct = model_dim // num_local_experts
    tmp_cnt = num_tokens // (num_local_experts * DIST_ENV.get_world().size() // args.topk)
    # print(ct, tmp_cnt, num_tokens)
    # construct special input
    for j in range(num_tokens):

        if args.topk == 1:
            x[:, j, ct * (j % num_local_experts):ct * (j % num_local_experts + 1)] = (j %
                                                                                    num_local_experts + 1) / 2
        elif args.topk == num_local_experts * DIST_ENV.get_world().size():
            x[:, j, :] = 0.1
        else:
            t_idx = j // tmp_cnt
            x[:, j, ct * (t_idx * args.topk):ct * ((t_idx + 1) * args.topk)] = (t_idx + 1) / 5


    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=args.hidden_size,
        num_attention_heads=4,
        num_moe_experts=args.num_moe_experts,
        use_cpu_initialization=True,
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=False,
        bias_activation_fusion=True,
        moe_router_load_balancing_type="sinkhorn",
        moe_router_topk=args.topk,
        moe_grouped_gemm=True,
        add_bias_linear=False,
    )

    moe_ctx = MoeMlp1Ctx(
        b=args.batch_size,
        s=args.num_tokens,
        h=args.hidden_size,
        ffn_size=args.model_dim,
        nexperts=args.num_moe_experts,
        topk=args.topk,
        input_dtype=INP_DTYPE_MAP[args.dtype],
        output_dtype=OUT_DTYPE_MAP[args.dtype],
        dist=args.dist,
        fast_accum=args.fast_accum,
        weight_groups=args.weight_groups,
    )

    # megatron_moe = MoE_layer_megatron(transformer_config).cuda().to(torch.bfloat16)
    flux_moe = MoE_layer_flux(transformer_config, moe_ctx).cuda().to(torch.bfloat16)

    if args.expert_shape in ['ab->ac']:
        x = x.reshape(-1, args.model_dim)
    # output1 = megatron_moe(x)

    for i in range(5):
        output2 = flux_moe(x)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    iters = 10
    for i in range(iters):
        torch.distributed.barrier()
        output2 = flux_moe(x)

    end_event.record()
    end_event.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    if RANK == 0:
        print("Elapsed time: ", elapsed_time / iters)