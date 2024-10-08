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
from torch.profiler import profile, record_function, ProfilerActivity

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
torch.manual_seed(RANK)

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


def generate_choosed_experts(splits, num_tokens, topk, device):
    # scatter_index(dst2src): [B*S, K]
    # scatter_weight: [B*S, K]
    # gather_index(src2dst): [B*S*K, ]
    # gather_weight: [B*S*K, ]

    # generate choosed experts
    choosed_experts = torch.zeros((num_tokens, topk), dtype=torch.int64)
    bin_counter = torch.clone(splits)
    offsets = torch.cumsum(splits, dim=0) - splits
    for tid in range(num_tokens):
        bin_size, bins = torch.topk(bin_counter, topk)
        choosed_experts[tid] = bins
        bin_counter[bins] -= 1

    return choosed_experts

def generate_scatter_index(splits, num_tokens, topk, device):
    # scatter_index(dst2src): [B*S, K]
    # scatter_weight: [B*S, K]
    # gather_index(src2dst): [B*S*K, ]
    # gather_weight: [B*S*K, ]

    # generate choosed experts
    choosed_experts = torch.zeros((num_tokens, topk), dtype=torch.int64)
    bin_counter = torch.clone(splits)
    offsets = torch.cumsum(splits, dim=0) - splits
    for tid in range(num_tokens):
        bin_size, bins = torch.topk(bin_counter, topk)
        choosed_experts[tid] = bins
        bin_counter[bins] -= 1

    # generate scatter index
    scatter_index = torch.zeros((num_tokens, topk), dtype=torch.int64)
    for i in range(num_tokens):
        for j in range(topk):
            eid = choosed_experts[i][j]
            scatter_index[i][j] = bin_counter[eid] + offsets[eid]
            bin_counter[eid] += 1
    return choosed_experts, scatter_index.to(device)



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

        print("self.ep_size: ", self.tp_size, self.ep_size)
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

            self.choosed_experts_all_token, self.scatter_index = generate_scatter_index(
                self.splits_cpu, self.ntokens, self.topk, device
            )
            self.scatter_index = self.scatter_index.to(torch.int32)

            exp_tokens = [[] for _ in range(args.num_moe_experts)]
            top_index = [[] for _ in range(args.num_moe_experts)]
            for tid in range(len(self.choosed_experts_all_token)):
                for _rank, eid in enumerate(self.choosed_experts_all_token[tid]):
                    exp_tokens[eid].append(tid)
                    top_index[eid].append(_rank)

            t_tokens = torch.tensor(sum(exp_tokens, []), dtype=torch.int32).cuda()
            t_topk_index = torch.tensor(sum(top_index, []), dtype=torch.int32).cuda()

            routing_idx = [0] * (10240 * 5)
            for i in range(10240 * 5):
                token_id = t_tokens[i].item()
                topk_id = t_topk_index[i].item()
                pos = token_id * args.topk + topk_id
                routing_idx[pos] = i
            t_routing_index = torch.tensor(routing_idx, dtype=torch.int32).cuda()

            eid_start = self.ep_rank * 8
            ep_rank_m_start = 0
            for i in range(eid_start):
                ep_rank_m_start += self.splits_cpu[i]
            M_cur_ep_rank = 12800 # This can vary
            ep_rank_m_end = ep_rank_m_start + M_cur_ep_rank

            self.new_index = (
                args.topk * t_tokens[ep_rank_m_start:ep_rank_m_end]
                + t_topk_index[ep_rank_m_start:ep_rank_m_end]
            )

            if RANK == 0:
                print("t_tokens: ", t_tokens, t_tokens.size())
                print("t_topk_index:", t_topk_index, t_topk_index.size())
                print("t_routing_index:", t_routing_index, t_topk_index.size())
                print("self.scatter_index:", self.scatter_index.view(-1), self.scatter_index.view(-1).size())
                print("new_index: ", self.new_index, self.new_index.size())
                print("Splits:", self.splits_cpu.tolist(), "Sum:", sum(self.splits_cpu.tolist()))
                print("choosed_experts_all_token:", self.choosed_experts_all_token, self.choosed_experts_all_token.size())
            # self.choosed_experts_all_token = generate_choosed_experts(self.splits_cpu, self.ntokens, self.topk, device)
            self.choosed_experts = self.choosed_experts_all_token[self.ep_rank * 1280 : (self.ep_rank+1) * 1280].to(torch.int32).cuda()

            self.gate_weight = torch.rand((self.ntokens, topk), dtype=input_dtype, device=device)
            gather_index, _ = moe_utils.calculate_gather_index_weight(
                self.scatter_index, self.gate_weight
            )
            self.gather_index = gather_index.to(torch.int32)

            n_experts_per_rank = args.num_moe_experts // args.ep_world_size
            ep_rank = TP_GROUP.rank() // args.tp_world_size
            tp_rank = TP_GROUP.rank() % args.tp_world_size
            eid_start = ep_rank * n_experts_per_rank
            eid_end = eid_start + n_experts_per_rank
            M_cur_ep_rank = torch.sum(self.splits_cpu[eid_start:eid_end]).item() # 12800
            local_K = args.model_dim // args.tp_world_size # 2560
            self.gelu_output = torch.zeros((M_cur_ep_rank, local_K), dtype=input_dtype).cuda()
            self._weight = torch.rand((n_experts_per_rank, args.model_dim, local_K), dtype=input_dtype).cuda() - 0.5


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

            torch.distributed.all_gather_into_tensor(self.inputs, self.inputs_shard, group=TP_GROUP)
            self.scatter_inputs.copy_(torch.index_select(self.inputs, dim=0, index=self.gather_index))
            self.dispatched_input = self.scatter_inputs[self.ep_rank * 12800 : self.ep_rank * 12800 + 12800]
            self.tokens_per_expert = torch.tensor([1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600])

            # if torch.distributed.get_rank() == 1:
            #     print("self.dispatched_input: ", self.dispatched_input, self.dispatched_input.size())

            torch.cuda.synchronize()

    def clear_outputs(self):
        for i in range(self.weight_groups):
            self.outputs[i].fill_(0.0)

    def get_outputs_clone(self):
        return [out.clone() for out in self.outputs]


def perf_torch(ctx: MoeMlp1Ctx):
    input = ctx.gelu_output  # From Flux phase1
    weight = ctx._weight

    acc = 0
    output_list = []
    full_output = torch.zeros((51200, 5120), dtype=torch.bfloat16, device=input.device)
    for exp_id in range(weight.size(0)):
        exp_w = weight[exp_id]
        Mi = ctx.splits_cpu[exp_id + TP_GROUP.rank() // 2 * 8]
        exp_input = input[acc : acc + Mi]
        acc += Mi
        output_list.append(torch.matmul(exp_input, exp_w.t()))

    output = torch.concat(output_list)
    # if RANK == 0:
    #     print("output: ", output, output.size())
    output1 = torch.zeros_like(full_output)
    output1[ctx.new_index] = output
    full_output += output1
    topk_reduce = full_output.view(
        (51200 // 5, 5, 5120)
    ).sum(1)
    output2 = torch.zeros(
        (full_output.size(0) // TP_GROUP.size() // args.topk, full_output.size(1)),
        dtype=topk_reduce.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.reduce_scatter_tensor(output2, topk_reduce, group=TP_GROUP)

    if RANK == 0:
        print("perf_torch output: ", output2, output2.size())

    return output2


def perf_flux(ctx: MoeMlp1Ctx):

    input = ctx.gelu_output  # From Flux phase1
    weight = ctx._weight

    op = flux.GemmGroupedV3GatherRS(
        32, 2*5120*5, 5120, 5, RANK, 8, 2, 4, 1
    )

    output = op.forward_gather_rs(
        input,
        weight,
        ctx.splits_cpu,
        ctx.scatter_index.view(-1),
        None,
        None,
        None,
        False,
    )

    if RANK == 0:
        print("perf_flux output: ", output, output.size())

    return output

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


class MoE_layer_megatron_wo_gate(torch.nn.Module):
    def __init__(self, config, ctx):
        super().__init__()

        self.ctx = ctx
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_moe_experts, moe_grouped_gemm=True)
        # self._moe_layer = MoELayer(config, transformer_layer_spec.submodules.mlp.submodules)
        self._moe_layer = MoELayer_wo_gate(config, submodules=transformer_layer_spec.submodules.mlp.submodules)

    def forward(self):
        result = self._moe_layer(self.ctx.dispatched_input, self.ctx.tokens_per_expert)
        return result

class MoE_layer_megatron_wo_gate_v2(torch.nn.Module):
    def __init__(self, config, ctx):
        super().__init__()

        self.ctx = ctx
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_moe_experts, moe_grouped_gemm=False)
        self._moe_layer = MoELayer_wo_gate_v2(config, submodules=transformer_layer_spec.submodules.mlp.submodules)

        for name, param in self._moe_layer.named_parameters():
            # print("name, param.size: ", name, " ", param.size())
            if "experts.linear_fc1.weight" in name:
                # print("param.size: ", param.size())
                param.data = self.ctx.weights[0][int(name[-1])]
            if "experts.linear_fc2.weight" in name:
                param.data = self.ctx._weight[int(name[-1])]

    def forward(self):
        reshaped_tensor = self.ctx.inputs_shard.reshape(640, 2, 5120)
        # self.ctx.choosed_experts_all_token = self.ctx.choosed_experts_all_token.cuda()
        result = self._moe_layer(self.ctx.gate_weight, self.ctx.choosed_experts, reshaped_tensor)

        full_output = torch.zeros((51200, 5120), dtype=result.dtype, device=self.ctx.inputs_shard.device)
        output1 = torch.zeros_like(full_output)
        output1[self.ctx.new_index] = result
        full_output += output1
        topk_reduce = full_output.view((full_output.size(0) // args.topk, args.topk, full_output.size(1))).sum(1)
        output2 = torch.zeros(
            (full_output.size(0) // TP_GROUP.size() // args.topk, full_output.size(1)),
            dtype=topk_reduce.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        torch.distributed.reduce_scatter_tensor(output2, topk_reduce, group=TP_GROUP)

        return output2


class MoE_layer_flux(torch.nn.Module):
    def __init__(self, config: TransformerConfig, ctx):
        super().__init__()

        self.ctx = ctx
        input_dtype = INP_DTYPE_MAP[args.dtype]
        output_dtype = OUT_DTYPE_MAP[args.dtype]
        tp_group = TP_GROUP
        ep_group = EP_GROUP

        self.activation_func = config.activation_func
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

        n_dim = args.model_dim # Check this!
        self.flux_rs_op = flux.GemmGroupedV3GatherRS(args.num_moe_experts, flux_m_max, n_dim, args.topk, RANK, WORLD_SIZE, 
                                                args.tp_world_size, args.ep_world_size, 1)


    def forward(self, hidden_states):

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

        self.ctx.gelu_output = self.activation_func(self.ctx.outputs[0])

        # weighted_gelu_output = torch.ops.FasterTransformer.GatedLinearUnit_forward_swiglu(
        #     ctx.outputs[0], ctx.outputs[1], scattered_gate_weight
        # )
        #     print("_input shape: ", self._input.size())
        #     print("_weight shape: ", self._weight.size())
        #     print("self.ctx.scatter_index: ", self.ctx.scatter_index.size())

        # if RANK == 2:
        #     print("Flux gelu_output: ", self.ctx.gelu_output.size(), self.ctx.gelu_output)
        mlp_output = self.flux_rs_op.forward_gather_rs(
            self.ctx.gelu_output,
            self.ctx._weight,
            self.ctx.splits_cpu,
            self.ctx.scatter_index.view(-1),
            None,
            None,
            None,
            False,
        )

        return mlp_output



if __name__ == "__main__":

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
        ffn_hidden_size=args.hidden_size,
        num_attention_heads=4,
        num_moe_experts=args.num_moe_experts,
        use_cpu_initialization=True,
        activation_func=torch.nn.functional.gelu,
        gated_linear_unit=False,
        bias_activation_fusion=False,
        moe_router_load_balancing_type="none",
        moe_router_topk=args.topk,
        moe_grouped_gemm=False,
        moe_extended_tp=False,
        add_bias_linear=False,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        sequence_parallel=True,
        tp_comm_overlap=True,
    )


    flux.init_flux_shm(TP_GROUP)
    init_ep_group(args.ep_world_size)
    # parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)

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

    flux_moe = MoE_layer_flux(transformer_config, moe_ctx).cuda().to(torch.bfloat16)
    for i in range(5):
        output1 = flux_moe(x)
    # perf_flux(moe_ctx)
    # perf_torch(moe_ctx)


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    iters = 100
    for i in range(iters):
        torch.distributed.barrier()
        output1 = flux_moe(x)

    end_event.record()
    end_event.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    if RANK == 0:
        # print("moe_ctx.outputs shape:", moe_ctx.outputs[0].size())
        # print(moe_ctx.outputs[0])
        print("Flux time: ", elapsed_time / iters)

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=4)
    # megatron_moe = MoE_layer_megatron_wo_gate(transformer_config, moe_ctx).cuda().to(torch.bfloat16)
    # print("choosed_experts: ", moe_ctx.choosed_experts, moe_ctx.choosed_experts.size())
    # print("gate_weight: ", moe_ctx.gate_weight, moe_ctx.gate_weight.size())
    megatron_moe = MoE_layer_megatron_wo_gate_v2(transformer_config, moe_ctx).cuda().to(torch.bfloat16)
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    output2 = megatron_moe()

    start_event.record()
    for i in range(iters):
        torch.distributed.barrier()
        output2 = megatron_moe()

    end_event.record()
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    if RANK == 0:
        print("Megatron time: ", elapsed_time / iters)
        # print("Flux output shape: ", output1.size(), output1)
        # print("Megatron output shape: ", output2.size(), output2)