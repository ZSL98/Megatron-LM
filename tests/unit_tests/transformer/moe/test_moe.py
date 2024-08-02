import torch
from torch import nn
import torch.distributed as dist

import argparse

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_tokens', type=int, default=32)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_local_experts', type=int, default=2)
parser.add_argument('--dtype', type=str, default='bfloat16')
parser.add_argument('--top', type=int, default=2)
parser.add_argument('--mode', type=str, default='single')
parser.add_argument('--expert_shape', type=str, default='abc-abd')
args = parser.parse_args()
args.expert_shape = args.expert_shape.replace('-', '->')
dist.init_process_group(backend=dist.Backend.NCCL, rank=args.local_rank,
                        world_size=args.world_size, init_method='env://')
dist_rank, dist_world_size = dist.get_rank(), dist.get_world_size()



batch_size = args.batch_size
num_tokens = args.num_tokens
model_dim = args.model_dim
hidden_size = args.hidden_size
num_local_experts = args.num_local_experts
top_value = args.top
device = torch.device('cuda', args.local_rank)

parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
# model_parallel_cuda_manual_seed(123)

class FFN(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, dim, expert_shape):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.expert_shape = expert_shape

    def forward(self, x) -> torch.Tensor:
        if self.expert_shape in ['abc->abd', 'ab->ac']:
            return self.fc(x)
        elif self.expert_shape in ['abc->adbe']:
            y = self.fc(x)
            return y.reshape(y.shape[0], y.shape[1], 2, -1).permute(0, 2, 1, 3)
        elif self.expert_shape in ['abc->dabe']:
            y = self.fc(x)
            return y.reshape(y.shape[0], y.shape[1], 2, -1).permute(2, 0, 1, 3)
        else:
            raise NotImplementedError


x = torch.tensor(torch.zeros([batch_size, num_tokens, model_dim], dtype=torch.float32, device='cpu').detach(
).numpy(), dtype=torch.bfloat16, requires_grad=False, device=device)
ct = model_dim // num_local_experts
tmp_cnt = num_tokens // (num_local_experts * dist_world_size // args.top)
print(ct, tmp_cnt, num_tokens)
# construct special input
for j in range(num_tokens):

    if args.top == 1:
        x[:, j, ct * (j % num_local_experts):ct * (j % num_local_experts + 1)] = (j %
                                                                                  num_local_experts + 1) / 2
    elif args.top == num_local_experts * dist_world_size:
        x[:, j, :] = 0.1
    else:
        t_idx = j // tmp_cnt
        x[:, j, ct * (t_idx * args.top):ct * ((t_idx + 1) * args.top)] = (t_idx + 1) / 5



class ExampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self._moe_layer = FFN(dim=model_dim, expert_shape=args.expert_shape).to(device)
        # self._moe_layer = MoE(hidden_size=model_dim,
        #                       expert=self._moe_layer,
        #                       num_experts=num_local_experts * dist_world_size,
        #                       k=top_value,
        #                       expert_shape=args.expert_shape).to(device)
        num_moe_experts = 4
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=False,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="sinkhorn",
            moe_router_topk=2,
            moe_grouped_gemm=True,
            add_bias_linear=False,
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=True)
        self._moe_layer = MoELayer(transformer_config, transformer_layer_spec.submodules.mlp.submodules)

    def forward(self, input):
        result, _ = self._moe_layer(input)
        return result


torch.manual_seed(dist_rank)
model = ExampleModel().cuda().to(torch.bfloat16)


if args.expert_shape in ['ab->ac']:
    x = x.reshape(-1, args.model_dim)
output = model(x)
