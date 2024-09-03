# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch
import os
import flux
from flux import pynvshmem
from flux import moe_utils
from fmoe import FMoETransformerMLP
from tutel.impls.moe_layer import MOELayer as tutel_moelayer

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.test_forward.utils import generate_scatter_index

import torch.distributed as dist

class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


# class MixtralFluxMoELayer(MegatronModule, ABC):
#     """Base class for a mixture of experts layer.

#     Args:
#         config (TransformerConfig): Configuration object for the transformer model.
#     """
#     tp_group = parallel_state.get_tensor_model_parallel_group()
#     ep_group = parallel_state.get_expert_model_parallel_group()

#     tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
#     ep_world_size = parallel_state.get_expert_model_parallel_world_size()

#     # DIST_ENV = flux.get_dist_env()
#     TP_GROUP = torch.distributed.group.WORLD
#     flux.init_flux_shm(TP_GROUP)
#     tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=ep_group)
#     flux_m_max = 1 * 4096 * 2
#     bf16_moe_args = flux.MoeArguments(
#         max_ntokens=flux_m_max // 2,
#         hidden=4096,
#         ffn_hidden=14336,
#         nexperts=8,
#         topk=2,
#         input_dtype=torch.bfloat16,
#         output_dtype=torch.bfloat16,
#     )
#     flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, bf16_moe_args)
#     RANK = int(os.environ.get("RANK", 0))
#     flux_rs_op = flux.GemmGroupedV3GatherRS(8, flux_m_max, 4096, 
#                                                 2, RANK, 8, tp_world_size, ep_world_size, 1)

#     def __init__(self, config: TransformerConfig, layer_number: int = None):
#         super(BaseMoELayer, self).__init__(config)
#         self.config = config
#         self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
#         assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

#         if self.config.moe_extended_tp:
#             self.num_local_experts = self.config.num_moe_experts
#             local_expert_indices_offset = 0
#         else:
#             assert self.config.num_moe_experts % self.expert_parallel_size == 0
#             self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
#             local_expert_indices_offset = (
#                 parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
#             )

#         self.local_expert_indices = [
#             local_expert_indices_offset + i for i in range(self.num_local_experts)
#         ]
#         assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
#         self.router = None
#         self.experts = None
#         self.token_dispatcher = None
#         self.layer_number = layer_number

#     @abstractmethod
#     def forward(self, hidden_states):
#         pass

#     def set_layer_number(self, layer_number: int):
#         self.layer_number = layer_number
#         self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            # print("indices: ", indices.size())
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, indices
            )
            # print("dispatched_input: ", dispatched_input.size())
            # print("tokens_per_expert: ", tokens_per_expert)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # print("expert_output: ", expert_output.size())
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

class MoELayer_wo_te(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_wo_te, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.experts = GroupedMLP(self.num_local_experts, self.config)
        # self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            # print("hidden_states: ", hidden_states.size())
            probs, indices = self.router(hidden_states)
            # print("indices: ", indices.size())
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, indices
            )
            # print("dispatched_input: ", dispatched_input.size())
            # print("tokens_per_expert: ", tokens_per_expert)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # print("expert_output: ", expert_output.size())
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

class MoELayer_wo_gate(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_wo_gate, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, dispatched_input: torch.Tensor, tokens_per_expert):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(dispatched_input, tokens_per_expert):
            # probs, indices = self.router(hidden_states)
            # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            #     hidden_states, probs, indices
            # )
            # tokens_per_expert = torch.tensor([3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200])
            # print("dispatched_input: ", dispatched_input.size(), "; tokens_per_expert: ", tokens_per_expert, "; rank: ", torch.distributed.get_rank())
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # print("expert_output: ", expert_output.size())
            # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            # print("outputoutput: ", output.size())
            return expert_output

        expert_output = custom_forward(dispatched_input, tokens_per_expert)

        return expert_output


class MoELayer_wo_gate_v2(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_wo_gate_v2, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, probs, indices, hidden_states):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(probs, indices, hidden_states):
            probs0, indices0 = self.router(hidden_states)
            if torch.distributed.get_rank() == 0:
                print("probs size: ", probs0.size(), probs0)
                print("indices size: ", indices0.size(), indices0)
                print("hidden_states size: ", hidden_states.size(), hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs0, indices
            )
            # tokens_per_expert = torch.tensor([3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200])
            if torch.distributed.get_rank() == 0:
                print("dispatched_input: ", dispatched_input.size(), "; tokens_per_expert: ", tokens_per_expert, "; rank: ", torch.distributed.get_rank())
                # print("dispatched_input: ", dispatched_input, dispatched_input.size())
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            if torch.distributed.get_rank() == 0:
                print("expert_output: ", expert_output.size())
            # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            # print("outputoutput: ", output.size())
            return expert_output, mlp_bias

        output, mlp_bias = custom_forward(probs, indices, hidden_states)

        return output, mlp_bias


class MoELayer_wo_gate_v3(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None, use_te: bool = True
    ):
        self.submodules = submodules
        super(MoELayer_wo_gate_v3, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules) and use_te:
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, probs, indices, hidden_states):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(probs, indices, hidden_states):
            probs0, indices0 = self.router(hidden_states)
            # if torch.distributed.get_rank() == 2:
            #     print("probs size: ", probs0.size(), probs0)
            #     print("indices size: ", indices.size(), indices)
            #     print("hidden_states size: ", hidden_states.size(), hidden_states)
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs0, indices
            )
            # tokens_per_expert = torch.tensor([3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200])
            # if torch.distributed.get_rank() == 2:
            #     print("dispatched_input: ", dispatched_input.size(), "; tokens_per_expert: ", tokens_per_expert, "; rank: ", torch.distributed.get_rank())
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # if torch.distributed.get_rank() == 0:
            #     print("expert_output: ", expert_output.size())
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            # print("outputoutput: ", output.size())
            return output, mlp_bias

        output, mlp_bias = custom_forward(probs, indices, hidden_states)

        return output, mlp_bias


class MoELayer_tutel_mixtral(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_tutel_mixtral, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.tutel_moe_layer = tutel_moelayer(
            gate_type={'type': 'top', 'k': 2},
            model_dim=4096,
            experts={
                'count_per_node': 1,
                'type': 'ffn', 
                'hidden_size_per_expert': 14336, 
                'activation_fn': lambda x: torch.nn.functional.silu(x)
            },
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        ).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 4096), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.tutel_moe_layer(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_tutel_qwen2(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_tutel_qwen2, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.tutel_moe_layer = tutel_moelayer(
            gate_type={'type': 'top', 'k': 4},
            model_dim=2048,
            experts={
                'count_per_node': 8,
                'type': 'ffn', 
                'hidden_size_per_expert': 5632, 
                'activation_fn': lambda x: torch.nn.functional.silu(x)
            },
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        ).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 2048), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 2048), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.tutel_moe_layer(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_tutel_phi(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_tutel_phi, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.tutel_moe_layer = tutel_moelayer(
            gate_type={'type': 'top', 'k': 2},
            model_dim=4096,
            experts={
                'count_per_node': 2,
                'type': 'ffn', 
                'hidden_size_per_expert': 6400, 
                'activation_fn': lambda x: torch.nn.functional.silu(x)
            },
            scan_expert_func = lambda name, param: setattr(param, 'skip_allreduce', True),
        ).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 4096), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.tutel_moe_layer(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_fastermoe_mixtral(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_fastermoe_mixtral, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.fastermoe = FMoETransformerMLP(num_expert=1, 
                                    d_model=4096, 
                                    d_hidden=14336,
                                    world_size=8,
                                    top_k=2).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 4096), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.fastermoe(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_fastermoe_qwen2(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_fastermoe_qwen2, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.fastermoe = FMoETransformerMLP(num_expert=8, 
                                    d_model=2048, 
                                    d_hidden=5632,
                                    world_size=8,
                                    top_k=4).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 2048), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 2048), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.fastermoe(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_fastermoe_phi(BaseMoELayer):
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_fastermoe_phi, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        self.fastermoe = FMoETransformerMLP(num_expert=2, 
                                    d_model=4096, 
                                    d_hidden=6400,
                                    world_size=8,
                                    top_k=2).cuda().to(torch.bfloat16)
        device = torch.cuda.current_device()
        self.input = torch.randn((512, 4096), dtype=torch.bfloat16, device=device)
        self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor):
        _ = self.fastermoe(self.input)

        return self.mlp_output, self.mlp_output


class MoELayer_uniform_distribution_qwen2(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_uniform_distribution_qwen2, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute
        device = torch.cuda.current_device()
        self.fake_hidden_states = torch.rand((512, 1, 2048), dtype=torch.bfloat16, device=device)
        # self.splits_cpu = torch.tensor([2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048], dtype=torch.int32, device=device)
        self.splits_cpu = torch.tensor([256]*64, dtype=torch.int32, device=device)
        self.choosed_experts_all_token, _ = generate_scatter_index(
                self.splits_cpu, 4096, config.moe_router_topk, device
            )
        # print("choosed_experts_all_token: ", self.choosed_experts_all_token.size())
        self.ep_rank = parallel_state.get_expert_model_parallel_rank()
        token_per_rank = 512
        # self.indices = self.choosed_experts_all_token[self.ep_rank * token_per_rank : (self.ep_rank+1) * token_per_rank].to(torch.int32).cuda()
        self.indices = self.choosed_experts_all_token[0 : token_per_rank].to(torch.int32).cuda()
        # print("self.indices: ", self.indices.size(), self.ep_rank)


    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward():
            probs0, indices0 = self.router(self.fake_hidden_states)
            # print("self.fake_hidden_states: ", self.fake_hidden_states.size())
            # print("self.indices: ", self.indices.size())
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                self.fake_hidden_states, probs0, self.indices
            )
            # print("dispatched_input: ", dispatched_input.size())
            # print("tokens_per_expert: ", tokens_per_expert)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias
        
        output, mlp_bias = custom_forward()

        return output, mlp_bias


class MoELayer_uniform_distribution_mixtral(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_uniform_distribution_mixtral, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute
        device = torch.cuda.current_device()
        self.fake_hidden_states = torch.rand((4096, 1, 4096), dtype=torch.bfloat16, device=device)
        # self.splits_cpu = torch.tensor([2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048], dtype=torch.int32, device=device)
        self.splits_cpu = torch.tensor([1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], dtype=torch.int32, device=device)
        self.choosed_experts_all_token, _ = generate_scatter_index(
                self.splits_cpu, 4096, config.moe_router_topk, device
            )
        print("choosed_experts_all_token: ", self.choosed_experts_all_token.size())
        self.ep_rank = parallel_state.get_expert_model_parallel_rank()
        token_per_rank = 4096
        # self.indices = self.choosed_experts_all_token[self.ep_rank * token_per_rank : (self.ep_rank+1) * token_per_rank].to(torch.int32).cuda()
        self.indices = self.choosed_experts_all_token[0 : token_per_rank].to(torch.int32).cuda()
        # print("self.indices: ", self.indices.size(), self.ep_rank)


    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward():
            probs0, indices0 = self.router(self.fake_hidden_states)
            # print("self.fake_hidden_states: ", self.fake_hidden_states.size())
            # print("self.indices: ", self.indices.size())
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                self.fake_hidden_states, probs0, self.indices
            )
            # print("dispatched_input:{}".format(dispatched_input.size()))
            # print("tokens_per_expert: ", tokens_per_expert)
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            return output, mlp_bias
        
        output, mlp_bias = custom_forward()

        return output, mlp_bias


class MoELayer_flux_uniform_distribution_qwen2(BaseMoELayer):

    _initialized = False
    flux_ag_op = None
    flux_rs_op = None
                                                
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_flux_uniform_distribution_qwen2, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        device = torch.cuda.current_device()
        # self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)
        # self.mlp_bias = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

        self.activation_func = self.config.activation_func

        tp_group = parallel_state.get_tensor_model_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()

        # DIST_ENV = flux.get_dist_env()
        TP_GROUP = torch.distributed.group.WORLD
        flux.init_flux_shm(TP_GROUP)
        tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=ep_group)
        flux_m_max = 1 * 4096 * 4
        bf16_moe_args = flux.MoeArguments(
            max_ntokens=flux_m_max // 2,
            hidden=2048,
            ffn_hidden=5632,
            nexperts=64,
            topk=4,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
        RANK = int(os.environ.get("RANK", 0))

        if not MoELayer_flux_uniform_distribution_mixtral._initialized:
            MoELayer_flux_uniform_distribution_mixtral.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, bf16_moe_args)
            MoELayer_flux_uniform_distribution_mixtral.flux_rs_op = flux.GemmGroupedV3GatherRS(64, flux_m_max, 2048, 
                                                        4, RANK, 8, tp_world_size, ep_world_size, 1)
            MoELayer_flux_uniform_distribution_mixtral._initialized = True

            MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states = torch.rand((512, 2048), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.weights = torch.rand((64 // ep_world_size, self.config.ffn_hidden_size//tp_world_size, self.config.hidden_size), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_gpu = torch.tensor([256]*64, dtype=torch.int32, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu = torch.tensor([256]*64, dtype=torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.choosed_experts_all_token, MoELayer_flux_uniform_distribution_mixtral.scatter_index = generate_scatter_index(
                    MoELayer_flux_uniform_distribution_mixtral.splits_cpu, 4096, config.moe_router_topk, device
                )
            MoELayer_flux_uniform_distribution_mixtral.scatter_index = MoELayer_flux_uniform_distribution_mixtral.scatter_index.to(torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.outputs = [torch.zeros((4096 * 4 // ep_world_size, self.config.ffn_hidden_size//tp_world_size), dtype=torch.bfloat16, device=device)]
            MoELayer_flux_uniform_distribution_mixtral.weight2 = torch.rand((self.config.num_moe_experts // ep_world_size, self.config.hidden_size, self.config.ffn_hidden_size // tp_world_size), dtype=torch.bfloat16).cuda()


    def forward(self, hidden_states):

        probs0, indices0 = self.router(MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states.reshape(512, 1, 2048))
        MoELayer_flux_uniform_distribution_mixtral.flux_ag_op.forward_multiple_weights(
            inputs_shard=MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states,
            weights=[MoELayer_flux_uniform_distribution_mixtral.weights],
            splits_gpu=MoELayer_flux_uniform_distribution_mixtral.splits_gpu,
            scatter_index=MoELayer_flux_uniform_distribution_mixtral.scatter_index,
            output_scale=None,
            outputs_buf=MoELayer_flux_uniform_distribution_mixtral.outputs,
            fast_accum=False,
        )

        intermediate_output = self.activation_func(MoELayer_flux_uniform_distribution_mixtral.outputs[0])
        # print("intermediate_output: ", intermediate_output.size())
        mlp_output = MoELayer_flux_uniform_distribution_mixtral.flux_rs_op.forward_gather_rs(
            intermediate_output,
            MoELayer_flux_uniform_distribution_mixtral.weight2,
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu,
            MoELayer_flux_uniform_distribution_mixtral.scatter_index.view(-1),
            None,
            None,
            None,
            False,
        )
        # print("mlp_output: ", mlp_output.size())
        mlp_output = mlp_output.unsqueeze(1)

        return mlp_output, mlp_output


class MoELayer_flux_uniform_distribution_mixtral(BaseMoELayer):

    _initialized = False
    flux_ag_op = None
    flux_rs_op = None
                                                
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_flux_uniform_distribution_mixtral, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        device = torch.cuda.current_device()
        # self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)
        # self.mlp_bias = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

        self.activation_func = self.config.activation_func

        tp_group = parallel_state.get_tensor_model_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()

        # DIST_ENV = flux.get_dist_env()
        TP_GROUP = torch.distributed.group.WORLD
        flux.init_flux_shm(TP_GROUP)
        tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=ep_group)
        flux_m_max = 1 * 4096 * 2
        bf16_moe_args = flux.MoeArguments(
            max_ntokens=flux_m_max // 2,
            hidden=4096,
            ffn_hidden=14336,
            nexperts=8,
            topk=2,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
        RANK = int(os.environ.get("RANK", 0))

        if not MoELayer_flux_uniform_distribution_mixtral._initialized:
            MoELayer_flux_uniform_distribution_mixtral.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, bf16_moe_args)
            MoELayer_flux_uniform_distribution_mixtral.flux_rs_op = flux.GemmGroupedV3GatherRS(8, flux_m_max, 4096, 
                                                        2, RANK, 8, tp_world_size, ep_world_size, 1)
            MoELayer_flux_uniform_distribution_mixtral._initialized = True

            MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states = torch.rand((512, 4096), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.weights = torch.rand((8//ep_world_size, self.config.ffn_hidden_size//tp_world_size, self.config.hidden_size), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_gpu = torch.tensor([1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], dtype=torch.int32, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu = torch.tensor([1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024], dtype=torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.choosed_experts_all_token, MoELayer_flux_uniform_distribution_mixtral.scatter_index = generate_scatter_index(
                    MoELayer_flux_uniform_distribution_mixtral.splits_cpu, 4096, config.moe_router_topk, device
                )
            MoELayer_flux_uniform_distribution_mixtral.scatter_index = MoELayer_flux_uniform_distribution_mixtral.scatter_index.to(torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.outputs = [torch.zeros((4096 * 2 // ep_world_size, self.config.ffn_hidden_size//tp_world_size), dtype=torch.bfloat16, device=device)]
            MoELayer_flux_uniform_distribution_mixtral.weight2 = torch.rand((self.config.num_moe_experts // ep_world_size, self.config.hidden_size, self.config.ffn_hidden_size // tp_world_size), dtype=torch.bfloat16).cuda()


    def forward(self, hidden_states):

        probs0, indices0 = self.router(MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states.reshape(512, 1, 4096))
        MoELayer_flux_uniform_distribution_mixtral.flux_ag_op.forward_multiple_weights(
            inputs_shard=MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states,
            weights=[MoELayer_flux_uniform_distribution_mixtral.weights],
            splits_gpu=MoELayer_flux_uniform_distribution_mixtral.splits_gpu,
            scatter_index=MoELayer_flux_uniform_distribution_mixtral.scatter_index,
            output_scale=None,
            outputs_buf=MoELayer_flux_uniform_distribution_mixtral.outputs,
            fast_accum=False,
        )

        intermediate_output = self.activation_func(MoELayer_flux_uniform_distribution_mixtral.outputs[0])
        # print("intermediate_output: ", intermediate_output.size())
        mlp_output = MoELayer_flux_uniform_distribution_mixtral.flux_rs_op.forward_gather_rs(
            intermediate_output,
            MoELayer_flux_uniform_distribution_mixtral.weight2,
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu,
            MoELayer_flux_uniform_distribution_mixtral.scatter_index.view(-1),
            None,
            None,
            None,
            False,
        )
        # print("mlp_output: ", mlp_output.size())
        mlp_output = mlp_output.unsqueeze(1)

        return mlp_output, mlp_output


class MoELayer_flux_uniform_distribution_phi(BaseMoELayer):

    _initialized = False
    flux_ag_op = None
    flux_rs_op = None
                                                
    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer_flux_uniform_distribution_phi, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        device = torch.cuda.current_device()
        # self.mlp_output = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)
        # self.mlp_bias = torch.rand((512, 1, 4096), dtype=torch.bfloat16, device=device)

        self.activation_func = self.config.activation_func

        tp_group = parallel_state.get_tensor_model_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        ep_world_size = parallel_state.get_expert_model_parallel_world_size()

        # DIST_ENV = flux.get_dist_env()
        TP_GROUP = torch.distributed.group.WORLD
        flux.init_flux_shm(TP_GROUP)
        tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=ep_group)
        flux_m_max = 1 * 4096 * 2
        bf16_moe_args = flux.MoeArguments(
            max_ntokens=flux_m_max // 2,
            hidden=4096,
            ffn_hidden=6400,
            nexperts=16,
            topk=2,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
        RANK = int(os.environ.get("RANK", 0))

        if not MoELayer_flux_uniform_distribution_mixtral._initialized:
            MoELayer_flux_uniform_distribution_mixtral.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env, bf16_moe_args)
            MoELayer_flux_uniform_distribution_mixtral.flux_rs_op = flux.GemmGroupedV3GatherRS(16, flux_m_max, 4096, 
                                                        2, RANK, 8, tp_world_size, ep_world_size, 1)
            MoELayer_flux_uniform_distribution_mixtral._initialized = True

            MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states = torch.rand((512, 4096), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.weights = torch.rand((16//ep_world_size, self.config.ffn_hidden_size//tp_world_size, self.config.hidden_size), dtype=torch.bfloat16, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_gpu = torch.tensor([512]*16, dtype=torch.int32, device=device)
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu = torch.tensor([512]*16, dtype=torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.choosed_experts_all_token, MoELayer_flux_uniform_distribution_mixtral.scatter_index = generate_scatter_index(
                    MoELayer_flux_uniform_distribution_mixtral.splits_cpu, 4096, config.moe_router_topk, device
                )
            MoELayer_flux_uniform_distribution_mixtral.scatter_index = MoELayer_flux_uniform_distribution_mixtral.scatter_index.to(torch.int32)
            MoELayer_flux_uniform_distribution_mixtral.outputs = [torch.zeros((4096 * 2 // ep_world_size, self.config.ffn_hidden_size//tp_world_size), dtype=torch.bfloat16, device=device)]
            MoELayer_flux_uniform_distribution_mixtral.weight2 = torch.rand((self.config.num_moe_experts // ep_world_size, self.config.hidden_size, self.config.ffn_hidden_size // tp_world_size), dtype=torch.bfloat16).cuda()


    def forward(self, hidden_states):

        probs0, indices0 = self.router(MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states.reshape(512, 1, 4096))
        MoELayer_flux_uniform_distribution_mixtral.flux_ag_op.forward_multiple_weights(
            inputs_shard=MoELayer_flux_uniform_distribution_mixtral.fake_hidden_states,
            weights=[MoELayer_flux_uniform_distribution_mixtral.weights],
            splits_gpu=MoELayer_flux_uniform_distribution_mixtral.splits_gpu,
            scatter_index=MoELayer_flux_uniform_distribution_mixtral.scatter_index,
            output_scale=None,
            outputs_buf=MoELayer_flux_uniform_distribution_mixtral.outputs,
            fast_accum=False,
        )

        intermediate_output = self.activation_func(MoELayer_flux_uniform_distribution_mixtral.outputs[0])
        # print("intermediate_output: ", intermediate_output.size())
        mlp_output = MoELayer_flux_uniform_distribution_mixtral.flux_rs_op.forward_gather_rs(
            intermediate_output,
            MoELayer_flux_uniform_distribution_mixtral.weight2,
            MoELayer_flux_uniform_distribution_mixtral.splits_cpu,
            MoELayer_flux_uniform_distribution_mixtral.scatter_index.view(-1),
            None,
            None,
            None,
            False,
        )
        # print("mlp_output: ", mlp_output.size())
        mlp_output = mlp_output.unsqueeze(1)

        return mlp_output, mlp_output