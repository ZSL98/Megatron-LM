from megatron.core import parallel_state
import flux
import torch
import torch.distributed as dist


if __name__ == "__main__":
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=4)

    # TP_GROUP = parallel_state.get_tensor_model_parallel_group()
    EP_GROUP = parallel_state.get_expert_model_parallel_group()

    TP_GROUP = torch.distributed.group.WORLD
    # EP_GROUP = None

    print(parallel_state.get_tensor_model_parallel_world_size())
    print(parallel_state.get_expert_model_parallel_world_size())
    flux.init_flux_shm(torch.distributed.group.WORLD)

    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=EP_GROUP)

    print("After DistEnvTPWithEP")