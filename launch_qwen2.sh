#!/bin/bash
# libflux_cuda.so maybe installed under /usr/local/lib or ~/.local/lib/ by pip3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# add flux python package to PYTHONPATH
export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=1

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

nproc_per_node=$(nvidia-smi --list-gpus | wc -l)
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port="23456"
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
IB_HCA=mlx5


export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3

MODEL_ARGS=(
    --model-name qwen2
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 1408
    --num-attention-heads 16
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 64
    --expert-model-parallel-size 8
    --moe-router-load-balancing-type none # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 4
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-layer-type flux
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)


CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} ${additional_args} test_forward_gpt.py\
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} $@ --model-name qwen2"

echo ${CMD}
${CMD}

ret=$?
exit $ret
