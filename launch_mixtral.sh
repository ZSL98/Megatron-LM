#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# add flux python package to PYTHONPATH
export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=1

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

nproc_per_node=${ARNOLD_WORKER_GPU:=$(nvidia-smi --list-gpus | wc -l)}
nnodes=${ARNOLD_WORKER_NUM:=1}
if [ $ARNOLD_WORKER_NUM == 1 ]; then # single machine. use no NVSHMEM_REMOTE_TRANSPORT
  export NVSHMEM_REMOTE_TRANSPORT=none
fi
node_rank=${ARNOLD_ID:=0}
master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="23456"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]] || [[ "$ARNOLD_DEVICE_TYPE" == *H800* ]]; then
  IB_HCA=mlx5
else
  IB_HCA=$ARNOLD_RDMA_DEVICE:1
fi

if [[ -n $NCCL_SOCKET_IFNAME ]]; then # check if NCCL_SOCKET_IFNAME exists
  ip link >/dev/null || sudo apt install iproute2
  ip link | grep $NCCL_SOCKET_IFNAME
  if [ $? -eq 0 ]; then
    echo "NCCL_SOCKET_IFNAME ${NCCL_SOCKET_IFNAME} exists"
  else
    echo "NCCL_SOCKET_IFNAME ${NCCL_SOCKET_IFNAME} not exist. unset NCCL_SOCKET_IFNAME"
    unset NCCL_SOCKET_IFNAME # let NCCL make the decisions
  fi
fi

if [ "$ARNOLD_RDMA_DEVICE" != "" ]; then
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=0}
  export NCCL_IB_HCA=${NCCL_IB_HCA:=$IB_HCA}
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}

  ## env var setting for ARNOLD machines
  if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]] || [[ "$ARNOLD_DEVICE_TYPE" == *L20* ]]; then
    export NVSHMEM_IB_ENABLE_IBGDA=1
    export NVSHMEM_IBGDA_SUPPORT=1
    export NVSHMEM_IB_GID_INDEX=3
  fi
  if [[ "$ARNOLD_DEVICE_TYPE" == *PCI* ]]; then
    # for merlin PCIE machine
    export NVSHMEM_HCA_LIST=mlx5_0
  fi
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
fi

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 2
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
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
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)


CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} ${additional_args} $@ \
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]}"

echo ${CMD}
${CMD}

ret=$?
exit $ret
