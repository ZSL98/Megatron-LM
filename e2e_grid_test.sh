#!/bin/bash

# TP_SIZE=(1 2 4 8)
# MOE_LAYER_TYPE=(default te tutel fastermoe flux skip)
# NUM_TOKENS=(8192 4096)
# TP_SIZE=(1 2 4 8)
# MOE_LAYER_TYPE=(default te tutel fastermoe flux skip)

NUM_TOKENS=(8192)
TP_SIZE=(4 2 1)
MOE_LAYER_TYPE=(default te tutel fastermoe flux skip)

# Loop over all combinations of num_token and num_moe_experts
for num_tokens in "${NUM_TOKENS[@]}"; do
    for tp_size in "${TP_SIZE[@]}"; do
        for moe_layer_type in "${MOE_LAYER_TYPE[@]}"; do
            # Run the Python script with the current combination of arguments

            TEST_TYPE="e2e" TOPK=2 SEQ_LEN="$num_tokens" TP_SIZE="$tp_size" ./launch_mixtral.sh --tensor-model-parallel-size "$tp_size" --expert-model-parallel-size "$((8 / tp_size))" --moe-layer-type "$moe_layer_type" --seq-length "$num_tokens"
            TEST_TYPE="e2e" TOPK=4 SEQ_LEN="$num_tokens" TP_SIZE="$tp_size" ./launch_qwen2.sh   --tensor-model-parallel-size "$tp_size" --expert-model-parallel-size "$((8 / tp_size))" --moe-layer-type "$moe_layer_type" --seq-length "$num_tokens"
            TEST_TYPE="e2e" TOPK=2 SEQ_LEN="$num_tokens" TP_SIZE="$tp_size" ./launch_phi.sh     --tensor-model-parallel-size "$tp_size" --expert-model-parallel-size "$((8 / tp_size))" --moe-layer-type "$moe_layer_type" --seq-length "$num_tokens"

        done
    done
done