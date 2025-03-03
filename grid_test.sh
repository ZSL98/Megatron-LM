#!/bin/bash

NUM_TOKENS=(8192) 
EXPERT_NUM=(8)
TOPK=(2)
TP_SIZE=(2)

# Loop over all combinations of num_token and num_moe_experts
for num_tokens in "${NUM_TOKENS[@]}"; do
    for expert_num in "${EXPERT_NUM[@]}"; do
        for topk in "${TOPK[@]}"; do
            for tp_size in "${TP_SIZE[@]}"; do
                
                # Run the Python script with the current combination of arguments
                TEST_TYPE="single" TOPK="$topk" SEQ_LEN="$num_tokens" TP_SIZE="$tp_size" ./launch.sh ./tests/unit_tests/transformer/moe/test_moe.py --num_tokens "$num_tokens" --num_moe_experts "$expert_num" --topk "$topk" --tp_world_size "$tp_size" --ep_world_size "$((8 / tp_size))"
                
            done
        done
    done
done