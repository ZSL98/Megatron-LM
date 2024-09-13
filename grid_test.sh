#!/bin/bash

# # Define the ranges for grid search
# NUM_TOKENS=(2048 8192 32768) 
# EXPERT_NUM=(32)
# TOPK=(2)
# TP_SIZE=(8)

# # Loop over all combinations of num_token and num_moe_experts
# for NUM_TOKENS in "${NUM_TOKENS[@]}"; do
#     for EXPERT_NUM in "${EXPERT_NUM[@]}"; do
#         for TOPK in "${TOPK[@]}"; do
#             for TP_SIZE in "${TP_SIZE[@]}"; do
#                 echo "Running test with num_token=$NUM_TOKENS and num_moe_experts=$EXPERT_NUM"
                
#                 # Run the Python script with the current combination of arguments
#                 ./launch.sh ./tests/unit_tests/transformer/moe/test_moe.py --num_tokens $NUM_TOKENS --num_moe_experts $EXPERT_NUM --topk $TOPK --tp_world_size $TP_SIZE --ep_world_size $((8 / TP_SIZE))
                
#                 echo "Completed test with num_token=$NUM_TOKENS and num_moe_experts=$EXPERT_NUM"
#             done
#         done
#     done
# done

# NUM_TOKENS=(16384) 
# EXPERT_NUM=(8 16 32)
# TOPK=(5)
# TP_SIZE=(8)

# # Loop over all combinations of num_token and num_moe_experts
# for NUM_TOKENS in "${NUM_TOKENS[@]}"; do
#     for EXPERT_NUM in "${EXPERT_NUM[@]}"; do
#         for TOPK in "${TOPK[@]}"; do
#             for TP_SIZE in "${TP_SIZE[@]}"; do
#                 echo "Running test with num_token=$NUM_TOKENS and num_moe_experts=$EXPERT_NUM"
                
#                 # Run the Python script with the current combination of arguments
#                 ./launch.sh ./tests/unit_tests/transformer/moe/test_moe.py --num_tokens $NUM_TOKENS --num_moe_experts $EXPERT_NUM --topk $TOPK --tp_world_size $TP_SIZE --ep_world_size $((8 / TP_SIZE))
                
#                 echo "Completed test with num_token=$NUM_TOKENS and num_moe_experts=$EXPERT_NUM"
#             done
#         done
#     done
# done

NUM_TOKENS=(8192) 
EXPERT_NUM=(8 16 32 64)
TOPK=(1 2 3 4 5)
TP_SIZE=(8)

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