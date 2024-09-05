#!/bin/bash


TP_SIZE=(1) 
MOE_LAYER_TYPE=(default te tutel fastermoe flux skip)

# Loop over all combinations of num_token and num_moe_experts
for TP_SIZE in "${TP_SIZE[@]}"; do
    for MOE_LAYER_TYPE in "${MOE_LAYER_TYPE[@]}"; do
        
        # Run the Python script with the current combination of arguments
        TP_SIZE=$TP_SIZE ./launch_mixtral.sh --tensor-model-parallel-size $TP_SIZE --expert-model-parallel-size $((8 / TP_SIZE)) --moe-layer-type $MOE_LAYER_TYPE
        TP_SIZE=$TP_SIZE ./launch_qwen2.sh   --tensor-model-parallel-size $TP_SIZE --expert-model-parallel-size $((8 / TP_SIZE)) --moe-layer-type $MOE_LAYER_TYPE
        TP_SIZE=$TP_SIZE ./launch_phi.sh     --tensor-model-parallel-size $TP_SIZE --expert-model-parallel-size $((8 / TP_SIZE)) --moe-layer-type $MOE_LAYER_TYPE
        
    done
done