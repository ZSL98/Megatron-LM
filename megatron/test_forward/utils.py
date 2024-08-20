import torch

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