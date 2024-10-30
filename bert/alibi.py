import torch
import torch.nn as nn


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


query = torch.randn(128, 64)
key = torch.randn(128, 64)

attn_weight = torch.matmul(query, key.T)

a = attn_weight[0]
b = attn_weight[-1]


# before_bias = torch.cosine_similarity(attn_weight[0], attn_weight[-1], dim=-1)
# print(f"similarity before bias: {before_bias}")

constant = get_relative_positions(128)
m = get_alibi_slope(8)

print(constant)

attn_weight = attn_weight + constant


# after_bias = torch.cosine_similarity(attn_weight[0], attn_weight[-1], dim=-1)
# print(f"similarity after bias: {after_bias}")
