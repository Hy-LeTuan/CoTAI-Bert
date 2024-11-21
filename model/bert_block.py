from optparse import Values
import torch.nn as nn
import torch
from mlp import MLP
from attention import GQAAttention, FlashGQAAttention
from layernorm import RMSNorm


class BertBlock(nn.Module):
    def __init__(self, layer_id: int, hidden_state, mlp_dim, num_heads, num_kv_heads, base=10000, flash=False, device="cpu"):
        assert hidden_state % num_heads == 0, print(
            f"Hidden state is {hidden_state} not divisible by number of heads: {num_heads}")

        if flash and device != "cuda":
            raise ValueError("Device must be CUDA to enable flash attention")

        super().__init__()
        head_dim = hidden_state // num_heads

        if flash:
            self.attention = FlashGQAAttention(
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, hidden_state=hidden_state, base=base, device=device)
        else:
            self.attention = GQAAttention(
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, hidden_state=hidden_state, base=base, device=device)

        self.mlp = MLP(mlp_dim=mlp_dim, hidden_state=hidden_state)
        self.layer_id = layer_id

        self.attention_norm = RMSNorm(hidden_size=hidden_state)
        self.mlp_norm = RMSNorm(hidden_size=hidden_state)

    def forward(self, x, position_ids, mask=None):
        h = x + self.attention(self.attention_norm(x), position_ids)

        out = h + self.mlp(x)

        return out


if __name__ == "__main__":
    device = "cpu"
    hidden_state = 512
    num_kv_heads = 4
    num_heads = 8

    x = torch.randn(4, 128, 512).to(device)
    position_ids = torch.stack([torch.arange(128)
                               for _ in range(4)], dim=0).to(device)

    block = BertBlock(0, hidden_state=hidden_state, mlp_dim=3072, num_heads=num_heads, num_kv_heads=num_kv_heads,
                      base=10000, flash=False).to(device)

    output = block(x, position_ids)
    print(output.shape)
