import torch.nn as nn
import torch


def get_inv_freq(hidden_state, base=10000, device="cpu"):
    assert hidden_state % 2 == 0, print(
        f"Hidden state must be divisible by 2 to allow rotatory embedding, but received hidden state {hidden_state}")
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_state, 2,
                      dtype=torch.int64).float().to(device) / hidden_state))

    return inv_freq


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotatory_pos_embed(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotatoryEmbedding(nn.Module):
    def __init__(self, hidden_state, base, device="cpu"):
        super().__init__()

        self.device = device

        # initialize theta frequency
        inv_freq = get_inv_freq(
            hidden_state=hidden_state, base=base, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None,
                                          :, None].float().expand(position_ids.shape[0], -1, -1)
        position_ids_expanded = position_ids[:,
                                             None, :].float()

        # [[theta_1 * m0, theta_1 * m1, theta_1 * m2....]
        # [theta_2 * m0, theta_2 * m1, theta_2 * m2....]]
        freqs = (inv_freq_expanded.float() @
                 position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, embed_dim)

        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype).to(self.device), sin.to(dtype=x.dtype).to(self.device)


if __name__ == "__main__":
    hidden_state = 256
    seq_len = 64
    x = torch.randn(4, seq_len, hidden_state)

    attention = RotatoryEmbedding(
        hidden_state=hidden_state, base=10000, device="cpu")

    position_ids = torch.stack([torch.arange(seq_len)
                               for _ in range(4)], dim=0)
    res = attention(x, position_ids)

    res = apply_rotatory_pos_embed(torch.randn(4, 128, 512),
                                   torch.randn(4, 128, 512), cos=torch.randn(128, 512), sin=torch.randn(128, 512))
