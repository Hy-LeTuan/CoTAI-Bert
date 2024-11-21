import torch.nn as nn
import torch
import math


def get_inv_freq(hidden_state, base=10000, device="cpu"):
    assert hidden_state % 2 == 0, print(
        f"Hidden state must be divisible by 2 to allow rotatory embedding, but received hidden state {hidden_state}")
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_state, 2,
                      dtype=torch.int64).float().to(device) / hidden_state))

    return inv_freq


def get_yarn_positional_embed(dim: int, original_max_position_embeddings: int, base: float = 10000.0, scale: int = 16, beta: int = 32, alpha: int = 1, mscale: float = 0.707,  max_position_embeddings: int = 2048, finetune=False):
    """
    https://medium.com/@zaiinn440/linear-rope-vs-ntk-vs-yarn-vs-cope-d33587ddfd35

    current alpha and beta values are tuned for Llama models. In case of smaller models, a rescale is imperative
    """

    # calculate base RoPE
    pos_freqs = base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    # calculate range of frequency bands
    low = max(math.floor(dim * math.log(original_max_position_embeddings /
              (beta * 2 * math.pi)))/(2 * math.log(base)), 0)
    high = min(math.ceil(dim * math.log(original_max_position_embeddings /
               (alpha * 2 * math.pi)))/(2 * math.log(base)), dim-1)

    # calculate ramp function for NTK by parts
    linear_func = (torch.arange(
        dim//2, dtype=torch.float32) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1).float().to(
        device=pos_freqs.device)

    inv_freq_mask = 1 - ramp_func
    inv_freq = inv_freq_interpolation * \
        (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    _mscale = float((0.1 * math.log(scale) + 1.0) * mscale)

    if finetune:
        t = torch.arange(max_position_embeddings, device=inv_freq.device,
                         dtype=inv_freq.dtype)
    else:
        t = torch.arange(original_max_position_embeddings, device=inv_freq.device,
                         dtype=inv_freq.dtype)

    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = emb.cos() * _mscale
    freqs_sin = emb.sin() * _mscale
    return freqs_cos, freqs_sin


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
    def __init__(self, hidden_state, base, max_position_embeddings=2048, device="cpu"):
        super().__init__()

        self.device = device
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.hidden_state = hidden_state

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


class DynamicNTKRotatoryEmbedding(RotatoryEmbedding):
    def __init__(self, *args, scaling_factor=1.0):
        super().__init__(*args)
        self.scaling_factor = scaling_factor

    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len /
                 self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.hidden_state / (self.hidden_state - 2))

            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.hidden_state, 2,
                         dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos, sin = super().forward(x, position_ids)
        return cos, sin


class YarnRotatoryEmbedding(RotatoryEmbedding):
    def __init__(self, *args, original_max_position_embeddings, scale, beta, alpha, mscale):
        super().__init__(*args)
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scale = scale
        self.beta = beta
        self.alpha = alpha
        self.mscale = mscale

    def forward(self, x, position_ids):
        bs, _, _ = x.shape
        print(bs)
        cos, sin = get_yarn_positional_embed(self.hidden_state, self.original_max_position_embeddings,
                                             self.base, self.scale, self.beta, self.alpha, self.mscale, self.max_position_embeddings)
        cos = cos[None, :, :].expand(bs, -1, -1)
        sin = sin[None, :, :].expand(bs, -1, -1)
        return cos, sin


if __name__ == "__main__":
    hidden_state = 256
    seq_len = 64
    x = torch.randn(4, seq_len, hidden_state)

    attention = DynamicNTKRotatoryEmbedding(
        hidden_state, 10000, 2048, "cpu", scaling_factor=1.0)

    yarn_attention = YarnRotatoryEmbedding(
        hidden_state, 10000, 2048, "cpu", original_max_position_embeddings=seq_len, scale=16, beta=32, alpha=1, mscale=0.707)

    position_ids = torch.stack([torch.arange(seq_len)
                               for _ in range(4)], dim=0)

    cos, sin = yarn_attention(x, position_ids)
    cos1, sin1 = attention(x, position_ids)

    print("yarn attention------")
    print(cos.shape)
    print(sin.shape)

    print("other attention-----")
    print(cos1.shape)
    print(sin1.shape)
