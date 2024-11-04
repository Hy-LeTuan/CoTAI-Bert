from flash_attn import flash_attn_func
import torch.nn as nn
import torch
import math
from rotatory_positional_embedding import RotatoryEmbedding, apply_rotatory_pos_embed


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


def get_alibi_weight(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class GQAAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, hidden_state, base=10000, device="cpu"):
        super().__init__()
        assert num_heads % num_kv_heads == 0, print(
            f"Number of heads {num_heads} is not divisible by number of Key-Value heads {num_kv_heads}")

        self.num_heads = num_heads  # number of queries
        self.num_kv_heads = num_kv_heads  # number of keys and values
        self.num_groups = num_heads // num_kv_heads  #
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_state, num_heads*head_dim)
        self.k_proj = nn.Linear(hidden_state, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_state, num_kv_heads * head_dim)

        # positional embedding
        self.rot_embed = RotatoryEmbedding(
            hidden_state=head_dim, base=base, device=device)

        # final projection to turn back to hidden state
        self.o_proj = nn.Linear(num_heads*head_dim, hidden_state)

        # alibi weight and slope
        self.register_buffer("m", get_alibi_slope(num_heads=num_heads))

    def repeat_kv(self, kv):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = kv.shape

        if self.num_groups == 1:
            return kv

        kv = kv[:, :, None, :, :].expand(
            batch, num_key_value_heads, self.num_groups, slen, head_dim)

        return kv.reshape(batch, num_key_value_heads * self.num_groups, slen, head_dim)

    def forward(self, x, position_ids):
        bs, seq, _ = x.size()

        q = self.q_proj(x).view(bs, seq, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)

        # apply rotational embedding
        cos, sin = self.rot_embed(v, position_ids)
        q, k = apply_rotatory_pos_embed(q, k, cos=cos, sin=sin)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        attn_weights = torch.matmul(
            q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # calculate attention score and upcast to float32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.attention_dropout, training=self.training)

        # calculate final attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, seq, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class FlashGQAAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, hidden_state, base=10000, device="cuda"):
        """
        Flash attention version of the ordinary GQA attention. This module is only usable on a device with CUDA enabled
        """
        super().__init__()

        # assertions to prevent runtime error
        assert num_heads % num_kv_heads == 0, print(
            f"Number of heads {num_heads} is not divisible by number of Key-Value heads {num_kv_heads}")
        assert device == "cuda", print(
            f"Current device {device} is not suitable for flash attention. Device required: CUDA")

        self.device = device

        self.num_heads = num_heads  # number of queries
        self.num_kv_heads = num_kv_heads  # number of keys and values
        self.num_groups = num_heads // num_kv_heads  #
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_state, num_heads*head_dim)
        self.k_proj = nn.Linear(hidden_state, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_state, num_kv_heads * head_dim)

        # positional embedding
        self.rot_embed = RotatoryEmbedding(
            hidden_state=head_dim, base=base, device=device)

        # final projection to turn back to hidden state
        self.o_proj = nn.Linear(num_heads*head_dim, hidden_state)

        # alibi weight and slope
        self.register_buffer("m", get_alibi_slope(num_heads=num_heads))

    def forward(self, x, position_ids):
        bs, seq, _ = x.size()

        q = self.q_proj(x).view(bs, seq, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)

        # apply rotational positional embedding before doing flash attention
        cos, sin = self.rot_embed(v, position_ids)
        q, k = apply_rotatory_pos_embed(q, k, cos=cos, sin=sin)

        # flash attention requires bfloat16 data type
        q = q.transpose(1, 2).bfloat16().to("cuda")
        k = k.transpose(1, 2).bfloat16().to("cuda")
        v = v.transpose(1, 2).bfloat16().to("cuda")

        attn_output = flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, alibi_slopes=None, deterministic=False).to(x.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, seq, -1)
        attn_output = self.o_proj(attn_output)

        print("output-----")
        print(attn_output.shape)


if __name__ == "__main__":
    device = "cuda"
    hidden_state = 512
    num_kv_heads = 4
    num_heads = 8
    head_dim = hidden_state // num_heads

    x = torch.randn(4, 128, 512).to(device)
    position_ids = torch.stack([torch.arange(128)
                               for _ in range(4)], dim=0).to(device)

    # att = GQAAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
    #                    head_dim=head_dim, hidden_state=hidden_state)

    att = FlashGQAAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                            head_dim=head_dim, hidden_state=hidden_state).to(device)

    att(x, position_ids=position_ids)
