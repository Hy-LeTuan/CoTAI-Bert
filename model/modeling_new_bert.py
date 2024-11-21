import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.utils import ModelOutput


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(
        seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


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


def get_inv_freq(hidden_state, base=10000, device="cpu"):
    assert hidden_state % 2 == 0, print(
        f"Hidden state must be divisible by 2 to allow rotatory embedding, but received hidden state {hidden_state}")
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_state, 2,
                      dtype=torch.int64).float().to(device) / hidden_state))

    return inv_freq


def get_yarn_positional_embed(
    dim: int,
    original_max_position_embeddings: int,
    base: float = 10000.0,
    scale: int = 16,
    beta: int = 32,
    alpha: int = 1,
    mscale: float = 0.707,
    max_position_embeddings: int = 2048,
    finetune=False
):
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
              (beta * 2 * math.pi))) / (2 * math.log(base)), 0)
    high = min(math.ceil(dim * math.log(original_max_position_embeddings /
               (alpha * 2 * math.pi))) / (2 * math.log(base)), dim - 1)

    # calculate ramp function for NTK by parts
    linear_func = (torch.arange(
        dim // 2, dtype=torch.float32) - low) / (high - low)
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

    return torch.cat((-x2, x1), dim=-1).to(x.device)


def apply_rotatory_pos_embed(
    q,
    k,
    cos,
    sin,
    unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ROTATORY EMBEDDINGS
class RotatoryEmbedding(nn.Module):
    def __init__(
            self,
            hidden_state,
            base,
            max_position_embeddings=2048,
            device="cpu"
    ):
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
    def __init__(
        self,
        hidden_state,
        base,
        max_position_embeddings,
        device,
        scaling_factor=1.0
    ):
        super().__init__(
            hidden_state=hidden_state,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device,
        )
        self.scaling_factor = scaling_factor

    def forward(self, x, position_ids):
        seq_len = torch.max(position_ids) + 1

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) - (
                self.scaling_factor - 1)) ** (self.hidden_state / (self.hidden_state - 2))

            inv_freq = 1.0 / (base ** (torch.arange(0, self.hidden_state,
                              2, dtype=torch.int64).float().to(x.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        cos, sin = super().forward(x, position_ids)
        return cos, sin


class YarnRotatoryEmbedding(RotatoryEmbedding):
    def __init__(
            self,
            hidden_state,
            base,
            max_position_embeddings,
            device,
            original_max_position_embeddings,
            scale,
            beta,
            alpha,
            mscale,
            finetune=False
    ):
        super().__init__(
            hidden_state=hidden_state,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scale = scale
        self.beta = beta
        self.alpha = alpha
        self.mscale = mscale
        self.finetune = finetune

    def forward(self, x, position_ids):
        bs, _, seq_len, _ = x.shape

        cos, sin = get_yarn_positional_embed(
            dim=self.hidden_state,
            original_max_position_embeddings=seq_len,
            base=self.base,
            scale=self.scale,
            beta=self.beta,
            alpha=self.alpha,
            mscale=self.mscale,
            max_position_embeddings=self.max_position_embeddings,
            finetune=self.finetune
        )
        cos = cos[None, :, :].expand(bs, -1, -1).to(x.device)
        sin = sin[None, :, :].expand(bs, -1, -1).to(x.device)
        return cos, sin


# ATTENTION MODULES
class GQAAttention(nn.Module):
    def __init__(
            self,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_state,
            base=10000,
            device="cpu"
    ):
        super().__init__()
        self.hidden_state = hidden_state
        self.head_dim = head_dim
        self.base = base
        self.device = device

        assert num_heads % num_kv_heads == 0, print(
            f"Number of heads {num_heads} is not divisible by number of Key-Value heads {num_kv_heads}")

        self.num_heads = num_heads  # number of queries
        self.num_kv_heads = num_kv_heads  # number of keys and values
        self.num_groups = num_heads // num_kv_heads  #
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_state, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_state, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(
            hidden_state, num_kv_heads * head_dim, bias=False)

        # positional embedding
        self.rot_embed = RotatoryEmbedding(
            hidden_state=head_dim, base=base, device=device)

        # final projection to turn back to hidden state
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_state, bias=False)

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

    def forward(self, x, attention_mask, position_ids):
        bs, seq, _ = x.size()

        q = self.q_proj(x).view(bs, seq, self.num_heads,
                                self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq, self.num_kv_heads,
                                self.head_dim).transpose(1, 2)

        # apply rotational embedding
        cos, sin = self.rot_embed(v, position_ids)
        cos = cos.to(self.device)
        sin = sin.to(self.device)

        q, k = apply_rotatory_pos_embed(q, k, cos=cos, sin=sin)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        attn_weights = torch.matmul(
            q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attn_weights = attn_weights + attention_mask

        # calculate attention score and upcast to float32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # calculate final attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, seq, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output


class DynamicNTKGQAAttention(GQAAttention):
    def __init__(
            self,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_state,
            base=10000,
            max_position_embeddings=2048,
            device="cpu",
            scaling_factor=1.0
    ):
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_state=hidden_state,
            base=base,
            device=device
        )
        self.rot_embed = DynamicNTKRotatoryEmbedding(
            hidden_state=hidden_state,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device,
            scaling_factor=scaling_factor
        )

    def forward(self, x, mask, position_ids):
        return super().forward(x, mask, position_ids)


class YarnGQAAttention(GQAAttention):
    def __init__(
            self,
            *args,
            original_max_position_embeddings=128,
            max_position_embeddings=2048,
            scale=16,
            beta=32,
            alpha=1,
            mscale=0.707
    ):
        super().__init__(*args)
        self.rot_embed = YarnRotatoryEmbedding(
            self.head_dim,
            self.base,
            max_position_embeddings,
            self.device,
            original_max_position_embeddings=original_max_position_embeddings,
            scale=scale,
            beta=beta,
            alpha=alpha,
            mscale=mscale
        )

    def forward(self, x, mask, position_ids):
        return super().forward(x, mask, position_ids)


class FlashGQAAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        hidden_state,
        base=10000,
        max_position_embeddings=2048,
        device="cuda"
    ):
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

        self.q_proj = nn.Linear(hidden_state, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_state, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(
            hidden_state, num_kv_heads * head_dim, bias=False)

        # positional embedding
        self.rot_embed = RotatoryEmbedding(
            hidden_state=head_dim, base=base, max_position_embeddings=max_position_embeddings, device=device)

        # final projection to turn back to hidden state
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_state, bias=False)

        # alibi weight and slope
        self.register_buffer("m", get_alibi_slope(num_heads=num_heads))

    def _upad_input(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(
                batch_size * kv_seq_len,
                num_key_value_heads,
                head_dim
            ),
            indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(
                batch_size * kv_seq_len,
                num_key_value_heads,
                head_dim
            ),
            indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length):
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout_p=0.0, softmax_scale=None, causal=False)

        return attn_output

    def forward(self, x, attention_mask, position_ids):
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

        attn_output = self._flash_attention_forward(
            query_states=q, key_states=k, value_states=v, attention_mask=attention_mask, query_length=seq)

        attn_output = attn_output.to(x.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, seq, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DynamicNTKFlashGQAAttention(FlashGQAAttention):
    def __init__(
            self,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_state,
            base,
            max_position_embeddings,
            device,
            scaling_factor
    ):
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_state=hidden_state,
            base=base,
            device=device
        )

        self.rot_embed = DynamicNTKRotatoryEmbedding(
            hidden_state=head_dim,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device,
            scaling_factor=scaling_factor
        )

    def forward(self, x, attention_mask, position_ids):
        return super().forward(x, attention_mask, position_ids)


class YarnFlashGQAAttention(FlashGQAAttention):
    def __init__(
            self,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_state,
            base,
            max_position_embeddings,
            device,
            original_max_position_embeddings,
            scale,
            beta,
            alpha,
            mscale,
            finetune=False,
    ):
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_state=hidden_state,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device
        )

        self.rot_embed = YarnRotatoryEmbedding(
            hidden_state=head_dim,
            base=base,
            max_position_embeddings=max_position_embeddings,
            device=device,
            original_max_position_embeddings=original_max_position_embeddings,
            scale=scale,
            beta=beta,
            alpha=alpha,
            mscale=mscale,
            finetune=finetune
        )

    def forward(self, x, attention_mask, position_ids):
        return super().forward(x, attention_mask, position_ids)

# FEED FORWARD NETWORK


class MLP(nn.Module):
    def __init__(self, mlp_dim, hidden_state):
        super().__init__()

        self.w1 = nn.Linear(hidden_state, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, hidden_state, bias=False)
        self.w3 = nn.Linear(hidden_state, mlp_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))


# LAYER NORM
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


# BERT BLOCK
class BertBlock(nn.Module):
    def __init__(
            self,
            layer_id: int,
            hidden_state,
            mlp_dim, num_heads,
            num_kv_heads,
            base=10000,
            max_position_embeddings=2048,
            flash=False,
            device="cpu",
            immediate=False
    ):
        assert hidden_state % num_heads == 0, print(
            f"Hidden state is {hidden_state} not divisible by number of heads: {num_heads}")

        if flash is True and device != "cuda":
            raise ValueError("Device must be CUDA to enable flash attention")

        super().__init__()
        self.flash = flash
        self.immediate = immediate
        self.head_dim = hidden_state // num_heads

        if flash:
            self.attention = FlashGQAAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                hidden_state=hidden_state,
                base=base,
                max_position_embeddings=max_position_embeddings,
                device=device
            )
        else:
            self.attention = GQAAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                hidden_state=hidden_state,
                base=base,
                device=device
            )

        self.mlp = MLP(mlp_dim=mlp_dim, hidden_state=hidden_state)
        self.layer_id = layer_id

        self.attention_norm = RMSNorm(hidden_size=hidden_state)
        self.mlp_norm = RMSNorm(hidden_size=hidden_state)

    def _forward(self, x, position_ids, mask):
        h = x + self.attention(self.attention_norm(x), mask, position_ids)
        out = self.mlp_norm(h) + self.mlp(x)

        return out

    def forward(self, x, position_ids, mask):
        if self.immediate:
            out = self._forward(x, position_ids, mask)
            out = self._forward(out, position_ids, mask)
            return out

        h = x + self.attention(self.attention_norm(x), mask, position_ids)
        out = self.mlp_norm(h) + self.mlp(x)

        return out


class DynamicNTKBertBlock(BertBlock):
    def __init__(
            self,
            layer_id,
            hidden_state,
            mlp_dim,
            num_heads,
            num_kv_heads,
            base=10000,
            max_position_embeddings=2048,
            flash=True,
            device="cpu",
            immediate=False,
            scaling_factor=1.0
    ):
        super().__init__(
            layer_id=layer_id,
            hidden_state=hidden_state,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            base=base,
            max_position_embeddings=max_position_embeddings,
            flash=flash,
            device=device,
            immediate=immediate
        )

        if self.flash:
            self.attention = DynamicNTKFlashGQAAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                hidden_state=hidden_state,
                base=base,
                max_position_embeddings=max_position_embeddings,
                device=device,
                scaling_factor=scaling_factor
            )
        else:
            self.attention = DynamicNTKGQAAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                hidden_state=hidden_state,
                base=base,
                max_position_embeddings=max_position_embeddings,
                device=device,
                scaling_factor=scaling_factor
            )

    def forward(self, x, position_ids, mask=None):
        return super().forward(x, position_ids, mask)


class YarnBertBlock(BertBlock):
    def __init__(
        self,
        layer_id,
        hidden_state,
        mlp_dim,
        num_heads,
        num_kv_heads,
        base=10000,
        flash=False,
        device="cpu",
        immediate=False,
        original_max_position_embeddings=128,
        max_position_embeddings=2048,
        scale=16,
        beta=32,
        alpha=1,
        mscale=0.707
    ):
        super().__init__(
            layer_id=layer_id,
            hidden_state=hidden_state,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            base=base,
            max_position_embeddings=max_position_embeddings,
            flash=flash,
            device=device,
            immediate=immediate
        )

        if self.flash:
            self.attention = YarnFlashGQAAttention(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.head_dim,
                hidden_state=hidden_state,
                base=base,
                max_position_embeddings=max_position_embeddings,
                device=device,
                original_max_position_embeddings=original_max_position_embeddings,
                scale=scale,
                beta=beta,
                alpha=alpha,
                mscale=mscale
            )
        else:
            self.attention = YarnGQAAttention(
                num_heads,
                num_kv_heads,
                self.head_dim,
                hidden_state,
                base,
                device,
                original_max_position_embeddings=original_max_position_embeddings,
                max_position_embeddings=max_position_embeddings,
                beta=beta,
                alpha=alpha,
                scale=scale,
                mscale=mscale
            )

    def forward(self, x, position_ids, mask=None):
        return super().forward(x, position_ids, mask)


class CotaiBert(nn.Module):
    def __init__(
        self,
        num_blocks,
        hidden_state,
        mlp_dim,
        num_heads,
        num_kv_heads,
        base,
        flash=False,
        device="cpu",
        original_max_position_embeddings=128,
        max_position_embeddings=2048,
        scale=16,
        beta=32,
        alpha=1,
        mscale=0.707,
        scaling_factor=1.0,
        yarn=False,
        ntk=False,
        mask_id=52290,
        immediate=False
    ):
        super().__init__()

        vocab_size = 52293
        self.immediate = immediate

        self.register_buffer("mask_id", torch.tensor(
            [mask_id], dtype=torch.int), persistent=True)
        self.embed = nn.Embedding(vocab_size, hidden_state, padding_idx=52289)

        if not yarn and not ntk:
            self.blocks = nn.ModuleList([BertBlock(
                layer_id=id,
                hidden_state=hidden_state,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                base=base,
                max_position_embeddings=max_position_embeddings,
                flash=flash,
                device=device,
                immediate=immediate
            ).to(device) for id in range(num_blocks)])
        else:
            if ntk:
                self.blocks = nn.ModuleList([DynamicNTKBertBlock(
                    layer_id=id,
                    hidden_state=hidden_state,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    base=base,
                    max_position_embeddings=max_position_embeddings,
                    flash=flash,
                    device=device,
                    immediate=immediate,
                    scaling_factor=scaling_factor
                ).to(device) for id in range(num_blocks)])
            elif yarn:
                self.blocks = nn.ModuleList([YarnBertBlock(
                    id,
                    hidden_state,
                    mlp_dim,
                    num_heads,
                    num_kv_heads,
                    base,
                    flash,
                    device,
                    immediate,
                    original_max_position_embeddings=original_max_position_embeddings,
                    max_position_embeddings=max_position_embeddings,
                    beta=beta,
                    alpha=alpha,
                    scale=scale,
                    mscale=mscale
                ).to(device) for id in range(num_blocks)])

    def _forward(self, x, attention_mask, position_ids):
        for block in self.blocks:
            x = block(x, position_ids, attention_mask)

        return x

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            **kwargs
    ):
        x = self.embed(input_ids)

        # forward pass through model
        if self.immediate:
            x = self._forward(x, attention_mask, position_ids)
        else:
            x = self._forward(x, attention_mask, position_ids)
            x = self._forward(x, attention_mask, position_ids)

        mask_id = self.mask_id[0]
        x = x[input_ids == mask_id]

        logits = F.linear(x, self.embed.weight)

        return logits


class Hfwrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        bs, seq_len = input_ids.shape
        position_ids = torch.stack([torch.arange(seq_len)
                                   for _ in range(bs)]).to(input_ids.device)
        logits = self.model(input_ids, attention_mask,
                            None, position_ids, **kwargs)
        mask_map = input_ids == self.model.mask_id

        if labels is not None:
            labels = labels[mask_map]
            loss = self.loss(logits, labels)
            return ModelOutput(loss=loss, logits=logits, mask_map=mask_map, labels=labels)
        else:
            return ModelOutput(logits=logits, mask_map=mask_map, labels=labels)


if __name__ == "__main__":
    original_max_position_embeddings = 256
    max_position_embeddings = 2048
    num_heads = 9
    num_kv_heads = 3
    hidden_state = 576
    vocab_size = 52293 - 5  # calculate the number of non-speical tokens in tokenizer
    mlp_dim = 1536
    head_dim = hidden_state // num_heads
    base = 10000
    batch_size = 4
    device = "cuda"
    flash = True
    num_blocks = 15
    yarn = False
    ntk = False
    scaling_factor = 1.0
    immediate = False
    mask_id = 52290
    alpha = 1
    beta = 32
    scale = 16
    mscale = 0.707

    model = CotaiBert(
        num_blocks,
        hidden_state,
        mlp_dim,
        num_heads,
        num_kv_heads,
        base,
        flash=flash,
        device=device,
        original_max_position_embeddings=original_max_position_embeddings,
        max_position_embeddings=max_position_embeddings,
        scale=scale,
        beta=beta,
        alpha=alpha,
        mscale=mscale,
        scaling_factor=scaling_factor,
        yarn=yarn,
        ntk=ntk,
        mask_id=mask_id,
        immediate=immediate
    ).to(device)

    wrapper = Hfwrapper(model=model).to(device)

    # initialize input
    x = torch.randint(0, vocab_size, (batch_size,
                      original_max_position_embeddings)).to(device)
    labels = torch.randint(
        0, vocab_size, (batch_size, original_max_position_embeddings)).to(device)

    # manual masking
    x[0][0] = mask_id
    x[1][10] = mask_id
    x[1][20] = mask_id
    x[2][30] = mask_id
    x[3][60] = mask_id

    mask = None
    token_type_ids = None
    position_ids = torch.stack([torch.arange(
        original_max_position_embeddings) for _ in range(batch_size)]).to(device)

    model_output = wrapper(x, mask, labels)

    print("model output----")
    print(model_output)
