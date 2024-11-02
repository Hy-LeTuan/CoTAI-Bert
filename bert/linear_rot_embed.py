import torch.nn as nn
import torch


def get_inv_freq(hidden_state, base=10000, device="cpu"):
    assert hidden_state % 2 == 0, print(
        f"Hidden state must be divisible by 2 to allow rotatory embedding, but received hidden state {hidden_state}")
    res = base ** (torch.arange(0, hidden_state, 2,
                   dtype=torch.int64).float().to(device) / hidden_state)
    res = 1 / res
    return res


class RotatoryEmbedding(nn.Module):
    def __init__(self, hidden_state, base, device="cpu"):
        super().__init__()

        # initialize theta frequency modified without the exponent 2
        inv_freq = get_inv_freq(
            hidden_state=hidden_state, base=base, device=device)
        self.register_buffer("inv", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None,
                                          :, None].float().expand(position_ids.shape[0], -1, 1)


if __name__ == "__main__":
    res = get_theta(768)
    print(res.shape)
    print(res)
