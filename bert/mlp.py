import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, mlp_dim, hidden_state):
        super().__init__()

        self.w1 = nn.Linear(hidden_state, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, hidden_state, bias=False)
        self.w3 = nn.Linear(hidden_state, mlp_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))


if __name__ == "__main__":
    x = torch.randn(4, 128, 512)
    mlp = MLP(mlp_dim=3092, hidden_state=512)

    res = mlp(x)
    print(res.shape)
