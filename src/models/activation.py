import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """近似版GELU活性化関数を適用します。

        Args:
            x: 入力テンソル。

        Returns:
            GELU を適用したテンソル。
        """
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
