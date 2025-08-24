import torch
import torch.nn as nn


class GELU(nn.Module):
    """GELU（Gaussian Error Linear Unit）活性化関数の実装。

    Transformerモデルでよく使用される活性化関数で、ReLUの滑らかな代替として機能します。
    近似版の実装を使用しています。
    """

    def __init__(self):
        """GELUクラスのコンストラクタ。"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """近似版GELU活性化関数を適用します。

        Args:
            x (torch.Tensor): 入力テンソル。任意の形状。

        Returns:
            torch.Tensor: GELU活性化関数を適用したテンソル。入力と同じ形状。
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


class Switch(nn.Module):
    """Swish活性化関数（SiLUとも呼ばれる）の実装。

    Swish(x) = x * sigmoid(W*x + b)の形で実装されています。
    線形変換を含むパラメータ化されたバージョンです。
    """

    def __init__(self, d_in: int, d_out: int):
        """Switchクラスのコンストラクタ。

        Args:
            d_in (int): 入力次元数。
            d_out (int): 出力次元数。
        """
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swish活性化関数を適用します。

        Args:
            x (torch.Tensor): 入力テンソル。形状は(..., d_in)。

        Returns:
            torch.Tensor: Swish活性化関数を適用したテンソル。形状は(..., d_out)。
        """
        return x * torch.sigmoid(self.linear(x))


class SwiGLU(nn.Module):
    """SwiGLU（Swish Gated Linear Unit）活性化関数の実装。

    GLU（Gated Linear Unit）の一種で、Swish活性化関数を使用します。
    PaLMやLLaMAなどの大規模言語モデルで使用される活性化関数です。
    SwiGLU(x) = Swish(W1*x) ⊙ W2*x の形で実装されています。
    """

    def __init__(self, d_in: int, d_out: int):
        """SwiGLUクラスのコンストラクタ。

        Args:
            d_in (int): 入力次元数。
            d_out (int): 出力次元数。
        """
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out)
        self.linear2 = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU活性化関数を適用します。

        Args:
            x (torch.Tensor): 入力テンソル。形状は(..., d_in)。

        Returns:
            torch.Tensor: SwiGLU活性化関数を適用したテンソル。形状は(..., d_out)。
        """
        swish = self.linear1(x)
        swish = swish * torch.sigmoid(swish)

        gate = self.linear2(x)
        return swish * gate


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    # (-5 <= x <= 5, -1 <= y <= 5)の範囲で、GLUとSwish, SwiGLUのx-yグラフを1つのグラフで描画する
    x = torch.linspace(-5, 5, 100)
    gelu = GELU()
    swish = Switch(d_in=1, d_out=1)
    swiglu = SwiGLU(d_in=1, d_out=1)

    plt.figure(figsize=(12, 6))
    plt.title("Activation Functions")
    plt.plot(x.numpy(), gelu(x.unsqueeze(1)).detach().numpy(), label="GELU")
    plt.plot(x.numpy(), swish(x.unsqueeze(1)).detach().numpy(), label="Swish")
    plt.plot(x.numpy(), swiglu(x.unsqueeze(1)).detach().numpy(), label="SwiGLU")
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-1, 5)
    plt.show()
