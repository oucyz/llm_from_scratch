import torch
import torch.nn as nn

from .self_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """複数の因果自己注意ヘッドを並列に適用するラッパ。

        引数:
            d_in: 入力埋め込み次元。
            d_out: 各ヘッド出力の総和次元。
            context_length: コンテキスト長（マスク作成に使用）。
            dropout: ドロップアウト率。
            num_heads: ヘッド数。
            qkv_bias: Q/K/V の線形層にバイアスを付与するか。
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """各ヘッドの出力を最後の次元で連結して返します。"""
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """GPT-2スタイルのマルチヘッド注意。

        引数:
            d_in: 入力埋め込み次元。
            d_out: 出力埋め込み次元（ヘッド数で割り切れること）。
            context_length: コンテキスト長（因果マスクのサイズ）。
            dropout: ドロップアウト率。
            num_heads: ヘッド数。
            qkv_bias: Q/K/V にバイアスを与えるか。
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前方計算を行い、コンテキストに基づく表現を返します。"""
        batch, num_tokens, _d_in = x.shape

        keys = self.W_key(x).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = (
            self.W_query(x).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        )
        values = (
            self.W_value(x).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        )
        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = (attention_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    batch = 2
    d_in = 3
    d_out = 2
    context_length = 6
    dropout = 0.5
    num_heads = 2

    x = torch.randn(batch, context_length, d_in)
    multi_head_attention = MultiHeadAttentionWrapper(
        d_in, d_out, context_length, dropout, num_heads
    )
    output = multi_head_attention(x)
    print(output.shape)
    print(output)

    # GPT-2 style multi-head attention
    batch = 2
    d_in = 768
    d_out = 768 * 12
    context_length = 1024
    dropout = 0.5
    num_heads = 12
    x = torch.randn(batch, context_length, d_in)
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)
    output_mha = mha(x)
    print(output_mha.shape)
    print(output_mha)
