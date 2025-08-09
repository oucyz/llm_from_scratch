import torch
import torch.nn as nn


class SelfAttentionV3(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自己注意を計算します（シンプル版）。

        引数:
            x: 入力（系列長 x d_in）。

        戻り値:
            文脈ベクトル（系列長 x d_out）。
        """
        query = self.W_query(x)  # 系列長 x d_out
        key = self.W_key(x)  # 系列長 x d_out
        attention_scores = query @ key.T  # 系列長 x 系列長

        # マスクを適用して上三角行列を無効化
        context_length = attention_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
        attention_weights = torch.softmax(attention_scores / key.shape[-1] ** 0.5, dim=-1)

        value = self.W_value(x)  # 系列長 x d_out
        context_vec = attention_weights @ value  # 系列長 x d_out
        return context_vec


class CausalAttention(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, context_length: int, dropout: float, bias: bool = False
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """因果マスク付き自己注意を計算します。

        引数:
            x: 入力テンソル（B, T, d_in）。

        戻り値:
            文脈ベクトル（B, T, d_out）。
        """
        _batch, num_tokens, _d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.d_out**0.5, dim=-1)

        attention_weights = self.dropout(attention_weights)

        values = self.W_value(x)
        context_vec = attention_weights @ values

        return context_vec


if __name__ == "__main__":
    n = 6
    d_in = 8
    d_out = 8
    embedding = torch.tensor(torch.rand(n, d_in), dtype=torch.float32)
    print(embedding.shape)  # torch.Size([6, 8])

    self_attention = SelfAttentionV3(d_in=d_in, d_out=d_out, bias=False)
    output = self_attention(embedding)
    print(output)

    x = torch.stack([embedding, embedding], dim=0)
    print(x.shape)  # torch.Size([2, 6, 8])
    causal_attention = CausalAttention(
        d_in=d_in, d_out=d_out, context_length=n, dropout=0.5, bias=False
    )
    output = causal_attention(x)
    print(output)
