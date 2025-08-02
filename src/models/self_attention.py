import torch
import torch.nn as nn


class SelfAttentionV2(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.W_query(x)  # 系列長 x d_out
        key = self.W_key(x)  # 系列長 x d_out
        attention_weights = query @ key.T  # 系列長 x 系列長
        normalized_attention_weights = torch.softmax(
            attention_weights / key.shape[-1] ** 0.5, dim=-1
        )

        value = self.W_value(x)  # 系列長 x d_out
        context_vec = normalized_attention_weights @ value  # 系列長 x d_out
        return context_vec


if __name__ == "__main__":
    n = 6
    d_in = 8
    d_out = 8
    embedding = torch.tensor(torch.rand(n, d_in), dtype=torch.float32)
    print(embedding.shape)  # torch.Size([6, 8])

    self_attention = SelfAttentionV2(d_in=d_in, d_out=d_out, bias=False)
    output = self_attention(embedding)
    print(output)
