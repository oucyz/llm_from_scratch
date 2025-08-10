from typing import Optional

import torch

from src.config.config import GPT_CONFIG_124M
from src.models.dummy_gpt_model import GPTModel


def generate_text_simple(
    model: torch.nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """貪欲法によりトークンを生成します。

    各ステップで直近の ``context_size`` トークンをモデルに入力し、語彙分布の
    argmax を次のトークンとして末尾に追加する処理を ``max_new_tokens`` 回繰り返します。

    Args:
        model: 入力トークン列 ``(B, T)`` をロジット ``(B, T, V)`` に写像する呼び出し可能オブジェクト。
        idx: 生成の出発点となるトークン列（dtype: ``torch.long``、形状 ``(B, T0)``）。
        max_new_tokens: 生成する新規トークン数。
        context_size: 各生成ステップで参照する直近トークン数（モデルのコンテキスト長）。

    Returns:
        torch.Tensor: 生成後のトークン列。形状 ``(B, T0 + max_new_tokens)``。

    Notes:
        - 形状記法: B = バッチ、T = 時系列/コンテキスト長、V = 語彙サイズ。
          モデルは ``(B, T)`` を受け取り ``(B, T, V)`` のロジットを返すことを想定しています。
        - 本実装は貪欲デコード（argmax）のみです。サンプリングや温度付きデコード等を行う場合は、
          確率からトークンを選択する箇所を調整してください。
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Batch x Context
        with torch.no_grad():
            logits = model(idx_cond)  # Batch x Context x Vocab
        probas = torch.softmax(logits[:, -1, :], dim=-1)  # B x V
        next_token = torch.argmax(probas, dim=-1, keepdim=True)  # B x 1
        idx = torch.cat((idx, next_token), dim=1)
    return idx


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """温度・top-kサンプリングに対応した生成ループ。

    Args:
        model: 言語モデル（``(B, T) -> (B, T, V)`` のロジットを返す）。
        idx: 生成の出発点となるトークン列（形状 ``(B, T0)``）。
        max_new_tokens: 生成する新規トークン数。
        context_size: 各生成ステップで参照する直近トークン数。
        temperature: 温度パラメータ。0.0 の場合は貪欲（argmax）。
        top_k: 上位 ``k`` 語彙のみからサンプリングする場合の ``k``。``None`` で無効。
        eos_id: 生成停止とする語彙ID（出現したら早期停止）。

    Returns:
        torch.Tensor: 生成後のトークン列。
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Batch x Context
        with torch.no_grad():
            logits = model(idx_cond)  # Batch x Context x Vocab
        logits = logits[:, -1, :]  # Batch x Vocab

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_logits[:, -1], torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == "__main__":
    import tiktoken

    text = [
        "Once upon a time, there was a brave knight who",
        "昔むかしあるところに、勇敢な騎士がいました。",
    ]

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text[1])
    encoded = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    print(encoded)

    model = GPTModel(GPT_CONFIG_124M).eval()
    output = generate_text_simple(
        model=model,
        idx=encoded,
        max_new_tokens=6,
        context_size=int(GPT_CONFIG_124M["context_length"]),
    )
    print("output:", output)
    print("output (decoded):", tokenizer.decode(output[0].tolist()))

    output = generate(
        model=model,
        idx=encoded,
        max_new_tokens=25,
        context_size=int(GPT_CONFIG_124M["context_length"]),
        temperature=1.4,
        top_k=20,
    )
    print("output:", output)
    print("output (decoded):", tokenizer.decode(output[0].tolist()))
