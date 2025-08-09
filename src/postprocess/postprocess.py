import torch

from src.config.config import GPT_CONFIG_124M
from src.models.dummy_gpt_model import GPTModel


def generate_text_simple(model, idx, max_new_tokens: int, context_size: int):
    """貪欲法によりトークンを生成します。

    各ステップで直近の ``context_size`` トークンをモデルに入力し、語彙分布の
    argmax を次のトークンとして末尾に追加する処理を ``max_new_tokens`` 回繰り返します。

    Args:
        model: 入力トークン列 ``(B, T)`` をロジット ``(B, T, V)`` に写像する呼び出し可能オブジェクト
            （例: ``torch.nn.Module``）。
        idx (torch.Tensor): 生成の出発点となるトークン列（dtype: ``torch.long``、形状 ``(B, T0)``）。
        max_new_tokens (int): 生成する新規トークン数。
        context_size (int): 各生成ステップで参照する直近トークン数（モデルのコンテキスト長）。

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
