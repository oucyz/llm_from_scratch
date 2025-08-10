"""テキスト/トークン変換ヘルパーとデコード補助関数群。

本モジュールはテキストをトークンへ、トークンをテキストへ変換する関数や、
サンプリング関連のユーティリティを提供します。既存ロジックは保持し、
型アノテーションと日本語Googleスタイルdocstringのみを追加しています。
"""

from typing import Any

import tiktoken
import torch

from .models.dummy_gpt_model import GPTModel
from .postprocess.postprocess import generate_text_simple


def text_to_token_ids(text: str, tokenizer: Any) -> torch.Tensor:
    """テキストをトークンID列へエンコードし、バッチ次元を付与します。

    Args:
        text: トークナイズ対象の文字列。
        tokenizer: tiktoken互換のエンコーダ（``encode`` メソッドを持つ）。

    Returns:
        torch.Tensor: 形状 ``(1, seq_len)`` のトークンIDテンソル。

    Notes:
        - 特殊トークン ``<|endoftext|>`` を許可してエンコードします。
    """

    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Any) -> str:
    """バッチ1のトークンIDテンソルをテキストへデコードします。

    Args:
        token_ids: 形状 ``(1, seq_len)`` または ``(seq_len,)`` のトークンIDテンソル。
        tokenizer: tiktoken互換のデコーダ（``decode`` メソッドを持つ）。

    Returns:
        str: 復元された文字列。
    """

    flat = token_ids.squeeze(0)  # Remove batch dimension
    decoded = tokenizer.decode(flat.tolist())
    return decoded


def print_sampled_tokens(probas: torch.Tensor, inverse_vocab: Any) -> None:
    """確率分布からサンプリングしたトークンの頻度を出力します。

    Args:
        probas: カテゴリ分布（語彙サイズ）の確率テンソル。``(V,)`` を想定。
        inverse_vocab: トークンIDから文字列への写像（インデックスアクセス可能）。

    Returns:
        なし
    """
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """温度付きソフトマックスを適用します。

    Args:
        logits: ロジットのテンソル。
        temperature: 温度パラメータ。大きいほど分布が平坦になります。

    Returns:
        torch.Tensor: 温度スケーリング後のソフトマックス確率。
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=25,
        context_size=int(GPT_CONFIG_124M["context_length"]),
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    file_path = "src/data/the_verdict.txt"
    with open(file_path, "r") as f:
        verdict_text = f.read()

    total_characters = len(verdict_text)
    total_tokens = len(tokenizer.encode(verdict_text, allowed_special={"<|endoftext|>"}))
    print(f"Total characters in the verdict: {total_characters}")
    print(f"Total tokens in the verdict: {total_tokens}")

    train_ratio = 0.9
    split_idx = int(train_ratio * len(verdict_text))
    train_data = verdict_text[:split_idx]
    val_data = verdict_text[split_idx:]
