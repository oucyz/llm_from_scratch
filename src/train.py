"""学習ループと評価・可視化ユーティリティ。

本モジュールは簡易的なトレーニング手順、評価関数、プロット関数を提供します。
ロジックは既存のまま、型アノテーションと日本語Googleスタイルdocstringのみを追加しています。
"""

from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import tiktoken
import torch
from matplotlib.ticker import MaxNLocator

from .generate_text import text_to_token_ids, token_ids_to_text
from .models.dummy_gpt_model import GPTModel
from .models.loss import calc_loss_batch, calc_loss_loader
from .postprocess.postprocess import generate, generate_text_simple
from .preprocess.preprocess_text import create_dataloader_v1


def train_model_simple(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
) -> Tuple[List[float], List[float], List[int]]:
    """定期的な評価とサンプリングを行いながら言語モデルを学習します。

    Args:
        model: 学習対象のモデル。
        train_loader: 学習用の ``(input_batch, target_batch)`` を返すデータローダ。
        val_loader: 検証用の ``(input_batch, target_batch)`` を返すデータローダ。
        optimizer: モデルパラメータを更新するオプティマイザ。
        device: 学習に使用するTorchデバイス。
        num_epochs: 学習を行うエポック数。
        eval_freq: 何ステップごとに評価するか。
        eval_iter: 評価時に平均化するバッチ数。
        start_context: サンプル生成時のプロンプト文字列。
        tokenizer: ヘルパ関数と互換のトークナイザ。

    Returns:
        Tuple[List[float], List[float], List[int]]: 3つのリストからなるタプル。
        - train_losses: 各評価時点の学習損失の平均。
        - val_losses: 各評価時点の検証損失の平均。
        - track_tokens_seen: 各評価時点までに処理したトークン数。
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1} (Step {global_step:06d}): ",
                    f"Train Loss: {train_loss:.3f}",
                    f"Val Loss: {val_loss:.3f}",
                )
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    eval_iter: int,
) -> Tuple[float, float]:
    """学習/検証ローダで評価し、それぞれの平均損失を返します。

    Args:
        model: 評価対象のモデル。
        train_loader: 学習損失の見積もりに用いるローダ。
        val_loader: 検証損失の見積もりに用いるローダ。
        device: 評価に使用するTorchデバイス。
        eval_iter: それぞれの損失で平均化するバッチ数。

    Returns:
        Tuple[float, float]: ``(train_loss, val_loss)`` のタプル。
    """

    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    start_context: str,
) -> None:
    """学習済みまたは学習中のモデルから短いサンプルを生成して表示します。

    Args:
        model: 言語モデル本体。
        tokenizer: ヘルパ関数と互換のトークナイザ。
        device: 生成に使用するTorchデバイス。
        start_context: 生成を開始するためのプロンプト文字列。
    """

    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=1.0,
            top_k=10,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("Output text:\n", decoded_text.replace("\n", " "))
    model.train()


def plot_losses(
    epochs_seen: Sequence[float],
    tokens_seen: Sequence[int],
    train_losses: Sequence[float],
    val_losses: Sequence[float],
) -> None:
    """エポック数と処理トークン数に対する学習/検証損失をプロットします。

    Args:
        epochs_seen: 各記録点に対応するエポックの値。
        tokens_seen: 累計で処理したトークン数（第2のX軸）。
        train_losses: 記録された学習損失。
        val_losses: 記録された検証損失。
    """

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen")
    fig.tight_layout()
    plt.show()


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
        max_new_tokens=10,
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

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=int(GPT_CONFIG_124M["context_length"]),
        stride=int(GPT_CONFIG_124M["context_length"]),
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=int(GPT_CONFIG_124M["context_length"]),
        stride=int(GPT_CONFIG_124M["context_length"]),
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # print("train loader")
    # for x, y in train_loader:
    #     print("Input:", x.shape)
    #     print("Target:", y.shape)
    # print("val loader")
    # for x, y in val_loader:
    #     print("Input:", x.shape)
    #     print("Target:", y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "./output/gpt_model_and_adamw.pth",
    )
