from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """1バッチ分のクロスエントロピー損失を計算します。

    Args:
        input_batch: 入力テンソル。通常は (batch_size, seq_len) 形状のトークンID。
        target_batch: 教師トークンIDのテンソル (batch_size, seq_len)。
        model: 入力をロジット (batch_size, seq_len, vocab_size) に写像する
            自己回帰言語モデル。
        device: 計算に使用するTorchデバイス。

    Returns:
        バッチおよび時刻方向で平均したスカラーのクロスエントロピー損失。
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(
    data_loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> float:
    """データローダ全体（または先頭の指定バッチ数）に対する平均損失を計算します。

    Args:
        data_loader: ``(input_batch, target_batch)`` のペアを返すシーケンス。
            各テンソルは (batch_size, seq_len) を想定します。
        model: 評価対象のモデル。
        device: 評価を実行するTorchデバイス。
        num_batches: 評価に用いるバッチ数の上限。None の場合はローダ全体。

    Returns:
        指定バッチ数にわたる平均損失（float）。ローダが空の場合は ``nan``。
    """

    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches <= i:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches
