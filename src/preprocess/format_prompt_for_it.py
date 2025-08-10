from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..dl_it_data import download_and_load_file


def format_input(entry: Dict[str, str]) -> str:
    """命令と任意の追加入力を所定のプロンプト形式に整形します。

    Args:
        entry: 単一サンプルの辞書。少なくとも ``"instruction"``, ``"input"`` キーを持つことを想定。

    Returns:
        str: 整形済みのテキスト（"Instruction" と "Input" セクションを含む）。
    """
    instruction_text = (
        f"Below is an instruction that describes a task."
        f" Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def split_entries(
    entry: List[Dict[str, str]],
    train_ratio: float = 0.85,
    test_ratio: float = 0.1,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """リスト化されたサンプルを学習/検証/テストに分割します。

    Args:
        entry: サンプルのリスト。各要素は ``instruction``, ``input``, ``output`` 等を含む辞書。
        train_ratio: 学習データに割り当てる比率。
        test_ratio: テストデータに割り当てる比率（残りは検証）。

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        ``(train_data, val_data, test_data)`` のタプル。
    """
    train_portion = int(len(entry) * train_ratio)
    test_portion = int(len(entry) * test_ratio)

    train_data = entry[:train_portion]
    test_data = entry[train_portion : train_portion + test_portion]
    val_data = entry[train_portion + test_portion :]

    print("Training data:", len(train_data))
    print("Validation data:", len(val_data))
    print("Testing data:", len(test_data))
    return train_data, val_data, test_data


class InstructionDataset(Dataset):
    """命令追従のためのフルテキスト（命令+入力+応答）を符号化して保持するDataset。

    各サンプルは以下を連結したテキストをトークンID列へエンコードします：
    - Instruction + optional Input + Response

    Args:
        data: サンプルのリスト。
        tokenizer: tiktoken互換のエンコーダ（``encode`` メソッドを持つ）。
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer: Any) -> None:
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self) -> int:
        """データセット内のサンプル数を返します。"""
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> List[int]:
        """インデックスで指定したサンプルのトークンID列を返します。

        Args:
            idx: 取得するサンプルのインデックス。

        Returns:
            List[int]: エンコード済みトークンID列。
        """
        return self.encoded_texts[idx]


def custom_collate_fn(
    batch: List[List[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """命令追従タスク向けのカスタムcollate関数。

    右シフトしたターゲットを作成し、バッチ内の最大長に合わせてパディングします。
    末尾のパディング位置以降のターゲットは ``ignore_index`` でマスクします。

    Args:
        batch: 1サンプル=トークンID列（リスト[int]）のミニバッチ。
        pad_token_id: パディングに用いるトークンID。
        ignore_index: 損失計算で無視するインデックス。
        allowed_max_length: 許容する最大長（トランケーション）。``None`` で無効。
        device: 返却テンソルを配置するデバイス（例: "cpu" または ``torch.device("cuda")``）。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(inputs, targets)`` テンソル。
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, target_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(torch.tensor(inputs))
        target_lst.append(torch.tensor(targets))

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(target_lst).to(device)

    return inputs_tensor, targets_tensor


def create_instruction_tuning_dataset(
    entry: List[Dict[str, str]],
    tokenizer: Any,
    train_ratio: float = 0.85,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    batch_size: int = 8,
    allowed_max_length: int = 1024,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """命令追従データから学習/検証/テスト用のDataLoaderを作成します。

    Args:
        entry: サンプル辞書のリスト。
        tokenizer: tiktoken互換のエンコーダ。
        train_ratio: 学習データ比率。
        test_ratio: テストデータ比率。
        num_workers: DataLoaderのワーカ数。
        batch_size: バッチサイズ。
        allowed_max_length: 入力の最大長（トランケーション長）。
        device: 返却テンソルの配置デバイス。

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: ``(train_loader, val_loader, test_loader)``。
    """
    train_data, val_data, test_data = split_entries(entry, train_ratio, test_ratio)

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            custom_collate_fn, device=device, allowed_max_length=allowed_max_length
        ),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            custom_collate_fn, device=device, allowed_max_length=allowed_max_length
        ),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            custom_collate_fn, device=device, allowed_max_length=allowed_max_length
        ),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    file_path = "./output/instruction_data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)
    print("number of entries:", len(data))
    idx = 999
    print(data[idx])

    formatted_prompt = format_input(data[idx])
    print(formatted_prompt + f"\n\n### Response:\n{data[idx]['output']}")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training data:", len(train_data))
    print("Validation data:", len(val_data))
    print("Testing data:", len(test_data))

    inputs_1 = list(range(5))
    inputs_2 = list(range(2))
    inputs_3 = list(range(3))
    batch = [inputs_1, inputs_2, inputs_3]
    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

    train_data_loader, val_data_loader, test_data_loader = create_instruction_tuning_dataset(
        data,
        tokenizer=tiktoken.get_encoding("gpt2"),
        train_ratio=0.85,
        test_ratio=0.1,
        num_workers=0,
        batch_size=8,
        allowed_max_length=1024,
        device="cpu",
    )
    print("Train loader:")
    for inputs, targets in train_data_loader:
        print(inputs.shape, targets.shape)
        print(inputs[0])
        print(targets[0])
