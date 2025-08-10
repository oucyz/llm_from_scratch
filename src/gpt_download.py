# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


"""GPT-2 モデルのダウンロードとTensorFlowチェックポイントの読み込みユーティリティ。

本モジュールは以下を提供します。

- 指定したサイズのGPT-2モデルファイル群のダウンロード
- TensorFlowチェックポイントからの重み読み込みとPython辞書への展開

注意: 関数のロジックは既存実装を維持し、型アノテーションとdocstringのみを追加しています。
"""

# import requests
import json
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(
    model_size: str, models_dir: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """指定したサイズのGPT-2モデルをダウンロードし、設定と重みを読み込みます。

    Args:
        model_size: モデルサイズ。"124M"、"355M"、"774M"、"1558M" のいずれか。
        models_dir: モデルファイルを格納するベースディレクトリのパス。

    Returns:
        settings: ``hparams.json`` の内容（ハイパーパラメータ設定）。
        params: TensorFlowチェックポイントから展開した重み辞書（多重ネストされた辞書）。

    Raises:
        ValueError: ``model_size`` が許可された値でない場合。
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url: str, destination: str, backup_url: Optional[str] = None) -> None:
    """URL からファイルをダウンロードします（バックアップURLにフェイルオーバー対応）。

    既存ファイルがありサイズが一致する場合は再ダウンロードを行いません。進捗バーを表示します。

    Args:
        url: プライマリのダウンロードURL。
        destination: ダウンロード先のローカルパス。
        backup_url: プライマリが失敗した際に試行するバックアップURL（任意）。

    Returns:
        なし

    Notes:
        - ネットワークエラー時はエラーメッセージを標準出力に表示します。
    """

    def _attempt_download(download_url: str) -> bool:
        """単一URLからのダウンロード試行（成功時 True）。"""
        with urllib.request.urlopen(download_url) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
            progress_bar_description = os.path.basename(download_url)
            with tqdm(
                total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description
            ) as progress_bar:
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """TensorFlowのGPT-2チェックポイントから重みを読み込み、辞書に詰め替えます。

    チェックポイント内の変数名を解析し、PyTorchに取り込みやすいネスト辞書へ変換します。

    Args:
        ckpt_path: ``tf.train.latest_checkpoint`` で得られるチェックポイントのベースパス。
        settings: GPT-2 のハイパーパラメータ設定（``hparams.json``）。

    Returns:
        チェックポイントから取得した重みを格納した辞書。``{"blocks": [...], ...}`` の形式。
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params: Dict[str, Any] = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
