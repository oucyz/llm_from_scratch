import json
import os
import urllib.request
from typing import Any


def download_and_load_file(file_path: str, url: str) -> Any:
    """URLからJSONをダウンロードし、ローカル保存または読み込みして返します。

    Args:
        file_path: 保存/読み込みに使用するローカルのファイルパス。
        url: 取得元のJSONテキストURL。

    Returns:
        Any: ``json.loads`` で復元されたPythonオブジェクト（例: listやdict）。
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()

    data = json.loads(text_data)

    return data


if __name__ == "__main__":
    file_path = "./output/instruction_data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)
    print("number of entries:", len(data))
    print(data[50])
    print(data[999])
