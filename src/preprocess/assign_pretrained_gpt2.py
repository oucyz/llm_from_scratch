"""事前学習GPT-2重みを自作GPTModelへ割り当てるユーティリティ。

このモジュールは設定生成、形状検証付きのパラメータ代入、重みの一括割当を提供します。
ロジックは既存のままで、型アノテーションと日本語Googleスタイルdocstringのみを追加しています。
"""

from typing import Any, Dict

import numpy as np
import tiktoken
import torch

from ..config.config import GPT_CONFIG_124M, MODEL_CONFIGS
from ..generate_text import text_to_token_ids, token_ids_to_text
from ..gpt_download import download_and_load_gpt2
from ..models.dummy_gpt_model import GPTModel
from ..postprocess.postprocess import generate


def get_gpt2_small_config() -> Dict[str, Any]:
    """GPT-2 Small (124M) に準拠した設定ディクショナリを返します。

    Returns:
        Dict[str, Any]: GPT-2 Small の構成を表す辞書。
    """
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    model_name = "gpt2-small(124M)"
    NEW_CONFIG.update(MODEL_CONFIGS[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})
    return NEW_CONFIG


def assign(left: torch.Tensor, right: np.ndarray | torch.Tensor) -> torch.nn.Parameter:
    """形状を検証して右辺の値で ``torch.nn.Parameter`` を作成します。

    Args:
        left: 既存パラメータのテンソル（形状検証に使用）。
        right: 代入する値。NumPy配列またはTorchテンソル。

    Returns:
        torch.nn.Parameter: ``right`` を値に持つ ``Parameter``。

    Raises:
        ValueError: ``left`` と ``right`` の形状が一致しない場合。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: Dict[str, Any]) -> None:
    """TensorFlow由来の重み辞書を ``GPTModel`` インスタンスへ割り当てます。

    Args:
        gpt: 重みを割り当てるモデルインスタンス。
        params: ``download_and_load_gpt2`` で構築された重み辞書。

    Returns:
        なし
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


if __name__ == "__main__":
    gpt = GPTModel(get_gpt2_small_config())
    gpt.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings, params = download_and_load_gpt2("124M", "./output/gpt2")
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    print("Model loaded and weights assigned.")

    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tiktoken.get_encoding("gpt2")).to(device),
        max_new_tokens=10,
        context_size=get_gpt2_small_config()["context_length"],
        top_k=3,
        temperature=1.0,
    )
    print(
        "Output text:\n",
        token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2")).replace("\n", " "),
    )
