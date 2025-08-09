import torch
import torch.nn as nn

from ..config.config import GPT_CONFIG_124M
from .activation import GELU
from .multi_head_attention import MultiHeadAttention


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: dict[str, int | bool]):
        """最小構成のGPT風モデル。

        Args:
            cfg: 語彙サイズ、埋め込み次元、コンテキスト長、層数、ドロップ率などの設定。
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )

        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """トークンID列を受け取りロジットを返します。

        Args:
            in_idx: 入力インデックス（B, T）。

        Returns:
            ロジット（B, T, V）。
        """
        _batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, int | bool]):
        """ダミーのTransformerブロック（恒等変換）。"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力をそのまま返します。"""
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """ダミーのLayerNorm。内部に `layer_norm` が実装されている前提。"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`self.layer_norm` を呼び出して正規化したテンソルを返します。"""
        return self.layer_norm(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """最後の次元で平均・分散を計算し、正規化して学習可能なスケール・シフトを適用します。"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        """Transformer用の位置ごとの前向きネットワーク。GELUを中間に挟みます。"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向きに層を適用して出力を返します。"""
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int], use_shortcut: bool):
        """簡易DNN。順次Linear+GELUを適用し、`use_shortcut` は現状未使用です。"""
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU())
            for i in range(len(layer_sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """レイヤを逐次適用した結果を返します。"""
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        """Transformerブロック（MHA→残差、FF→残差）。"""
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformerブロックの前方計算。"""
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        """GPT風モデル本体。埋め込み、位置埋め込み、ブロック群、最終正規化と出力ヘッドを持つ。"""
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """トークンID列を入力し、語彙サイズに対するロジットを返します（B, T, V）。"""
        _batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(123)

    print("=" * 80)
    layer_norm = LayerNorm(emb_dim=5)
    x = torch.randn(2, 5)
    print(layer_norm(x).shape)

    print("mean:", layer_norm(x).mean(dim=-1))
    print("var:", layer_norm(x).var(dim=-1, unbiased=False))

    print("=" * 80)
    model = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, GPT_CONFIG_124M["emb_dim"])
    out = model(x)
    print(out.shape)
    total_prams = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_prams / 1e6:.2f}M")
    print(f"Total parameters: {total_prams:,}")

    layer_sizes = [3, 3, 3, 3, 3]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    model = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    output = model(sample_input)
    print(output)

    print("=" * 80)
    x = torch.randn(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    out = block(x)
    print(x.shape)
    print(out.shape)

    print("=" * 80)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(torch.randint(0, 5000, (2, 4)))
    print(out.shape)
    print(out)

    total_prams = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_prams / 1e6:.2f}M")
    print(f"Total parameters: {total_prams:,}")
    print(total_prams * 4 / (1024 * 1024))
    print(total_prams * 4 / (1024 * 1024 * 1024))
