GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

MODEL_CONFIGS = {
    "gpt2-small(124M)": {
        "emb_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
    },
    "gpt2-medium(355M)": {
        "emb_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
    },
    "gpt2-large(774M)": {
        "emb_dim": 1024,
        "num_layers": 36,
        "num_heads": 20,
    },
    "gpt2-xl(1558M)": {
        "emb_dim": 1280,
        "num_layers": 48,
        "num_heads": 25,
    },
}

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}
