import re

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_words(text: str) -> list[str]:
    result = re.split(r'([,.;:?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]


def create_vocab(words: list) -> dict[str, int]:
    vocab = sorted(set(words))
    vocab.extend(["<|endoftext|>", "<|unk|>"])
    return {word: idx for idx, word in enumerate(vocab)}


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocess = self._split_text_into_words(text)
        return [
            self.str_to_int[word] if word in self.str_to_int else self.str_to_int["<|unk|>"]
            for word in preprocess
        ]

    def decode(self, tokens: list[int]) -> str:
        text = " ".join(self.int_to_str[token] for token in tokens if token in self.int_to_str)
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)

    def _split_text_into_words(self, text: str) -> list[str]:
        result = re.split(r'([,.;:?_!"()\']|--|\s)', text)
        return [item.strip() for item in result if item.strip()]


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":
    content = read_text_file("src/data/the_verdict.txt")
    words = split_text_into_words(content)
    print(f"First 10 words in the verdict: {words[:10]}")

    vocab = create_vocab(words)
    print(f"Vocab size: {len(vocab)}")
    print(f"First 10 words in vocab: {list(vocab.items())[:10]}")
    print(f"Last 5 words in vocab: {list(vocab.items())[-5:]}")

    tokenizer = SimpleTokenizerV2(vocab)
    encoded = tokenizer.encode(content)
    print(f"Encoded: {encoded[:10]}")

    decoded = tokenizer.decode(encoded[:10])
    print(f"Decoded: {decoded}")

    assert decoded == "I HAD always thought Jack Gisburn rather a cheap genius"

    encoded = tokenizer.encode("this is a test, with punctuation! Does it work?")
    print(f"Encoded: {encoded[:10]}")

    decoded = tokenizer.decode(encoded[:10])
    print(f"Decoded: {decoded}")
    assert decoded == "this is a <|unk|>, with <|unk|>! <|unk|> it"

    # Using tiktoken for comparison
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode("this is a test, with punctuation! Does it work?")
    print(f"Encoded with tiktoken: {encoded[:10]}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded with tiktoken: {decoded}")
    assert decoded == "this is a test, with punctuation! Does it work?"

    # Testing unknown token handling
    unknown_text = "Akwirw ier"
    encoded_unknown = tokenizer.encode(unknown_text)
    print(f"Encoded unknown text: {encoded_unknown}")
    for token in encoded_unknown:
        print(tokenizer.decode([token]), end=" ")
    print()
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Decoded unknown text: {decoded_unknown}")

    # Create DataLoader
    dataloader = create_dataloader_v1(content, batch_size=1, max_length=4, stride=1, shuffle=False)
    for idx, batch in enumerate(dataloader):
        if idx >= 2:
            break
        input_ids, target_ids = batch
        print(f"Input IDs: {input_ids}")
        print(f"Target IDs: {target_ids}")

    dataloader = create_dataloader_v1(content, batch_size=1, max_length=2, stride=2, shuffle=False)
    for idx, batch in enumerate(dataloader):
        if idx >= 2:
            break
        input_ids, target_ids = batch
        print(f"Input IDs: {input_ids}")
        print(f"Target IDs: {target_ids}")

    dataloader = create_dataloader_v1(content, batch_size=8, max_length=4, stride=4, shuffle=False)
    for idx, batch in enumerate(dataloader):
        if idx >= 2:
            break
        input_ids, target_ids = batch
        print(f"Input IDs: {input_ids}")
        print(f"Target IDs: {target_ids}")
