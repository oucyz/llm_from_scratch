import re


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_words(text: str) -> list[str]:
    result = re.split(r'([,.;:?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]


def create_vocab(words: list) -> dict[str, int]:
    vocab = sorted(set(words))
    return {word: idx for idx, word in enumerate(vocab)}


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}
    def encode(self, text: str) -> list[int]:
        preprocess = self._split_text_into_words(text)
        return [self.str_to_int[word] for word in preprocess if word in self.str_to_int]

    def decode(self, tokens: list[int]) -> str:
        text = " ".join(self.int_to_str[token] for token in tokens if token in self.int_to_str)
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)

    def _split_text_into_words(self, text: str) -> list[str]:
        result = re.split(r'([,.;:?_!"()\']|--|\s)', text)
        return [item.strip() for item in result if item.strip()]


if __name__ == "__main__":
    content = read_text_file("src/data/the_verdict.txt")
    words = split_text_into_words(content)
    print(f"First 10 words in the verdict: {words[:10]}")

    vocab = create_vocab(words)
    print(f"Vocab size: {len(vocab)}")
    print(f"First 10 words in vocab: {list(vocab.items())[:10]}")

    tokenizer = SimpleTokenizerV1(vocab)
    encoded = tokenizer.encode(content)
    print(f"Encoded: {encoded[:10]}")

    decoded = tokenizer.decode(encoded[:10])
    print(f"Decoded: {decoded}")

    assert decoded == "I HAD always thought Jack Gisburn rather a cheap genius"
