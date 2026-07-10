"""Byte-level BPE for v2/v3, trained on the corpus itself.

I use the HuggingFace `tokenizers` library for the BPE machinery (it's the
Rust one, training takes seconds) but the vocab is 100% ours — 8192 merges
learned from TinyStories. Same encode/decode interface as CharVocab so the
rest of the code doesn't care which tokenizer a checkpoint carries.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 8192


class BPETokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok

    def __len__(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token for streaming output."""
        return self.tok.decode([token_id])

    def to_json(self) -> str:
        return self.tok.to_str()

    @classmethod
    def from_json(cls, s: str) -> "BPETokenizer":
        return cls(Tokenizer.from_str(s))

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        return cls(Tokenizer.from_file(path))

    @classmethod
    def train(cls, files: list[str], vocab_size: int = VOCAB_SIZE) -> "BPETokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(files, trainer)
        return cls(tok)
