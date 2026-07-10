"""v1 data pipeline: Tiny Shakespeare, character-level.

The "tokenizer" here is literally a dict from chars to ints. I kept it that
simple on purpose — v1 was about getting the whole pipeline working end to
end before caring about tokenization (that's tokenizer.py, used by v2/v3).
"""

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DATASET_PATH = os.path.join(DATA_DIR, "shakespeare.txt")


def download_dataset() -> str:
    """Fetch Tiny Shakespeare if not already on disk; return the text."""
    if not os.path.exists(DATASET_PATH):
        import requests

        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading dataset from {DATASET_URL} ...")
        resp = requests.get(DATASET_URL, timeout=30)
        resp.raise_for_status()
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            f.write(resp.text)
    with open(DATASET_PATH, encoding="utf-8") as f:
        return f.read()


class CharVocab:
    """char <-> int, both directions. That's the whole tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        # chars the corpus never had just get dropped
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_dataset(val_fraction: float = 0.1):
    """Returns (train_ids, val_ids, vocab) with ids as uint16 numpy arrays."""
    text = download_dataset()
    vocab = CharVocab(text)
    ids = np.array(vocab.encode(text), dtype=np.uint16)
    split = int(len(ids) * (1 - val_fraction))
    return ids[:split], ids[split:], vocab


def get_batch(
    ids: np.ndarray, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random batch of (x, y) where y is x shifted one left — every position
    predicts the next token. Works on plain arrays and memmaps alike."""
    starts = np.random.randint(0, len(ids) - block_size - 1, size=batch_size)
    x = torch.stack(
        [torch.from_numpy(ids[s : s + block_size].astype(np.int64)) for s in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(ids[s + 1 : s + 1 + block_size].astype(np.int64))
            for s in starts
        ]
    )
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
