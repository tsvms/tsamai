"""TinyStories -> tokenizer + token bins for v2.

Expects data/tinystories/{train,val}.txt (the TinyStoriesV2-GPT4 files from
HuggingFace, roneneldan/TinyStories). Trains the BPE tokenizer on a 200MB
sample (merge statistics converge long before the full 2GB), then encodes
everything into uint16 .bin files that train.py memory-maps.
"""

import os

import numpy as np
from tqdm import tqdm

from tokenizer import BPETokenizer

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tinystories")
TOKENIZER_PATH = os.path.join(DIR, "tokenizer.json")
SAMPLE_BYTES = 200 * 1024 * 1024  # plenty for the merge statistics
ENCODE_CHUNK_BYTES = 32 * 1024 * 1024


def train_tokenizer() -> BPETokenizer:
    if os.path.exists(TOKENIZER_PATH):
        print("tokenizer already trained, loading")
        return BPETokenizer.from_file(TOKENIZER_PATH)
    sample_path = os.path.join(DIR, "sample.txt")
    with open(os.path.join(DIR, "train.txt"), encoding="utf-8") as f:
        sample = f.read(SAMPLE_BYTES)
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"training BPE tokenizer on {len(sample)/1e6:.0f}MB sample ...")
    tok = BPETokenizer.train([sample_path])
    tok.tok.save(TOKENIZER_PATH)
    os.remove(sample_path)
    print(f"tokenizer saved: {len(tok)} tokens")
    return tok


def encode_file(tok: BPETokenizer, txt_path: str, bin_path: str):
    if os.path.exists(bin_path):
        print(f"{bin_path} exists, skipping")
        return
    total = os.path.getsize(txt_path)
    parts: list[np.ndarray] = []
    with open(txt_path, encoding="utf-8") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(txt_path)
    ) as bar:
        while True:
            chunk = f.read(ENCODE_CHUNK_BYTES)
            if not chunk:
                break
            ids = tok.encode(chunk)
            parts.append(np.array(ids, dtype=np.uint16))
            bar.update(len(chunk.encode("utf-8")))
    arr = np.concatenate(parts)
    assert len(tok) < 65536, "uint16 overflow"
    arr.tofile(bin_path)
    print(f"{bin_path}: {len(arr):,} tokens")


def main():
    for split in ("train", "val"):
        if not os.path.exists(os.path.join(DIR, f"{split}.txt")):
            raise SystemExit(f"missing {DIR}/{split}.txt — download TinyStories first")
    tok = train_tokenizer()
    for split in ("train", "val"):
        encode_file(
            tok,
            os.path.join(DIR, f"{split}.txt"),
            os.path.join(DIR, f"{split}.bin"),
        )
    print("done")


if __name__ == "__main__":
    main()
