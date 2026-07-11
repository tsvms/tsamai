"""Dialogue corpus for the v3 chat fine-tune.

Formats DailyDialog (~11k human conversations) plus a 100k slice of SODA
into:

    User: Hi! How are you?
    TsamAI: I am fine, thank you!
    <|endoftext|>

Plain "User:"/"TsamAI:" text markers instead of special chat tokens — no new
vocab entries means no embedding surgery on the v2 weights. Must be encoded
with the v2 tokenizer for the same reason.
"""

import os
import re

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import BPETokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "data", "dialog")
TOKENIZER_PATH = os.path.join(BASE, "data", "tinystories", "tokenizer.json")
SODA_TRAIN = 700_000  # scan this many, keep the ones that survive filtering
EOT = "<|endoftext|>"

_space_before_punct = re.compile(r"\s+([.,!?;:'’])")
_multi_space = re.compile(r"\s{2,}")


def clean(utterance: str) -> str:
    """DailyDialog comes pre-tokenized (' Say , Jim , how about ... ?')
    so the spacing has to be stitched back together."""
    s = _space_before_punct.sub(r"\1", utterance.strip())
    s = s.replace("’", "'").replace(" n't", "n't")
    return _multi_space.sub(" ", s)


def format_dialogue(utterances: list[str]) -> str:
    lines = []
    for i, utt in enumerate(utterances):
        utt = clean(utt)
        if not utt:
            continue
        speaker = "User" if i % 2 == 0 else "TsamAI"
        lines.append(f"{speaker}: {utt}")
    if len(lines) < 2:
        return ""
    return "\n".join(lines) + f"\n{EOT}\n"


def collect() -> tuple[list[str], list[str]]:
    train_texts: list[str] = []
    val_texts: list[str] = []

    print("loading DailyDialog ...")
    dd = load_dataset("li2017dailydialog/daily_dialog", revision="refs/convert/parquet")
    for row in dd["train"]:
        train_texts.append(format_dialogue(row["dialog"]))
    for split in ("validation", "test"):
        for row in dd[split]:
            val_texts.append(format_dialogue(row["dialog"]))

    print("loading SODA subset ...")
    # SODA speakers address each other by name ("Hey Sarah!") — a chat model
    # trained on that greets *you* with a random invented name. Filter out
    # any dialogue where a speaker's name leaks into the text itself.
    def name_free(row) -> bool:
        text = " ".join(row["dialogue"]).lower()
        return not any(
            name and name.lower() in text for name in set(row["speakers"])
        )

    soda_train = load_dataset("allenai/soda", split=f"train[:{SODA_TRAIN}]")
    kept = 0
    for row in tqdm(soda_train, desc="soda train"):
        if name_free(row):
            train_texts.append(format_dialogue(row["dialogue"]))
            kept += 1
    print(f"soda: kept {kept:,}/{SODA_TRAIN:,} dialogues after the name filter")
    soda_val = load_dataset("allenai/soda", split="validation[:4000]")
    for row in soda_val:
        if name_free(row):
            val_texts.append(format_dialogue(row["dialogue"]))

    return [t for t in train_texts if t], [t for t in val_texts if t]


def encode_split(tok: BPETokenizer, texts: list[str], bin_path: str):
    ids: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), 2000), desc=os.path.basename(bin_path)):
        chunk = "".join(texts[i : i + 2000])
        ids.append(np.array(tok.encode(chunk), dtype=np.uint16))
    arr = np.concatenate(ids)
    arr.tofile(bin_path)
    print(f"{bin_path}: {len(arr):,} tokens")


def main():
    if not os.path.exists(TOKENIZER_PATH):
        raise SystemExit("v2 tokenizer missing — run prepare_tinystories.py first")
    os.makedirs(DIR, exist_ok=True)
    tok = BPETokenizer.from_file(TOKENIZER_PATH)
    train_texts, val_texts = collect()
    print(f"{len(train_texts):,} train / {len(val_texts):,} val dialogues")
    encode_split(tok, train_texts, os.path.join(DIR, "train.bin"))
    encode_split(tok, val_texts, os.path.join(DIR, "val.bin"))
    print("done")


if __name__ == "__main__":
    main()
