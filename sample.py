"""Terminal generation — quick way to poke at a checkpoint without the app.

    python sample.py --prompt "ROMEO:" --tokens 500
    python sample.py --checkpoint checkpoints/v2.pt --prompt "Once upon a time"

Also home of load_model(), which app.py uses too. Checkpoints carry their
own tokenizer so any of v1/v2/v3 loads with the same call.
"""

import argparse
import sys

import torch

from config import GPTConfig
from model import GPT


class CharTokenizer:
    """Wraps a v1 char table in the same interface the BPE tokenizer has."""

    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def decode_token(self, token_id: int) -> str:
        return self.itos[token_id]


def load_model(checkpoint_path: str = "checkpoints/ckpt.pt", device: str | None = None):
    """Load a self-contained checkpoint. Returns (model, tokenizer, meta)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if ckpt.get("tokenizer_type") == "bpe":
        from tokenizer import BPETokenizer

        tokenizer = BPETokenizer.from_json(ckpt["tokenizer_json"])
    else:
        tokenizer = CharTokenizer(ckpt["vocab_stoi"])

    meta = {"iter": ckpt["iter"], "val_loss": ckpt["val_loss"], "device": device}
    return model, tokenizer, meta


def main():
    parser = argparse.ArgumentParser(description="Sample text from TsamAI")
    parser.add_argument("--prompt", default="\n", help="seed text to continue")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--checkpoint", default="checkpoints/ckpt.pt")
    args = parser.parse_args()

    try:
        model, tokenizer, meta = load_model(args.checkpoint)
    except FileNotFoundError:
        sys.exit(f"no checkpoint at {args.checkpoint} — run train.py first")

    print(
        f"[TsamAI {model.num_params()/1e6:.1f}M | {meta['device']} | "
        f"val loss {meta['val_loss']:.3f}]\n"
    )

    ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n") or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=meta["device"])

    print(args.prompt, end="", flush=True)
    for token_id in model.generate(
        idx, args.tokens, temperature=args.temperature, top_k=args.top_k
    ):
        print(tokenizer.decode_token(token_id), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
