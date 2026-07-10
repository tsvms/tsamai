"""The training loop.

    python train.py                        # v1, minutes
    python train.py --preset v2            # v2, a few hours
    python train.py --preset v3            # chat fine-tune, ~40 min
    python train.py --preset v2 --resume   # pick up an interrupted run

Nothing exotic: AdamW (weight decay on matrices only), warmup + cosine decay,
grad clipping, autocast on CUDA. Saves two files per run — <name>.pt is the
best-val inference checkpoint (self-contained: weights + config + tokenizer),
<name>.pt.resume is the full training state so --resume can continue.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from config import PRESETS
from data import get_batch, load_dataset
from model import GPT

DATA_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TINYSTORIES_DIR = os.path.join(DATA_BASE, "tinystories")
BIN_DIRS = {"v2": TINYSTORIES_DIR, "v3": os.path.join(DATA_BASE, "dialog")}


def get_lr(it: int, cfg) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, gpt_cfg, cfg, ctx):
    model.eval()
    out = {}
    for name, ids in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(ids, gpt_cfg.block_size, cfg.batch_size, cfg.device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def load_data(preset: str):
    """Returns (train_ids, val_ids, tokenizer_payload). The payload goes
    into the checkpoint so inference never needs the data pipeline."""
    if preset == "v1":
        train_ids, val_ids, vocab = load_dataset()
        return train_ids, val_ids, {"tokenizer_type": "char", "vocab_stoi": vocab.stoi}

    # v2 and v3 share the tokenizer; only the token bins differ
    tok_path = os.path.join(TINYSTORIES_DIR, "tokenizer.json")
    train_bin = os.path.join(BIN_DIRS[preset], "train.bin")
    val_bin = os.path.join(BIN_DIRS[preset], "val.bin")
    for p in (tok_path, train_bin, val_bin):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} — run the matching prepare_*.py first")
    # memmap so the 600M-token array never has to fit in RAM
    train_ids = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_ids = np.memmap(val_bin, dtype=np.uint16, mode="r")
    with open(tok_path, encoding="utf-8") as f:
        tokenizer_json = f.read()
    return train_ids, val_ids, {"tokenizer_type": "bpe", "tokenizer_json": tokenizer_json}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gpt_cfg, cfg = PRESETS[args.preset]
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (this will be slow)")
        cfg.device = "cpu"

    torch.manual_seed(1337)

    train_ids, val_ids, tok_payload = load_data(args.preset)
    if tok_payload["tokenizer_type"] == "char":
        gpt_cfg.vocab_size = len(tok_payload["vocab_stoi"])
    else:
        from tokenizer import BPETokenizer

        gpt_cfg.vocab_size = len(BPETokenizer.from_json(tok_payload["tokenizer_json"]))
    print(f"[{args.preset}] dataset: {len(train_ids):,} train / {len(val_ids):,} val "
          f"tokens, vocab {gpt_cfg.vocab_size}")

    model = GPT(gpt_cfg).to(cfg.device)
    print(f"model: {model.num_params()/1e6:.2f}M parameters on {cfg.device}")

    if cfg.init_checkpoint and not args.resume:
        init = torch.load(
            cfg.init_checkpoint, map_location=cfg.device, weights_only=True
        )
        model.load_state_dict(init["model_state"])
        print(f"initialized weights from {cfg.init_checkpoint} "
              f"(val {init['val_loss']:.4f})")

    # weight-decay the matrices, leave layernorm gains & co alone
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        fused=cfg.device.startswith("cuda"),
    )

    resume_path = cfg.checkpoint_path + ".resume"
    start_iter = 0
    best_val = float("inf")
    if args.resume:
        if not os.path.exists(resume_path):
            raise SystemExit(f"--resume: no state at {resume_path}")
        state = torch.load(resume_path, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_iter = state["iter"] + 1
        best_val = state["best_val"]
        print(f"resumed at iter {start_iter} (best val {best_val:.4f})")

    use_amp = cfg.device.startswith("cuda")
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    splits = {"train": train_ids, "val": val_ids}
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    t0 = time.time()

    for it in range(start_iter, cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, splits, gpt_cfg, cfg, ctx)
            dt = time.time() - t0
            print(
                f"iter {it:6d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | {dt:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "gpt_config": gpt_cfg.__dict__,
                        "iter": it,
                        "val_loss": best_val,
                        **tok_payload,
                    },
                    cfg.checkpoint_path,
                )
                print(f"        saved checkpoint (val {best_val:.4f})")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "iter": it,
                    "best_val": best_val,
                },
                resume_path,
            )

        if it == cfg.max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(train_ids, gpt_cfg.block_size, cfg.batch_size, cfg.device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    print(f"done. best val loss {best_val:.4f}, "
          f"checkpoint at {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
