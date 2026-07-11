"""All the knobs in one place. Architecture in GPTConfig, training recipe
in TrainConfig, and the three presets (v1/v2/v3) at the bottom."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256   # max context length (characters)
    vocab_size: int = 65    # overwritten at runtime from the dataset vocab
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False      # bias-less Linears/LayerNorms train a bit better here


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 250    # evaluate train/val loss every N iters
    eval_iters: int = 100       # batches averaged per evaluation
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 1   # micro-batches per optimizer step (for small VRAM)
    device: str = "cuda"        # train.py falls back to cpu if cuda missing
    checkpoint_path: str = "checkpoints/ckpt.pt"
    init_checkpoint: str | None = None  # fine-tuning: start from these weights


# v1: char-level Shakespeare, trains in minutes — my sanity check.
# v2: BPE + TinyStories, ~30M params, a few hours (prepare_tinystories.py first).
# v3: chat fine-tune on top of v2 (prepare_dialog.py first).

PRESETS = {
    "v1": (
        GPTConfig(),
        TrainConfig(),
    ),
    "v2": (
        GPTConfig(
            block_size=512,
            vocab_size=8192,  # overwritten from the trained tokenizer
            n_layer=8,
            n_head=8,
            n_embd=512,
            dropout=0.1,      # with 600M tokens overfitting isn't the enemy anymore
        ),
        TrainConfig(
            batch_size=16,            # batch 32 OOMs on my 6GB card...
            grad_accum_steps=2,       # ...so: 2 micro-batches, same effective 16k tokens/step
            max_iters=36000,          # ~600M tokens = 20 tokens/param (Chinchilla rule)
            eval_interval=1000,
            eval_iters=50,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=1000,
            checkpoint_path="checkpoints/v2.pt",
        ),
    ),
    "v3": (
        GPTConfig(  # must match v2 exactly — we load its weights
            block_size=512,
            vocab_size=8192,
            n_layer=8,
            n_head=8,
            n_embd=512,
            dropout=0.1,
        ),
        TrainConfig(
            batch_size=16,
            grad_accum_steps=2,
            max_iters=7000,           # ~2 passes over the 57M dialogue tokens
            eval_interval=500,
            eval_iters=50,
            learning_rate=1e-4,       # low LR: nudge v2 toward chat, don't erase it
            min_lr=1e-5,
            warmup_iters=100,
            checkpoint_path="checkpoints/v3.pt",
            init_checkpoint="checkpoints/v2.pt",
        ),
    ),
}
