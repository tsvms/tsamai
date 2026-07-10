# TsamAI

A small language model I built and trained from scratch on my laptop, with a
little desktop app to chat with it. No APIs, no pretrained weights — every
parameter in this thing was trained on my own GPU (an RTX 4050 with 6GB,
which turned out to be enough).

I started this because I wanted to actually understand how LLMs work, not
just call one. The best way I know to understand something is to build it.

## What it can do

Three checkpoints, trained in stages:

- **v1** — character-level GPT (10M params) trained on Shakespeare. My "hello
  world". Takes ~12 minutes to train and produces surprisingly convincing
  fake Shakespeare.
- **v2** — proper BPE tokenizer + 30M params, trained on ~540M tokens of
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
  Writes coherent little stories in simple English.
- **v3** — v2 fine-tuned on ~111k everyday dialogues (DailyDialog + SODA).
  This one you can actually chat with.

Be honest with your expectations: v3 chats like a small kid with a short
memory. It does small talk, it stays on topic for a few exchanges, and it
confidently makes things up if you ask it anything factual. That's what 30M
parameters gets you — and watching *why* that's the limit is kind of the
point of the project.

## Running it

```bash
uv venv --python 3.13 && source .venv/bin/activate
uv sync   # or: uv pip install torch numpy requests PySide6 tokenizers

python app.py
```

The app needs at least one trained checkpoint in `checkpoints/`. Either grab
the weights from the releases page, or train your own (below). CUDA is used
if available, otherwise it runs on CPU (fine for chatting, slow for training).

## Training from scratch

```bash
# v1 — quick sanity check that everything works (~12 min on my 4050)
python train.py

# v2 — the real one (~2.5h). Download TinyStories first:
#   the two TinyStoriesV2-GPT4 txt files -> data/tinystories/{train,val}.txt
python prepare_tinystories.py
python train.py --preset v2

# v3 — chat fine-tune on top of v2 (~40 min)
python prepare_dialog.py
python train.py --preset v3
```

Training can be interrupted; `--resume` picks up where it left off.

## How it's put together

```
config.py     all hyperparameters, three presets
data.py       v1 data pipeline (char-level)
tokenizer.py  byte-level BPE, 8192 vocab, trained on the corpus itself
model.py      the transformer (attention, blocks, generation)
train.py      training loop — AdamW, cosine LR, mixed precision
sample.py     generate from a checkpoint in the terminal
app.py        the desktop app (PySide6)
```

Checkpoints are self-contained (weights + config + tokenizer in one file),
so the app never touches the data pipeline.

Some decisions I had to make and what I learned from them:

- **Why 30M params and not more?** 6GB of VRAM, that's why. But also: I
  sized the training run with the Chinchilla rule of thumb (~20 tokens per
  parameter), and 30M params × 20 = 600M tokens ≈ exactly the TinyStories
  corpus. Bigger model would've been undertrained.
- **Why TinyStories and not "more data"?** I originally wanted to throw
  gigabytes of internet text at it. Turns out data quality beats quantity
  at this scale — a small model trained on clean, simple text learns real
  grammar; the same model on messy web text learns mush.
- **Why plain "User:" / "TsamAI:" markers instead of special chat tokens?**
  Adding new tokens after pretraining means resizing the embedding matrix.
  Plain text markers cost nothing and the model picks them up fine.
- **The v1 → v2 jump is the whole story of modern NLP** in miniature:
  same architecture, just better tokenization + more/better data + more
  compute = qualitatively different behavior.

## Ideas I might get to

- KV cache (generation currently re-runs the whole context per token, which
  is embarrassing but fine at this size)
- a Greek version — there's no reason the pipeline can't do it, I just need
  a clean Greek corpus
- quantized CPU inference + a proper packaged binary
