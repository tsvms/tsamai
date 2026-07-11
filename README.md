# TsamAI

A small language model I built and trained from scratch on my laptop, with a
little desktop app to chat with it. No APIs, no pretrained weights, no
fine-tuning someone else's model — every parameter in this thing was trained
on my own GPU, an RTX 4050 laptop card with 6GB of VRAM.

I started this because I wanted to actually understand how LLMs work, not
just call one. The best way I know to understand something is to build it.
The project went through three stages, and each stage taught me something
different — that's why all three models ship with the repo instead of just
the final one. Together they're basically the history of modern NLP replayed
on one laptop: v1 proves the machinery works, v2 shows what tokenization +
data + compute buy you, v3 shows what "instruction tuning" actually is.

**Quick taste**, same architecture, three stages:

> **v1** — prompt `ROMEO:`
> *"And thou art a heigh, a new draw of heaven! DUKE VINCENTIO: ..."*
>
> **v2** — prompt `Once upon a time`
> *"...in a small house, there lived a boy named Tim. Tim had a small toy
> box with a lock on it. He loved to play with his toys every day."*
>
> **v3** — a conversation:
> *you: hey* → *"Well, what's up?"*
> *you: I had a really long day at university.* → *"Oh, I see. Well, I hope
> you have a good day."*

---

## The three models

### v1 — "Shakespeare" (character-level GPT)

My hello world. The whole point of v1 was to get every piece of the pipeline
working — data loading, the transformer, the training loop, sampling — with
the simplest possible tokenizer: none. Every character is a token. The vocab
is literally the 65 distinct characters that appear in the corpus.

|  |  |
|---|---|
| parameters | 10.65M |
| architecture | 6 layers, 6 heads, 384 embedding dim |
| context | 256 characters |
| tokenizer | none (char ↔ int lookup table, 65 entries) |
| data | Tiny Shakespeare, ~1MB / ~1M characters |
| training | 5,000 iterations, batch 64×256, ~12 minutes |
| loss | 4.28 → **1.47** val (best, at iter 2000) |

What it does: give it the beginning of a scene and it continues in
convincing pseudo-Shakespeare — correct character names in caps, line
breaks in the right places, iambic-ish rhythm, complete nonsense meaning.
Which is exactly what a 10M model trained on 1MB can learn: *form*, not
*content*.

What v1 taught me: **overfitting, live**. The dataset is so small that
train loss kept falling (0.65 at the end) while val loss bottomed out at
iter 2000 (1.47) and then got *worse* (1.66 by the end). If I had naively
kept the final weights I'd have shipped a worse model. The checkpoint
logic saves on best-val for exactly this reason.

### v2 — "Stories" (BPE + TinyStories, the real one)

v2 is the same transformer idea grown up: a real tokenizer, 20× more
compute, and 500× more data — chosen carefully rather than maximally.

|  |  |
|---|---|
| parameters | 29.37M |
| architecture | 8 layers, 8 heads, 512 embedding dim |
| context | 512 tokens |
| tokenizer | byte-level BPE, 8,192 vocab, trained on the corpus itself |
| data | [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories), 542,737,971 training tokens |
| training | 36,000 iterations, effective batch 16k tokens/step, 2h 42m |
| loss | 9.06 → **1.224** val |

Three decisions here mattered more than any code:

**Why 8,192 vocab and not GPT-2's 50k?** With a 512-dim embedding, a 50k
vocab puts ~26M parameters in the embedding table alone — most of the
model's capacity would be a dictionary of rare tokens the corpus barely
uses. 8k keeps the embedding table at 4M and leaves the capacity for the
actual transformer. Small models need small vocabs.

**Why TinyStories and not "more data"?** My first instinct was to download
tens of GB of internet text. Turns out data *quality* beats quantity at
this scale — the TinyStories paper showed that models this size trained on
simple, clean text learn fluent grammar, while the same models on messy
web text learn mush. I verified this the cheap way: believed the paper.
The result speaks English properly, which for 30M parameters is kind of
remarkable.

**Why 36,000 iterations exactly?** The Chinchilla rule of thumb says a
model absorbs usefully about 20 tokens per parameter: 30M × 20 = 600M
tokens. At 16,384 tokens per step that's ~36k steps — which is almost
exactly one pass over TinyStories. The corpus and the model size just
happen to be a perfect match, which is half the reason I picked them.

What it does: write the first line of a story and it continues with real
narrative logic — characters keep their names, events follow causally,
stories end. It even stops by itself at the end of a story (it learned
what `<|endoftext|>` means from the corpus separators).

### v3 — "Chat" (dialogue fine-tune)

The step that turns a text-continuer into something you can talk to — the
same step (conceptually) that turns a base LLM into a chat assistant.
No new architecture, no new capability: just supervised fine-tuning on
conversations, so the model learns the *pattern* "when text looks like a
dialogue and it's your turn, produce a reply."

|  |  |
|---|---|
| starting point | v2 weights (val 1.224 on stories) |
| data | DailyDialog (~11k human dialogues) + a filtered slice of [SODA](https://huggingface.co/datasets/allenai/soda) (269,659 dialogues), 56.7M tokens |
| format | plain `User:` / `TsamAI:` markers, `<\|endoftext\|>` between conversations |
| training | 7,000 iterations at LR 1e-4 (10× lower than pretraining), 32 minutes |
| loss | **1.984** val on held-out dialogues |

Details that turned out to matter:

**Plain-text markers instead of special chat tokens.** Real chat models
use dedicated tokens like `<|user|>`. Adding new tokens after pretraining
means growing the tied embedding matrix and initializing the new rows —
surgery I didn't need. Literal `User:` text costs nothing and the model
picks up the pattern within the first few hundred steps.

**The name filter.** My first fine-tune greeted me with *"Hey, Damarina!"*
— SODA speakers know each other's names, so the model learned that chats
open with a name, and invented one for me. Fix: drop every dialogue where
a speaker's name leaks into the utterance text. That kept 270k of 700k
dialogues scanned, and the cleaned model stopped baptizing me.

**Low learning rate.** Fine-tuning at the pretraining LR would bulldoze
what v2 knows about English. At 1e-4 → 1e-5 the model keeps its language
and layers the dialogue behavior on top.

**An invisible warm-up.** The app silently prepends a short greeting
exchange to every conversation before your first message — a poor man's
system prompt. My first version of it asked *"what is your name?"*, which
put the model in introductions mode and it started naming everyone
involved. The current version *closes* the greeting ritual instead of
opening one. Prompting matters even at 30M parameters.

What to expect: honest small talk. It greets, it asks back, it reacts to
feelings, it stays on topic for a few exchanges (context is 512 tokens ≈
the last 5–8 messages). Ask it anything factual and it will confidently
make something up — 30M parameters have room for language, not knowledge.
It chats like a small kid with a short memory. It's my small kid with a
short memory.

---

## How it works, in one screen

The whole thing is a decoder-only transformer (GPT-2 family), written by
hand in `model.py` (~150 lines):

```
tokens → token embedding + position embedding
       → 8 × Block( layernorm → causal attention → layernorm → MLP )
       → layernorm → linear head → probability of every next token
```

- **Attention** moves information *between* positions ("the word at
  position 40 should care about the name at position 12").
- The **MLP** transforms each position independently.
- **Causal masking** means position t can only see positions ≤ t —
  otherwise predicting the next token would be cheating.
- Residual connections around both, so gradients survive 8 layers.

Generation is just: run the context, sample one token from the output
distribution (with temperature and top-k), append it, repeat. Everything
you type in the app and everything it answers passes through that loop.

The training recipe (`train.py`) is deliberately boring: AdamW with weight
decay on matrices only, linear warmup into cosine LR decay, gradient
clipping, bf16 autocast, gradient accumulation. Checkpoints are
self-contained — weights + architecture config + tokenizer in one file —
so inference never touches the data pipeline.

## The app

PySide6, one window, no chrome. Pick a model in the top bar; v3 opens a
conversation, v1/v2 continue whatever you start writing. I type in
sans-serif, the model answers in serif — that's the entire "chat bubble"
system and I like it better than bubbles. A `tune` toggle hides the
temperature / top-k / length sliders until you want them. Generation
streams token-by-token from a worker thread; Enter sends, Shift+Enter is
a newline, the send arrow becomes a stop button while it writes.

## Things that went wrong (kept for honesty)

- **CUDA OOM on the first v2 launch.** Batch 32×512 doesn't fit in 6GB
  next to a desktop session. Gradient accumulation (2 × batch 16) gives
  the identical effective batch at half the peak memory.
- **The final model is not the best model** (v1 overfitting, above).
- **"Hey, Damarina!"** (the name filter story, above).
- **The priming that backfired** (introductions mode, above).
- **Ugly fonts.** The first UI styled text with fonts I don't have
  installed; Qt silently substituted something rough. Check `fc-list`
  before styling anything.

## Running it

```bash
git clone https://github.com/tsvms/tsamai && cd tsamai
uv venv --python 3.13 && source .venv/bin/activate
uv sync   # or: uv pip install torch numpy requests PySide6 tokenizers datasets

python app.py
```

The app needs at least one checkpoint in `checkpoints/`. Grab the trained
weights from the [releases page](https://github.com/tsvms/tsamai/releases)
(instructions there — no training needed), or train your own:

```bash
# v1 — sanity check the whole pipeline (~12 min on a laptop GPU)
python train.py

# v2 — download the two TinyStoriesV2-GPT4 .txt files from HuggingFace
#      into data/tinystories/{train,val}.txt first  (~2.7h)
python prepare_tinystories.py
python train.py --preset v2

# v3 — chat fine-tune on top of v2  (~35 min)
python prepare_dialog.py
python train.py --preset v3
```

Every training run writes a `.resume` file; `--resume` continues an
interrupted run. CUDA is used when available; CPU works too (fine for
chatting, patience required for training).

## Layout

```
config.py     every hyperparameter, three presets — change things here
data.py       v1 pipeline: download shakespeare, char vocab, batches
tokenizer.py  byte-level BPE (8192 merges learned from the corpus)
prepare_tinystories.py   corpus → tokenizer + memory-mapped token bins
prepare_dialog.py        dialogues → cleaned, formatted, encoded bins
model.py      the transformer — start reading here
train.py      the training loop, all three presets
sample.py     terminal generation + load_model()
app.py        the desktop app
docs/notes.md my running notes on decisions and mistakes
```

## Ideas I might get to

- KV cache — generation currently re-runs the whole context for every
  token, which is embarrassing but fine at this size
- a Greek version; the pipeline doesn't care about the language, I just
  need a clean Greek corpus
- quantized CPU inference and a packaged binary
- letting it learn a persona ("I'm TsamAI") without re-triggering the
  introductions bug

## Thanks

- Andrej Karpathy's *"Let's build GPT"* — the reason model.py holds no
  mysteries for me anymore
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li) for proving
  small models can speak, and for the dataset
- [DailyDialog](http://yanran.li/dailydialog) and
  [SODA](https://huggingface.co/datasets/allenai/soda) for the conversations
