# notes to self

Decisions and things I figured out along the way, mostly so future me
remembers why the code looks the way it does.

## sizing the model (v2)

The 4050 has 6GB. After the desktop takes its share I realistically get
~4.5GB for training. A 30M param model with batch 16×512 and bf16 autocast
sits around 3.5GB — that's the ceiling with some breathing room.

First attempt at batch 32 OOM'd. Fixed with gradient accumulation (2 micro
batches of 16) instead of shrinking the effective batch — same math, half
the peak memory.

## chinchilla math

Rule of thumb from the Chinchilla paper: ~20 training tokens per parameter.

    30M params × 20 = 600M tokens
    TinyStories V2  ≈ 542M tokens  ✓ almost exactly right

So one epoch-ish over the whole corpus is the correctly-sized run, which
came out to 36k iterations at 16k tokens/iter. Neat when it works out.

## overfitting in v1

Tiny dataset (1MB), so train loss kept falling while val loss turned around
at ~iter 2000. Standard. The trick that matters: always checkpoint on best
*val* loss, never the final iteration. The final v1 model had train 0.65 /
val 1.66 — the saved one has val 1.47.

## tokenizer choice

8192 BPE vocab, byte-level so nothing is ever out-of-vocabulary. Trained it
on a 200MB sample of the corpus — BPE merge statistics converge way before
that, no point feeding it all 2GB. Kept the vocab small on purpose: with
n_embd=512, a GPT-2-sized vocab (50k) would put ~26M params in the embedding
table alone — most of the model's capacity wasted on rare tokens the corpus
barely uses.

## chat format (v3)

Went with plain text markers:

    User: Hi!
    TsamAI: Hello! How are you?
    <|endoftext|>

instead of proper special tokens like <|user|>, because adding tokens after
pretraining means resizing tied embeddings and I didn't want the surgery.
The stop condition at inference is just "model started writing 'User:'" —
crude but works. <|endoftext|> was already in the vocab from pretraining
(TinyStories uses it as a story separator) so the model already knows what
it means. Lucky.

## generation streaming

The app streams token by token from a QThread. One subtlety: in chat mode I
hold back the last few characters before showing them, so if the model
starts generating the user's next turn ("\nUser: ...") I can cut it before
it flashes on screen.

## fonts (the ugly-text saga)

The first version of the UI looked rough around the text. Cause: I styled
it with fonts I don't have installed (Inter, Georgia) and Qt silently
substituted something bad. Lesson: check `fc-list` first, then pick fonts.
