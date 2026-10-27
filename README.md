# supra-3m-molab

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/bapt433/supra-3m-molab/blob/main/notebook.py)

A self-contained reproduction of the winning **Supra2.5-Pico 3M** training recipe, scaled to
**100M tokens** so it runs end to end in a few minutes on molab's free RTX Pro 6000 (96 GB).

Everything is driven from one marimo notebook: `notebook.py`. It downloads this repo, pulls the
tokenised shards from the `data-v1` release, trains, and plots the loss.

## The run

| Item | Value |
| --- | --- |
| Parameters | 2,982,480 |
| Shape | d=144, 10 blocks, SwiGLU 376, ctx 1024 |
| Attention | GQA 3:1, head_dim 48, QK-RMSNorm, partial RoPE 24/48, XSA, sigmoid gate |
| Residuals | softmax attention residuals over all previous block outputs |
| Tokenizer | 4,104 ids (4,096 BPE + 8 length tokens), tied embeddings |
| Batch | 256 x 1024, no accumulation = 262,144 tokens/step |
| Optimiser | Muon 0.02 on 2D weights, AdamW 6e-3 on the rest, betas 0.9/0.95, clip 1.0 |
| Schedule | 381 steps, warmup 17, stable to 80%, cosine to zero |
| Precision | bf16 autocast, torch.compile |
| Tokens | 100,000,000 |

The batch is the same *effective* batch as the 12B run (64 x 1024 x 4 accumulation steps), so the
optimiser dynamics are unchanged. The warmup keeps the 12B run's 4.369% share of total steps
(2,000 / 45,776 = 17 / 381). Everything else is copied verbatim from `train_ddp.py`.

## Run it

In molab, create a notebook from this repository and open `notebook.py`, then run all cells.
First run downloads about 110 MB of tokenised data (compressed); training itself is roughly one
minute of GPU time on the RTX Pro 6000.

Locally:

```bash
python train.py --data-dir ./data --out-dir ./runs/test --tokens 100e6 \
  --bs 256 --seq 1024 --grad-accum 1 --compile
```

## Data

`data-v1` release: the same six tokenised sources and the same three-phase mixture as the 12B
run, sliced to 120M tokens.

| Source | Tokens shipped |
| --- | --- |
| ufw (Ultra-FineWeb L2) | 34M |
| txt (TXT360-v2) | 10M |
| cos (Cosmopedia) | 24M |
| math (UltraData-Math) | 17M |
| synth (grade-school generator) | 26M |
| l3 (Ultra-FineWeb L3) | 9M |

Mixture phases as fractions of the run:
0-30% `(0.45, 0.15, 0.20, 0.10, 0.10, 0.00)`,
30-80% `(0.28, 0.08, 0.20, 0.14, 0.20, 0.10)`,
80-100% `(0.05, 0.00, 0.20, 0.20, 0.40, 0.15)`.

## Files

- `model.py`: the exact architecture of the 12B run, unchanged
- `train.py`: single-GPU trainer, same maths, vectorised batch sampler, resume support
- `notebook.py`: marimo notebook for molab
- `tokenizer.json`: the 4,104-id tokenizer used to build the shards
- `configs/win3m_100m.json`: the full parameter set of this run

## What this is not

It is a pipeline and throughput test, not a quality result. 100M tokens is 33 tokens per
parameter, so the loss will be far above the 12B run's. Compare throughput and the loss curve
shape, not the absolute loss.
