import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
# Supra2.5-Pico 3M: 100M-token reproduction

The exact winning 12B recipe, scaled down to **100M tokens** as a pipeline and
throughput test on molab's RTX Pro 6000 (96 GB).

- Model: 2,982,480 params, d=144, 10 blocks, SwiGLU 376, GQA 3:1, partial RoPE 24/48,
  QK-norm, XSA, attention residuals, tied 4,104-token embeddings, ctx 1024
- Batch: **256 x 1024, no accumulation** = 262,144 tokens/step (identical effective
  batch to the 12B run's 64 x 1024 x 4)
- Optimiser: Muon 0.02 (2D weights) + AdamW 6e-3, betas 0.9/0.95, grad clip 1.0
- Schedule: 381 steps, warmup 17 (same 4.369% share as 2,000/45,776), stable to 80%,
  cosine to zero
- Data: the same tokenised shards and the same three-phase mixture as the 12B run

Expected wall clock: about 1 minute of training once the data is on disk, plus
download and torch.compile.
"""
    )
    return


@app.cell
def _(mo):
    import json
    import os
    import pathlib
    import shutil
    import subprocess
    import sys
    import tarfile
    import urllib.request

    REPO = "bapt433/supra-3m-molab"
    TAG = "data-v1"
    ASSET = "slm3m_100m.tar.gz"

    WORK = pathlib.Path.cwd() / "supra3m_work"
    SRC = WORK / "src"
    DATA = WORK / "data"
    WORK.mkdir(parents=True, exist_ok=True)

    def fetch(url, dst):
        if dst.exists() and dst.stat().st_size > 0:
            print(f"have {dst.name} ({dst.stat().st_size/2**20:.1f} MiB)")
            return
        print(f"downloading {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(dst)
        print(f"saved {dst.name} ({dst.stat().st_size/2**20:.1f} MiB)")

    repo_tar = WORK / "repo.tar.gz"
    fetch(f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main", repo_tar)
    if SRC.exists():
        shutil.rmtree(SRC)
    with tarfile.open(repo_tar) as t:
        t.extractall(WORK / "_repo")
    root = next((WORK / "_repo").iterdir())
    shutil.move(str(root), SRC)
    shutil.rmtree(WORK / "_repo", ignore_errors=True)
    print("source:", sorted(p.name for p in SRC.iterdir()))

    data_tar = WORK / "data.tar.gz"
    fetch(f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}", data_tar)
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    with tarfile.open(data_tar) as t:
        t.extractall(DATA)
    counts = {p.stem: json.load(open(p))["tokens"] for p in sorted(DATA.glob("*.json"))}
    print("token shards:", counts, "total", f"{sum(counts.values())/1e6:.1f}M")

    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")

    mo.output.append(
        mo.md(
            f"**Env ready.** torch {torch.__version__}, "
            f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, "
            f"{sum(counts.values())/1e6:.0f}M tokens staged in `{DATA}`."
        )
    )
    return DATA, SRC, json, subprocess, sys, torch


@app.cell
def _(DATA, SRC, json, mo, subprocess, sys, torch):
    cmd = [
        sys.executable, "train.py",
        "--data-dir", str(DATA),
        "--out-dir", str(SRC / "runs" / "test100m"),
        "--tokens", "100e6",
        "--bs", "256",
        "--seq", "1024",
        "--grad-accum", "1",
        "--lr-muon", "0.02",
        "--lr-adam", "6e-3",
        "--stable", "0.80",
        "--cooldown-start", "0.80",
        "--ckpt-every", "100",
        "--log-every", "10",
        "--dtype", "bf16",
        "--compile",
        "--resume",
    ]
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    rc = proc.wait()
    print("exit", rc)

    out = SRC / "runs" / "test100m"
    summary = json.load(open(out / "summary.json")) if (out / "summary.json").exists() else None
    mo.output.append(
        mo.md(
            "**Run finished.** " + (
                f"{summary['tokens']/1e6:.1f}M tokens, final loss {summary['final_loss']:.4f}, "
                f"{summary['tok_PER_s']:,} tok/s, {summary['wall_s']:.0f}s wall."
                if summary else f"exit code {rc}, no summary.json"
            )
        )
    )
    return (out,)


@app.cell
def _(out):
    import csv

    log = out / "log.csv"
    rows = list(csv.DictReader(open(log))) if log.exists() else []
    print(f"{len(rows)} log points; loss is the mean of the last 10 steps of each window")

    fig = None
    if rows:
        try:
            import matplotlib.pyplot as plt
            steps = [int(r["step"]) for r in rows]
            losses = [float(r["loss"]) for r in rows]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, losses, marker="o", ms=3, lw=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("training loss")
            ax.set_title("Supra2.5-Pico 3M, 100M tokens")
            ax.grid(alpha=0.3)
            fig.tight_layout()
        except Exception as e:
            print("plot skipped:", e)
            print("last losses:", [r["loss"] for r in rows[-10:]])

    fig
    return


if __name__ == "__main__":
    app.run()
