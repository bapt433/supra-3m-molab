"""Single-GPU pretraining for the 2.98M Supra2.5-Pico model.

This is the exact recipe of the winning 12B-token run (index 12.96):
bs 64 x 1024 x grad-accum 4 = 262,144 tokens/step, Muon 0.02 + AdamW 6e-3,
2,000-step warmup, stable to 80%, cosine to zero, bf16, torch.compile.

The effective batch is preserved here with bs 256 x 1024 x accum 1.
Only the total token count changes, so the schedule is rescaled:
warmup keeps its 4.369% share of the run (2000 / 45776 steps).

    python train.py --data-dir ./data --out-dir ./runs/test --tokens 100e6 \
        --bs 256 --seq 1024 --grad-accum 1 --compile
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import model as M

PHASES = [
    (0.00, (0.45, 0.15, 0.20, 0.10, 0.10, 0.00)),
    (0.30, (0.28, 0.08, 0.20, 0.14, 0.20, 0.10)),
    (0.70, (0.05, 0.00, 0.20, 0.20, 0.40, 0.15)),   # cooldown, 60% arithmetic
]
SOURCES = ["ufw", "txt", "cos", "math", "synth", "l3"]

# 2,000 warmup steps over 45,776 steps in the 12B run
WARMUP_FRAC = 2000 / 45776


def mixture(frac):
    w = PHASES[0][1]
    for start, weights in PHASES:
        if frac >= start:
            w = weights
    return w


class Shards:
    def __init__(self, data_dir):
        self.data = {}
        for n in SOURCES:
            p = os.path.join(data_dir, f"{n}.bin")
            j = os.path.join(data_dir, f"{n}.json")
            if not (os.path.exists(p) and os.path.exists(j)):
                continue
            meta = json.load(open(j))
            self.data[n] = (np.memmap(p, dtype=np.uint16, mode="r"), meta["tokens"])
        if not self.data:
            raise SystemExit(f"no token shards found under {data_dir}")
        print("loaded:", {k: v[1] for k, v in self.data.items()}, flush=True)

    def batch(self, rng, weights, bs, T):
        """Same distribution as the original per-example loop, vectorised.

        Every sequence gets a uniform random start inside its source shard.
        """
        names = list(self.data)
        w = np.array([weights[SOURCES.index(n)] for n in names], dtype=np.float64)
        w = w / w.sum()
        choose = rng.choice(len(names), size=bs, p=w)
        out = np.empty((bs, T + 1), dtype=np.int64)
        span = np.arange(T + 1)
        for si, name in enumerate(names):
            mask = choose == si
            k = int(mask.sum())
            if k == 0:
                continue
            arr, n = self.data[name]
            starts = rng.integers(0, max(1, n - T - 2), size=k)
            out[mask] = arr[starts[:, None] + span[None, :]]
        return out


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                o = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                p.add_(o, alpha=-lr * scale)


def split_params(model):
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(k in name for k in ("embed", "head", "attnres")) and min(p.shape) >= 8:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


def lr_at(step, total, warmup, stable_frac, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < stable_frac * total:
        return peak
    p = (step - stable_frac * total) / max(1, (1 - stable_frac) * total)
    return peak * 0.5 * (1 + math.cos(math.pi * p))


def strip_compile(state):
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./runs/test")
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--bs", type=int, default=256, help="sequences per step")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adam", type=float, default=6e-3)
    ap.add_argument("--warmup", type=int, default=-1, help="-1 = 4.369% of total steps, as in the 12B run")
    ap.add_argument("--stable", type=float, default=0.80)
    ap.add_argument("--cooldown-start", type=float, default=0.80)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global PHASES
    PHASES = [(0.00, PHASES[0][1]), (0.30, PHASES[1][1]), (a.cooldown_start, PHASES[2][1])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"gpu {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_capability(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    shards = Shards(a.data_dir)
    net = M.Model(vocab=4104, d=144, blocks=10, inter=376, ctx=a.seq).to(device)
    if a.compile:
        try:
            net = torch.compile(net)
            print("compiled", flush=True)
        except Exception as e:
            print(f"compile failed, running eager: {e}", flush=True)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"params {nparam:,}", flush=True)

    muon_p, adam_p = split_params(net)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    opt_adam = torch.optim.AdamW(adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)

    tokens_per_step = a.bs * a.seq * a.grad_accum
    total_steps = int(a.tokens / tokens_per_step)
    warmup = a.warmup if a.warmup >= 0 else max(1, round(total_steps * WARMUP_FRAC))
    print(f"tokens/step {tokens_per_step:,}  total steps {total_steps:,}  warmup {warmup}", flush=True)

    step = 0
    ckpt_path = os.path.join(a.out_dir, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        (net._orig_mod if hasattr(net, "_orig_mod") else net).load_state_dict(ck["model"])
        opt_adam.load_state_dict(ck["adam"])
        opt_muon.load_state_dict(ck["muon"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)

    log_path = os.path.join(a.out_dir, "log.csv")
    if not os.path.exists(log_path):
        open(log_path, "w").write("step,tokens,loss,lr_adam,tok_per_s\n")

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.dtype]
    if device.type == "cpu":
        amp = None
    rng = np.random.default_rng(a.seed * 100003 + 7)
    net.train()
    t_start = time.time()
    t_last = t_start
    losses = []

    while step < total_steps:
        frac = step / total_steps
        w = mixture(frac)
        lr_m = lr_at(step, total_steps, warmup, a.stable, a.lr_muon)
        lr_a = lr_at(step, total_steps, warmup, a.stable, a.lr_adam)
        opt_adam.param_groups[0]["lr"] = lr_a
        opt_muon.param_groups[0]["lr"] = lr_m

        opt_adam.zero_grad(set_to_none=True)
        opt_muon.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            b = shards.batch(rng, w, a.bs, a.seq)
            x = torch.from_numpy(b[:, :-1]).to(device)
            y = torch.from_numpy(b[:, 1:]).to(device)
            if amp is not None:
                with torch.autocast("cuda", dtype=amp):
                    _, loss = net(x, y)
            else:
                _, loss = net(x, y)
            (loss / a.grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt_adam.step()
        opt_muon.step()

        li = loss.item()
        if not math.isfinite(li):
            print(f"NaN loss at step {step}, aborting", flush=True)
            break
        losses.append(li)
        step += 1

        if step % a.log_every == 0 or step == total_steps:
            now = time.time()
            sps = a.log_every / (now - t_last) if step % a.log_every == 0 else 0.0
            ml = sum(losses[-a.log_every:]) / len(losses[-a.log_every:])
            tps = sps * tokens_per_step
            done = step * tokens_per_step
            amt = f"{done/1e6:.1f}M" if done >= 1e6 else f"{done/1e3:.0f}K"
            print(f"step {step}/{total_steps} tok {amt} "
                  f"loss {ml:.4f} lr {lr_a:.2e} {tps:.0f} tok/s "
                  f"elapsed {(now-t_start):.1f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{step*tokens_per_step},{ml:.5f},{lr_a:.6g},{tps:.0f}\n")
            t_last = now

        if step % a.ckpt_every == 0 or step == total_steps:
            state = (net._orig_mod if hasattr(net, "_orig_mod") else net).state_dict()
            torch.save({"model": strip_compile(state),
                        "adam": opt_adam.state_dict(),
                        "muon": opt_muon.state_dict(),
                        "step": step, "args": vars(a)}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    wall = time.time() - t_start
    summary = {
        "params": nparam,
        "tokens": step * tokens_per_step,
        "steps": step,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
        "wall_s": round(wall, 1),
        "tok_PER_s": round(step * tokens_per_step / wall) if wall > 0 else None,
        "tokens_per_step": tokens_per_step,
        "warmup": warmup,
        "args": vars(a),
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done", json.dumps({k: summary[k] for k in ("tokens", "final_loss", "wall_s", "tok_PER_s")}), flush=True)


if __name__ == "__main__":
    main()
