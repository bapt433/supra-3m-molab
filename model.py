"""Candidate architecture for the sub-3M SLM project (param-budget prototype).

Design points under test:
  - pre-norm RMSNorm, SwiGLU, tied embeddings
  - GQA (3 query / 1 KV head, head_dim 48) with QK-norm
  - partial RoPE (rotary_dim of head_dim)
  - XSA: subtract the projection of the attention output on the self value vector
  - Attention Residuals (AttnRes): softmax over all previous layer outputs, learned query per layer
  - optional logit cap
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


def apply_rope(x, cos, sin, rotary_dim):
    # x: (B, H, T, D)
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    c, s = cos, sin
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    xr = torch.stack((o1, o2), dim=-1).flatten(-2)
    return torch.cat((xr, xp), dim=-1)


class Attention(nn.Module):
    def __init__(self, d, n_q, n_kv, rotary_dim, use_xsa=True):
        super().__init__()
        self.n_q, self.n_kv = n_q, n_kv
        self.hd = d // n_q
        self.rotary_dim = rotary_dim
        self.use_xsa = use_xsa
        self.wq = nn.Linear(d, n_q * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wo = nn.Linear(n_q * self.hd, d, bias=False)
        self.gate = nn.Linear(d, n_q * self.hd, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)   # start near identity
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_q, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin, self.rotary_dim)
        k = apply_rope(k, cos, sin, self.rotary_dim)
        rep = self.n_q // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.use_xsa:
            vn = F.normalize(v, dim=-1)
            proj = (y * vn).sum(-1, keepdim=True)
            y = y - proj * vn
        y = y.transpose(1, 2).reshape(B, T, -1)
        y = y * torch.sigmoid(self.gate(x))
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, n_q, n_kv, rotary_dim, inter):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_q, n_kv, rotary_dim)
        self.n2 = RMSNorm(d)
        self.mlp = SwiGLU(d, inter)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab=4096, d=144, blocks=10, inter=420, n_q=3, n_kv=1,
                 rotary_dim=24, ctx=1024, rope_theta=10000.0, attnres=True,
                 logit_cap=None, apply_repeat=None):
        super().__init__()
        self.ctx, self.rotary_dim, self.attnres = ctx, rotary_dim, attnres
        self.logit_cap = logit_cap
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_q, n_kv, rotary_dim, inter) for _ in range(blocks)])
        self.attnres_q = nn.Parameter(torch.randn(blocks, d) * 0.02) if attnres else None
        self.final_norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied
        # which stored blocks to apply twice, e.g. [3,4,5]
        self.apply_repeat = apply_repeat or []
        self._rope_cache = {}
        self.apply(self._init)
        # residual branches scaled down as depth grows
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, std=0.02 / math.sqrt(2 * blocks))
            nn.init.normal_(blk.mlp.w2.weight, std=0.02 / math.sqrt(2 * blocks))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def rope(self, T, device):
        key = (T, device)
        if key not in self._rope_cache:
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
            t = torch.arange(T, device=device).float()
            freqs = torch.outer(t, inv)
            self._rope_cache[key] = (freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :])
        return self._rope_cache[key]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self.rope(T, idx.device)
        h = self.embed(idx)
        hs = [h]
        hs_n = [F.normalize(h, dim=-1)]
        for i, blk in enumerate(self.blocks):
            if self.attnres:
                stack = torch.stack(hs, 0)               # (L, B, T, D)
                sn = torch.stack(hs_n, 0)                # normalized once per source
                q = F.normalize(self.attnres_q[i], dim=-1)
                logits = torch.einsum("lbt d,d->lbt", sn, q) / math.sqrt(self.embed.embedding_dim)
                w = torch.softmax(logits, dim=0)
                x = torch.einsum("lbt,lbt d->bt d", w, stack)
            else:
                x = h
            h = blk(x, cos, sin)
            hs.append(h)
            if self.attnres:
                hs_n.append(F.normalize(h, dim=-1))
        h = self.final_norm(h)
        logits = self.head(h)
        if self.logit_cap is not None:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def n_params(**kw):
    m = Model(**kw)
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    cfgs = {
        "B  d144 b10 f420 4096": dict(vocab=4096, d=144, blocks=10, inter=420),
        "B2 d144 b10 f432 4096": dict(vocab=4096, d=144, blocks=10, inter=432),
        "C  d160 b10 f384 2048": dict(vocab=2048, d=160, blocks=10, inter=384, n_q=4, n_kv=2),
        "D  d144 b11 f352 4096": dict(vocab=4096, d=144, blocks=11, inter=352),
    }
    for name, c in cfgs.items():
        n = n_params(**c)
        print(f"{name:24} {n:>10,}  ({n/1e6:.3f}M)")
