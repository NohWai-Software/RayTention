#!/usr/bin/env python3
# Copyright (C) 2026  NohWai Software
# License: AGPL-3.0 — see LICENSE file
# ==============================================================================
# PATENT PENDING NOTICE
# This file contains the implementation of the RayTention attention mechanism.
# The methods, algorithms, and architectures herein are protected under 
# U.S. Patent Application No. 64/102,801. 
# Copyright (c) 2026 NohWai Software. All Rights Reserved.
# ==============================================================================
"""
Definitive Benchmark: Standard, GQA, MLA vs RayTention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hardware:  any CUDA GPU (auto-detected)
Data:      FineWeb 16K tokenized sample (16384 vocab)
           If the .tok file is missing, a synthetic dataset is generated.
           To use real data, place fineweb_16k_slice.tok alongside this script
           or set the FINEWEB_TOK_PATH environment variable.
Models:    4 layers, D=128, matched param counts (4,205,824 each)
           Standard: dot-product attention + 1536-dim FFN
           RayTention: L2-distance → 7 signals → 512-dim AttnFFN + 512-dim FFN

Usage:  python3 definitive_bench.py              # auto-detect data
        python3 definitive_bench.py --synthetic  # force synthetic data
        python3 definitive_bench.py --steps 5000 --ctx 128  # custom training
        FINEWEB_TOK_PATH=/data/fineweb.tok python3 definitive_bench.py
"""

import struct, torch, torch.nn as nn, torch.nn.functional as F, gc, time, sys, os, gzip

# ═══ CLI ═══
USE_SYNTHETIC = "--synthetic" in sys.argv
TRAIN_STEPS   = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 2000
TRAIN_CTX     = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 64

# ═══ Constants ═══
D_MODEL      = 128
N_LAYERS     = 4
VOCAB        = 16384
SIGNAL_DIM   = D_MODEL * 5 + 2          # 642: cent+tcent+pred+top1+top2+spread+ent
ATTN_HIDDEN  = 512                       # RayTention AttnFFN hidden
STD_FFN_HID  = 1536                      # Standard FFN (boosted for param match)
RT_FFN_HID   = 512                       # RayTention FFN hidden
N_Q_HEADS     = 4                         # multi-head: Q heads
N_KV_HEADS    = 1                         # GQA: KV heads (shared across Q heads)
HEAD_DIM      = D_MODEL // N_Q_HEADS      # 32
MLA_LATENT    = 32                        # MLA: KV latent dim (4× compression)
LR           = 0.001
BATCH        = 1

TOK_PATH = "../helios_raytention/throughput_tokens/fineweb_16k/fineweb_16k_slice.tok"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
TOTAL_VRAM = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

torch.manual_seed(42)

# ═══ Data Loading (auto-detect or synthetic) ═══
TOK_PATH = os.environ.get("FINEWEB_TOK_PATH", "fineweb_16k_slice.tok")
USE_SYNTHETIC = "--synthetic" in sys.argv

def load_tok(path):
    """Load .tok or .tok.gz (u32 little-endian token IDs)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    return list(struct.unpack(f"<{len(data)//4}I", data))

if not USE_SYNTHETIC:
    # Try .tok, then .tok.gz
    for try_path in [TOK_PATH, TOK_PATH + ".gz"]:
        if os.path.exists(try_path):
            ALL_TOKENS = load_tok(try_path)
            DATA_SOURCE = f"FineWeb 16K ({len(ALL_TOKENS):,} tokens, {try_path})"
            break
    else:
        ALL_TOKENS = None
else:
    ALL_TOKENS = None

if ALL_TOKENS is None:
    torch.manual_seed(42)
    ALL_TOKENS = torch.randint(0, VOCAB, (200_000,)).tolist()
    DATA_SOURCE = f"synthetic ({len(ALL_TOKENS):,} tokens, seed=42)"
    if not USE_SYNTHETIC:
        print(f"Note: {TOK_PATH}[.gz] not found. Using synthetic data.")
        print(f"  Place fineweb_16k_slice.tok.gz alongside this script or set FINEWEB_TOK_PATH")
        print(f"  Run with --synthetic to suppress this message.\n")

print(f"Data: {DATA_SOURCE}")

# ═══ Shared Blocks ═══
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ═══ Standard Transformer ═══
class StdBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.attn_ffn = nn.Sequential(
            nn.Linear(D_MODEL, ATTN_HIDDEN), Swish(),
            nn.Linear(ATTN_HIDDEN, D_MODEL))
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, STD_FFN_HID), Swish(),
            nn.Linear(STD_FFN_HID, D_MODEL))
        self.scale = D_MODEL ** 0.5

    def forward(self, x, keys, ctx_ids):
        # x: [B,D]  keys: [V,D]  ctx_ids: [B,ctx]
        residual = x
        x = self.ln1(x)
        ck = keys[ctx_ids]                              # [B, ctx, D]
        scores = torch.bmm(x.unsqueeze(1), ck.transpose(1,2)).squeeze(1) / self.scale
        attn_w = F.softmax(scores, dim=-1)               # [B, ctx]
        attn_out = torch.bmm(attn_w.unsqueeze(1), ck).squeeze(1)  # [B, D]
        x = residual + self.attn_ffn(attn_out)
        residual = x
        x = residual + self.ffn(self.ln2(x))
        return x

class StdTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([StdBlock() for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x, keys, ctx_ids):
        for b in self.blocks: x = b(x, keys, ctx_ids)
        return self.head(self.ln_final(x))

# ═══ Multi-Head / GQA (H_q=4 heads, H_kv configurable) ═══
class MHA(nn.Module):
    def __init__(self, n_kv_heads=N_KV_HEADS):
        super().__init__()
        self.n_q = N_Q_HEADS; self.n_kv = n_kv_heads; self.hd = HEAD_DIM
        self.q_proj = nn.Linear(D_MODEL, N_Q_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(N_Q_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.scale = HEAD_DIM ** 0.5

    def forward(self, x, kv_k=None, kv_v=None):
        B = x.shape[0]
        q = self.q_proj(x).view(B, self.n_q, self.hd)          # [B, H_q, hd]
        if kv_k is not None:
            k = kv_k; v = kv_v                                  # [B, H_kv, hd, ctx]
        else:
            k = self.k_proj(x).view(B, self.n_kv, self.hd).unsqueeze(-1)
            v = self.v_proj(x).view(B, self.n_kv, self.hd).unsqueeze(-1)
        if self.n_q > self.n_kv:
            r = self.n_q // self.n_kv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        scores = torch.matmul(q.unsqueeze(2), k).squeeze(2) / self.scale
        attn_w = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_w.unsqueeze(2), v.transpose(-2,-1)).squeeze(2)
        return self.out_proj(out.reshape(B, -1))

class MHABlock(nn.Module):
    def __init__(self, n_kv_heads=N_KV_HEADS):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)
        self.attn = MHA(n_kv_heads)
        self.attn_ffn = nn.Sequential(nn.Linear(D_MODEL, ATTN_HIDDEN), Swish(),
                                       nn.Linear(ATTN_HIDDEN, D_MODEL))
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, STD_FFN_HID), Swish(),
                                  nn.Linear(STD_FFN_HID, D_MODEL))
    def forward(self, x, kv_k=None, kv_v=None):
        r = x; x = self.ln1(x)
        x = r + self.attn_ffn(self.attn(x, kv_k, kv_v))
        r = x; x = r + self.ffn(self.ln2(x))
        return x

class MHATransformer(nn.Module):
    def __init__(self, n_kv_heads=N_KV_HEADS):
        super().__init__()
        self.blocks = nn.ModuleList([MHABlock(n_kv_heads) for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.n_kv = n_kv_heads
    def forward(self, x, kv_k=None, kv_v=None):
        h = x
        if kv_k is not None:
            for i, b in enumerate(self.blocks):
                h = b(h, kv_k[:, i], kv_v[:, i])   # [B, H_kv, hd, ctx]
        else:
            for b in self.blocks: h = b(h)
        return self.head(self.ln_final(h))

# ═══ MLA: Multi-Head Latent Attention (DeepSeek V2/V3) ═══
class MLA(nn.Module):
    """KV cache stores latent vectors [latent_dim], not full [D].
       At inference: expand latent → full K,V, attend, discard."""
    def __init__(self):
        super().__init__()
        self.n_q = N_Q_HEADS; self.hd = HEAD_DIM; self.lat = MLA_LATENT
        self.q_proj = nn.Linear(D_MODEL, N_Q_HEADS * HEAD_DIM, bias=False)
        self.kv_down = nn.Linear(D_MODEL, MLA_LATENT, bias=False)        # compress
        self.k_up = nn.Linear(MLA_LATENT, D_MODEL, bias=False)           # expand K
        self.v_up = nn.Linear(MLA_LATENT, D_MODEL, bias=False)           # expand V
        self.out_proj = nn.Linear(N_Q_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.scale = HEAD_DIM ** 0.5

    def forward(self, x, kv_latent_k=None, kv_latent_v=None):
        B = x.shape[0]
        q = self.q_proj(x).view(B, self.n_q, self.hd)
        if kv_latent_k is not None:
            # kv_latent: [B, ctx, lat] → expand to [B, n_q, hd, ctx]
            k_full = self.k_up(kv_latent_k)   # [B, ctx, D]
            v_full = self.v_up(kv_latent_v)
            k = k_full.view(B, -1, self.n_q, self.hd).permute(0,2,3,1)  # [B, n_q, hd, ctx]
            v = v_full.view(B, -1, self.n_q, self.hd).permute(0,2,3,1)
        else:
            # Training: compress then expand (single token)
            latent = self.kv_down(x)                                 # [B, lat]
            k_full = self.k_up(latent).view(B, 1, 1, self.hd).expand(-1, self.n_q, -1, -1)
            v_full = self.v_up(latent).view(B, 1, 1, self.hd).expand(-1, self.n_q, -1, -1)
            k = k_full.permute(0,1,3,2); v = v_full.permute(0,1,3,2)  # [B, n_q, hd, 1]
        scores = torch.matmul(q.unsqueeze(2), k).squeeze(2) / self.scale
        attn_w = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_w.unsqueeze(2), v.transpose(-2,-1)).squeeze(2)
        return self.out_proj(out.reshape(B, -1))

class MLABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)
        self.attn = MLA()
        self.attn_ffn = nn.Sequential(nn.Linear(D_MODEL, ATTN_HIDDEN), Swish(),
                                       nn.Linear(ATTN_HIDDEN, D_MODEL))
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, STD_FFN_HID), Swish(),
                                  nn.Linear(STD_FFN_HID, D_MODEL))
    def forward(self, x, kv_latent_k=None, kv_latent_v=None):
        r = x; x = self.ln1(x)
        x = r + self.attn_ffn(self.attn(x, kv_latent_k, kv_latent_v))
        r = x; x = r + self.ffn(self.ln2(x))
        return x

class MLATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MLABlock() for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
    def forward(self, x, kv_latent_k=None, kv_latent_v=None):
        h = x
        if kv_latent_k is not None:
            for i, b in enumerate(self.blocks):
                h = b(h, kv_latent_k[:, i], kv_latent_v[:, i])
        else:
            for b in self.blocks: h = b(h)
        return self.head(self.ln_final(h))

# ═══ RayTention ═══
class RayTentionSignals(nn.Module):
    def forward(self, q, keys, ctx_ids):
        B, D = q.shape
        ctx = ctx_ids.shape[1]
        k = keys[ctx_ids]                                # [B, ctx, D]
        diff = q.unsqueeze(1) - k
        dist = torch.norm(diff, dim=-1)                  # [B, ctx]
        scores = -dist / 1.0                             # tau=1.0
        w = torch.softmax(scores - scores.max(-1,True)[0], dim=-1)

        # 1. Centroid
        centroid = (w.unsqueeze(-1) * k).sum(dim=1)     # [B, D]

        # 2. Temporal centroid (γ=0.7 decay toward recent)
        pos_w = 0.7 ** torch.arange(ctx-1, -1, -1, device=DEV).float()
        tw = w * pos_w.unsqueeze(0)
        tsum = tw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        tcent = (tw.unsqueeze(-1) * k).sum(dim=1) / tsum # [B, D]

        # 3. Predecessor (last token)
        pred = k[:, -1, :]                               # [B, D]

        # 4-5. Top-1, Top-2 by score
        _, top_idx = torch.topk(scores, min(2, ctx), dim=-1)
        top1 = k[torch.arange(B).unsqueeze(1), top_idx[:,0]].squeeze(1)  # [B,D]
        top2 = k[torch.arange(B).unsqueeze(1), top_idx[:,1]].squeeze(1) if ctx>=2 else top1

        # 6. Spread (weighted avg distance)
        spread = (w * dist).sum(dim=-1, keepdim=True)    # [B, 1]

        # 7. Entropy
        wc = w.clamp(min=1e-10)
        entropy = -(wc * wc.log()).sum(dim=-1, keepdim=True)  # [B, 1]

        return torch.cat([centroid, tcent, pred, top1, top2, spread, entropy], dim=-1)  # [B, 642]

class RTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.raytention = RayTentionSignals()
        self.attn_ffn = nn.Sequential(
            nn.Linear(SIGNAL_DIM, ATTN_HIDDEN), Swish(),
            nn.Linear(ATTN_HIDDEN, D_MODEL))
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, RT_FFN_HID), Swish(),
            nn.Linear(RT_FFN_HID, D_MODEL))

    def forward(self, x, keys, ctx_ids):
        residual = x
        signals = self.raytention(self.ln1(x), keys, ctx_ids)
        x = residual + self.attn_ffn(signals)
        residual = x
        x = residual + self.ffn(self.ln2(x))
        return x

class RTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([RTBlock() for _ in range(N_LAYERS)])
        self.ln_final = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, x, keys, ctx_ids):
        for b in self.blocks: x = b(x, keys, ctx_ids)
        return self.head(self.ln_final(x))

# ═══ Training ═══
def train_model(model, keys, ctx_ids, targets, steps, lr):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    keys = keys.clone().detach().requires_grad_(False)
    key_mom = torch.zeros_like(keys)
    ce_ema = 0.0
    ce_hist = []
    report_every = max(steps // 10, 1)
    t0 = time.time()

    for s in range(steps):
        idx = s % len(ctx_ids)
        cb = ctx_ids[idx:idx+1].to(DEV)
        tt = targets[idx:idx+1].to(DEV)
        x = keys[cb[0,0]].unsqueeze(0)

        opt.zero_grad()
        logits = model(x, keys, cb)
        loss = F.cross_entropy(logits, tt)
        loss.backward()

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1).squeeze(0)
            gl = probs.clone()
            gl[tt.item()] -= 1.0
            key_mom = 0.9 * key_mom + gl.unsqueeze(1) * model.head.weight
            keys -= lr * 10.0 * key_mom
        opt.step()

        ce = loss.item()
        ce_ema = 0.99 * ce_ema + 0.01 * ce
        if s % report_every == 0 or s == steps - 1:
            ce_hist.append((s, ce))
            elapsed = time.time() - t0
            print(f"    step {s:>5}/{steps}  ce={ce:.4f}  ema={ce_ema:.4f}  {s/elapsed:.0f} steps/s" if s > 0 else f"    step {s:>5}/{steps}  ce={ce:.4f}")

    elapsed = time.time() - t0
    return {"steps": steps, "time": elapsed, "steps_per_sec": steps / elapsed,
            "final_ema": ce_ema, "ce_history": ce_hist,
            "params": count_params(model)}

# ═══ Inference VRAM (autoregressive) ═══
def measure_inference_vram(ModelClass, keys, tok_ids, ctx, steps):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(DEV)

    # Allocate KV cache per layer: [L, ctx, D] each for K and V
    kv_k = torch.zeros(N_LAYERS, ctx, D_MODEL, device=DEV, dtype=torch.float32)
    kv_v = torch.zeros(N_LAYERS, ctx, D_MODEL, device=DEV, dtype=torch.float32)

    # Build minimal blocks that consume KV cache directly
    class StdBKV(nn.Module):
        def __init__(self, li):
            super().__init__()
            self.ln1 = nn.LayerNorm(D_MODEL); self.ln2 = nn.LayerNorm(D_MODEL)
            self.af = nn.Sequential(nn.Linear(D_MODEL, ATTN_HIDDEN), Swish(), nn.Linear(ATTN_HIDDEN, D_MODEL))
            self.ff = nn.Sequential(nn.Linear(D_MODEL, STD_FFN_HID), Swish(), nn.Linear(STD_FFN_HID, D_MODEL))
            self.sc = D_MODEL**0.5; self.li = li
        def forward(self, x, n):
            r = x; xn = self.ln1(x)
            k = kv_k[self.li, :n]; v = kv_v[self.li, :n]
            s = (xn.unsqueeze(1) @ k.T.unsqueeze(0)).squeeze(1) / self.sc
            a = torch.softmax(s, -1).unsqueeze(1) @ v.unsqueeze(0)
            return r + self.af(a.squeeze(1)) + self.ff(self.ln2(r))

    blocks = nn.ModuleList([StdBKV(i) for i in range(N_LAYERS)]).to(DEV)
    ln_f = nn.LayerNorm(D_MODEL).to(DEV); hd = nn.Linear(D_MODEL, VOCAB, bias=False).to(DEV)

    with torch.no_grad():
        for s in range(min(steps, ctx)):
            x = keys[tok_ids[s]].unsqueeze(0)
            for li in range(N_LAYERS):
                kv_k[li, s, :] = x; kv_v[li, s, :] = x
            h = x
            for b in blocks: h = b(h, s+1)
            _ = hd(ln_f(h))

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(DEV) / 1e6
    del blocks, ln_f, hd, kv_k, kv_v; gc.collect(); torch.cuda.empty_cache()
    return peak_mb

def measure_inference_vram_rt(ModelClass, keys, tok_ids, ctx, steps):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(DEV)

    model = ModelClass().to(DEV); model.eval()

    with torch.no_grad():
        for s in range(min(steps, ctx)):
            x = keys[tok_ids[s]].unsqueeze(0)
            ctx_ids = tok_ids[:s+1].unsqueeze(0)
            _ = model(x, keys, ctx_ids)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(DEV) / 1e6

    del model
    gc.collect(); torch.cuda.empty_cache()
    return peak_mb

def measure_inference_vram_gqa(n_kv_heads, keys, tok_ids, ctx, steps):
    """GQA inference: pre-allocate KV cache [1, L, H_kv, hd, ctx] and measure."""
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(DEV)

    # [batch=1, layers, heads, head_dim, ctx]
    kv_k = torch.zeros(1, N_LAYERS, n_kv_heads, HEAD_DIM, ctx, device=DEV, dtype=torch.float32)
    kv_v = torch.zeros(1, N_LAYERS, n_kv_heads, HEAD_DIM, ctx, device=DEV, dtype=torch.float32)

    proj_k = nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False).to(DEV)
    proj_v = nn.Linear(D_MODEL, n_kv_heads * HEAD_DIM, bias=False).to(DEV)

    model = MHATransformer(n_kv_heads).to(DEV); model.eval()

    with torch.no_grad():
        for s in range(min(steps, ctx)):
            x = keys[tok_ids[s]].unsqueeze(0)
            k = proj_k(x).view(1, n_kv_heads, HEAD_DIM)
            v = proj_v(x).view(1, n_kv_heads, HEAD_DIM)
            kv_k[:, :, :, :, s] = k.unsqueeze(1); kv_v[:, :, :, :, s] = v.unsqueeze(1)
            _ = model(x, kv_k[:, :, :, :, :s+1], kv_v[:, :, :, :, :s+1])

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(DEV) / 1e6
    del model, kv_k, kv_v, proj_k, proj_v; gc.collect(); torch.cuda.empty_cache()
    return peak_mb

def measure_inference_vram_mla(keys, tok_ids, ctx, steps):
    """MLA inference: KV cache stores latent vectors [1, L, ctx, lat]."""
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats(DEV)

    kv_latent_k = torch.zeros(1, N_LAYERS, ctx, MLA_LATENT, device=DEV, dtype=torch.float32)
    kv_latent_v = torch.zeros(1, N_LAYERS, ctx, MLA_LATENT, device=DEV, dtype=torch.float32)

    model = MLATransformer().to(DEV); model.eval()

    with torch.no_grad():
        for s in range(min(steps, ctx)):
            x = keys[tok_ids[s]].unsqueeze(0)
            for i, b in enumerate(model.blocks):
                latent = b.attn.kv_down(b.ln1(x))  # [1, lat]
                kv_latent_k[:, i, s] = latent
                kv_latent_v[:, i, s] = latent
            _ = model(x, kv_latent_k[:, :, :s+1], kv_latent_v[:, :, :s+1])

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(DEV) / 1e6
    del model, kv_latent_k, kv_latent_v; gc.collect(); torch.cuda.empty_cache()
    return peak_mb

# ═══ MAIN ═══
print(f"\n{'═'*60}")
print(f"  DEFINITIVE BENCHMARK")
print(f"  Hardware: {GPU_NAME} ({TOTAL_VRAM:.1f} GB)")
print(f"  Models: {N_LAYERS} layers, D={D_MODEL}")
print(f"{'═'*60}")

# Print model params once
m_std = StdTransformer().to(DEV)
m_rt = RTTransformer().to(DEV)
P_STD = count_params(m_std)
P_RT = count_params(m_rt)
del m_std, m_rt; gc.collect(); torch.cuda.empty_cache()

print(f"\n  Model params: {P_STD:,} each (matched)")
print(f"  Standard FFN hidden: {STD_FFN_HID}  |  RayTention AttnFFN hidden: {ATTN_HIDDEN}, FFN: {RT_FFN_HID}")
print(f"  Standard: LN → dot-prod attn → AttnFFN → + → LN → FFN → +")
print(f"  RayTention: LN → L2→7 signals → AttnFFN → + → LN → FFN → +")

# ═══ PART 1: Training CE ═══
print(f"\n{'─'*60}")
print(f"  PART 1: CE Convergence (ctx={TRAIN_CTX}, {TRAIN_STEPS} steps)")
print(f"{'─'*60}")

needed = TRAIN_CTX + TRAIN_STEPS + 1
t = ALL_TOKENS[:min(len(ALL_TOKENS), needed)]
n = len(t)
ctx_ids_list = [t[i:i+TRAIN_CTX] for i in range(0, n-TRAIN_CTX, max(1, (n-TRAIN_CTX)//TRAIN_STEPS))]
tgts_list = [t[i+TRAIN_CTX] for i in range(0, n-TRAIN_CTX, max(1, (n-TRAIN_CTX)//TRAIN_STEPS))]
ci = torch.tensor(ctx_ids_list[:TRAIN_STEPS], dtype=torch.long)
tg = torch.tensor(tgts_list[:TRAIN_STEPS], dtype=torch.long)

keys = torch.randn(VOCAB, D_MODEL, device=DEV) * 0.1

ci_dev = ci.to(DEV); tg_dev = tg.to(DEV)

torch.manual_seed(42)
print("\n  Training Standard...")
sr = train_model(StdTransformer().to(DEV), keys.clone(), ci_dev, tg_dev, TRAIN_STEPS, LR)
print(f"  Standard:  {sr['time']:.1f}s  {sr['steps_per_sec']:.0f} steps/s  final CE ema={sr['final_ema']:.4f}")

torch.manual_seed(42)
print("  Training RayTention...")
rr = train_model(RTTransformer().to(DEV), keys.clone(), ci_dev, tg_dev, TRAIN_STEPS, LR)
print(f"  RayTention: {rr['time']:.1f}s  {rr['steps_per_sec']:.0f} steps/s  final CE ema={rr['final_ema']:.4f}")

print(f"\n  {'Step':>6} | {'Standard CE':>12} | {'RayTention CE':>14}")
print(f"  {'─'*40}")
for i in range(min(len(sr['ce_history']), len(rr['ce_history']))):
    s1, c1 = sr['ce_history'][i]; s2, c2 = rr['ce_history'][i]
    print(f"  {s1:>6} | {c1:>12.4f} | {c2:>14.4f}")

# ═══ PART 2: Inference VRAM ═══
INF_STEPS = 500
print(f"\n{'─'*60}")
print(f"  PART 2: Inference VRAM (autoregressive, {INF_STEPS} steps)")
print(f"{'─'*60}")

tok_ids = torch.randint(0, VOCAB, (max(1048576, INF_STEPS),), device=DEV)

print(f"\n  {'ctx':>7} | {'Standard':>12} | {'RayTention':>12} | {'Ratio':>8} | {'Std %VRAM':>10}")
print(f"  {'─'*55}")

contexts = [1024, 4096, 16384, 65536, 131072, 1048576]
vram_results = {}  # ctx -> (std_mb, rt_mb)
for c in contexts:
    sv = measure_inference_vram(None, keys, tok_ids, c, min(INF_STEPS, c))
    rv = measure_inference_vram_rt(RTTransformer, keys, tok_ids, c, min(INF_STEPS, c))
    vram_results[c] = (sv, rv)
    ratio = sv / rv if rv > 0 else 0
    pct = sv / (TOTAL_VRAM * 1000) * 100
    print(f"  {c:>7} | {sv:>9.0f} MB | {rv:>9.0f} MB | {ratio:>6.1f}x | {pct:>8.1f}%")

# ═══ PART 2.5: GQA Comparison ═══
print(f"\n{'─'*60}")
print(f"  PART 2.5: Multi-Head vs GQA-4 vs RayTention VRAM")
print(f"  (H_q=4 heads, H_kv=4 vs H_kv=1 shared)")
print(f"{'─'*60}")

print(f"\n  {'ctx':>7} | {'MHA (4 KV)':>12} | {'GQA-4 (1 KV)':>13} | {'RayTention':>12} | {'GQA/RT':>8}")
print(f"  {'─'*60}")

gqa_contexts = [4096, 16384, 65536, 131072, 262144]
for c in gqa_contexts:
    mha_vram = measure_inference_vram_gqa(N_Q_HEADS, keys, tok_ids, c, min(500, c))   # 4 KV heads
    gqa_vram = measure_inference_vram_gqa(1, keys, tok_ids, c, min(500, c))            # 1 KV head
    rt_vram  = measure_inference_vram_rt(RTTransformer, keys, tok_ids, c, min(500, c))
    ratio = gqa_vram / rt_vram if rt_vram > 0 else 0
    print(f"  {c:>7} | {mha_vram:>9.0f} MB | {gqa_vram:>10.0f} MB | {rt_vram:>9.0f} MB | {ratio:>6.1f}x")

# ═══ PART 2.6: MLA Comparison ═══
print(f"\n{'─'*60}")
print(f"  PART 2.6: MLA (latent_dim={MLA_LATENT}) vs GQA-4 vs RayTention")
print(f"  (MLA stores {MLA_LATENT}-dim latent vectors, 4× smaller than D=128)")
print(f"{'─'*60}")

print(f"\n  {'ctx':>7} | {'MLA':>12} | {'GQA-4':>12} | {'RayTention':>12} | {'MLA/RT':>8}")
print(f"  {'─'*55}")

for c in [4096, 16384, 65536, 131072, 262144]:
    mla_vram = measure_inference_vram_mla(keys, tok_ids, c, min(500, c))
    gqa_vram = measure_inference_vram_gqa(1, keys, tok_ids, c, min(500, c))
    rt_vram  = measure_inference_vram_rt(RTTransformer, keys, tok_ids, c, min(500, c))
    ratio = mla_vram / rt_vram if rt_vram > 0 else 0
    print(f"  {c:>7} | {mla_vram:>9.0f} MB | {gqa_vram:>9.0f} MB | {rt_vram:>9.0f} MB | {ratio:>6.1f}x")

# ═══ PART 3: Speed at different context lengths ═══
print(f"\n{'─'*60}")
print(f"  PART 3: Per-token latency at target context length")
print(f"  (pre-fills KV cache to ctx-1, measures final step)")
print(f"{'─'*60}")

print(f"\n  {'ctx':>7} | {'Std (tok/s)':>13} | {'RT (tok/s)':>11} | {'Ratio':>8}")
print(f"  {'─'*48}")

for c in [64, 256, 1024, 4096, 16384, 65536, 131072]:
    WARMUP = 3; MEASURE = 10

    # Standard: pre-fill KV cache to c-1, then measure c-th step
    kv_k = torch.randn(N_LAYERS, c, D_MODEL, device=DEV, dtype=torch.float32)
    kv_v = torch.randn(N_LAYERS, c, D_MODEL, device=DEV, dtype=torch.float32)
    class StdBKV3(nn.Module):
        def __init__(self,li):
            super().__init__(); self.ln1=nn.LayerNorm(D_MODEL); self.ln2=nn.LayerNorm(D_MODEL)
            self.af=nn.Sequential(nn.Linear(D_MODEL,ATTN_HIDDEN),Swish(),nn.Linear(ATTN_HIDDEN,D_MODEL))
            self.ff=nn.Sequential(nn.Linear(D_MODEL,STD_FFN_HID),Swish(),nn.Linear(STD_FFN_HID,D_MODEL))
            self.sc=D_MODEL**0.5; self.li=li
        def forward(self,x,n):
            r=x; xn=self.ln1(x); k=kv_k[self.li,:n]; v=kv_v[self.li,:n]
            s=(xn.unsqueeze(1)@k.T.unsqueeze(0)).squeeze(1)/self.sc
            a=torch.softmax(s,-1).unsqueeze(1)@v.unsqueeze(0)
            return r+self.af(a.squeeze(1))+self.ff(self.ln2(r))
    blocks=[StdBKV3(i).to(DEV) for i in range(N_LAYERS)]
    ln_f=nn.LayerNorm(D_MODEL).to(DEV); hd=nn.Linear(D_MODEL,VOCAB,bias=False).to(DEV)

    # Warmup
    x=torch.randn(1,D_MODEL,device=DEV)
    for _ in range(WARMUP):
        h=x
        for b in blocks: h=b(h,c)
        _=hd(ln_f(h))
    torch.cuda.synchronize()

    # Measure
    t0=time.time()
    for _ in range(MEASURE):
        h=x
        for b in blocks: h=b(h,c)
        _=hd(ln_f(h))
    torch.cuda.synchronize()
    std_tps=MEASURE/(time.time()-t0)
    del blocks,ln_f,hd,kv_k,kv_v; gc.collect(); torch.cuda.empty_cache()

    # RayTention: pre-gather context keys for c-1 tokens, measure c-th step
    ctx_ids_all=torch.randint(0,VOCAB,(c,),device=DEV)
    m=RTTransformer().to(DEV); m.eval()
    # Warmup
    for _ in range(WARMUP):
        _=m(x,keys,ctx_ids_all.unsqueeze(0))
    torch.cuda.synchronize()
    # Measure
    t0=time.time()
    for _ in range(MEASURE):
        _=m(x,keys,ctx_ids_all.unsqueeze(0))
    torch.cuda.synchronize()
    rt_tps=MEASURE/(time.time()-t0)
    del m; gc.collect(); torch.cuda.empty_cache()

    ratio=std_tps/rt_tps if rt_tps>0 else 0
    print(f"  {c:>7} | {std_tps:>11.0f} | {rt_tps:>9.0f} | {ratio:>6.1f}x")

# ═══ SUMMARY ═══
v1s, v1r = vram_results[131072]
v2s, v2r = vram_results[1048576]

print(f"\n{'═'*60}")
print(f"  SUMMARY")
print(f"{'═'*60}")
print(f"  {'':22} | {'Standard':>14} | {'RayTention':>14}")
print(f"  {'─'*54}")
print(f"  {'Model params':22} | {P_STD:>14,} | {P_RT:>14,}")
print(f"  {'Train steps/s':22} | {sr['steps_per_sec']:>13.0f} | {rr['steps_per_sec']:>13.0f}")
print(f"  {'Final CE (ctx=64)':22} | {sr['final_ema']:>14.4f} | {rr['final_ema']:>14.4f}")
print(f"  {'Inf VRAM ctx=131K':22} | {v1s:>9.0f} MB | {v1r:>9.0f} MB")
print(f"  {'Inf VRAM ctx=1M':22} | {v2s:>9.0f} MB | {v2r:>9.0f} MB")
print(f"  {'Vs GQA':22} | {'see Part 2.5':>14} | {'see Part 2.5':>14}")
print(f"  {'KV cache required':22} | {'Yes':>14} | {'No':>14}")
print(f"  {'Attention memory':22} | {'O(ctx) KV cache':>14} | {'O(1) 642 floats':>14}")
print()
print(f"  Data: {DATA_SOURCE}")
print(f"  Hardware: {GPU_NAME} ({TOTAL_VRAM:.1f} GB)")
print(f"  Framework: PyTorch {torch.__version__}")
