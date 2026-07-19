#!/usr/bin/env python3
# Copyright (C) 2026  NohWai Software
# License: AGPL-3.0 — see LICENSE file
#
# =============================================================================
# PATENT PENDING NOTICE
# This file contains the implementation of the RayTention attention mechanism.
# The methods, algorithms, and architectures herein are protected under
# U.S. Patent Application No. 64/102,801.
# Copyright (c) 2026 NohWai Software. All Rights Reserved.
# =============================================================================
#
# RayTention — Reference Implementation
#
# ============================================================================
# What is RayTention?
# ============================================================================
#
# RayTention is a drop-in replacement for standard self-attention that works
# with any model dimension. Instead of computing Q·K dot products, it uses
# negative squared Euclidean distance -||Q - K||².
#
# After the softmax, it extracts 10 structured "signal" vectors + 2 scalars
# per token — a fixed-size summary of the attention distribution regardless
# of context length. These signals feed into any downstream network (standard
# FFN, MoE router, etc.).
#
# ============================================================================
# Quick Example
# ============================================================================
#
#     import torch
#     from raytention import RayTention
#
#     rt = RayTention(d_model=256)  # works with any d_model
#     embed = torch.nn.Embedding(vocab_size, 256)
#
#     hidden = embed(input_ids)
#     signals = rt(hidden, embed.weight, input_ids)  # [B, T, 10*d_model+2]
#
# ============================================================================
# What Does It Output?
# ============================================================================
#
# Per token:
#
#   • 10 vector signals
#   • 2 scalar statistics
#
# The vectors represent different "views" of the attention distribution:
#
#   1. Centroid
#   2. Primacy
#   3. Sharp Temporal
#   4. Moderate Temporal
#   5. Slow Temporal
#   6. Predecessor
#   7. Top-1 Key
#   8. Top-2 Key
#   9. Recency
#  10. Antitop
#
# Scalars:
#
#   • Spread
#   • Entropy
#
# ============================================================================
# File Layout
# ============================================================================
#
#   _compute_scores()
#       Computes Euclidean attention scores.
#
#   _extract_signals()
#       Converts attention weights into structured signals.
#
#   forward()
#       Calls the two stages and returns the final normalized signal tensor.
#
# Everything below is intentionally written for clarity rather than maximum
# performance. It serves as a reference implementation that optimized CUDA,
# Triton, or other backends should reproduce exactly.
#
# ============================================================================
# Training Memory Note
# ============================================================================
#
# This reference implementation materializes the full [B, T, T] score matrix
# in _compute_scores(). At T=2048 with d=768, that's ~134 MB for batch 8 —
# manageable. At T=32768, it's ~34 GB — not.
#
# The CUDA implementation uses chunked streaming (CHUNK_K=2048) to reduce
# this to O(T·C) by processing keys in 2048-token windows, accumulating
# partial signal state across chunks. This is the same tiling strategy
# FlashAttention uses, applied to signal accumulation.
#
# RayTention does NOT eliminate O(T²) compute — every token must compare
# against every past token. What it eliminates is the O(T) accumulation
# of KV state across tokens during inference. That's where the memory
# win lives: no KV cache, ever.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class RayTention(nn.Module):
    """
    Euclidean Softmax Attention with Structured Signal Extraction.

    Replaces standard Q·K attention with negative squared Euclidean distance
    -||q_i - k_j||² and extracts 10 time-weighted signal channels per token.

    Works with any d_model — the signal dimension is always 10*d_model + 2.
    """

    def __init__(
        self,
        d_model: int = 768,
        max_seq: int = 2048,
        tau: float = 1.0,
        gammas: tuple = (0.99, 0.5, 0.88, 0.993, 0.3),
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.signal_dim = d_model * 10 + 2  # 10 vectors + 2 scalars

        # Learnable parameters
        self.tau = nn.Parameter(torch.tensor(tau))
        self.gammas = nn.Parameter(torch.tensor(gammas, dtype=torch.float32))

    def _compute_scores(
        self,
        queries: torch.Tensor,       # [B, T, D]
        embed_weight: torch.Tensor,  # [V, D]
        input_ids: torch.Tensor,     # [B, T]
    ) -> torch.Tensor:
        """
        Compute negative squared Euclidean distance for all (query, key) pairs
        within the causal window. Returns [B, T, T].
        """
        B, T, D = queries.shape

        # Gather key embeddings: keys[b, t, k] = embed_weight[input_ids[b, k]]
        keys = embed_weight[input_ids]  # [B, T, D]

        # Compute pairwise negative squared Euclidean distances
        # scores[b, i, j] = -||q[b,i] - k[b,j]||²
        q_expand = queries.unsqueeze(2)   # [B, T, 1, D]
        k_expand = keys.unsqueeze(1)      # [B, 1, T, D]
        dist2 = ((q_expand - k_expand) ** 2).sum(dim=-1)  # [B, T, T]
        scores = -dist2  # negative distance: closer = higher score

        # Causal mask: cannot attend to future positions
        causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool().unsqueeze(0), float('-inf'))

        return scores

    def _extract_signals(
        self,
        scores: torch.Tensor,        # [B, T, T]
        keys: torch.Tensor,          # [B, T, D]
        tau: torch.Tensor,           # scalar
        gammas: torch.Tensor,        # [5]
    ) -> torch.Tensor:
        """
        Extract 10 structured signal channels + 2 scalar channels per token
        from the softmax-weighted key embeddings.

        Returns [B, T, signal_dim] where signal_dim = 10*D + 2.
        """
        B, T, _ = scores.shape
        D = keys.shape[-1]  # model dimension (not sequence length)
        signal_dim = 10 * D + 2
        signals = torch.zeros(B, T, signal_dim, device=scores.device, dtype=scores.dtype)

        # Temperature-scaled softmax over keys (causal)
        # weights[b, i, :i+1] = softmax(scores[b, i, :i+1] / tau)
        # Future positions are -inf, so exp(-inf) = 0
        weights = F.softmax(scores / tau.clamp(min=0.01), dim=-1)  # [B, T, T]

        # Unpack gammas
        g_prim, g_sharp, g_mod, g_slow, g_rec = gammas.unbind()

        # Position indices for temporal weighting
        # pos_idx[j] = j (forward index for primacy)
        pos_fwd = torch.arange(T, device=scores.device, dtype=torch.float32)  # [T]
        # pos_bwd[j] = T-1-j (backward index for recency/temporal)
        pos_bwd = (T - 1) - pos_fwd  # [T]

        # Compute normalization sums for temporally-weighted channels
        # For each query i, only keys j ≤ i are non-zero
        eps = 1e-8

        # Primacy: forward-weighted (γ_prim^j), emphasizes early context
        primacy_w = weights * (g_prim ** pos_fwd)  # [B, T, T]
        primacy_sum = primacy_w.sum(dim=-1, keepdim=True).clamp(min=eps)  # [B, T, 1]
        primacy_normed = primacy_w / primacy_sum  # [B, T, T]

        # Sharp temporal: backward-weighted (γ_sharp^(T-1-j)), very recent
        sharp_w = weights * (g_sharp ** pos_bwd)
        sharp_sum = sharp_w.sum(dim=-1, keepdim=True).clamp(min=eps)
        sharp_normed = sharp_w / sharp_sum

        # Moderate temporal: γ_mod^(T-1-j)
        mod_w = weights * (g_mod ** pos_bwd)
        mod_sum = mod_w.sum(dim=-1, keepdim=True).clamp(min=eps)
        mod_normed = mod_w / mod_sum

        # Slow temporal: γ_slow^(T-1-j)
        slow_w = weights * (g_slow ** pos_bwd)
        slow_sum = slow_w.sum(dim=-1, keepdim=True).clamp(min=eps)
        slow_normed = slow_w / slow_sum

        # Recency: γ_rec^(T-1-j)
        rec_w = weights * (g_rec ** pos_bwd)
        rec_sum = rec_w.sum(dim=-1, keepdim=True).clamp(min=eps)
        rec_normed = rec_w / rec_sum

        # --- Channel 0: Centroid (standard softmax attention) ---
        signals[..., 0*D : 1*D] = (weights @ keys)  # [B, T, D]  @ [B, T, D] → [B, T, D]

        # --- Channel 1: Primacy (forward emphasis) ---
        signals[..., 1*D : 2*D] = (primacy_normed @ keys)

        # --- Channel 2: Sharp temporal (very recent) ---
        signals[..., 2*D : 3*D] = (sharp_normed @ keys)

        # --- Channel 3: Moderate temporal ---
        signals[..., 3*D : 4*D] = (mod_normed @ keys)

        # --- Channel 4: Slow temporal (long-range) ---
        signals[..., 4*D : 5*D] = (slow_normed @ keys)

        # --- Channel 5-7: Predecessor, Top-1, Top-2 ---
        # Predecessor: last causal key (j = i)
        # We need the last key for each query. For causal, that's j = i.
        # We can use the identity mask to extract the diagonal.
        for b in range(B):
            for i in range(T):
                # Predecessor = key at position i (the most recent in causal window)
                if i < T:
                    signals[b, i, 5*D : 6*D] = keys[b, i]

                # Top-1 key (highest softmax weight)
                top1_idx = weights[b, i].argmax(dim=-1)
                signals[b, i, 6*D : 7*D] = keys[b, top1_idx]

                # Top-2 key (second highest weight)
                w_i = weights[b, i].clone()
                w_i[top1_idx] = -1  # exclude top-1
                top2_idx = w_i.argmax(dim=-1)
                signals[b, i, 7*D : 8*D] = keys[b, top2_idx]

        # --- Channel 8: Recency (very recent) ---
        signals[..., 8*D : 9*D] = (rec_normed @ keys)

        # --- Channel 9: Antitop (key with smallest non-zero softmax weight) ---
        for b in range(B):
            for i in range(T):
                w_i = weights[b, i].clone()
                # Find smallest non-zero weight (exclude future = 0)
                nonzero_mask = w_i > 1e-10
                if nonzero_mask.any():
                    antitop_idx = w_i[nonzero_mask].argmin()
                    # Map back to original index
                    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
                    signals[b, i, 9*D : 10*D] = keys[b, nonzero_indices[antitop_idx]]

        # --- Scalar channel 0: Spread (weighted avg distance) ---
        # Replace -inf with a large negative number to avoid NaN
        finite_scores = torch.where(scores == float('-inf'), torch.tensor(-1e10, device=scores.device), scores)
        spread = (weights * (-finite_scores * self.tau)).sum(dim=-1)  # [B, T]
        signals[..., 10*D] = spread

        # --- Scalar channel 1: Entropy ---
        entropy = -(weights * weights.clamp(min=1e-10).log()).sum(dim=-1)  # [B, T]
        signals[..., 10*D + 1] = entropy

        # --- L2 normalize the entire signal vector per token ---
        norm = signals.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-10)
        signals = signals / norm

        return signals

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, D] — query embeddings
        embed_weight: torch.Tensor,   # [V, D] — embedding table (acts as keys)
        input_ids: torch.Tensor,      # [B, T] — token IDs for key lookup
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Query embeddings [batch, seq_len, d_model]
            embed_weight: Embedding weight matrix [vocab_size, d_model]
            input_ids: Token IDs for key lookup [batch, seq_len]

        Returns:
            signals: Structured signal tensor [batch, seq_len, 10*d_model + 2]
        """
        B, T, D = hidden_states.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # 1. Euclidean distance scoring
        scores = self._compute_scores(hidden_states, embed_weight, input_ids)

        # 2. Key lookup
        keys = embed_weight[input_ids]  # [B, T, D]

        # 3. Signal extraction
        signals = self._extract_signals(scores, keys, self.tau, self.gammas)

        return signals


# ============================================================================
# Integration Recipes
# ============================================================================
#
# Recipe 1 — Replace standard self-attention in a transformer block:
#
#   class TransformerBlock(nn.Module):
#       def __init__(self, d_model):
#           super().__init__()
#           self.rt = RayTention(d_model=d_model)
#           self.proj = nn.Linear(10 * d_model + 2, d_model)
#           self.ffn = nn.Sequential(
#               nn.Linear(d_model, 4 * d_model), nn.GELU(),
#               nn.Linear(4 * d_model, d_model))
#           self.norm1 = nn.RMSNorm(d_model)
#           self.norm2 = nn.RMSNorm(d_model)
#
#       def forward(self, x, emb_w, ids):
#           s = self.rt(self.norm1(x), emb_w, ids)
#           x = x + self.proj(s)
#           x = x + self.ffn(self.norm2(x))
#           return x
#
# Recipe 2 — Feed signals into an MoE router:
#
#   class MoEBlock(nn.Module):
#       def __init__(self, d_model, n_experts):
#           super().__init__()
#           self.rt = RayTention(d_model=d_model)
#           self.router = nn.Linear(10*d_model + 2 + d_model, n_experts)
#           self.experts = nn.ModuleList([...])
#
#       def forward(self, x, emb_w, ids):
#           s = self.rt(x, emb_w, ids)
#           logits = self.router(torch.cat([s, x], dim=-1))
#           # ... top-k gating, expert dispatch ...
#           return x
#
# Recipe 3 — Anneal temperature during training:
#
#   rt.tau.data.fill_(1.0 / (1.0 + step / 5000.0))


# ═══════════════════════════════════════════════════════════════
# Minimal Working Example (run with: python reference_raytention.py)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Setup
    B, T, D, V = 2, 64, 256, 10000  # batch, seq, model_dim, vocab

    rt = RayTention(d_model=D, max_seq=T)
    embed = nn.Embedding(V, D)

    # Dummy data
    input_ids = torch.randint(0, V, (B, T))
    hidden = embed(input_ids)  # In practice, this comes from previous layers

    # Forward pass
    signals = rt(hidden, embed.weight, input_ids)

    print(f"Input:  {hidden.shape}")              # [2, 64, 256]
    print(f"Output: {signals.shape}")              # [2, 64, 2562] = [B, T, 10*D+2]
    print(f"Signal breakdown:")
    print(f"  Centroid:        signals[..., {0*D}:{1*D}]")
    print(f"  Primacy:         signals[..., {1*D}:{2*D}]")
    print(f"  Sharp temporal:  signals[..., {2*D}:{3*D}]")
    print(f"  Moderate temp:   signals[..., {3*D}:{4*D}]")
    print(f"  Slow temporal:   signals[..., {4*D}:{5*D}]")
    print(f"  Predecessor:     signals[..., {5*D}:{6*D}]")
    print(f"  Top-1 key:       signals[..., {6*D}:{7*D}]")
    print(f"  Top-2 key:       signals[..., {7*D}:{8*D}]")
    print(f"  Recency:         signals[..., {8*D}:{9*D}]")
    print(f"  Antitop:         signals[..., {9*D}:{10*D}]")
    print(f"  Spread:          signals[..., {10*D}]")
    print(f"  Entropy:         signals[..., {10*D+1}]")
    print(f"  L2 norm per token: {signals.norm(dim=-1).mean():.4f} (should be ~1.0)")
    print(f"\nLearnable params: tau={rt.tau.item():.4f}, gammas={rt.gammas.tolist()}")
