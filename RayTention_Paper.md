# RayTention: Euclidean Softmax Attention with Structured Signal Extraction

## Reference Implementation

A pure PyTorch reference is provided at `reference_raytention.py`. It mirrors the CUDA kernels exactly and works with any model dimension. Usage:

```python
rt = RayTention(d_model=256)  # works with any d_model
signals = rt(hidden_states, embed_weight, input_ids)  # [B, T, 10*d_model + 2]
```

## Executive Summary

RayTention is a causal self-attention mechanism that replaces the standard $QK^\top$ dot product with negative squared Euclidean distance $-\|q_i - k_j\|^2_2$ as the compatibility score. It applies a softmax over these distance scores (with learned temperature $\tau$) and extracts **10 structured signal channels + 2 scalar channels** per query position.

The key innovation is NOT in replacing dot-product with Euclidean distance (that's a minor change). The innovation is **structured signal extraction**: instead of producing one attended vector, RayTention produces 10 time-aware signal channels (centroid, primacy, sharp/moderate/slow temporal, predecessor, top-1, top-2, recency, antitop) plus spread and entropy scalars. These signals form a fixed-size representation of the attention context — $10d + 2$ floats regardless of sequence length — making the KV cache unnecessary.

---

## 1. Forward Pass

### 1.1 Scoring: Euclidean Distance

For each query position $i$ and each causal key position $j \leq i$:

$$\text{score}(i, j) = -\|q_i - k_j\|^2_2 = -\sum_{d=1}^{D} (q_{i,d} - k_{j,d})^2$$

Keys are looked up from a weight-tied embedding table: $k_j = E[\text{input\_ids}[j]]$. Causal masking sets $\text{score}(i, j) = -\infty$ for $j > i$.

### 1.2 Softmax with Learned Temperature

$$w_{ij} = \frac{\exp(\text{score}(i, j) / \tau)}{\sum_{k=0}^{i} \exp(\text{score}(i, k) / \tau)}$$

Temperature $\tau$ is a learnable parameter (default 1.0, typically annealed during training).

### 1.3 Signal Extraction: 10 Vector Channels + 2 Scalars

The signal dimension is $S = 10d + 2$, where $d$ is the model dimension.

| Channel | Offset | Formula | Meaning |
|---|---|---|---|
| Centroid | $0 \cdot d$ | $\sum_j w_{ij} \cdot k_j$ | Standard softmax attention |
| Primacy | $1 \cdot d$ | $\sum_j w_{ij} \cdot \gamma_{\text{prim}}^j \cdot k_j \;/\; \text{norm}$ | Forward emphasis ($\gamma=0.99$) |
| Sharp temporal | $2 \cdot d$ | $\sum_j w_{ij} \cdot \gamma_{\text{sharp}}^{T-1-j} \cdot k_j \;/\; \text{norm}$ | Very recent context ($\gamma=0.5$) |
| Moderate temporal | $3 \cdot d$ | $\sum_j w_{ij} \cdot \gamma_{\text{mod}}^{T-1-j} \cdot k_j \;/\; \text{norm}$ | Mid-term context ($\gamma=0.88$) |
| Slow temporal | $4 \cdot d$ | $\sum_j w_{ij} \cdot \gamma_{\text{slow}}^{T-1-j} \cdot k_j \;/\; \text{norm}$ | Long-range context ($\gamma=0.993$) |
| Predecessor | $5 \cdot d$ | $k_i$ | The immediately preceding key |
| Top-1 key | $6 \cdot d$ | $k_{\text{argmax } w_{i\cdot}}$ | Key with highest attention weight |
| Top-2 key | $7 \cdot d$ | $k_{\text{argmax}_2 w_{i\cdot}}$ | Second-highest weight key |
| Recency | $8 \cdot d$ | $\sum_j w_{ij} \cdot \gamma_{\text{rec}}^{T-1-j} \cdot k_j \;/\; \text{norm}$ | Very recent ($\gamma=0.3$) |
| Antitop | $9 \cdot d$ | $k_{\text{argmin}_{\text{nonzero}} w_{i\cdot}}$ | Key furthest from attention |
| **Spread** | $10d$ | $\sum_j w_{ij} \cdot (-\text{score}(i,j) \cdot \tau)$ | Weighted average distance (scalar) |
| **Entropy** | $10d + 1$ | $-\sum_j w_{ij} \cdot \log(w_{ij})$ | Shannon entropy of softmax (scalar) |

The five gammas $(0.99, 0.5, 0.88, 0.993, 0.3)$ are learnable parameters. Each temporally-weighted channel is independently normalized.

### 1.4 L2 Normalization

The entire $S$-dimensional signal vector is L2-normalized per token before being passed downstream:

$$\text{signals}[i] = \frac{\text{signals}[i]}{\|\text{signals}[i]\|_2}$$

---

## 2. Signal Channels — Design Rationale

The 10 channels capture different aspects of the attention distribution:

- **Centroid** is the standard attention output — what a normal transformer would produce
- **Primacy, Sharp, Moderate, Slow, Recency** are temporally-weighted variants that emphasize different timescales (from very recent to very distant). The exponential weighting $\gamma^{\text{pos}}$ means recent positions contribute more in recency channels and distant positions contribute more in primacy
- **Predecessor, Top-1, Top-2, Antitop** are direct key copies — they give downstream networks raw embedding vectors rather than blended averages
- **Spread and Entropy** are global statistics about the attention distribution — how dispersed the attention is and how confident the model is

These signals feed into a small projection network (the "AttnFFN") that produces the layer output, exactly like standard attention — but without ever storing or re-reading a full KV cache.

---

## 3. CUDA Implementation

### 3.1 Non-Chunked Forward (sequences $\leq$ 2048)

Two CUDA kernels:

**Kernel 1 — Euclidean Distance Scoring**
- Grid: $(BT \times BT)$, Block: 128 threads — one block per (query, key) pair
- Computes $-\sum_d (q_d - k_d)^2$ with parallel reduction
- Causal mask: writes $-10^{30}$ for $k > q$
- Keys read from BF16 embedding table (fits in L2 cache)

**Kernel 2 — Signal Extraction**
- Grid: $(BT)$, Block: 256 threads — one block per query position
- Online softmax with max-reduction for numerical stability
- All 10 vector channels computed in parallel across threads
- L2 normalization fused in-place

### 3.2 Chunked Streaming (sequences $>$ 2048)

For long sequences, the forward switches to chunked mode with `CHUNK_K = 2048`:

```
accum_init(accumulator)
for each chunk of 2048 keys:
    compute distances for ALL queries vs this chunk
    accum_chunk(accumulator, scores, chunk_offset)
signals = accum_finalize(accumulator)
```

Each chunk computes pairwise distances between all $BT$ queries and the $k$ keys in that chunk window. The accumulator merges contributions across chunks using the gamma-weighted formulas from Section 1.3.

**Why chunking is necessary:** Without chunking, the forward pass materializes a full $T \times T$ score matrix. At $T{=}2048$ with $d{=}768$ and batch 8, that's ~134 MB — manageable. At $T{=}32768$, it's ~34 GB — not. Chunking with $C{=}2048$ reduces training memory from $O(T^2)$ to $O(T \cdot C)$, the same tiling strategy FlashAttention uses. RayTention does not eliminate the $O(T^2)$ compute (nothing can — every token must compare against every past token), but it eliminates the $O(T)$ *accumulation* of KV state across tokens during inference.

### 3.3 OptiX Backend (optional)

When compiled with OptiX 7.5 and $BT \geq 4096$, distance scoring uses NVIDIA GPU ray tracing. A GAS (Geometry Acceleration Structure) is built with one triangle per context position. Each query launches a ray that finds the closest keys via hardware-accelerated BVH traversal, potentially offloading the $O(T^2)$ distance computation to RT cores.

Signal extraction (Section 1.3) is identical regardless of score computation method.

### 3.4 Backward Pass

A single fused kernel backpropagates gradients through the softmax, distance computation, and key lookups:

1. **Query gradient** ($\partial L / \partial q_i$): Flows back into the hidden state
2. **Key embedding gradient** ($\partial L / \partial E$): Scattered atomically into the embedding matrix
3. **Gamma gradient** ($\partial L / \partial \gamma$): 5 scalars accumulated across all tokens
4. **Temperature gradient** ($\partial L / \partial \tau$): Single scalar

### 3.5 BF16 Embedding Table

The embedding matrix is maintained in BF16 as a shadow copy. Before each forward pass, FP32 weights are cast to BF16. The BF16 buffer ($V \times d \times 2$ bytes) fits in L2 cache, enabling fast key lookups during distance scoring.

---

## 4. Implementation Status

Measured on RTX 5080 (16.6 GB), CUDA 13.3, Rust + cuBLAS.

| Metric | Value |
|---|---|
| Model size | 268.5M params (DM=768, depth=4) |
| Vocabulary | 49,152 tokens |
| BF16 embedding | ~75 MB (L2-cache resident) |
| Signal dimension | 7,682 floats per token |
| Chunked streaming | Implemented (CHUNK_K=2048, $O(T \cdot C)$ training memory) |
| KV cache required | None (signals replace it) |

> Stable training throughput benchmarks are in progress. The chunked streaming kernels, OptiX backend, and fused backward pass are all implemented and functional.

---

## 5. Comparison to Standard Attention

| Feature | Standard Transformer | RayTention |
|---|---|---|
| Compatibility function | $q \cdot k$ | $-\|q - k\|^2$ |
| Normalization | Softmax | Softmax (same) |
| Output per token | 1 vector ($d$-dim) | 10 vectors + 2 scalars ($10d + 2$-dim) |
| Key storage | FP16/BF16 K,V per layer | BF16 embedding table (weight-tied) |
| Causal masking | Masked fill $-\infty$ | Same: $-10^{30}$ for $k > q$ |
| Temperature | $1/\sqrt{d_k}$ | Learned $\tau$ |
| Memory (T=2048) | $O(T^2)$ attention | $O(T^2)$ scores |
| Memory (inference) | $O(T)$ KV cache | $O(1)$ signal buffer |
| Chunked mode | FlashAttention-style tiling | Accumulator-based streaming ($O(T \cdot C)$) |
| Training memory | $O(T^2)$ attention matrix | $O(T^2)$ score matrix → $O(T \cdot C)$ with chunking |

### Inference Memory Scaling

KV cache formula: $2 \times L \times H_{kv} \times d_{head} \times T \times 2\text{ bytes}$ (bf16). Model: $d_{model}{=}4096$, $L{=}32$, $d_{head}{=}128$.

| Context | MHA ($H_{kv}{=}32$) | GQA-8 (Llama 3) | GQA-4 | MQA ($H_{kv}{=}1$) | MLA (latent=576) | Sliding Window (W=4K) | **RayTention** |
|---|---|---|---|---|---|---|---|
| 16K | 8.6 GB | 2.1 GB | 1.1 GB | 0.3 GB | 1.2 GB | 0.5 GB | **0** |
| 131K | 69 GB | 17 GB | 8.6 GB | 2.1 GB | 9.7 GB | 0.5 GB | **0** |
| 262K | 137 GB | 34 GB | 17 GB | 4.3 GB | 19 GB | 0.5 GB | **0** |
| 1M | 524 GB | 131 GB | 66 GB | 16 GB | 74 GB | 0.5 GB | **0** |

- **RayTention** has zero KV cache. The signal vector ($10d + 2$ floats) is an activation, not stored state — it's consumed and discarded.
- **Sliding Window** is also $O(1)$ but blind to tokens outside the window. RayTention preserves full-context information in the signal summary.
- **Flash Attention** is omitted — it computes exact standard attention, same KV cache as MHA. It's a speed optimization, not a memory one.

---

## 6. Where RayTention Wins

| Scenario | Why |
|---|---|
| Long-context inference | No KV cache — memory is $O(1)$, not $O(T)$ |
| Multi-tenant serving | No per-user KV cache — memory scales with model weights, not context length |
| Edge deployment | Runs on hardware where KV cache would exceed memory |
| Retrieval / RAG | Compute signals once, reuse across queries |
| Interpretability | Centroid, spread, entropy are human-readable features |

## Where Standard Wins

| Scenario | Why |
|---|---|
| Short contexts ($<$1K) | KV cache negligible; optimized matmul faster |
| Training throughput | FlashAttention + fused kernels highly tuned |
| Ecosystem maturity | PyTorch/HF integration, proven at scale |

---

## 7. Future Work

- **RT Core acceleration** — L2 distance scoring maps naturally to ray tracing hardware
- **Incremental signals** — Update centroid, spread, top-k in $O(1)$ per new token
- **Signal reuse across layers** — Compute signals once, feed to deeper layers
- **Scaling laws** — Quality at $d = 512, 768, 4096, 8192$
- **Downstream perplexity** — Full FineWeb training, standard benchmarks

---

## Citation

```bibtex
@software{raytention2026,
  title     = {RayTention: Zero-KV-Cache Attention via Geometric Signal Extraction},
  year      = {2026},
  url       = {https://github.com/NohWai-Software/RayTention}
}
```

## License

AGPL-3.0 — see [LICENSE](https://github.com/NohWai-Software/RayTention/blob/main/LICENSE)

---

*Patent Pending — U.S. Patent Application No. 64/102,801. All rights reserved by NohWai Software.*
