# Context Length Impact on Decode Pareto Front

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, decode, context length, KV cache, TPOT, throughput, Pareto front, B_max, KV-paging capacity, long-context

---

# Table of Contents

- [1. Where S Enters the Decode Equations](#1-where-s-enters-the-decode-equations)
- [2. The Three Effects of Growing S](#2-the-three-effects-of-growing-s)
  - [2.1 B_max Shrinks Like 1/S (KV Capacity)](#21-b_max-shrinks-like-1s-kv-capacity)
  - [2.2 T_kv Grows Linearly With S (Memory-Bound Slope)](#22-t_kv-grows-linearly-with-s-memory-bound-slope)
  - [2.3 Attention FLOPs Grow Linearly With S (Compute Bound)](#23-attention-flops-grow-linearly-with-s-compute-bound)
- [3. Why TPOT@B=1 Is Nearly S-Invariant](#3-why-tpotb1-is-nearly-s-invariant)
- [4. Why the Throughput Corner Collapses](#4-why-the-throughput-corner-collapses)
- [5. B* Under Long Context](#5-b-under-long-context)
- [6. Empirical: GPT-1.8T MoE on B200, S ∈ {4k, 8k, 12k}](#6-empirical-gpt-18t-moe-on-b200-s--4k-8k-12k)
- [7. Pareto Front Shape Change](#7-pareto-front-shape-change)
- [8. Mitigation Levers](#8-mitigation-levers)
- [9. Key Takeaways](#9-key-takeaways)

---

# 1. Where S Enters the Decode Equations

During autoregressive decode at context length S, each new token attends to all S−1 previous tokens. The sequence length S enters the per-step cost in three distinct places, which we call the **three S-axes**:

| Quantity | Formula (per device) | Scaling with S |
|----------|----------------------|----------------|
| KV memory footprint (per sequence) | M_kv = (L/PP) · 2·S·H_kv·b / (TP·SP) | **linear** |
| KV traffic per token (per sequence) | T_kv = (L/PP) · 2·S·H_kv·b / (TP·SP) | **linear** |
| Attention FLOPs per token | F_attn = (L/PP) · 4·S·H / (TP·SP) | **linear** |
| Weight memory M_θ | (L/PP) · P_layer · b / (TP·EP) | **S-independent** |
| Weight traffic T_θ | same as M_θ | **S-independent** |

Weights don't care about S. Everything attention-related scales linearly with S. This asymmetry drives all of the long-context effects below.

---

# 2. The Three Effects of Growing S

## 2.1 B_max Shrinks Like 1/S (KV Capacity)

Under KV paging with block size BLK_KV, the maximum number of concurrent sequences B_max is determined by available HBM after weights and activations:

$$
B_\text{max} \;=\; \Big\lfloor \frac{M_\text{HBM} - M_\theta - M_\text{act} - M_\text{sys}}{\lceil S / \text{BLK}_{KV} \rceil \cdot M_\text{block} \cdot \phi} \Big\rfloor
$$

Since M_kv per sequence grows linearly with S, **B_max scales as ~1/S** once KV dominates the residual HBM. Doubling S roughly halves B_max. This is the dominant effect on the throughput corner of the Pareto front.

## 2.2 T_kv Grows Linearly With S (Memory-Bound Slope)

In batched decode, memory traffic is split between shared and per-sequence terms:

$$
T_\text{step} \;=\; T_\theta + B \cdot T_\text{kv}(S)
$$

For small B, T_θ dominates. But the per-sequence term T_kv grows with S, so the "slope" of memory-bound TPOT versus B steepens as S grows:

$$
\text{TPOT}_\text{mem}(B, S) \;=\; \frac{T_\theta + B \cdot T_\text{kv}(S)}{B \cdot B_m}
$$

As S grows, the B-value at which T_kv starts to matter shrinks. Long context turns batching into a steeper tradeoff.

## 2.3 Attention FLOPs Grow Linearly With S (Compute Bound)

Per-token attention FLOPs scale linearly with S (each new token does dot products with all S past KV entries). For transformers dominated by dense matmul FLOPs (MoE layers or dense FFN), attention is a small fraction at short S but **grows as S grows**. Rough proportions for GPT-1.8T MoE (H=20480, I_moe=14336, k=2):

| S | Attention FLOPs / (MoE + attention) FLOPs |
|---|------------------------------------------|
| 4k | ~9% |
| 8k | ~17% |
| 12k | ~24% |

So at S=12k, roughly a quarter of the compute-bound throughput floor comes from attention — and that fraction **does not amortize over B** (it scales with B, like KV reads). This compounds with the B_max shrinkage to collapse the throughput corner.

---

# 3. Why TPOT@B=1 Is Nearly S-Invariant

At B=1, the decode step is almost always **memory-bound** and weight-dominated:

$$
\text{TPOT}(B{=}1) \approx \frac{T_\theta + T_\text{kv}(S)}{B_m}
$$

Plugging numbers (GPT-1.8T MoE, FP4, TP=8, EP=8):

- T_θ per device ≈ 1.5 GB (weights loaded once per step)
- T_kv at S=4k per device ≈ 0.03 GB (much smaller than T_θ)
- T_kv at S=12k per device ≈ 0.09 GB

T_kv(S=12k) is still **3% of T_θ**, so the single-token TPOT barely moves. Our sweep confirms this:

| S | TPOT@B=1 (FP4, 64-GPU) | Change |
|---|------------------------|--------|
| 4k | 4.65 ms | baseline |
| 8k | 4.66 ms | +0.2% |
| 12k | 4.67 ms | +0.4% |

**The interactivity corner of the Pareto front barely shifts with S** — provided you can still fit one sequence. Long context does not hurt single-user latency (it only hurts how many concurrent users you can serve).

---

# 4. Why the Throughput Corner Collapses

At large B (throughput corner), TPOT transitions to compute-bound:

$$
\text{TPOT}_\text{compute}(B, S) \;=\; \frac{B \cdot (F_\text{dense} + F_\text{attn}(S))}{R_\text{gpu}}
$$

Per-GPU throughput is B / (TPOT · num_gpus). The practical throughput corner is not at some abstract "B → ∞" but at **B = B_max**, because the KV paging allocator runs out of blocks. Two effects compound:

1. **B_max shrinks** ~1/S, so you run fewer concurrent sequences per device.
2. **F_attn grows** ~S, so even within the compute-bound regime, per-token work increases.

For GPT-1.8T MoE FP4 at TP=8, EP=8, 64 GPUs:

| S | B_max | TPS/GPU @ B_max | Change |
|---|-------|-----------------|--------|
| 4k | 1079 | 541.8 | baseline |
| 8k | 540 | 348.9 | −36% |
| 12k | 360 | 233.9 | −57% |

Throughput drops by **~1/√S** initially, then faster as attention grows in importance. The effect is stronger than a naive 1/S prediction because B_max shrinks and the per-token cost grows.

---

# 5. B* Under Long Context

B* is the batch size where memory-bound crosses compute-bound:

$$
B^* \;=\; \frac{T_\theta \cdot R_\text{gpu}}{F_\text{token} \cdot B_m - T_\text{kv}(S) \cdot R_\text{gpu}}
$$

As S grows, the denominator shrinks (T_kv increases). If T_kv · R exceeds F_token · B_m, B* → ∞ (the "KV wall"). Qualitatively:

| S | Denominator behavior | B* behavior |
|---|----------------------|-------------|
| Small | Dense FLOPs dominate; denominator ≈ F·B_m | B* small, easy to reach |
| Medium | T_kv erodes the denominator | B* grows |
| Large | T_kv · R > F · B_m → denominator negative | **B* = ∞**, permanently memory-bound |

For our B200 configs, B* is already ∞ across all the S values tested (the MoE compute is small relative to HBM bandwidth). Longer S doesn't change the regime classification — only the magnitude of the throughput available.

---

# 6. Empirical: GPT-1.8T MoE on B200, S ∈ {4k, 8k, 12k}

Measured via `sandbox/cluster_size_pareto_sweep.py` at the 64-GPU TP=8, EP=8 baseline. Precision rows share the same cluster; the only change is S.

### FP4 (bytes_per_param=0.5, peak_flops=9000 TF):

| S | TPOT@B=1 | B_max | TPS/GPU @ B_max |
|---|----------|-------|-----------------|
| 4k | 4.65 ms | 1079 | 541.8 |
| 8k | 4.66 ms | 540 | 348.9 |
| 12k | 4.67 ms | 360 | 233.9 |

### FP8 (bytes_per_param=1.0, peak_flops=4500 TF):

| S | TPOT@B=1 | B_max | TPS/GPU @ B_max |
|---|----------|-------|-----------------|
| 4k | 5.95 ms | 474 | 253.9 |
| 8k | 5.97 ms | 237 | 153.8 |
| 12k | 5.99 ms | 158 | 103.1 |

Both precisions show identical **fractional** behavior: B_max halves per doubling of S, throughput drops proportionally, TPOT@B=1 unchanged. FP8 just starts at a worse position and follows the same curve.

### Recovery via TP scaling

Doubling TP halves KV footprint per device — equivalent to halving S for KV capacity purposes. So TP=16 at S=8k has the same B_max as TP=8 at S=4k:

| Config | TP | S | B_max |
|--------|----|---|-------|
| Baseline | 8 | 4k | 1079 |
| "Recovered" | 16 | 8k | 1145 |

**TP doubling is the cleanest counter-move for context doubling**, provided you have the devices and the TP all-reduce cost stays hidden under the overlap budget.

---

# 7. Pareto Front Shape Change

Putting it all together, the Pareto curve (TPS/GPU on x-axis, 1/TPOT on y-axis) transforms as S grows:

- **Interactivity ceiling** (top of curve, at B=1): roughly fixed — weight-dominated.
- **Throughput ceiling** (right of curve, at B=B_max): compresses by >1/S.
- **Curve length**: shortens as B_max shrinks — fewer valid B points, less dynamic range for the operator to choose.
- **Curve slope** (memory-to-compute transition): steepens — the KV term in memory traffic ramps up faster with B.

Visually, the Pareto front **keeps its head but loses its tail**. The operator's scheduling flexibility degrades along with raw throughput.

---

# 8. Mitigation Levers

Each lever attacks a different part of the KV problem:

| Lever | What it does | S-shift equivalent | Cost |
|-------|--------------|-------------------|------|
| **TP ↑ (or SP ↑)** | Shards KV heads/seq dim across more devices | TP doubling ≡ S halving for KV | Adds TP all-reduce cost; needs H_kv ≥ TP |
| **KV quantization** (FP4/INT4) | Halves or quarters `b` in M_kv and T_kv | FP4 KV ≡ S halving | Accuracy risk at long context |
| **GQA / MQA** | Lower H_kv (fewer KV heads) | H_kv ↓ 4× ≡ S ↓ 4× | Architectural choice (baked into model) |
| **Sliding window / sparse attn** | Attention only attends to last W ≪ S tokens | Caps effective S at W | Accuracy tradeoff; breaks long-range |
| **Chunked prefill** | Spreads prefill over multiple steps | Doesn't reduce decode cost | Helps TTFT, not TPOT or B_max |
| **KV offload / disagg** | Move inactive KV to DDR or remote | Extends effective HBM at high α | Startup latency on reactivation |
| **HBM capacity ↑** | More sequences fit | Linear B_max increase | Hardware |

Ranked by effect-per-effort in current practice: GQA > KV quantization > TP scaling > sliding window > offload.

---

# 9. Key Takeaways

1. **Interactivity is S-insensitive; throughput is S-sensitive.** TPOT@B=1 barely moves with context, but per-GPU throughput at B_max drops faster than 1/S.
2. **The Pareto front contracts from the throughput side.** The curve keeps its interactivity ceiling but loses length along the throughput axis.
3. **TP doubling ≡ S halving.** A clean substitution when you have the devices and your model's H_kv supports larger TP. Beyond H_kv, SP takes over.
4. **FP4 vs FP8 behavior is identical in shape, only in magnitude.** Precision downgrade moves the baseline; S growth is orthogonal to it.
5. **B* mostly doesn't move**, but throughput at B_max does. The "KV wall" is already crossed on modern MoE/B200 configs, so long context manifests as a smaller B_max, not a regime change.
6. **Architecture choices (GQA, sliding window) dominate system choices.** A model with H_kv=16 vs H_kv=128 is an 8× effective-S advantage that no amount of TP scaling can recover.
