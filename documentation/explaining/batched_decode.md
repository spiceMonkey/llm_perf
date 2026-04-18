# Batched Decoding: Why It Matters and How It Works

**Author:** Yue Lu  
**Date:** November 2025  

**Keywords:**  
LLM inference, batched decoding, TPOT, throughput, latency, roofline, memory-bound, compute-bound, KV cache, weight amortization, operational intensity

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Motivation: The Single-Token Decode Problem](#1-motivation-the-single-token-decode-problem)
- [2. What Changes When We Batch?](#2-what-changes-when-we-batch)
  - [2.1 Weights Are Shared, KV Reads Are Not](#21-weights-are-shared-kv-reads-are-not)
  - [2.2 Batch Size B vs. Context Length S](#22-batch-size-b-vs-context-length-s)
- [3. Per-Step Cost Breakdown at Batch Size B](#3-per-step-cost-breakdown-at-batch-size-b)
  - [3.1 FLOPs Per Step](#31-flops-per-step)
  - [3.2 Memory Traffic Per Step](#32-memory-traffic-per-step)
  - [3.3 Operational Intensity OI(B)](#33-operational-intensity-oib)
- [4. The Crossover Batch Size B*](#4-the-crossover-batch-size-b)
  - [4.1 Derivation from the Roofline Ridge Point](#41-derivation-from-the-roofline-ridge-point)
  - [4.2 Weight-Dominated Approximation](#42-weight-dominated-approximation)
  - [4.3 Validity Domain](#43-validity-domain)
- [5. Batched TPOT: Three Regimes](#5-batched-tpot-three-regimes)
  - [5.1 Memory-Bound Regime (B << B*)](#51-memory-bound-regime-b--b)
  - [5.2 Compute-Bound Regime (B >> B*)](#52-compute-bound-regime-b--b)
  - [5.3 Regime Summary Table](#53-regime-summary-table)
- [6. Throughput-Latency Pareto Curve](#6-throughput-latency-pareto-curve)
  - [6.1 Three Zones](#61-three-zones)
  - [6.2 Throughput–Interactivity Pareto Axes](#62-throughputinteractivity-pareto-axes)
- [7. Worked Example: LLaMA-3 70B on 8 H100 GPUs](#7-worked-example-llama-3-70b-on-8-h100-gpus)
- [8. FAQ: Common Questions About Batched Decode](#8-faq-common-questions-about-batched-decode)

---

<div style="page-break-before: always;"></div>

# 1. Motivation: The Single-Token Decode Problem

During autoregressive decoding, each generated token requires a full forward pass through every layer of the model. At B=1 (a single user request), this means:

- **All model weights** are loaded from HBM to the compute units for **every single token**.
- **One matrix-vector multiply** per weight matrix (gemv, not gemm).
- **One KV cache read** per attention layer.

The operational intensity (FLOPs per byte of HBM traffic) at B=1 is approximately 2/b (where b is bytes per parameter), which for bf16 (b=2) gives OI of about 1. Modern GPUs have ridge points of 100--300 FLOPs/byte, so **single-token decode utilizes less than 1% of the GPU's compute capacity**. The GPU spends almost all its time waiting for memory.

This is the fundamental problem that batching solves.

---

# 2. What Changes When We Batch?

## 2.1 Weights Are Shared, KV Reads Are Not

The key asymmetry in batched decoding:

| Resource | B = 1 | B sequences |
|----------|-------|-------------|
| Weight loads per step | 1 full read of all weights | **Same** 1 full read (shared) |
| FLOPs per step | F_token | B x F_token |
| KV cache reads per step | 1 user's KV history | B independent KV reads |

**Weights are loaded once and reused across all B tokens.** This is the source of all batching efficiency. Each weight matrix is fetched from HBM exactly once per step, but now it performs B matrix-vector multiplications instead of one. From the perspective of each individual sequence, the effective cost of loading weights is amortized by a factor of B.

**KV cache reads scale linearly with B.** Each sequence has its own prompt history stored in KV cache. User A's cached key-value pairs are completely unrelated to User B's — there is no sharing. The GPU must read each user's KV cache independently.

This creates a simple cost structure:

> **T_step = T_theta (shared, loaded once) + B x T_KV (per-user, loaded B times)**

At small B, the shared weight term dominates and batching is essentially "free" — more users, same memory traffic, more FLOPs. At large B, the per-user KV term dominates and the benefit of weight sharing saturates.

## 2.2 Batch Size B vs. Context Length S

A common misconception: "batching" and "longer context" are not the same thing, even though both increase the amount of KV cache in HBM. They correspond to different axes of the problem:

| Dimension | What it means | KV memory effect | FLOPs effect |
|-----------|--------------|-----------------|--------------|
| Batch size B | Number of **independent users** decoded in the same step | B separate KV histories | B x F_token (independent tokens) |
| Context length S | Length of **one user's** prompt + generated history | Larger per-user KV cache (grows with S) | More KV bytes to read per token; attention FLOPs grow with S |

**Increasing B** adds more independent sequences — more weight reuse, more total FLOPs, separate KV caches. This is the multi-user serving scenario.

**Increasing S** makes each individual sequence's KV cache larger — more memory consumed per user, more attention FLOPs and KV bytes per token, but no additional weight reuse.

In matrix form, the KV cache memory is B x S x (per-token KV size). B and S both contribute to total KV footprint, but their performance implications are different: B amortizes weight traffic (good for throughput), while S increases per-token KV traffic (shifts the memory-bound / compute-bound tradeoff).

---

# 3. Per-Step Cost Breakdown at Batch Size B

## 3.1 FLOPs Per Step

Each of the B tokens independently performs the same computation through every layer. Total FLOPs per decode step:

> **F_step = B x F_token**

where F_token is the per-device per-token FLOPs defined in [../modeling/decode.md §3](../modeling/decode.md#3-compute-flops-per-token).

## 3.2 Memory Traffic Per Step

> **T_step = T_theta + B x T_KV**

The two terms are: weight traffic (loaded once, shared) and KV cache traffic (per-user, loaded B times). The critical point: T_theta does **not** multiply by B, while T_KV does.

## 3.3 Operational Intensity OI(B)

$$
OI(B) = \frac{B \times F_{token}}{T_{\theta} + B \times T_{KV}}
$$

**Two limiting cases:**

- **Small B** (weight-dominated): OI(B) grows linearly with B, since the fixed weight traffic T_theta dominates the denominator.
- **Large B** (KV-dominated): OI(B) saturates at F_token / T_KV, since per-user KV traffic overwhelms the shared weight term.

---

# 4. The Crossover Batch Size B*

## 4.1 Derivation from the Roofline Ridge Point

The roofline ridge point is R_GPU / BW_mem (peak FLOPs / effective memory bandwidth). Setting OI(B*) equal to the ridge point and solving for B*:

$$
B^{*} = \frac{T_{\theta} \times R_{GPU}}{F_{token} \times BW_{mem} - T_{KV} \times R_{GPU}}
$$

This is the batch size at which the system transitions from memory-bound to compute-bound. Below B\*, the GPU is starved for data; above B\*, the GPU is saturated with compute.

## 4.2 Weight-Dominated Approximation

When the context length S is short and KV traffic is small relative to weight traffic, the formula simplifies:

$$
B^{*} \approx \frac{T_{\theta}}{F_{token}} \times \frac{R_{GPU}}{BW_{mem}}
$$

The first factor (T_theta / F_token) is the inverse single-token OI (bytes per FLOP). The second factor (R_GPU / BW_mem) is the ridge point. Their product is the batch size needed to bridge the gap between single-token OI and the ridge point.

## 4.3 Validity Domain

B\* exists (is finite and positive) only when:

> **F_token x BW_mem > T_KV x R_GPU**

When this condition is violated — typically at very long context lengths where each token's KV read is so large that per-token traffic alone exceeds the roofline threshold — the system remains memory-bound for **all** batch sizes. In this regime, B\* should be interpreted as infinity: no amount of batching can push the OI past the ridge point.

---

# 5. Batched TPOT: Three Regimes

The per-step local execution time under the roofline model is:

$$
t_{local}(B) = \max\left(\frac{B \times F_{token}}{R_{GPU}}, \frac{T_{\theta} + B \times T_{KV}}{BW_{mem}}\right)
$$

Including communication overlap:

$$
t_{\text{step,user}}(B) = t_{local}(B) + \max(0, t_{comm} - \rho \cdot t_{local}(B))
$$

The **Time Per Output Token** experienced by each individual sequence:

$$
TPOT(B) = \frac{t_{\text{step,user}}(B)}{B}
$$

## 5.1 Memory-Bound Regime (B << B*)

Weight traffic dominates the denominator. Step time is approximately constant (bounded by weight reads):

$$
t_{local}(B) \approx \frac{T_{\theta}}{BW_{mem}} \quad \Rightarrow \quad TPOT(B) \approx \frac{T_{\theta}}{B \times BW_{mem}}
$$

**TPOT drops as ~1/B** — each additional user in the batch is essentially free from a latency perspective. The GPU was already loading all the weights for one user; now those same weights serve B users. This is the "free lunch" regime.

## 5.2 Compute-Bound Regime (B >> B*)

FLOPs dominate. Step time grows linearly with B:

$$
t_{local}(B) \approx \frac{B \times F_{token}}{R_{GPU}} \quad \Rightarrow \quad TPOT(B) \approx \frac{F_{token}}{R_{GPU}}
$$

**TPOT flattens** to a constant determined by the compute rate. Adding more sequences no longer improves per-sequence latency — the GPU is already fully utilized.

## 5.3 Regime Summary Table

| Regime | Condition | Step time | TPOT | Throughput |
|--------|-----------|-----------|------|------------|
| Memory-bound | B << B* | Flat (weight-read time dominates) | Drops as 1/B | Grows linearly with B |
| Crossover | B = B* | At the ridge point | Minimum for given throughput | Optimal GPU utilization |
| Compute-bound | B >> B* | Grows linearly with B | Flat (set by compute rate) | Approaches ceiling R/F |

---

# 6. Throughput-Latency Pareto Curve

As B sweeps from 1 to infinity, the pair (Throughput, TPOT) traces a Pareto frontier:

- **Throughput** (total tokens per second, all users): Throughput(B) = B / t_step_user(B)
- **TPOT** (per-user latency, seconds per output token): TPOT(B) = t_step_user(B) / B

## 6.1 Three Zones

**Zone 1 — Memory-bound (B < B*):**
Both throughput and TPOT improve as B increases. Weight traffic is amortized across more tokens. Each additional sequence is essentially "free" from a bandwidth perspective.

> Throughput(B) ~ B x BW_mem / T_theta  
> TPOT(B) ~ T_theta / (B x BW_mem)

**Zone 2 — Crossover (B ~ B*):**
The system hits the ridge point. This is the "knee" of the Pareto curve — the operating point with the best throughput-per-unit-TPOT ratio. Further batching starts to cost latency.

**Zone 3 — Compute-bound (B > B*):**
Throughput plateaus. TPOT remains approximately constant.

> Throughput(B) -> R_GPU / F_token  
> TPOT(B) ~ F_token / R_GPU

## 6.2 Throughput–Interactivity Pareto Axes

Production LLM inference benchmarks plot **Throughput/GPU** against **Interactivity** (= 1/TPOT). The three zones map directly:

- Moving left-to-right corresponds to increasing B from memory-bound through crossover into compute-bound.
- **Maximum throughput** (rightmost): B >> B\*, low interactivity, high TPOT.
- **Maximum interactivity** (B=1): low throughput, minimum TPOT.
- Production deployments target an operating point near B\*, balancing GPU utilization with per-request latency SLAs.

---

# 7. Worked Example: LLaMA-3 70B on 8 H100 GPUs

Consider LLaMA-3 70B deployed with TP=8 on 8 H100 GPUs (bf16, b=2).

**Model parameters (per device after TP=8):**

| Symbol | Value | Notes |
|--------|-------|-------|
| L | 80 | Layers |
| H | 8192 | Hidden dimension |
| H_kv | 1024 | GQA with 8 KV heads (n_kv=8, d=128) |
| I | 28672 | SwiGLU FFN intermediate dim |
| S | 2048 | Decode context length |
| b | 2 | bf16 bytes per parameter |

**Per-device traffic (TP=8, PP=1, SP=1):**

- Weight traffic: T_theta — all weights read once per step.
- KV traffic per user: T_KV = L x 2 x S x H_kv x b / TP = 80 x 2 x 2048 x 1024 x 2 / 8 = 83.9 MB

**H100 specs:**

| Parameter | Value |
|-----------|-------|
| R_GPU | 989 TFLOPS (bf16 tensor core) |
| BW_mem | ~2.5 TB/s (effective HBM3 BW) |
| Ridge point | 989 / 2.5 = ~396 FLOPs/byte |

**Single-token OI (B=1):**
OI(1) = 2/b = 1 FLOP/byte — far below the ridge point of 396. The GPU is < 0.3% compute-utilized at B=1.

**Crossover B*:**
Using the weight-dominated approximation (valid because T_theta >> T_KV at S=2048), B\* ~ (T_theta / F_token) x (R_GPU / BW_mem) ~ 396 / 2 ~ 198.

This means the system transitions to compute-bound at around B ~ 200 sequences.

**TPOT scaling:**

| B | Regime | Relative TPOT | Notes |
|---|--------|---------------|-------|
| 1 | Memory-bound | 1x (baseline) | GPU < 1% utilized |
| 10 | Memory-bound | ~0.1x | 10 users, ~same step time, 10x better TPOT |
| 100 | Memory-bound | ~0.01x | "Free lunch" continues |
| ~200 | Crossover (B*) | Minimum | Ridge point reached |
| 500 | Compute-bound | ~F/R (flat) | Adding users no longer helps TPOT |

The table illustrates the key insight: from B=1 to B~200, TPOT improves by roughly 200x with nearly no penalty — this is the free lunch of weight amortization. Beyond B\*, TPOT stabilizes and throughput approaches the compute ceiling.

---

# 8. FAQ: Common Questions About Batched Decode

**Q: Is batched decode for serving multiple users, or for a single user generating a longer response?**

Batched decode serves **multiple independent users simultaneously**. Each of the B sequences in the batch belongs to a different user (or a different request). They share the same forward pass through the model weights but maintain completely independent KV caches.

A single user generating a longer response is still B=1 — the context length S grows as more tokens are generated, but the batch size does not change. Longer output means more decode steps (higher N_out), not a larger batch.

**Q: Why are weights shared but KV cache reads are not?**

The model weights (W_Q, W_K, W_V, W_O, FFN matrices) are the **same** for every user — they define the learned model, not the user's conversation. When the GPU loads these weights from HBM, it can apply them to all B tokens in the batch before moving to the next weight matrix.

KV cache entries, on the other hand, are **specific to each user's conversation history**. User A's key-value pairs encode *their* past tokens, and User B's encode an entirely different conversation. There is no reuse across users. The GPU must read each user's KV cache separately.

This is why the traffic formula is T_theta + B x T_KV: one weight read, B KV reads.

**Q: Does batching increase per-user latency?**

In the memory-bound regime (B < B\*), **no** — TPOT actually *decreases* with B. The GPU was already spending most of its time loading weights for one user; batching reuses those same weight reads for B users, so the step time barely changes while TPOT = step_time / B drops.

In the compute-bound regime (B > B\*), TPOT stabilizes at F_token / R_GPU and does not degrade further. The step time grows with B, but dividing by B cancels that growth.

The worst case for per-user latency is not high B in the compute-bound regime — it is high B *and* high S together, where KV cache traffic can dominate both the memory budget (risk of out-of-memory) and the traffic budget (pushing B\* toward infinity).

**Q: What limits the maximum batch size in practice?**

HBM capacity. Each sequence's KV cache consumes:

$$
M_{KV,per\_seq} = \frac{L}{PP} \times \frac{2 \cdot S \cdot H_{kv} \cdot b}{TP \cdot SP}
$$

The total KV footprint is B x M_KV_per_seq, which must fit in HBM alongside model weights and activation buffers. The maximum B is therefore:

$$
B_{max} = \left\lfloor \frac{M_{HBM} - M_{\theta} - M_{act} - M_{sys}}{M_{KV,per\_seq}} \right\rfloor
$$

See [kv.md](../modeling/kv.md) for paged KV cache analysis with block-level fragmentation.

**Q: How does batched decode interact with continuous batching (iteration-level scheduling)?**

The analysis in this document assumes a **static batch** — B sequences are decoded together for the duration of each step. In practice, serving systems use **continuous batching**: new requests can join the batch at any step, and completed requests leave immediately.

The per-step analysis is identical: at any given step, B_active sequences are in flight, and the cost structure is T_theta + B_active x T_KV. The difference is that B_active varies over time rather than being fixed. The formulas in this document apply to each individual step with the instantaneous B.

---

## Cross-References

- [../modeling/decode.md §6.4](../modeling/decode.md#64-batch-size-scaling-and-throughputlatency-tradeoff) — Formal derivation of all batched decode equations
- [../modeling/decode.md §4](../modeling/decode.md#4-compute-vs-memory-bound-roofline-model) — Roofline model and operational intensity
- [prefill.md](../modeling/prefill.md) — Prefill latency analysis (complementary to decode)
- [e2e.md](../modeling/e2e.md) — End-to-end metric assembly (TTFT + TPOT)
- [kv.md](../modeling/kv.md) — KV paging and HBM capacity limits on B
