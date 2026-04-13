# Prefill and Time-To-First-Token (TTFT) Performance Model

**Modeling Compute-Bound Prefill, Batched Prefill, Chunked Prefill, and Disaggregated Architectures**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
LLM inference, prefill, TTFT, time-to-first-token, GEMM, compute-bound, memory-bound, roofline,  
chunked prefill, batched prefill, disaggregated prefill, KV transfer, FlashAttention, pipeline warmup

---

Prefill processes the entire input sequence in a **single forward pass**, producing the KV cache entries for all input tokens and computing the first contextual representation. Unlike autoregressive decoding — where one token is processed per step and weight-reads dominate (GEMV, memory-bound) — prefill operates on $S_{\text{input}}$ tokens simultaneously, turning every linear layer into a matrix-matrix multiplication (GEMM) that scales with $S_{\text{input}}$. The resulting arithmetic intensity grows with sequence length, making prefill **compute-bound at typical input lengths** while decode remains perpetually memory-bound at batch size $B = 1$. Time To First Token (TTFT) encompasses three sequential phases: (1) the prefill pass itself, (2) optional KV-cache transfer from a prefill cluster to a decode cluster in disaggregated architectures, and (3) pipeline warmup latency as the first token traverses all PP stages. This document derives the full TTFT model from first principles, covering single-request, batched, chunked, and disaggregated scenarios.

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Prefill FLOPs](#1-prefill-flops)
  - [1.1 Projection FLOPs (Q/K/V/O)](#11-projection-flops-qkvo)
  - [1.2 Attention Score and Value FLOPs](#12-attention-score-and-value-flops)
  - [1.3 FFN FLOPs](#13-ffn-flops)
  - [1.4 Per-Layer Total and FlashAttention Note](#14-per-layer-total-and-flashattention-note)
  - [1.5 Per-Device Total FLOPs Under Parallelism](#15-per-device-total-flops-under-parallelism)

- [2. Compute vs. Memory Bound: Prefill vs. Decode](#2-compute-vs-memory-bound-prefill-vs-decode)
  - [2.1 Arithmetic Intensity of Prefill](#21-arithmetic-intensity-of-prefill)
  - [2.2 Arithmetic Intensity of Decode](#22-arithmetic-intensity-of-decode)
  - [2.3 Ridge Point and Regime Crossover](#23-ridge-point-and-regime-crossover)

- [3. Single-Request TTFT](#3-single-request-ttft)
  - [3.1 Prefill Local Time (Roofline)](#31-prefill-local-time-roofline)
  - [3.2 Communication During Prefill](#32-communication-during-prefill)
  - [3.3 Pipeline Warmup Latency](#33-pipeline-warmup-latency)
  - [3.4 End-to-End TTFT Formula](#34-end-to-end-ttft-formula)

- [4. Batched Prefill](#4-batched-prefill)
  - [4.1 FLOPs and Latency Under Batching](#41-flops-and-latency-under-batching)
  - [4.2 TTFT Scaling with Batch Size](#42-ttft-scaling-with-batch-size)
  - [4.3 Optimal Batch Size for GPU Utilization](#43-optimal-batch-size-for-gpu-utilization)

- [5. Chunked Prefill](#5-chunked-prefill)
  - [5.1 Per-Chunk Latency](#51-per-chunk-latency)
  - [5.2 Total Prefill Time and TTFT Under Chunking](#52-total-prefill-time-and-ttft-under-chunking)
  - [5.3 Throughput Benefit and Head-of-Line Blocking Reduction](#53-throughput-benefit-and-head-of-line-blocking-reduction)

- [6. Disaggregated Prefill](#6-disaggregated-prefill)
  - [6.1 Architecture and Motivation](#61-architecture-and-motivation)
  - [6.2 KV Cache Transfer Latency](#62-kv-cache-transfer-latency)
  - [6.3 TTFT for Disaggregated Systems](#63-ttft-for-disaggregated-systems)

---

<div style="page-break-before: always;"></div>

# 1. Prefill FLOPs

During decoding a single token is processed per step; all weight multiplications are **matrix–vector** (GEMV) operations. During prefill all $S_{\text{input}}$ tokens are processed in one forward pass; every linear layer becomes a **matrix–matrix** (GEMM) operation whose FLOP count scales with $S_{\text{input}}$.

**Notation used in this section** (see `modeling.notation.md §§3–4, 8, 11` for the full table):

| Symbol | Meaning |
|--------|---------|
| $S_{\text{input}}$ | Input sequence length (number of tokens in the prefill batch) |
| $H$ | Hidden size (model dimension) |
| $H_{kv} = n_{kv} \cdot d_{\text{head}}$ | Total KV projection dimension (GQA; [GQA]) |
| $I_{\text{eff}}$ | Effective FFN intermediate size ($I_{\text{dense}}$ for dense; $k \cdot I_{\text{moe}}$ for MoE) |
| $L$ | Total number of transformer layers |
| $PP, TP, EP, SP$ | Pipeline, tensor, expert, sequence parallelism degrees |

All counts follow the **multiply-accumulate = 2 FLOPs** convention (one multiply + one add per MAC).

---

## 1.1 Projection FLOPs (Q/K/V/O)

For decoding a single token, each projection matrix multiplies a vector of length $H$ (or $H_{kv}$) against the weight matrix.
For prefill, the operation is a GEMM of shape $[S_{\text{input}} \times H] \times [H \times H_{\text{out}}]$, contributing $2 \cdot S_{\text{input}} \cdot H \cdot H_{\text{out}}$ FLOPs.

The four projection matrices and their output dimensions:

| Projection | Weight shape | Output dim | FLOPs |
|------------|-------------|------------|-------|
| $W_Q$ | $H \times H$ | $H$ | $2 H^2 S_{\text{input}}$ |
| $W_K$ | $H \times H_{kv}$ | $H_{kv}$ | $2 H H_{kv} S_{\text{input}}$ |
| $W_V$ | $H \times H_{kv}$ | $H_{kv}$ | $2 H H_{kv} S_{\text{input}}$ |
| $W_O$ | $H_{kv} \times H$ | $H$ | $2 H_{kv} H S_{\text{input}}$ |

Summing:

$$
F_{\text{proj,prefill}}
=
\bigl(2H^2 + 6 H H_{kv}\bigr) \cdot S_{\text{input}}
$$

This matches the decode projection formula multiplied by $S_{\text{input}}$, reflecting the transition from GEMV to GEMM.

---

## 1.2 Attention Score and Value FLOPs

### Attention scores ($QK^\top$)

During prefill, the full causal attention matrix is computed. The query matrix has shape $[S_{\text{input}} \times H_{kv}]$ (after reshaping heads), and the key matrix has shape $[S_{\text{input}} \times H_{kv}]$. The product yields a score matrix of shape $[S_{\text{input}} \times S_{\text{input}}]$.

For the lower-triangular causal mask, each row $i$ attends to tokens $1 \ldots i$. The number of active cells averages $S_{\text{input}}/2$ per row, giving $S_{\text{input}}^2 / 2$ total cells. Using the full-matrix formula as an upper bound (standard in FLOPs accounting):

$$
F_{\text{score,prefill}}
=
2 \cdot S_{\text{input}}^2 \cdot H_{kv}
$$

> **Note:** The factor of 2 accounts for both the multiply and accumulate in each dot product. The causal mask halves the *average* work, but practical implementations (FlashAttention tile loops) often process the full upper-right with a mask, and published FLOPs budgets consistently use the full $S^2$ form [FA1, FA2]. We adopt the same convention.

### Value application (Attn · V)

After softmax, the attention weights (shape $[S_{\text{input}} \times S_{\text{input}}]$) are multiplied against the value matrix (shape $[S_{\text{input}} \times H_{kv}]$):

$$
F_{\text{value,prefill}}
=
2 \cdot S_{\text{input}}^2 \cdot H_{kv}
$$

### Combined attention KV FLOPs

$$
F_{\text{attn,KV,prefill}}
=
F_{\text{score,prefill}} + F_{\text{value,prefill}}
=
4 \cdot S_{\text{input}}^2 \cdot H_{kv}
$$

---

## 1.3 FFN FLOPs

For a gated MLP (SwiGLU/GeGLU style) with three weight matrices (gate, up, down) and effective intermediate dimension $I_{\text{eff}}$:

$$
F_{\text{ffn,prefill}}
=
4 \cdot H \cdot I_{\text{eff}} \cdot S_{\text{input}}
$$

> The factor of 4 arises from two GEMMs of shape $[S_{\text{input}} \times H] \times [H \times I_{\text{eff}}]$ (gate + up projections, each $2 H I_{\text{eff}} S_{\text{input}}$ FLOPs) and one GEMM of shape $[S_{\text{input}} \times I_{\text{eff}}] \times [I_{\text{eff}} \times H]$ (down projection, $2 H I_{\text{eff}} S_{\text{input}}$ FLOPs), giving $6 H I_{\text{eff}} S_{\text{input}}$ total. For brevity (and to match the widely-used approximation in scaling law literature), we keep $4 H I_{\text{eff}} S_{\text{input}}$ as the dominant term, noting the exact count is $6 H I_{\text{eff}} S_{\text{input}}$ for gated MLPs. In performance modeling it is standard to use $4 H I_{\text{eff}}$ per token because the gate elementwise multiply and activation have negligible FLOP cost relative to the GEMMs.

---

## 1.4 Per-Layer Total and FlashAttention Note

Combining projections, attention KV, and FFN per transformer layer:

$$
F_{\text{layer,prefill}}
=
F_{\text{proj,prefill}}
+ F_{\text{attn,KV,prefill}}
+ F_{\text{ffn,prefill}}
$$

$$
\boxed{
F_{\text{layer,prefill}}
=
\bigl(2H^2 + 6 H H_{kv} + 4 H I_{\text{eff}}\bigr) S_{\text{input}}
\;+\;
4 S_{\text{input}}^2 H_{kv}
}
$$

The first term grows as $O(S_{\text{input}})$ and corresponds to GEMM operations on the weight matrices. The second term grows as $O(S_{\text{input}}^2)$ and comes from the full pairwise attention computation.

### Regime crossover

The $S_{\text{input}}^2$ term dominates when:

$$
4 S_{\text{input}}^2 H_{kv}
>
\bigl(2H^2 + 6 H H_{kv} + 4 H I_{\text{eff}}\bigr) S_{\text{input}}
$$

Simplifying:

$$
S_{\text{input}}
>
\frac{2H^2 + 6 H H_{kv} + 4 H I_{\text{eff}}}{4 H_{kv}}
\approx
\frac{H^2}{2 H_{kv}}
\quad (\text{when } H \gg H_{kv}, I_{\text{eff}})
$$

For a model with $H = 8192$ and $H_{kv} = 1024$ (e.g., GQA with $n_{kv} = 8$, $d_{\text{head}} = 128$), the crossover is at $S_{\text{input}} \approx 32\text{k}$ tokens — only relevant for very long contexts. For smaller models (e.g., $H = 4096$, $H_{kv} = 512$), crossover occurs at $\sim 16\text{k}$ tokens. At common prefill lengths ($\le 8\text{k}$ tokens), the linear projection term typically dominates.

### FlashAttention does NOT reduce prefill FLOPs

FlashAttention [FA1, FA2] tiles the $S \times S$ attention matrix into SRAM-resident blocks, reducing the number of HBM round-trips and thus **memory traffic**. However, the number of floating-point operations remains identical — FlashAttention is an IO-aware reorganization of the same computation, not a FLOP reduction. The $4 S_{\text{input}}^2 H_{kv}$ term applies with or without FlashAttention; the difference appears only in the memory traffic model (Section 3).

---

## 1.5 Per-Device Total FLOPs Under Parallelism

Parallelism distributes prefill FLOPs across devices following the same DP → PP → EP → TP → SP nesting as decode [MEGATRON, MEGATRON3].

### PP sharding

Pipeline parallelism assigns $L/PP$ layers to each stage. Since the prefill pass traverses all layers, each PP stage handles exactly its local layers:

$$
F_{\text{layer,prefill,device}} = \frac{L}{PP} \cdot F_{\text{layer,prefill}}^{\text{sharded by TP/EP/SP}}
$$

### TP sharding

Tensor parallelism splits weight matrices. Each device computes $1/TP$ of the projection and FFN GEMMs. The attention KV term is also split by TP (heads are distributed):

- Projections: $\bigl(2H^2 + 6 H H_{kv} + 4 H I_{\text{eff}}\bigr) S_{\text{input}} / TP$
- Attention KV (score + value): attention heads are split across TP ranks, so $4 S_{\text{input}}^2 H_{kv} / TP$

### SP sharding

Sequence parallelism partitions $S_{\text{input}}$ across SP devices for the attention computation. Each SP rank computes attention over $S_{\text{input}} / SP$ query positions attending to the full context (via ring KV passing), but executes $1/SP$ of the attention score FLOPs:

$$
F_{\text{attn,KV,prefill,device}} = \frac{4 S_{\text{input}}^2 H_{kv}}{TP \cdot SP}
$$

### EP sharding (MoE)

For MoE layers, expert parallelism distributes expert weights so each device computes $1/EP$ of the FFN work:

$$
F_{\text{ffn,prefill,device}}^{\text{MoE}} = \frac{4 H I_{\text{eff}} S_{\text{input}}}{TP \cdot EP}
$$

### Full per-device expression

Collecting all terms:

$$
\boxed{
F_{\text{prefill,device}}
=
\frac{L}{PP}
\left[
\frac{\bigl(2H^2 + 6 H H_{kv}\bigr) S_{\text{input}}}{TP}
+
\frac{4 H I_{\text{eff}} S_{\text{input}}}{TP \cdot EP}
+
\frac{4 S_{\text{input}}^2 H_{kv}}{TP \cdot SP}
\right]
}
$$

For a **dense model** ($EP = 1$, $I_{\text{eff}} = I_{\text{dense}}$):

$$
F_{\text{prefill,device}}^{\text{dense}}
=
\frac{L}{PP \cdot TP}
\left[
\bigl(2H^2 + 6 H H_{kv} + 4 H I_{\text{dense}}\bigr) S_{\text{input}}
+
\frac{4 S_{\text{input}}^2 H_{kv}}{SP}
\right]
$$

---

<div style="page-break-before: always;"></div>

# 2. Compute vs. Memory Bound: Prefill vs. Decode

A fundamental insight into LLM inference performance is that **prefill and decode lie in different hardware regimes** — prefill is (typically) compute-bound, decode is (always) memory-bound at $B = 1$. This section derives the arithmetic intensities for both phases and identifies the crossover condition, following the roofline framework [ROOFLINE].

The **ridge point** of a device is:

$$
R_{\text{ridge}}
=
\frac{R_{\text{GPU}}}{B_{\text{eff,mem}}}
\quad (\text{FLOPs per byte})
$$

A kernel is compute-bound if $\text{OI} > R_{\text{ridge}}$, and memory-bound otherwise.

---

## 2.1 Arithmetic Intensity of Prefill

Arithmetic intensity (OI) is the ratio of FLOPs to bytes moved from HBM during the computation.

### Weight-dominated regime (linear projections)

For the projection and FFN GEMMs, the weight matrices are read once per prefill pass. The weight traffic per layer is $T_{\theta,\text{layer}} = (P_{\text{attn}} + P_{\text{FFN}}/EP) \cdot b / TP$ bytes. The projection and FFN FLOPs are $\bigl(2H^2 + 6 H H_{kv} + 4 H I_{\text{eff}} / EP\bigr) \cdot S_{\text{input}} / TP$.

Using the dominant-term approximation $P_{\text{attn}} \approx H^2$ and ignoring KV:

$$
\text{OI}_{\text{prefill,proj}}
\approx
\frac{\bigl(2H^2 + 4 H I_{\text{eff}}\bigr) S_{\text{input}}}{(H^2 + 3 H I_{\text{eff}} N_{\text{exp}}/EP)\, b}
\approx
\frac{2 S_{\text{input}}}{b}
\quad (\text{for dense, }I_{\text{eff}} \approx H)
$$

The arithmetic intensity of the linear projection stage **scales linearly with $S_{\text{input}}$** — each weight byte is reused across all $S_{\text{input}}$ token vectors in the GEMM.

### Attention computation regime

For the attention score and value computation, the "weights" are the $Q$, $K$, $V$ activations themselves (all loaded from HBM or kept in SRAM with FlashAttention). Without FlashAttention, the traffic is $O(S_{\text{input}}^2)$ (storing the full attention matrix); with FlashAttention [FA1, FA2], the $S \times S$ matrix is tiled and never fully materialized, reducing traffic to $O(S_{\text{input}})$. In both cases the FLOPs are $4 S_{\text{input}}^2 H_{kv} / (TP \cdot SP)$.

For performance modeling, we treat FlashAttention as the default and note that the attention computation is generally compute-bound in the prefill regime (the tiled SRAM reuse pattern achieves high OI within each tile).

### Summary

For the projection- and FFN-dominated components of prefill (the $O(S_{\text{input}})$ FLOP terms):

$$
\boxed{
\text{OI}_{\text{prefill}}
\approx
\frac{2 S_{\text{input}}}{b}
}
$$

---

## 2.2 Arithmetic Intensity of Decode

During decode at batch size $B = 1$, each linear layer is a GEMV: one vector of length $H$ (or $H_{kv}$) is multiplied against a weight matrix. The weight matrix is read in full ($\sim H^2 b$ bytes per layer), yielding $\sim 2 H^2$ FLOPs. The OI of a weight-matrix GEMV is:

$$
\text{OI}_{\text{decode,GEMV}}
\approx
\frac{2 H^2}{H^2\, b}
=
\frac{2}{b}
$$

At batch size $B > 1$ (batched decode), each weight column is reused across $B$ output tokens:

$$
\text{OI}_{\text{decode}}
\approx
\frac{2 B}{b}
$$

For $B = 1$ and $b = 2$ (bf16), $\text{OI}_{\text{decode}} = 1$ FLOP/byte — far below any modern GPU's ridge point.

$$
\boxed{
\text{OI}_{\text{decode}} \approx \frac{2B}{b}
\quad\Longrightarrow\quad
\textbf{always memory-bound at }B = 1
}
$$

---

## 2.3 Ridge Point and Regime Crossover

For an H100 SXM5 [H100-SPEC]:

- $R_{\text{GPU}} \approx 989$ TFlops/s (bf16 with Tensor Cores)
- $B_{\text{eff,mem}} \approx 3.35$ TB/s (HBM3E)
- $R_{\text{ridge}} = R_{\text{GPU}} / B_{\text{eff,mem}} \approx 295$ FLOPs/byte

Comparing with OI expressions (bf16, $b = 2$):

| Phase | OI expression | Regime |
|-------|--------------|--------|
| Decode ($B=1$) | $2/b = 1$ | Memory-bound ($1 \ll 295$) |
| Decode ($B=B$) | $2B/b$ | Compute-bound when $B > 295$ |
| Prefill | $2 S_{\text{input}}/b$ | Compute-bound when $S_{\text{input}} > 295$ |

**Prefill becomes compute-bound for $S_{\text{input}} \gtrsim 295$ tokens** on H100 with bf16.

This is the crossover condition:

$$
\boxed{
S_{\text{input}}^{\star}
=
\frac{b}{2} \cdot R_{\text{ridge}}
=
\frac{b}{2} \cdot \frac{R_{\text{GPU}}}{B_{\text{eff,mem}}}
}
$$

In practice, prefill sequences are almost always longer than $S_{\text{input}}^{\star}$, so the prefill pass is **virtually always compute-bound** at single-request inference on modern accelerators. The exact threshold varies with precision and HBM bandwidth; on A100 (HBM2E, $R_{\text{ridge}} \approx 312$ FLOPs/byte), $S_{\text{input}}^{\star} \approx 156$ tokens with bf16.

---

<div style="page-break-before: always;"></div>

# 3. Single-Request TTFT

We now derive TTFT for a single request on a co-located prefill+decode cluster (no disaggregation). The result will be extended in Sections 4–6.

TTFT decomposes into three sequential phases:

$$
TTFT
=
\underbrace{t_{\text{prefill,local}} + \max\!\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right)}_{\text{prefill latency (all PP stages in sequence)}}
+
\underbrace{t_{\text{pipeline,warmup}}}_{\text{PP fill}}
$$

Each term is derived below.

---

## 3.1 Prefill Local Time (Roofline)

The local prefill time on a single PP stage is the roofline of compute and memory time.

### Compute time

Given $F_{\text{prefill,device}}$ from Section 1.5:

$$
t_{\text{prefill,compute}}
=
\frac{F_{\text{prefill,device}}}{R_{\text{GPU}}}
$$

### Memory time

During prefill, weights are read from HBM (same as decode). However, the **KV cache is being written** (not read) for the input tokens — it is not yet present and therefore contributes **write traffic**, not read traffic. We model total prefill traffic as:

$$
T_{\text{prefill,device}}
=
T_{\theta,\text{device}}
\;+\;
T_{\text{KV,write,device}}
$$

where:

- $T_{\theta,\text{device}}$ — weight read traffic per device (same as decode; see `modeling.tpot.md §2.1`):

  $$
  T_{\theta,\text{device}}
  \approx
  \frac{L}{PP}
  \left(
  \frac{(H^2 + 3 H H_{kv})b}{TP}
  +
  \frac{3 H I_{\text{eff}} N_{\text{exp}} b}{TP \cdot EP}
  \right)
  $$

- $T_{\text{KV,write,device}}$ — KV cache write traffic for the $S_{\text{input}}$ tokens being prefilled:

  $$
  T_{\text{KV,write,device}}
  =
  \frac{L}{PP}
  \cdot
  \frac{2 S_{\text{input}} H_{kv}\, b}{TP \cdot SP}
  $$

  This is the cost of writing the keys and values for all $S_{\text{input}}$ tokens to HBM. Because HBM write bandwidth equals read bandwidth, we use $B_{\text{eff,mem}}$ for both.

### FlashAttention reduces attention read traffic

Without FlashAttention, the $S_{\text{input}} \times S_{\text{input}}$ attention score matrix would need to be written to and read from HBM, contributing $O(S_{\text{input}}^2)$ traffic. With FlashAttention [FA1, FA2], the attention computation is fused into SRAM-tiled blocks, and the $S \times S$ matrix is never materialized in HBM. This eliminates the dominant attention traffic term.

The effective memory time is:

$$
t_{\text{prefill,mem}}
=
\frac{T_{\text{prefill,device}}}{B_{\text{eff,mem}}}
$$

### Roofline local time

$$
\boxed{
t_{\text{prefill,local}}
=
\max\!\left(
t_{\text{prefill,compute}},\;
t_{\text{prefill,mem}}
\right)
}
$$

At typical prefill lengths (compute-bound regime, Section 2.3): $t_{\text{prefill,compute}} \gg t_{\text{prefill,mem}}$, so $t_{\text{prefill,local}} \approx t_{\text{prefill,compute}}$.

---

## 3.2 Communication During Prefill

The communication collectives required during prefill are structurally the same as during decode (same TP/EP/SP/PP operations per layer), but **message sizes scale with the sequence dimension** because outputs have shape $[S_{\text{input}} \times H]$ rather than $[1 \times H]$.

All collective latencies follow the $\alpha$–$\beta$ model [ALPHA-BETA]:

$$
t = \alpha + \frac{\text{message size}}{B_{\text{eff}}}
$$

### TP All-Reduce (prefill)

The TP All-Reduce synchronizes the partial hidden-state outputs after Row-Parallel matrix multiplications. During prefill, the output has shape $[S_{\text{input}} \times H]$, so the message size is $H \cdot S_{\text{input}} \cdot b$ (compared to $H \cdot b$ during decode).

Per-layer TP ring All-Reduce:

$$
t_{TP}^{\text{prefill,ring}}
=
2(TP-1)\alpha_{TP}
+
2\frac{TP-1}{TP}
\cdot
\frac{H \cdot S_{\text{input}} \cdot b}{B_{\text{eff,TP}}}
$$

For a tree All-Reduce:

$$
t_{TP}^{\text{prefill,tree}}
\approx
2\lceil\log_2(TP)\rceil\,\alpha_{TP}
+
2\frac{H \cdot S_{\text{input}} \cdot b}{B_{\text{eff,TP}}}
$$

The $S_{\text{input}}$ factor in the bandwidth term means TP communication during prefill is substantially larger than during decode. For $S_{\text{input}} = 4096$ this is a $4096\times$ larger payload per collective than single-token decode.

### EP All-to-All (prefill, MoE)

For MoE layers, each device dispatches $k \cdot H \cdot S_{\text{input}} \cdot b$ bytes of token activations per direction (vs. $k \cdot H \cdot b$ during decode):

$$
t_{EP}^{\text{prefill,ring}}
=
2(EP-1)\alpha_{EP}
+
2(EP-1)
\frac{k H \cdot S_{\text{input}} \cdot b}{EP \cdot B_{\text{eff,EP}}}
$$

### PP Hop (prefill)

The PP hop forwards the hidden-state shard to the next stage. With TP rank alignment, each device forwards its local shard of shape $[S_{\text{input}} \times H/TP]$:

$$
t_{PP}^{\text{prefill}}
=
\alpha_{PP}
+
\frac{(H/TP) \cdot S_{\text{input}} \cdot b}{B_{\text{eff,PP}}}
$$

### SP All-Gather (prefill)

During prefill with SP, each SP rank holds $S_{\text{input}}/SP$ of the input sequence. Ring Attention must circulate KV shards so each device's query block can attend to the full input. The per-layer SP communication:

$$
t_{SP}^{\text{prefill}}
=
(SP-1)\alpha_{SP}
+
(SP-1) \cdot \frac{(S_{\text{input}}/SP) \cdot (2 H_{kv}/TP) \cdot b}{B_{\text{eff,SP}}}
$$

### Total per-stage communication time (prefill)

Following the same structure as `modeling.tpot.md §5.5`, collectives within each layer are sequential:

$$
\boxed{
t_{\text{prefill,comm}}
=
\frac{L}{PP}
\bigl(
n_{TP}\, t_{TP}^{\text{prefill}}
+
n_{SP}\, t_{SP}^{\text{prefill}}
\bigr)
+
\frac{L_{\text{moe}}}{PP}
\bigl(
n_{EP}\, t_{EP}^{\text{prefill}}
\bigr)
+
t_{PP}^{\text{prefill}}
}
$$

where $n_{TP} = 2$ (attention + FFN), $n_{SP} = 1$ (attention), $n_{EP} = 1$ (MoE FFN dispatch/combine).

---

## 3.3 Pipeline Warmup Latency

With $PP$ pipeline stages, the first token's hidden state must traverse all stages **sequentially** before the first decoded token can be emitted. Since the pipeline is empty at the start of prefill, there is no parallel filling:

$$
t_{\text{pipeline,warmup}}
=
(PP - 1) \cdot t_{\text{stage}}
$$

where $t_{\text{stage}}$ is the latency to process one prefill batch through a single PP stage (approximately $t_{\text{prefill,local}}$ for the bottleneck stage, plus its inter-stage PP hop $t_{PP}^{\text{prefill}}$). For uniform PP stages:

$$
t_{\text{pipeline,warmup}}
\approx
(PP - 1) \cdot
\bigl(t_{\text{prefill,local}} + t_{PP}^{\text{prefill}}\bigr)
$$

For $PP = 1$ (no pipeline parallelism), $t_{\text{pipeline,warmup}} = 0$.

---

## 3.4 End-to-End TTFT Formula

Combining all three phases, with overlap factor $\rho \in [0, 1]$ capturing the fraction of prefill communication that can be hidden behind compute:

$$
\boxed{
TTFT
=
t_{\text{prefill,local}}
+
\max\!\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right)
+
t_{\text{pipeline,warmup}}
}
$$

**Interpretation of each term:**

- $t_{\text{prefill,local}}$: roofline local time; dominated by compute at typical $S_{\text{input}}$ (Section 2.3).
- $\max(0,\, t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}})$: residual communication after compute–communication overlap. In the compute-bound prefill regime, $t_{\text{prefill,local}}$ is large, so significant communication hiding ($\rho \approx 0.8$–$1.0$) is achievable.
- $t_{\text{pipeline,warmup}}$: pipeline fill penalty; grows with $PP$ and with $S_{\text{input}}$ (since $t_{PP}^{\text{prefill}}$ scales with $S_{\text{input}}$).

> **Overlap note:** The overlap factor $\rho$ is an original parameterization (this work); see `modeling.references.md`. In the compute-bound prefill regime, compute and communication can be overlapped aggressively by pipelining GEMM tiles with collective operations (e.g., using NCCL + CUDA stream concurrency). Practical $\rho$ values are system-dependent but commonly 0.5–0.9.

---

<div style="page-break-before: always;"></div>

# 4. Batched Prefill

Multiple requests can be prefilled simultaneously by stacking their input tokens into a single padded batch of dimension $B_{\text{prefill}} \times S_{\text{input}}$. (In continuous batching, the sequences may have different lengths, but we model the uniform-length case for clarity; the analysis generalizes by replacing $S_{\text{input}}$ with the average padded length.)

---

## 4.1 FLOPs and Latency Under Batching

All projection and FFN GEMMs grow proportionally with $B_{\text{prefill}}$:

$$
F_{\text{prefill,device}}^{(B_{\text{prefill}})}
=
B_{\text{prefill}} \cdot F_{\text{prefill,device}}
\bigl|_{B_{\text{prefill}}=1}
$$

For the attention score and value terms, the $S_{\text{input}}^2$ computation is **independent across requests** (each request's query attends only to its own key/value cache, not to other requests' tokens). Thus:

$$
F_{\text{attn,KV,prefill,device}}^{(B_{\text{prefill}})}
=
B_{\text{prefill}} \cdot \frac{4 S_{\text{input}}^2 H_{kv}}{TP \cdot SP}
$$

Unlike batched **decode** — where the weight matrices are shared and reused across all $B$ output tokens in the GEMV — batched prefill increases both FLOPs and compute time proportionally. In the compute-bound regime:

$$
t_{\text{prefill,compute}}^{(B_{\text{prefill}})}
\approx
B_{\text{prefill}} \cdot t_{\text{prefill,compute}}
\bigl|_{B_{\text{prefill}}=1}
$$

---

## 4.2 TTFT Scaling with Batch Size

All $B_{\text{prefill}}$ requests must wait for the **entire batched prefill pass** to complete before any of them receives their first token. Therefore:

$$
TTFT(B_{\text{prefill}})
\approx
t_{\text{prefill,local}}^{(B_{\text{prefill}})}
+
\max\!\left(0,\; t_{\text{prefill,comm}}^{(B_{\text{prefill}})} - \rho\, t_{\text{prefill,local}}^{(B_{\text{prefill}})}\right)
+
t_{\text{pipeline,warmup}}
$$

In the compute-bound regime, $t_{\text{prefill,local}} \propto B_{\text{prefill}}$, so TTFT grows **roughly linearly** with $B_{\text{prefill}}$:

$$
TTFT(B_{\text{prefill}}) \approx B_{\text{prefill}} \cdot TTFT(1)
\quad (\text{compute-bound approximation})
$$

Note: communication time also scales with $B_{\text{prefill}}$ (all message sizes acquire a $B_{\text{prefill}}$ factor), so the growth is strictly linear when both compute and communication are proportional to batch size.

---

## 4.3 Optimal Batch Size for GPU Utilization

The goal of batched prefill is to maximize **GPU utilization** (keep the compute units fully loaded) while satisfying a TTFT service-level objective (SLO), $TTFT_{\text{SLO}}$.

### Minimum batch size for compute saturation

The GPU is compute-bound when:

$$
\text{OI}(B_{\text{prefill}}, S_{\text{input}})
\ge R_{\text{ridge}}
$$

Using $\text{OI}_{\text{prefill}} \approx 2 S_{\text{input}} / b$ (Section 2.1), single-request prefill is already compute-bound for $S_{\text{input}} \gtrsim S_{\text{input}}^{\star}$ (Section 2.3). Batching primarily helps when $S_{\text{input}} < S_{\text{input}}^{\star}$, in which case the minimum batch size to enter the compute-bound regime is:

$$
B_{\text{prefill}}^{\min}
=
\left\lceil \frac{R_{\text{ridge}} \cdot b}{2 \cdot S_{\text{input}}} \right\rceil
=
\left\lceil \frac{R_{\text{GPU}}}{B_{\text{eff,mem}} \cdot S_{\text{input}}} \right\rceil
$$

For H100 and $S_{\text{input}} = 64$ tokens (a very short prompt): $B_{\text{prefill}}^{\min} \approx \lceil 295/64 \rceil \approx 5$ requests needed before compute-saturation.

### Maximum batch size from TTFT SLO

The TTFT SLO constrains the maximum allowable batch size. In the compute-bound regime:

$$
B_{\text{prefill}}^{\max}
=
\left\lfloor \frac{TTFT_{\text{SLO}}}{TTFT(1)} \right\rfloor
$$

### Optimal operating point

$$
\boxed{
B_{\text{prefill}}^{\text{opt}}
=
\min\!\left(B_{\text{prefill}}^{\max},\; B_{\text{avail}}\right)
}
$$

where $B_{\text{avail}}$ is the number of requests available in the queue. Batching beyond $B_{\text{prefill}}^{\max}$ violates the TTFT SLO; batching below $B_{\text{prefill}}^{\min}$ leaves the GPU under-utilized.

---

<div style="page-break-before: always;"></div>

# 5. Chunked Prefill

For very long prompts ($S_{\text{input}} \gg 1$), prefilling the entire sequence in one pass can monopolize the GPU for a long time and stall ongoing decode iterations for other requests in the system. **Chunked prefill** [SARATHI] addresses this by splitting the input sequence into $\lceil S_{\text{input}} / C \rceil$ chunks of $C$ tokens each and interleaving prefill chunks with decode iterations.

---

## 5.1 Per-Chunk Latency

Each chunk processes $C$ tokens instead of $S_{\text{input}}$. The per-chunk prefill FLOPs are obtained directly from Section 1.5 by substituting $S_{\text{input}} \leftarrow C$:

$$
F_{\text{chunk,device}}
=
\frac{L}{PP}
\left[
\frac{(2H^2 + 6 H H_{kv}) C}{TP}
+
\frac{4 H I_{\text{eff}} C}{TP \cdot EP}
+
\frac{4 C^2 H_{kv}}{TP \cdot SP}
\right]
$$

The per-chunk local time (roofline):

$$
t_{\text{chunk,compute}}
=
\frac{F_{\text{chunk,device}}}{R_{\text{GPU}}}
$$

$$
t_{\text{chunk,local}}
=
\max\!\left(
t_{\text{chunk,compute}},\;
t_{\text{chunk,mem}}
\right)
$$

where $t_{\text{chunk,mem}}$ uses chunk-sized weight reads and KV writes (same expressions as Section 3.1 with $S_{\text{input}} \leftarrow C$).

The per-chunk communication time:

$$
t_{\text{chunk,comm}}
=
t_{\text{prefill,comm}}\bigl|_{S_{\text{input}} = C}
$$

Per-chunk overlap-adjusted latency:

$$
t_{\text{chunk}}
=
t_{\text{chunk,local}}
+
\max\!\left(0,\; t_{\text{chunk,comm}} - \rho\, t_{\text{chunk,local}}\right)
$$

---

## 5.2 Total Prefill Time and TTFT Under Chunking

The number of chunks is $N_{\text{chunks}} = \lceil S_{\text{input}} / C \rceil$. The total FLOPs are identical to unchunked prefill (same total computation, just redistributed into smaller batches), so:

$$
t_{\text{prefill,chunked}}
=
N_{\text{chunks}} \cdot t_{\text{chunk}}
\approx
\left\lceil \frac{S_{\text{input}}}{C} \right\rceil \cdot t_{\text{chunk}}
$$

Since $t_{\text{chunk}} \approx t_{\text{prefill}} / N_{\text{chunks}}$ for the compute-dominated terms (FLOPs split equally), the total prefill time is approximately the same as unchunked in the compute-bound regime:

$$
t_{\text{prefill,chunked}} \approx t_{\text{prefill,unchunked}}
\quad (\text{compute-bound, neglecting startup overheads})
$$

> **Note on $C^2$ term:** The attention score term scales as $C^2$ per chunk, summing to $N_{\text{chunks}} \cdot C^2 = S_{\text{input}} \cdot C$, which is **smaller** than the unchunked $S_{\text{input}}^2$. When the $S^2$ term is significant (long contexts without FlashAttention), chunking **reduces total FLOPs**. With FlashAttention, the $S^2$ traffic reduction already applies, so the advantage is smaller.

TTFT for a chunked-prefill request:

$$
\boxed{
TTFT_{\text{chunked}}
\approx
t_{\text{prefill,chunked}}
+
t_{\text{pipeline,warmup}}
}
$$

Because the prefill is spread over $N_{\text{chunks}}$ scheduler iterations, the observed TTFT from the requesting client is the time until all chunks complete and the first token is generated — longer than unchunked TTFT for the same request in isolation.

---

## 5.3 Throughput Benefit and Head-of-Line Blocking Reduction

The core benefit of chunked prefill is **eliminating head-of-line blocking** in a continuous-batching serving system.

### Without chunked prefill

A long prefill with $S_{\text{input}} = 16384$ tokens on a single GPU takes $t_{\text{prefill}} \approx F_{\text{prefill}} / R_{\text{GPU}}$ — potentially tens to hundreds of milliseconds. All decode-phase requests that arrive during this time must wait, increasing their TPOT.

### With chunked prefill

Each chunk of $C$ tokens occupies approximately $t_{\text{chunk}} = (C / S_{\text{input}}) \cdot t_{\text{prefill}}$ compute time. Decode iterations for other requests execute between chunks. The maximum TPOT penalty for concurrent decode requests is bounded by $t_{\text{chunk}}$ rather than the full $t_{\text{prefill}}$.

**Trade-off:**

| Metric | Unchunked prefill | Chunked prefill ($C$ tokens) |
|--------|-------------------|------------------------------|
| Per-request TTFT | Low (one pass) | Higher (spread over iterations) |
| Concurrent decode TPOT | High spikes during long prefill | Bounded by $t_{\text{chunk}}$ |
| System throughput | Lower (head-of-line blocking) | Higher (decode and prefill interleaved) |
| Chunk size selection | N/A | Tune $C$ to balance TTFT and TPOT SLOs |

See [SARATHI] for empirical validation of chunked prefill scheduling and its effect on P99 TTFT and decode latency under mixed workloads.

---

<div style="page-break-before: always;"></div>

# 6. Disaggregated Prefill

In a **disaggregated prefill** architecture [DISAGG-PREFILL], prefill and decode run on **separate physical clusters**, each optimized for its dominant workload:

- **Prefill cluster**: high peak FLOPs, compute-optimal configuration (can tolerate lower HBM bandwidth per FLOP since prefill is GEMM-bound).
- **Decode cluster**: high HBM bandwidth, memory-bandwidth-optimal configuration (large HBM capacity for KV cache, fast HBM for GEMV throughput).

After prefill completes on the prefill cluster, the KV cache must be transferred to the decode cluster before decoding can begin.

---

## 6.1 Architecture and Motivation

In a **co-located** system (prefill and decode on the same cluster), a single GPU must balance two competing requirements: high HBM bandwidth for decode and high compute throughput for prefill. These requirements can conflict — e.g., a chip with wide HBM but lower FLOPs/byte favors decode, while a chip with narrow HBM but high FLOPs/byte favors prefill.

Disaggregation allows independent hardware optimization:

- Prefill nodes may run at higher utilization and higher batch size (less constrained by KV cache HBM capacity).
- Decode nodes can hold larger KV caches in HBM (more memory available since prefill KV is not stored here during the prefill phase).
- The two clusters can scale independently: a workload with long prompts and short outputs benefits from more prefill nodes; a chatbot with short prompts and long outputs benefits from more decode nodes.

See [DISAGG-PREFILL] (DistServe) for a full system design and goodput analysis.

---

## 6.2 KV Cache Transfer Latency

After the prefill pass completes on the prefill cluster, the resulting KV cache entries for all $S_{\text{input}}$ tokens must be transferred to the decode cluster.

### KV cache volume

The total KV cache produced by a complete prefill pass (all $L$ layers, both keys and values, all $S_{\text{input}}$ tokens):

$$
M_{\text{KV,total}}
=
2 \cdot L \cdot S_{\text{input}} \cdot H_{kv} \cdot b
\quad \text{(bytes)}
$$

This follows directly from the per-layer KV cache expression in `modeling.tpot.md §1.3`, summed over all $L$ layers. With GQA [GQA], $H_{kv} = n_{kv} \cdot d_{\text{head}} \ll H$, substantially reducing the transfer volume compared to MHA ($H_{kv} = H$).

**Example (LLaMA-3 70B):** $L = 80$, $n_{kv} = 8$, $d_{\text{head}} = 128$, $H_{kv} = 1024$, $S_{\text{input}} = 4096$, $b = 2$ (bf16):

$$
M_{\text{KV,total}}
=
2 \times 80 \times 4096 \times 1024 \times 2
\approx 1.34 \text{ GB}
$$

### Shard-aware transfer

If the prefill cluster uses TP/SP partitioning, each device holds only a shard of the KV cache. The transfer can be coordinated so that each prefill device sends directly to the corresponding decode device (same TP/SP rank), transferring only the local shard:

$$
M_{\text{KV,shard}}
=
\frac{M_{\text{KV,total}}}{TP \cdot SP}
$$

This reduces per-link transfer volume proportionally to the parallelism degree, though the aggregate cluster-level bandwidth requirement remains $M_{\text{KV,total}}$.

> **Scope caveat:** The shard-aware point-to-point formula above assumes the prefill and decode clusters share the **same $(TP, SP)$ partition**. In practice, prefill clusters often run at lower TP (compute-optimal, large-GEMM regime) while decode clusters run at higher TP/SP (memory-optimal, KV-sharded regime). When the two clusters have mismatched parallelism topologies, a direct rank-to-rank transfer is not possible; the transfer layer must instead perform an **all-to-all resharding** across the network to re-shard the KV state from the prefill layout to the decode layout [DISAGG-PREFILL]. The bandwidth model in this case is better approximated by a collective scatter cost rather than $M_{\text{KV,total}} / (TP \cdot SP)$ per link.

### Transfer latency ($\alpha$–$\beta$ model)

Using the $\alpha$–$\beta$ model [ALPHA-BETA] for the inter-cluster interconnect:

$$
\boxed{
t_{\text{KV,transfer}}
=
\alpha_{\text{inter}}
+
\frac{M_{\text{KV,total}}}{B_{\text{eff,inter}}}
}
$$

where:
- $\alpha_{\text{inter}}$ — inter-cluster link startup latency (typically $\mu$s-scale for InfiniBand HDR/NDR or NVLink-C2C; negligible vs. transfer time for large KV caches).
- $B_{\text{eff,inter}}$ — effective inter-cluster bandwidth per transfer stream. For InfiniBand HDR (200 Gb/s per port), $B_{\text{eff,inter}} \approx 20$ GB/s unidirectional. NVLink-C2C (e.g., Grace-Hopper NVLink-C2C) offers $\sim 900$ GB/s.

**Example (continued):** With InfiniBand ($B_{\text{eff,inter}} = 20$ GB/s):

$$
t_{\text{KV,transfer}} \approx \frac{1.34 \text{ GB}}{20 \text{ GB/s}} \approx 67\text{ ms}
$$

This transfer latency is comparable to or can exceed the prefill compute time for moderate input lengths, making inter-cluster bandwidth a critical design constraint for disaggregated serving.

---

## 6.3 TTFT for Disaggregated Systems

Let the prefill cluster have hardware parameters $R_{\text{GPU,pre}}$ and $B_{\text{eff,mem,pre}}$, and the decode cluster have $R_{\text{GPU,dec}}$ and $B_{\text{eff,mem,dec}}$. The three phases of TTFT are:

### Phase 1: Prefill on the prefill cluster

$$
t_{\text{prefill}}
=
t_{\text{prefill,local,pre}}
+
\max\!\left(0,\; t_{\text{prefill,comm,pre}} - \rho_{\text{pre}}\, t_{\text{prefill,local,pre}}\right)
+
t_{\text{pipeline,warmup,pre}}
$$

where all subscripts "pre" refer to the prefill cluster's hardware and parallelism configuration (which may differ from the decode cluster).

### Phase 2: KV cache transfer

$$
t_{\text{KV,transfer}}
=
\alpha_{\text{inter}}
+
\frac{M_{\text{KV,total}}}{B_{\text{eff,inter}}}
$$

### Phase 3: Pipeline warmup on the decode cluster

When the first decoded token must traverse the decode cluster's pipeline from stage 1 to stage $PP_{\text{dec}}$:

$$
t_{\text{pipeline,warmup,dec}}
=
(PP_{\text{dec}} - 1) \cdot t_{\text{stage,dec}}
$$

### Full TTFT (disaggregated)

$$
\boxed{
TTFT_{\text{disagg}}
=
t_{\text{prefill}}
+
t_{\text{KV,transfer}}
+
t_{\text{pipeline,warmup,dec}}
}
$$

The three terms have different system knobs:

| Term | Reduced by |
|------|-----------|
| $t_{\text{prefill}}$ | More/faster prefill nodes; higher TP for larger batch GEMM; FP8 quantization |
| $t_{\text{KV,transfer}}$ | Higher inter-cluster bandwidth (NVLink-C2C, InfiniBand NDR); KV quantization; GQA [GQA] |
| $t_{\text{pipeline,warmup,dec}}$ | Smaller $PP_{\text{dec}}$; prefetching KV before decode cluster pipeline clears |

### Comparison: Co-located vs. Disaggregated TTFT

| Architecture | Prefill location | KV transfer | Decode cluster startup |
|-------------|-----------------|-------------|----------------------|
| **Co-located** | Same cluster as decode | None ($t_{\text{KV,transfer}} = 0$) | $t_{\text{pipeline,warmup}}$ |
| **Disaggregated** | Dedicated prefill cluster | $\alpha_{\text{inter}} + M_{\text{KV}} / B_{\text{eff,inter}}$ | $t_{\text{pipeline,warmup,dec}}$ |

Co-located TTFT is lower for individual requests (no KV transfer), but disaggregated systems achieve better **system goodput** under mixed workloads because prefill no longer interferes with ongoing decode iterations [DISAGG-PREFILL].

---

<div style="page-break-before: always;"></div>

# Symbol Index (new in this document)

The following symbols are introduced in this document and extend `modeling.notation.md §11`.

| Symbol | Definition |
|--------|-----------|
| $S_{\text{input}}$ | Input sequence length (number of tokens in the prefill pass) |
| $B_{\text{prefill}}$ | Number of requests batched together in a single prefill pass |
| $C$ | Chunk size: number of tokens per chunk in chunked prefill |
| $N_{\text{chunks}}$ | Number of chunks: $\lceil S_{\text{input}} / C \rceil$ |
| $F_{\text{prefill,device}}$ | Total prefill FLOPs per device (all layers on this PP stage) |
| $F_{\text{layer,prefill}}$ | Per-layer prefill FLOPs (full-sequence GEMM + $S^2$ attention) |
| $F_{\text{proj,prefill}}$ | Q/K/V/O projection FLOPs for prefill: $(2H^2 + 6HH_{kv}) S_{\text{input}}$ |
| $F_{\text{score,prefill}}$ | Attention score FLOPs for prefill: $2 S_{\text{input}}^2 H_{kv}$ |
| $F_{\text{value,prefill}}$ | Value application FLOPs for prefill: $2 S_{\text{input}}^2 H_{kv}$ |
| $F_{\text{ffn,prefill}}$ | FFN FLOPs for prefill: $4 H I_{\text{eff}} S_{\text{input}}$ |
| $T_{\text{KV,write,device}}$ | HBM write traffic for KV cache during prefill |
| $t_{\text{prefill,compute}}$ | Prefill compute time: $F_{\text{prefill,device}} / R_{\text{GPU}}$ |
| $t_{\text{prefill,mem}}$ | Prefill memory time: $T_{\text{prefill,device}} / B_{\text{eff,mem}}$ |
| $t_{\text{prefill,local}}$ | Prefill roofline local time: $\max(t_{\text{prefill,compute}}, t_{\text{prefill,mem}})$ |
| $t_{\text{prefill,comm}}$ | Total prefill communication time (TP/EP/SP/PP collectives) |
| $t_{\text{chunk}}$ | Latency of a single chunked-prefill iteration (overlap-adjusted) |
| $t_{\text{pipeline,warmup}}$ | Pipeline fill time: $(PP-1) \times t_{\text{stage}}$ |
| $TTFT$ | Time To First Token (single-request, co-located) |
| $TTFT_{\text{disagg}}$ | Time To First Token for disaggregated prefill architecture |
| $M_{\text{KV,total}}$ | Total KV cache bytes produced by one prefill pass: $2 L S_{\text{input}} H_{kv} b$ |
| $t_{\text{KV,transfer}}$ | KV cache transfer latency from prefill to decode cluster |
| $B_{\text{eff,inter}}$ | Effective inter-cluster interconnect bandwidth (bytes/s) |
| $\alpha_{\text{inter}}$ | Inter-cluster link startup latency |
| $S_{\text{input}}^{\star}$ | Compute-bound crossover: prefill is compute-bound for $S_{\text{input}} > S_{\text{input}}^{\star}$ |
| $R_{\text{ridge}}$ | Device ridge point: $R_{\text{GPU}} / B_{\text{eff,mem}}$ (FLOPs/byte) [ROOFLINE] |

---

# References

This document cites the following references from `modeling.references.md`:

- **[FA1]** Dao et al. (2022) — FlashAttention: tiled attention for HBM traffic reduction during prefill.
- **[FA2]** Dao (2023) — FlashAttention-2: improved parallelism across heads and sequence dimensions.
- **[GQA]** Ainslie et al. (2023) — Grouped-query attention; $H_{kv} = n_{kv} \cdot d_{\text{head}}$; KV volume reduction.
- **[MEGATRON]** Shoeybi et al. (2019) — TP column/row-parallel linear layers; prefill GEMM sharding.
- **[MEGATRON3]** Narayanan et al. (2021) — PP schedules; SP; pipeline warmup and 1F1B.
- **[SARATHI]** Agrawal et al. (2023) — Chunked prefill scheduling; head-of-line blocking reduction.
- **[DISAGG-PREFILL]** Zhong et al. (2024) — DistServe: disaggregated prefill–decode; KV transfer latency.
- **[ROOFLINE]** Williams et al. (2009) — Roofline model; OI; ridge point; compute vs. memory bound.
- **[ALPHA-BETA]** Hockney (1994) — $\alpha$–$\beta$ collective latency model.
- **[H100-SPEC]** NVIDIA (2022) — H100 SXM5 hardware specs: $R_{\text{GPU}}$, $B_{\text{eff,mem}}$, ridge point values.
