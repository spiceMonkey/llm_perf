# Decode and Time-Per-Output-Token (TPOT) Performance Model

**Author:** Yue Lu  
**Date:** November 2025  

**Keywords:**  
LLM inference, Transformer, parallelism, tensor parallelism, expert parallelism, sequence parallelism, pipeline parallelism, distributed systems, KV cache, collective communication, latency, throughput, cluster topology, performance modeling

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Memory Footprint](#1-memory-footprint)
  - [1.0 Parameter Definitions (P vs W)](#10-parameter-definitions-p-vs-w)
  - [1.1 Model Parameter Memory](#11-model-parameter-memory)
  - [1.2 Activation Memory (Per-token Working Memory)](#12-activation-memory-per-token-working-memory)
  - [1.3 KV Cache Memory](#13-kv-cache-memory)
  - [1.4 Per-device Memory Footprint After Parallelism Sharding](#14-per-device-memory-footprint-after-parallelism-sharding)

- [2. Memory Traffic During Decoding](#2-memory-traffic-during-decoding)
  - [2.1 Model Parameter Traffic $T_{\theta,\text{device}}$](#21-model-parameter-traffic-t_theta_textdevice)
  - [2.2 Activation Traffic $T_{\text{act,device}}$](#22-activation-traffic-t_textactdevice)
  - [2.3 KV Cache Traffic $T_{\text{KV,device}}$](#23-kv-cache-traffic-t_textkvdevice)
  - [2.4 Total and Effective Traffic](#24-total-and-effective-traffic)
  - [2.5 Static Memory Footprint vs. Memory Traffic](#25-static-memory-footprint-vs-memory-traffic-important-distinction)

- [3. Compute (FLOPs) per Token](#3-compute-flops-per-token)
  - [3.1 Q/K/V and Output Projections](#31-qkv-and-output-projections)
  - [3.2 Attention Scores and Value Application](#32-attention-scores-and-value-application)
  - [3.3 FFN FLOPs (Unified Dense + MoE)](#33-ffn-flops-unified-dense--moe)
  - [3.4 LayerNorm and Elementwise FLOPs](#34-layernorm-and-elementwise-flops)
  - [3.5 Per-Device FLOPs per Layer Under TP, SP, EP, and PP](#35-per-device-flops-per-layer-under-tp-sp-ep-and-pp)
  - [3.6 Prefill FLOPs](#36-prefill-flops)

- [4. Compute vs. Memory Bound (Roofline Model)](#4-compute-vs-memory-bound-roofline-model)
  - [4.1 Operational Intensity (Ops:Byte)](#41-operational-intensity-opsbyte)
  - [4.2 Compute-Bound Time](#42-compute-bound-time)
  - [4.3 Memory-Bound Time](#43-memory-bound-time)
  - [4.4 Local Device Time Bound](#44-local-device-time-bound)

- [5. Communication Time During Decoding](#5-communication-time-during-decoding)
  - [5.1 Pipeline Parallel (PP) Hop](#51-pipeline-parallel-pp-hop)
  - [5.2 Expert Parallel (EP) All-to-All (MoE Dispatch and Combine)](#52-expert-parallel-ep-all-to-all-moe-dispatch-and-combine)
  - [5.3 Tensor Parallel (TP) Communication](#53-tensor-parallel-tp-communication)
  - [5.4 Sequence Parallel (SP) Communication](#54-sequence-parallel-sp-communication)
  - [5.5 Total Communication Time Per Token on a PP Stage](#55-total-communication-time-per-token-on-a-pp-stage)

- [6. End-to-End Latency, Throughput, and Partition Strategy](#6-end-to-end-latency-throughput-and-partition-strategy)
  - [6.1 Model Partition Strategy from HBM Constraints](#61-model-partition-strategy-from-hbm-constraints)
  - [6.2 Local and Networking Per-Token Latency](#62-local-and-networking-per-token-latency)
  - [6.3 TPS and TTPS — Pipeline Throughput](#63-tps-and-ttps--pipeline-throughput)
  - [6.4 Batch-Size Scaling and Throughput–Latency Tradeoff](#64-batch-size-scaling-and-throughputlatency-tradeoff)

---

<div style="page-break-before: always;"></div>

# 1. Memory Footprint

This section defines parameter sizes and memory footprint for a given set of model parameters. The memory footprint include those from model weights, per-token activation/working memories, and KV cache. We avoid model-wide parameter aggregation here and instead focus on **per-layer** quantities, because pipeline-parallel stages own disjoint sets of layers. All parameter definitions assume stored precision of $b$ bytes per element (e.g., bf16 = 2 bytes).

---

## 1.0 Parameter Definitions (P vs W)

For any weight matrix $W$, the parameter count is

$$
P(W) = \text{number of elements in } W
$$

Total stored parameter memory is

$$
M_\theta = P \cdot b
$$

---

## 1.1 Model Parameter Memory

### Embedding and LM Head Parameters

Modern LLM architectures (GPT-3/4, LLaMA families, PaLM, Qwen, DeepSeek, etc.) typically use embedding dimension $E = H$, so all internal projections operate on vectors in $\mathbb{R}^H$.

For the token embedding:

$$
W_{\text{emb}} \in \mathbb{R}^{V \times H},
\quad
P_{\text{emb}} = V H
$$

If the LM head is tied with embeddings, i.e. $W_{lm}=W_{emb}^T$ so that $W_{emb}$ can be re-used:

$$
P_{\text{lm}} = 0
$$

If untied:

$$
W_{\text{lm}} \in \mathbb{R}^{H \times V},
\quad
P_{\text{lm}} = V H
$$

### Attention Parameters

For hidden size $H$, head dimension $d_{\text{head}}$, and KV dimension $H_{kv} = n_{kv} \, d_{\text{head}}$ (supporting grouped-query attention where $n_{kv} \le n_q$ [GQA] and multi-query attention where $n_{kv} = 1$ [MQA]):

- $W_Q \in \mathbb{R}^{H \times H}$  
- $W_K \in \mathbb{R}^{H \times H_{kv}}$  
- $W_V \in \mathbb{R}^{H \times H_{kv}}$  
- $W_O \in \mathbb{R}^{H \times H}$  

In GQA, each of the $n_q$ query heads produces a $d_{\text{head}}$-dimensional output; the concatenated result across all query heads is $n_q \times d_{\text{head}} = H$-dimensional, regardless of how many KV heads are used. The output projection therefore always maps $\mathbb{R}^H \to \mathbb{R}^H$.

Parameter counts:

$$
P_Q = H^2, \qquad
P_K = H H_{kv}, \qquad
P_V = H H_{kv}, \qquad
P_O = H^2
$$

We define attention parameters:

$$
P_{\text{attn}} = P_Q + P_K + P_V + P_O
$$

### Unified FFN Parameters (Dense or MoE)

Each transformer layer contains an FFN module. In modern LLM architectures, this FFN is almost
always implemented as a gated MLP (GeLU/SiLU/GLU/SwiGLU–style) with **three** linear projections:

- a gate projection,
- an “up” projection (expansion),
- a “down” projection (contraction).

> **Convention:** This document assumes **gated FFN (SwiGLU/GeGLU style) throughout**, giving three weight matrices per FFN block. Standard (non-gated) FFN uses two matrices (up + down), yielding $2HI$ parameters. The gated form is used by LLaMA, Qwen, DeepSeek, and most modern LLMs.

For a hidden size $H$ and FFN intermediate dimension $I$, this yields an FFN parameter count

$$
P_{\text{FFN}} = 3 H I N_{\text{exp}}.
$$

Here $N_{\text{exp}}$ denotes the number of experts **per layer**:

- **For a dense MLP model:** $I = I_{\text{dense}}, \; N_{\text{exp}} = 1$.
- **For a MoE model:** $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$.

These two model cases are mutually exclusive **per layer**.

### LayerNorm parameters

LayerNorm or RMSNorm contain $\mathcal{O}(H)$ parameters (scale and optional bias). These are negligible compared to attention and FFN weights and are omitted in scaling formulas.

---

## 1.2 Activation Memory (Per-token Working Memory)

During autoregressive decoding, the model processes **one token at a time**. As a result, the only activations that need to be stored in memory are the **temporary, layer-local working buffers** used in the forward pass of the *current token*.  

These activations are **not reused** across layers or across tokens and therefore are:

- **not dependent on sequence length** $S$,
- **proportional to batch size** $B$ (each sequence in the batch needs its own working buffers),
- **not EP- or TP-sharded**,  
- and **extremely small** relative to model parameter memory (Section 1.1) and KV cache memory (Section 1.3).

Below we account for all activation tensors in a model layer that must be alive **concurrently** for one decoding token.

### Q, K, V projections

For the current hidden state $h \in \mathbb{R}^{H}$, the layer computes:

- $Q \in \mathbb{R}^{H}$
- $K \in \mathbb{R}^{H_{kv}}$
- $V \in \mathbb{R}^{H_{kv}}$

This contributes:

$$
H + 2H_{kv}
$$

### Attention score accumulation buffer (FlashAttention-like kernels)

Attention score computation normally requires a temporary buffer. FlashAttention-style fused kernels avoid storing full $S$-length score vectors and instead use a **single internal workspace** of size $H$ during streaming softmax.

This adds:

$$ + H
$$

### Attention output buffer

After applying attention weights to $V$ and combining across heads, we form the attention output $O_{\text{attn}} \in \mathbb{R}^{H}$.

This output must exist before the output projection is applied, contributing:

$$+ H$$

### FFN working buffer

Following attention and normalization, the FFN block needs at least one temporary buffer of size $H$ to hold either the FFN input or the FFN output before residual addition. Even with kernel fusion, this buffer cannot always overlap with the attention intermediates.

This adds:

$$+ H$$

Summing all simultaneously required buffers per sequence:

$$
P_{\text{act}} = 4H + 2H_{kv}
$$

In bytes, for a batch of $B$ sequences:
$$
M_{\text{act,layer}} = B \cdot (4H + 2H_{kv}) \cdot b
$$

This footprint is **small** compared to parameter memory and KV cache, even at large batch sizes. For example, at $B=128$, $H=8192$, $H_{kv}=1024$, $b=2$: $M_{\text{act,layer}} \approx 9$ MB per layer — negligible against hundreds of GB of parameter memory.

---

## 1.3 KV Cache Memory

Section 1.1 described the memory footprint of model parameters (static), and Section 1.2 covered the
activation memory required during decoding (per-token, dynamic). This section describes the **KV cache**, which is a *runtime* structure generated during the **pre-fill phase**, when the model processes the entire input sequence of length $S$.

During pre-fill, each attention layer produces:

- one key vector of dimension $H_{kv}$,
- one value vector of dimension $H_{kv}$,

for each input token. Because decoding adds only one new token at a time, the vast majority of KV memory comes from **pre-fill**, not decoding.

For a single attention layer, the KV cache consists of:

- Keys: $K \in \mathbb{R}^{S \times H_{kv}}$  
- Values: $V \in \mathbb{R}^{S \times H_{kv}}$

The KV cache size scales with $H_{kv} = n_{kv} d_{\text{head}}$; using grouped-query attention ($n_{kv} < n_q$) [GQA] or multi-query attention ($n_{kv} = 1$) [MQA] directly reduces this footprint.

Thus, the total KV elements for one layer are:

$$
P_{KV, layer} = S \cdot (2H_{kv}) = 2 S H_{kv}
$$

In bytes:

$$
M_{\mathrm{KV,layer}} =
2 S H_{kv} \cdot b
$$

This is **static** once pre-fill is complete; decoding contributes only an additional $2H_{kv} b$ per
generated token, which is negligible relative to the full cache.

---

## 1.4 Per-device Memory Footprint After Parallelism Sharding

So far we've completed the all the memory footprint estimation for a model layer. When we introduce different parallelism schemes, some of these memories would be sharded by one or more of these parallelism dimensions, resulting in a somewhat complicated memory aggregateion per device. We now describe how these parameters are distributed across devices under PP/EP/TP/SP, and then derive simple modeling approximations.

### Per-device Parameter Memory (TP, EP, PP)

Each transformer layer has two parameter groups:

- **Attention parameters** $P_{\text{attn}}$, that are sharded by **tensor parallelism (TP)** only.
- **FFN parameters** $P_{\text{FFN}}$ are sharded by **TP and EP**.
  Dense FFN layers are treated as a special case with $EP = 1$, so the same formula applies to both dense and MoE FFNs.

For a *single layer* on a device, the stored parameter memory is

$$
M_{\theta,\text{layer}} =
\frac{P_{\text{attn}}\, b}{TP}
\;+\;
\frac{P_{\text{FFN}}\, b}{TP \cdot EP}
$$

Pipeline parallelism (PP) assigns **disjoint sets of layers** to different stages. Let $L_s$ be the set of layers that live on PP stage $s$, and let $M_{\theta,\text{layer},\ell}$ be the per-layer memory from the expression above.

Excluding embeddings and LM head, the parameter memory per device on PP stage $s$ is

$$
M_{\theta,\text{layers}}^{(s)} =
\sum_{\ell \in L_s}
M_{\theta,\text{layer},\ell}
$$

Embeddings and LM head appear only on two stages:

- **Intermediate PP stages** (no embedding / LM head):
  $$
  M_{\theta,\text{device}}^{(\text{mid})} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  $$

- **First PP stage** (with token embedding):
  $$
  M_{\theta,\text{device}}^{(1)} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  \;+\;
  \frac{P_{\text{emb}}\, b}{TP}
  $$

- **Final PP stage** (with LM head):
  $$
  M_{\theta,\text{device}}^{(\text{PP})} =
  M_{\theta,\text{layers}}^{(\text{mid})}
  \;+\;
  \frac{P_{\text{lm}}\, b}{TP}
  $$

If each intermediate PP stage holds approximately $L/PP$ layers of similar size, and we use representative per-layer values $P_{\text{attn}}$ and $P_{\text{FFN}}$, then

$$
M_{\theta,\text{device}}^{(\text{mid})} =
\frac{L}{PP} M_{\theta,\text{layer}} =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
+
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
\right) b
$$

For capacity planning we use a worst-case PP stage budget, adding one $\frac{VH}{TP}b$ term to account for embedding/LM weights residing on boundary stages. Intermediate stages are slightly smaller. Therefore:

$$
M_{\theta,\text{device}} =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
\right) b
+\frac{VH}{TP} b
$$

For a **dense MLP model**: $I = I_{\text{dense}}$, and $N_{\text{exp}} = EP = 1$.

For a **MoE model**: $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$ and $EP \ge 1$.

### Mixed MoE/Dense Architectures

Many modern architectures use a **mixed** design where only some layers are MoE (e.g., alternating dense and MoE layers, or MoE only in deeper layers). For such models, the parameter memory must be computed separately for dense and MoE layers:

$$
M_{\theta,\text{device}} =
M_{\theta,\text{dense}} + M_{\theta,\text{moe}} + \frac{VH}{TP} b
$$

where:

$$
M_{\theta,\text{dense}} =
\frac{L_{\text{dense}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I_{\text{dense}}}{TP}
\right) b
$$

$$
M_{\theta,\text{moe}} =
\frac{L_{\text{moe}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I_{\text{moe}} N_{\text{exp}}}{TP \cdot EP}
\right) b
$$

Note that dense layers use $EP = 1$ and $N_{\text{exp}} = 1$ implicitly, while MoE layers use the specified $EP$ and $N_{\text{exp}}$ values.

### Per-device Activation Memory

The per-layer working activation footprint for a batch of $B$ sequences is $B \cdot (4H + 2H_{kv})$.

In standard sequential layer execution, only one layer's activation buffers are live at any time —
each layer's output overwrites the previous layer's buffer before the next layer begins. Therefore,
the per-device activation memory during decoding is simply one layer's worth:

$$
M_{\text{act,device}} =
B \cdot (4H + 2H_{kv}) \, b
$$

> **Note:** The $L/PP$ multiplier is *not* applied here because layers execute sequentially; earlier
> layers' activations are not retained while later layers execute. A $2\times$ factor could apply
> when double-buffering for PP communication overlap, but this is negligible in practice and omitted.

### Per-device KV Cache Memory

Only attention layers produce KV cache.  
The KV cache is sharded only across:

- **TP** — splits the channel dimension $H_{kv}$,  
- **SP** — splits the sequence dimension $S$.

EP and DP do not modify KV layout; PP only affects how many layers are assigned to a stage.

Therefore, the **per-device** KV memory footprint is:

$$
M_{\mathrm{KV,device}} =
\frac{L}{PP}
\frac{
    M_{\mathrm{KV,layer}}
}{
    TP \cdot SP
} =
\frac{L}{PP}
\frac{
     (2 S H_{kv}) b
}{
    TP \cdot SP
}
$$

which:

- For long-context inference (e.g., $S \in [16\mathrm{k}, 128\mathrm{k}]$), $S(2H_{kv})b$ is large enough that KV can exceed parameter memory unless aggressively reduced through TP and SP.
- Each decoded token adds only $2H_{kv} b$, which is negligible compared to the pre-fill KV footprint.

### Total Per-device Static Memory Footprint

Summing all the memory footprint we derive from section 1.1 - 1.4 together, we can therefore get the "minimum required" memory size for the device to host the model under a particular PP/EP/TP/SP partition.

$$
M_{\text{device}}^{\text{total}} = M_{\theta,\text{device}} + M_{\text{act,device}} + M_{\text{KV,device}}
$$

For **uniform architectures** (all dense or all MoE):

$$
M_{\text{device}}^{\text{total}} =
\frac{L}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{TP}
+
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
\right] b
+
B \cdot \frac{L}{PP} \cdot \frac{2 S H_{kv}}{TP \cdot SP} \cdot b
+
B(4H + 2H_{kv}) b
+\frac{VH}{TP} b
$$

For a dense model: $I = I_{\text{dense}}, \text{ } N_{\text{exp}}=EP=1$

And for a MoE model: $I = I_{\text{moe}}$

For **mixed MoE/dense architectures** (where $L_{\text{moe}} < L$):

$$
\begin{aligned}
M_{\text{device}}^{\text{total}} = \;&
\frac{L_{\text{dense}}}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{TP}
+
\frac{3 H I_{\text{dense}}}{TP}
\right] b \\
+\;&
\frac{L_{\text{moe}}}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{TP}
+
\frac{3 H I_{\text{moe}} N_{\text{exp}}}{TP \cdot EP}
\right] b \\
+\;&
B \cdot \frac{L}{PP} \cdot
\frac{2 S H_{kv}}{TP \cdot SP}
\cdot b \\
+\;&
B(4H + 2H_{kv}) b
+\frac{VH}{TP} b
\end{aligned}
$$

Note: Activation memory ($B(4H+2H_{kv})b$, one layer at a time) and KV cache apply to all $L$ layers. KV cache uses the total layer count $L/PP$ since all layers' KV tensors are concurrently resident.

---

<div style="page-break-before: always;"></div>

# 2. Memory Traffic During Decoding

Section 1 quantified the *static* memory footprint of the model — how many bytes of parameters, KV cache, and activations must **fit** in device HBM.

This section instead focuses on **memory traffic per generated token**, i.e., the bytes that must flow between HBM and compute cores *during decoding*. This traffic directly determines the memory-bound component of decoding performance (Section 4’s roofline model).

**Crucial Distinction for Decoding:**
In autoregressive decoding, each step generates one new token per active sequence. Unlike prefill — where weights are loaded once and reused across $S_{\text{input}}$ tokens within a single pass — decoding reloads the **entire model weight matrix** from HBM **every step**. Increasing the decode batch $B$ amortizes this fixed weight-load cost across $B$ tokens (per-token weight traffic = $T_\theta/B$), but does not change the fact that weights are reloaded each step.

Therefore, optimizations like FlashAttention or Fused-MLP do **not** reduce weight traffic; they only reduce the traffic of intermediate activations.

---

## 2.1 Model Parameter Traffic $T_{\theta,\text{device}}$

Following Section 1, we use:

- $P_{\text{attn}}$: Q/K/V/O projection parameters  
- $P_{\text{FFN}}$: dense FFN (or non-expert MoE core) parameters  

We drop $P_{\text{emb}}$ and $P_{\text{lm}}$ when modeling steady-state decoding, assuming they reside on boundary PP stages and do not bottleneck the central pipeline.

### Attention parameter traffic

Because $P_{\text{attn}}$ is defined **per layer**, a PP stage with $L_{\text{stage}} = L/PP$ layers has $L_{\text{stage}} P_{\text{attn}}$ attention parameters. These are sharded across $TP$ devices.

Since every weight must be read per token:

$$
T_{\theta,\text{attn}} =
\frac{L}{PP}
\cdot
\frac{P_{\text{attn}} \, b}{TP}
$$

### FFN parameter traffic

Similarly, the FFN parameters $P_{\text{FFN}}$ are sharded by both **TP** and **EP**. Although fused kernels (e.g., FlashMLP) avoid writing intermediate activations (like the gate tensor) to HBM, they still require reading the gate, up, and down projection weights fully.

$$
T_{\theta,\text{FFN}} =
\frac{L}{PP}\;
\frac{P_{\text{FFN}}}{TP \cdot EP}\; b
$$

### Final parameter-traffic expression

Combining these terms:

$$
T_{\theta,\text{device}}
\approx
\frac{L}{PP}
\left(
  \frac{P_{\text{attn}}}{TP}
  +
  \frac{P_{\text{FFN}}}{TP \cdot EP}
\right) b =
\frac{L}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
\right) b
$$

**For a dense MLP model:** $I = I_{\text{dense}}, \text{ } N_{\text{exp}} = EP = 1$

**And for a MoE model:** $I = I_{\text{moe}}$

### Mixed MoE/Dense Architectures

For mixed architectures, parameter traffic is computed separately for dense and MoE layers:

$$
T_{\theta,\text{device}} =
T_{\theta,\text{dense}} + T_{\theta,\text{moe}}
$$

where:

$$
T_{\theta,\text{dense}} =
\frac{L_{\text{dense}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I_{\text{dense}}}{TP}
\right) b
$$

$$
T_{\theta,\text{moe}} =
\frac{L_{\text{moe}}}{PP}\;
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I_{\text{moe}} N_{\text{exp}}}{TP \cdot EP}
\right) b
$$

---

## 2.2 Activation Traffic $T_{\text{act,device}}$

Section 1 showed that the per-layer activation footprint for a single decoding token is small. However, without optimization, the traffic to read/write these activations—especially the $S \times S$ attention scores—would be massive ($O(S^2)$).

### The Role of FlashAttention

**FlashAttention** [FA1, FA2] avoids materializing the $S \times S$ score matrix in HBM by streaming the tiled attention computation through on-chip SRAM. More precisely: Q, K, V reads remain $O(SH)$; the $O(S^2 d^2 / M)$ score matrix IO (per [FA1] Theorem 2, where $d = d_{\text{head}}$ and $M$ is SRAM size) is reduced to $O(S^2 d / \sqrt{M})$ via tiling, compared to $O(S^2 d)$ for standard attention. For large $S$ and modern GPU SRAM sizes, this makes the $O(S^2)$ term negligible and leaves KV reads as the dominant activation traffic.

Because FlashAttention drastically reduces the score matrix traffic, the residual activation traffic (hidden-state loads/stores, FFN buffers) is $O(H)$ per layer — negligible compared to the weight and KV cache terms for large models. We drop $T_{\text{act,device}}$ from the traffic model here. Residual kernel-level activation overhead is treated as an empirical correction in `framework.md`.

---

## 2.3 KV Cache Traffic $T_{\text{KV,device}}$

KV cache must be **fully read** for each new token to compute attention against the history.
For large $S$, the write term (appending the new token) is negligible compared to reading the history.

The per-layer KV access is approximately:

$$
T_{\text{KV,layer}}
\approx
2 S H_{kv} \, b
$$

Consistent with Section 1, KV is sharded by **TP** (channel/head dimension) and **SP** (sequence dimension). Thus each device only sees a $\frac{1}{TP \cdot SP}$ shard of this per-layer traffic.

For a PP stage with $L/PP$ layers, the **per-device KV traffic per token** is:

$$
T_{\text{KV,device}}
\approx
\frac{L}{PP}
\cdot
\frac{2 S H_{kv} \, b}{TP \cdot SP}
$$

FlashAttention does **not** reduce this term: keys and values from history must always be loaded to compute the current token's attention, regardless of tiling strategy.

---

## 2.4 Total and Effective Traffic

Combining the expressions derived in Sections 2.1–2.3 (with activation traffic dropped as negligible), the **effective** total per-token traffic is:

$$
T_{\text{token,device}}^{eff}
\approx
T_{\theta,\text{device}}
+
T_{\text{KV,device}}.
$$

Substituting yields the **final fully expanded expression**:

$$
T_{\text{token,device}}^{eff}
\;\approx\;
\frac{L}{PP}
\left[
\left(
\frac{2H^2 + 2 H H_{kv}}{TP}
+
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
\right) b
+
\frac{2 S H_{kv}}{TP \cdot SP} b
\right]
$$

The first group inside the brackets is **weight traffic** (loaded once per step), and the second term is **KV cache traffic** (per-token KV reads).

---

## 2.5 Static Memory Footprint vs. Memory Traffic (Important Distinction)

Sections 1 and 2 play different roles in the overall performance model:

**Static Memory Footprint (Section 1)** Determines whether a $(DP, PP, EP, TP, SP)$ configuration can *fit* on a device (Capacity Constraint)

$$
M_{\text{device}}^{\text{total}} \le M_{\text{HBM}}
$$

**Memory Traffic (Section 2)** Determines the *bandwidth-limited latency* per decoded token (Bandwidth Constraint)

$$
t_{\text{mem}} =
\frac{T_{\text{token,device}}^{eff}}
     {BW_{\text{mem}}}
$$

This distinction is critical: Section 1 tells us **which parallelism configurations are viable**, while Section 2 tells us **how fast decoding can proceed** for those viable configurations.

---

# 3. Compute (FLOPs) per Token

During inference, the FLOPs required to generate a token depend on the transformer layer structure. We distinguish between:

- **Prefill FLOPs (GEMM-dominant):** $O(S^2)$ across the full input sequence  
- **Decoding FLOPs (GEMV-dominant):** $O(S)$ for one additional token  

This section focuses on **decoding FLOPs**, which determine TPS throughput. All FLOPs below represent **per-token**, **per-layer**, **decoding** FLOPs.

---

## 3.1 Q/K/V and Output Projections

Q, K, and V projections are vector–matrix multiplications of shapes:

- Q: $[1 \times H] \cdot [H \times H]$  
- K: $[1 \times H] \cdot [H \times H_{kv}]$  
- V: $[1 \times H] \cdot [H \times H_{kv}]$  
- Output: $[1 \times H] \cdot [H \times H]$

### Projection FLOPs

$$
F_Q = 2H^2, \qquad
F_K = 2H H_{kv}, \qquad
F_V = 2H H_{kv}, \qquad
F_O = 2H^2.
$$

Where the factor 2 accounts for each multiply-accumulate pair in the standard GEMV convention.

### Total

$$
F_{\text{proj}} = 4H^2 + 4H H_{kv}
$$

If $H_{kv} = H$ (MHA), this reduces to $8H^2$.

## 3.2 Attention Scores and Value Application

During decoding, the newly generated token attends to all $S$ cached tokens in the KV cache for this layer. Conceptually, for each layer we can treat the cached keys and values as:

- $K_{\text{cache}} \in \mathbb{R}^{S \times H_{kv}}$  
- $V_{\text{cache}} \in \mathbb{R}^{S \times H_{kv}}$,

where $H_{kv} = n_{kv} d_{\text{head}}$ is the total KV projection dimension.

### Scores (Q · Kᵀ)

Each of the $n_q$ query heads independently computes a dot product against its corresponding (broadcast) KV head over $S$ cached positions. Per query head: $2 d_{\text{head}} S$ FLOPs. Summed over all $n_q$ query heads:

$$
F_{\text{score}} = 2 \, n_q \, d_{\text{head}} \, S = 2 S H
$$

> **GQA note:** In GQA ($n_{kv} < n_q$), each KV head is shared by $n_q / n_{kv}$ query heads [GQA]. The KV cache **memory** scales with $H_{kv}$ (only $n_{kv}$ unique heads are stored), but the attention **FLOPs** scale with $H$ because every query head independently computes attention scores and value-weighted sums.

### Value application (Attn · V)

After applying softmax to the scores, each of the $n_q$ query heads computes a weighted sum over the $S$ cached values of its corresponding value head. Per head: $2 d_{\text{head}} S$ FLOPs. Total:

$$
F_{\text{value}} = 2 S H
$$

### Total KV attention FLOPs

Combining the two:

$$
F_{\text{attn,KV}} = F_{\text{score}} + F_{\text{value}} = 4 S H
$$

This term captures the **sequence-length-dependent** attention cost during decoding. For MHA ($n_{kv} = n_q$, $H_{kv} = H$), this is numerically identical to the older $4SH_{kv}$ formulation; the distinction matters for GQA/MQA models where $H_{kv} < H$.

## 3.3 FFN FLOPs (Unified Dense + MoE)

To match the parameter definitions in Section 1.1, we express FFN FLOPs using a **unified formulation** that works for both dense FFN layers and MoE layers.

### Dense FFN FLOPs

For a gated FFN (the convention assumed throughout; see §1.1), a dense FFN consists of three GEMVs:

- a gate projection: $H \rightarrow I$,
- an up projection: $H \rightarrow I$, and
- a down (contraction) projection: $I \rightarrow H$.

Each GEMV costs $2HD$ FLOPs, giving:

$$
F_{\text{ffn,dense}} = 6 H I_{dense}
$$

> **Convention note — gated MLPs:** §1.1 counts **three** weight matrices for a gated FFN (gate, up, and down projections), giving $P_{\text{FFN}} = 3HI$ parameters. Each GEMV contributes $2HI$ FLOPs, so the exact total is $6HI$ — which this document uses throughout. The training scaling-law literature ([KAPLAN-SCALING], [CHINCHILLA]) uses $4HI$, which corresponds to a non-gated 2-matrix FFN with expansion ratio $4H$. Since this model targets **inference** of modern gated-MLP architectures (LLaMA, Qwen, DeepSeek, Mistral), we use the exact $6HI$ to ensure accurate compute-bound predictions (prefill TTFT, batched decode at $B > B^*$). With $6HI$ FLOPs and $3HI$ parameters, the FLOP-to-parameter ratio is exactly $2$ for every weight matrix, yielding the clean OI result $\text{OI} = 2/b$ without approximation.

Dense FFN layers always have $EP = 1$.

### MoE FFN FLOPs

For MoE layers, each token is routed to $k$ active experts (top-$k$ gating) [DEEPSPEED-MOE]; in practice $k=1$ [SWITCH] or $k=2$ [MIXTRAL]:

- the **router** is applied to the full hidden vector:
  $$
  F_{\text{router}} = 2 H N_{\text{exp}}
  $$
  where $N_{\text{exp}}$ is the total number of experts.

- for each of the $k$ selected experts for this token, the FFN computation is:
  $$
  F_{\text{expert}} = 6 H I_{\text{moe}}
  $$

Thus the MoE FFN FLOPs for one token in one layer are:

$$
F_{\text{ffn,moe}} =
F_{\text{router}}
+
k \cdot F_{\text{expert}} =
2 H N_{\text{exp}}
+
k (6 H I_{\text{moe}})
$$

### Unified FFN FLOP Term

We now define **effective FFN FLOP parameters**:

- Effective FFN “matrix” dimension:
  $$
  I_{\text{eff}} =
  \begin{cases}
  I_{\text{dense}}, & \text{dense layer}, \\
  k I_{\text{moe}}, & \text{MoE layer},
  \end{cases}
  $$

- Effective router multiplicity:
  $$
  N_{\text{eff}} =
  \begin{cases}
  0, & \text{dense layer}, \\
  N_{\text{exp}}, & \text{MoE layer}.
  \end{cases}
  $$

With these definitions, both dense and MoE FFN FLOPs can be written in a **single unified form**:

$$
F_{\text{ffn}} = 6H I_{\text{eff}} + 2H N_{\text{eff}}
$$

This matches:

- Dense layer:
  $F_{\text{ffn}} = 6H I_{\text{dense}} + 2H \cdot 0 = 6H I_{\text{dense}}$
- MoE layer:
  $F_{\text{ffn}} = 6H (k I_{\text{moe}}) + 2H N_{\text{exp}} = 6H I_{\text{moe}} k + 2H N_{\text{exp}}$

---

## 3.4 LayerNorm and Elementwise FLOPs

LayerNorm, RMSNorm, residual additions, and elementwise ops scale linearly with $H$ and are ~4 orders of magnitude smaller than the dominant FFN FLOPs for large models. We drop $F_{\text{norm}}$ from all per-device expressions. Norm overhead is an empirical correction handled in `framework.md`.

---

## 3.5 Per-Device FLOPs per Layer Under TP, SP, EP, and PP

For a single decoding token, the FLOPs for one transformer layer are:

$$
F_{\text{layer}}
\approx
F_{\text{proj}} + F_{\text{attn,KV}} + F_{\text{ffn}},
$$

where:

- $F_{\text{proj}} = 4H^2 + 4H H_{kv}$ (Section 3.1),
- $F_{\text{attn,KV}} = 4 S H$ (Section 3.2),
- $F_{\text{ffn}}$ is dense or MoE (Section 3.3).

$F_{\text{norm}}$ is dropped per Section 3.4.

To find **per-device FLOPs**, we apply sharding from TP, SP, EP and then multiply by the number of layers on the PP stage.

---

### TP (Tensor Parallelism)

TP shards all **GEMV/GEMM-like** FLOPs using column- and row-parallel linear layers [MEGATRON]:

- projections ($F_{\text{proj}}$),
- attention score/value FLOPs ($F_{\text{attn,KV}}$),
- FFN GEMMs (dense or MoE experts).

Thus:

$$
F_{\text{tensor}}^{\text{device}} = \frac{1}{TP} F_{\text{tensor}}
$$

---

### SP (Sequence Parallelism)

SP shards the **sequence dimension** across $SP$ ranks [MEGATRON3].  
Thus **only** the sequence-dependent FLOPs:

$$
F_{\text{attn,KV}} = 4 S H
$$

are reduced:

$$
F_{\text{attn,KV}}^{\text{device}} =
\frac{1}{TP \cdot SP}
(4 S H)
$$

SP does **not** reduce:

- $F_{\text{proj}}$ (single token),
- $F_{\text{ffn}}$ (dense or MoE),
- router FLOPs.

---

### EP (Expert Parallelism)

EP applies only to MoE layers:

- Router FLOPs use the full hidden state and are **not sharded**.
- Expert FFN GEMMs are sharded across EP:

$$
F_{\text{expert}}^{\text{device}} =
\frac{k}{EP} \,(6 H I_{\text{moe}})
$$

and may also be TP-sharded if FFN GEMMs follow the same path:

- If experts use TP GEMMs: divide by $TP$ again.
- If experts are **only EP-sharded**, do **not** include $1/TP$.

Dense layers always have $EP = 1$.

---

### PP (Pipeline Parallelism)

PP assigns whole layers to stages.  
Each device in a PP stage owns:

$$
\frac{L}{PP} \text{ layers}.
$$

Thus:

$$
F_{\text{token, device}}
\approx
\frac{L}{PP}
\left(
F_{\text{proj}}^{\text{device}} + F_{\text{attn,KV}}^{\text{device}} + F_{\text{ffn}}^{\text{device}}
\right)
$$

### Total Per-device FLOPs

Dropping the negligible $F_{\text{norm}}$ and also substituting everything yields the **final fully expanded expression** per-device FLOPs for a single decoded token:

$$
F_{\text{token,device}}
\;\approx\;
\frac{L}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{TP}
\;+\;
\frac{6H I_{\text{eff}}}{TP \cdot EP}
\;+\;
\frac{4 S H}{TP \cdot SP}
\;+\;
2H N_{\text{eff}}
\right)
$$

For a **dense MLP model**: $I_{\text{eff}} = I_{\text{dense}},\quad N_{\text{eff}} = 0,\quad EP = 1$

For a **MoE model**: $I_{\text{eff}} = k I_{\text{moe}},\quad N_{\text{eff}} = N_{\text{exp}},\quad EP \ge 1$

### Mixed MoE/Dense Architectures

For mixed architectures, FLOPs are computed separately for dense and MoE layers:

$$
F_{\text{token,device}} =
F_{\text{dense,device}} + F_{\text{moe,device}}
$$

**Dense layer FLOPs** (per device, for all dense layers on this PP stage):

$$
F_{\text{dense,device}} =
\frac{L_{\text{dense}}}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{TP}
\;+\;
\frac{6H I_{\text{dense}}}{TP}
\;+\;
\frac{4 S H}{TP \cdot SP}
\right)
$$

Note: Dense layers have no router FLOPs and use $EP = 1$.

**MoE layer FLOPs** (per device, for all MoE layers on this PP stage):

$$
F_{\text{moe,device}} =
\frac{L_{\text{moe}}}{PP}
\left(
\frac{4H^{2} + 4H H_{kv}}{TP}
\;+\;
\frac{6H k I_{\text{moe}}}{TP \cdot EP}
\;+\;
\frac{4 S H}{TP \cdot SP}
\;+\;
2H N_{\text{exp}}
\right)
$$

Note: The router term $2H N_{\text{exp}}$ is unsharded (applied to the full hidden state before expert selection).

---

## 3.6 Prefill FLOPs

Prefill FLOPs are covered in [prefill.md](prefill.md).

---

<div style="page-break-before: always;"></div>

# 4. Compute vs. Memory Bound (Roofline Model)

Sections 2 and 3 derived:

- the **per-token memory traffic** ($T_{\text{token,device}}^{eff}$), including FlashAttention effects.
- the **per-token FLOPs per device** on this PP stage ($F_{\text{token,device}}$), and  

We now combine them using a standard roofline model to determine the **local per-token latency**.

Let:

- $R_{\text{GPU}}$: sustained device compute throughput (FLOPs/s),  
- $BW_{\text{mem}}$: sustained device memory bandwidth (bytes/s).  

Both reflect *sustained* performance, not peak specs.

**Precision-aware compute peak.** Across the system database, ``peak_flops_TF`` stores the **FP16 dense per-chip peak** as the uniform reference. The framework derives the working-precision peak via linear byte-ratio scaling:

$$
R_{\text{GPU}}(b) = \mathrm{peak\_flops\_TF} \cdot \frac{2}{b}
$$

where $b$ is the model's `bytes_per_param` (FP16 = 2, FP8/INT8 = 1, FP4/INT4 = 0.5). For example, on GB200 NVL72 with peak_FP16 = 2250 TF/GPU, an FP4 model gets 9000 TF/GPU and an INT8 model gets 4500 TF/GPU. This matches Hopper / Blackwell / TPU Tensor Core families exactly. **Known limitation**: d-Matrix MXINT4 throughput is 4× MXINT8 (rather than 2×) due to block-sparse acceleration in the INT4 path; the linear-byte rule under-states d-Matrix INT4 / FP4 by 2× on those systems only.

---

## 4.1 Operational Intensity (Ops:Byte)

The **operational intensity** for decoding on this device is [ROOFLINE]:

$$
\text{OI} =
\frac{F_{\text{token,device}}}{T_{\text{token,device}}^{eff}}
\quad \text{(FLOPs per byte)}
$$

High-level interpretation:

- **High OI** → more FLOPs per byte → *compute-bound*  
- **Low OI** → fewer FLOPs per byte → *memory-bound*

This rating is compared to the device’s memory-to-compute ratio (the ridge point):

$$
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
$$

If:

$$
\text{OI} > \frac{R_{\text{GPU}}}{BW_{\text{mem}}}
\quad \Rightarrow \quad \text{compute-bound}
$$

else:

$$
\text{OI} < \frac{R_{\text{GPU}}}{BW_{\text{mem}}}
\quad \Rightarrow \quad \text{memory-bound}
$$

### Dominant-Term Approximation

In practice, the OI is often approximated using only the largest FLOP and traffic terms:

- FLOPs dominated by:
  $$
  \max\left( 2H^2,\; 6H I_{\text{eff}},\; 4 S H/SP \right)
  $$

- Memory traffic dominated by the KV term:
  $$
  \frac{2 S H_{kv}}{TP\cdot SP}\, b
  $$

Thus for long-context decoding (attention FLOPs dominate):

$$
\text{OI} \approx
\frac{4 S H/(TP\cdot SP)}{2 S H_{kv}/(TP\cdot SP)\; b}
= \frac{2H}{H_{kv}\, b}
$$

For MHA ($H_{kv} = H$), this reduces to $2/b$. For GQA models, $H/H_{kv} = n_q/n_{kv}$ amplifies the OI — e.g., for $n_q/n_{kv} = 8$ (LLaMA-3 70B) the OI is $16/b$. Even so, this is far below typical ridge points (~300 FLOPs/byte on H100), so long-context decode remains **memory-bound** in practice.

> **Note on batch size.** This subsection treats the $B = 1$ case to establish the fundamental memory-bound character of decode. The OI generalizes to $\text{OI}(B)$ once batching is introduced, which lifts the operational intensity by amortizing weight reads across $B$ tokens — that derivation, along with the crossover batch size $B^{\star}$, is in §6.

---

## 4.2 Compute-Bound Time

Given the per-token FLOPs on this device (Section 3.5):

$$
t_{\text{compute}} =
\frac{F_{\text{token,device}}}{R_{\text{GPU}}}
$$

This is the time assuming unlimited memory bandwidth.

---

## 4.3 Memory-Bound Time

Given the effective memory traffic per token (Section 2.4):

$$
t_{\text{mem}} =
\frac{T_{\text{token,device}}^{eff}}{BW_{\text{mem}}}
$$

This is the time assuming compute is free.

For long-context LLMs, this term is often dominant due to KV-cache reads.

---

## 4.4 Local Device Time Bound

The per-token latency for this PP stage is:

$$
t_{\text{local}} =
\max\left( t_{\text{compute}},\; t_{\text{mem}} \right)
$$

This is the roofline principle:

> **Decoding is limited by whichever is slower — compute throughput or memory bandwidth.**

---

<div style="page-break-before: always;"></div>

# 5. Communication Time During Decoding

Section 4 defined the **local per-token latency** on each device:

$$
t_{\text{local}} = \max(t_{\text{compute}}, t_{\text{mem}})
$$

We now incorporate the **inter-device communication time** that arises during decoding under distributed parallelism. In the nested parallelism structure used throughout this document:

$$
\textbf{PP} \;\rightarrow\; \textbf{EP} \;\rightarrow\; \textbf{TP} \;\rightarrow\; \textbf{SP}
$$

each axis contributes its own per-token communication term. All communication costs follow the standard $\alpha$–$\beta$ latency model [ALPHA-BETA]:

$$
t_{\text{comm}} = \alpha + \frac{\text{message size}}{B_{\text{eff}}}
$$

where $\alpha$ is the collective or hop latency, and $B_{\text{eff}}$ is the sustained bandwidth of the communication path.

The parameters $\alpha$ and $B_{\text{eff}}$ in this model are not abstract: they are **topology-dependent physical properties** of the underlying interconnect. Different parallelism domains—TP, EP, SP, and PP—may be mapped to **different network fabrics** or different portions of the same physical topology (e.g., NVSwitch star within a node, 2D/3D torus across nodes, or hybrid switch-plus-fabric designs). Consequently, each collective type sees its own communication characteristics, with potentially different latency constants and effective bandwidths. To keep the analysis general, we denote these as $\alpha_{XP}$ and $B_{\text{eff},XP}$ for TP, EP, SP, and PP respectively. Their actual numerical values depend on the system’s physical layout, routing scheme, and bisection bandwidth properties (e.g., constant-hop NVSwitch vs. hop-scaling torus fabrics). The following sections therefore use $\alpha_{XP}$ and $B_{\text{eff},XP}$ as **collective-specific, topology-aware** parameters, to be instantiated according to the actual deployment mapping.

**Delegation to `collectives.md`.** The shipped collective primitives (ring AR, double binary tree AR, ring AG / RS, pairwise A2A on star; dim-decomposed ring and bisection-bound A2A on torus; hierarchical RS → sub-AR → AG; in-network reduction via NVLS / Quantum SHARP / Tomahawk Ultra) are cost-modeled in `collectives.md §3–§6`, with contention coefficients $(\eta_\alpha, \eta_\beta)$ in `collectives.md §7`. This section instantiates those primitives with the decode-scale per-rank message sizes defined below; derivations and the $(\alpha_\mathrm{sum}, BW_\min)$ tier-chain accumulation live there. The $\alpha_{XP}$ and $BW_{XP}$ values used below are the fabric-chain span quantities from `notation.md §7`.

### Message sizes and their shard structure

To remain consistent with the compute and memory models, we strictly define the payload size for each collective type. Note the distinction between *storage size* (sharded) and *communication payload* (often full-width). Each shape below is given per token; for a decode step of batch size $B$ the payload scales linearly as $B \times (\text{per-token shape})$ because activation bytes scale with the number of sequences in the step. KV-gather (SP) scales with the number of sequences whose KV must be gathered, i.e., also $\propto B$.

- **PP (Pipeline Parallel):**
  Uses **activation shards** of width $\approx H/TP$ per token. Per step: $B \cdot H/TP$.
  *Rationale:* High-performance PP (e.g., Megatron-LM) preserves TP rank alignment, so only the local TP shard needs to be forwarded to the next stage.

- **EP (Expert Parallel):**
  Uses **full activations** of width $k \cdot H$ per token. Per step: $B \cdot k \cdot H$.
  *Rationale:* MoE routing sends token activations to experts. While the traffic is bidirectional (Dispatch + Combine), we model this by applying a factor of 2 to the *collective steps* in Section 5.2 rather than doubling the base message size here.

- **TP (Tensor Parallel):**
  Uses **full hidden state vectors** of width $H$ per token. Per step: $B \cdot H$.
  *Rationale:* Row-Parallel matrix multiplication produces a vector of **partial sums** that has the full width $H$. These must be All-Reduced across ranks, requiring the transfer of the full vector, not a shard.

- **SP (Sequence Parallel):**
  Uses **KV-cache blocks** of size $\frac{S}{SP} \cdot \frac{2H_{kv}}{TP}$ per sequence. Per step: $B \cdot \frac{S}{SP} \cdot \frac{2H_{kv}}{TP}$.
  *Rationale:* Ring Attention streams the distributed KV blocks around the ring. With $B$ concurrent sequences per step, each sequence's KV shard is streamed independently.

---

## 5.1 Pipeline Parallel (PP) Hop

Pipeline Parallelism (PP) forwards activations from one pipeline stage to the next [MEGATRON3]. Because TP is nested inside PP, high-performance implementations (e.g., Megatron-LM PP, DeepSpeed PP, NVIDIA NeMo, and FasterTransformer) preserve the **TP rank alignment** across all PP stages. That is, TP rank $i$ in stage $s$ corresponds directly to TP rank $i$ in stage $s{+}1$.  

This alignment has an important consequence:  
**each device only needs to forward its own TP shard of the hidden state**, not a full $H$-dimensional vector. The full activation is conceptually transferred across the PP boundary, but it is split naturally across $TP$ separate device-to-device links.

Thus the PP hop behaves as a **single, shard-sized point-to-point transfer**, with message size $\approx H/TP$ per device.

For a single token, the latency of this hop is modeled as:

$$
t_{PP} =
\alpha_{PP}
+
\frac{(H/TP)\, b}{BW_{\text{PP}}}
$$

This shard-preserving PP design avoids the extra TP collectives that would be required if stages exchanged full activations and then re-sharded them. Maintaining TP rank consistency across stages therefore yields a significantly faster pipeline, and is the standard strategy in modern LLM training and inference systems.

**Tier-aware PP cost (nested-layout convention).** $\alpha_{PP}$ and $\mathrm{BW}_{PP}$ above are *not* uniformly tier-0 fabric values. They are the latency and bandwidth of the **specific tier** the PP boundary physically crosses, which depends on where PP sits in the nested layout `DP → PP → EP → TP → SP` (innermost = highest-bandwidth tier). The framework's `partition_layout.assign_tier_per_axis` resolves a `(PartitionSpec, SystemSpec)` pair into a per-axis tier index by walking the fabric chain inner→outer, allocating each axis to the smallest tier whose cumulative reach holds the cumulative product of inner axes × this axis. For example, on d-Matrix squadrack (3-tier chain: 16 × 4 × 8):

- `PP=2, TP=8`: cumulative `8·2 = 16` ≤ tier-0 cap → **PP at tier 0** (pair-of-cards mesh, $\alpha=0.115$ μs, BW=64 GB/s).
- `PP=8, TP=8`: cumulative `8·8 = 64` ≤ tier-1 cap → **PP at tier 1** (PCIe, $\alpha=0.65$ μs).
- `PP=32, TP=8`: cumulative `8·32 = 256 > 64` → **PP at tier 2** (Ethernet, $\alpha=2.0$ μs, BW=50 GB/s).

On single-tier systems (e.g., NVL72), every axis collapses to tier 0 and the legacy tier-0 PP pricing is recovered exactly. The implementation is in `core/decode_model.py` and `core/prefill_model.py`; the helper lives at `core/primitives/partition_layout.py`. This is a worst-case-tier model — within a single PP cost call we use the *outermost* tier the boundary could possibly cross. A finer per-hop blend (some boundaries within tier 0, some across tier 1) is left as a future refinement; the worst-case form matches the conservative engineering view that "PP runs across servers" for sweeps where it does.

---

## 5.2 Expert Parallel (EP) All-to-All (MoE Dispatch and Combine)

MoE layers require exchanging token activations across the expert-parallel (EP) dimension via all-to-all routing [DEEPSPEED-MOE]. EP communication follows a **bidirectional dispatch-and-combine pattern**: token activations are routed from the source rank to the rank holding the selected expert (top-$k$), and the expert's output is then sent back to the source rank to be added to the residual stream. Because each token traverses the link twice, the collective cost is $2 \times$ a single-direction A2A.

Let $k$ denote the number of active experts per token. The per-direction message is $k H b$ bytes; the full Dispatch + Combine pair contributes $2 k H b$ bytes per device per token.

The shipped A2A primitive is pairwise direct-send (NCCL on star; bisection-bound pairwise on torus). Bruck / log-hop A2A does **not** ship and does not appear in the cost — see `collectives.md §5` for the primitive derivations. On a star topology, the per-token, per-layer EP A2A cost is:

$$
t_{EP} \;=\; 2(EP - 1)\,\alpha_{EP} \;+\; 2 \cdot \frac{EP - 1}{EP} \cdot \frac{k H \, b}{BW_{\text{EP}}}
$$

The factor of $2$ absorbs Dispatch + Combine; the primitive itself is pairwise direct-send per `collectives.md §5.1`. For torus EP fabrics, substitute the bisection-bound form of `collectives.md §5.2` with $M = k H b$. For dense models ($EP = 1$), $t_{EP} = 0$.

---

## 5.3 Tensor Parallel (TP) Communication

TP groups compute each layer in parallel across $TP$ devices using column- and row-parallel linear layers [MEGATRON]. The dominant collective is an **All-Reduce** at the end of the MLP (Row Parallel) and Attention (Output) blocks to sum partial results across ranks.

**Critical Note on Message Size:** unlike PP (which sends a shard), the TP All-Reduce operates on the **full hidden state vector** ($H$). Each device owns only a shard of the weights, but the partial output from Row Parallelism is a vector of size $H$ that must be reduced globally; the payload is $H b$ bytes per token, not $(H/TP) b$.

NCCL ships two AR algorithms on a star fabric — ring (large-$M$) and double binary tree (DBT, small-$M$). Selection is a manual tuner knob (`tuner.ar_algorithm`, default `"ring"`; see `collectives.md §3.1`). Both are pipelined and bandwidth-optimal; only the $n_\alpha$ coefficient differs. For the decode payload $M = H b$, the per-token, per-layer cost is:

$$
t_{TP}^{\text{ring}} \;=\; 2(TP - 1)\,\alpha_{TP} \;+\; 2 \cdot \frac{TP - 1}{TP} \cdot \frac{H b}{BW_{\text{TP}}}
$$

$$
t_{TP}^{\text{DBT}} \;=\; 2\,\lceil \log_2 TP \rceil \cdot \alpha_{TP} \;+\; 2 \cdot \frac{TP - 1}{TP} \cdot \frac{H b}{BW_{\text{TP}}}
$$

For torus TP fabrics (dim-decomposed ring, shipped on TPU / Trainium), substitute the torus AR form of `collectives.md §3.2` with $M = H b$. Derivation and the ring-vs-DBT empirical crossover behavior are in `collectives.md §3.1` (cost) and explainer `02 §2`.

---

## 5.4 Sequence Parallel (SP) Communication

Sequence Parallelism (SP) in inference typically refers to **Ring Attention** [RING-ATTN]. The KV cache is partitioned along the sequence dimension $S$; to compute attention for a new token the Query ($Q$) stays local and KV blocks rotate around the ring so that the local $Q$ attends to the full history. This is a **pass-KV** ring variant — the standard choice for KV-cache-dominated inference where KV is large relative to $Q$. (A pass-Q variant exists for training, where $Q$ is full-sequence; see [HUANG-CP-2024].)

The ring operation is effectively an **All-Gather** (streaming the distributed KV cache to every rank), not an All-Reduce. DeepSpeed-Ulysses [DEEPSPEED-ULYSSES] is an alternative SP approach using all-to-all instead of ring; unlike ring, it is bounded by the number of attention heads rather than the number of devices. Tree-based SP variants are theoretically possible but no production implementation ships them — KV shards are large and must be processed in sequence order. For modeling purposes, we assume **ring-style, pass-KV SP communication**, costed via the ring AG primitive of `collectives.md §4.1`.

### SP Ring Communication Latency

Substituting the decode per-rank KV shard $M_\mathrm{SP} = (S / SP) \cdot (2 H_{kv} / TP) \cdot b$ into the star ring AG cost of `collectives.md §4.1`:

$$
t_{SP} \;=\; (SP - 1)\,\alpha_{SP} \;+\; (SP - 1) \cdot \frac{(S / SP) \cdot (2 H_{kv} / TP) \cdot b}{BW_{\text{SP}}}
$$

The message size reflects TP and SP as orthogonal partitions of the head and sequence dimensions. For torus SP fabrics, use the torus AG form of `collectives.md §4.2` with the same $M_\mathrm{SP}$.

**Decode overlap note:** in single-token decode, per-token compute time is small, so communication overlap with compute ($\rho$) is unlikely to be significant for SP. Use $\rho \approx 0$ for SP when modeling decode latency.

---

## 5.5 Total Communication Time Per Token on a PP Stage

Sections 5.1–5.4 provide **per-token, per-layer** communication costs for each parallelism axis (TP, EP, SP), and a **per-token, per-hop** cost for PP. We now clarify how these terms combine to form the total per-token communication time on a given pipeline-parallel (PP) stage.

### Per-layer vs. per-stage normalization

A Transformer layer contains exactly one Attention block and one MLP/MoE block. Each block triggers a fixed number of communication collectives, and within each layer, TP, EP, and SP collectives are strictly ordered:

- **Attention**
  - 1 TP collective (Output Projection)
  - 1 SP collective (if SP is enabled)

- **MLP (dense)**
  - 1 TP collective (Output Projection)

- **MoE block**
  - 1 EP all-to-all
  - 1 TP collective (Expert Output Projection)

These collectives must complete before the token can advance to the next layer. Since a PP stage contains $L/PP$ *sequential* layers, the total communication work for that stage is:

- $n_{TP}$ TP collectives per layer  
- $n_{EP}$ EP collectives per layer (0 for dense layers, 1 for MoE layers)  
- $n_{SP}$ SP collectives per layer (1 if SP is enabled)

Because TP, EP, and SP operations within each layer depend on one another (e.g., TP → SP in attention and EP → TP in MoE), they are **strictly sequential** and do not overlap. Thus, the **per-token communication time accumulated over the entire PP stage** is:

$$
t_{\text{comm,stage}} =
\frac{L}{PP}
(
n_{TP}\, t_{TP}
+
n_{EP}\, t_{EP}
+
n_{SP}\, t_{SP}
)
$$

Where:

- $t_{EP}$, $t_{TP}$, $t_{SP}$ are the **per-token, per-layer** communication costs given in Sections 5.2–5.4.
- $n_{TP}$ is typically **2** (one for Attention, one for FFN).
- $n_{EP}$ is **1** for MoE layers and **0** for dense layers.
- $n_{SP}$ is **1** (occurs during Attention).

### Adding PP hop cost

The PP hop is different: it is a **per-token, per-hop** cost rather than a per-layer cost. A token is forwarded once from PP stage $s$ to stage $s{+}1$, with latency $t_{PP}$ as defined in Section 5.1.

Thus, the total per-token communication time for this stage is:

$$
t_{\text{comm}} =
\frac{L}{PP}
(
n_{TP}\, t_{TP}
+
n_{SP}\, t_{SP}
)
+
\frac{L_{\text{moe}}}{PP}
(
n_{EP}\, t_{EP}
)
+
t_{PP}
$$

### Interpretation

- The first term accumulates **TP and SP collectives** required by all $L/PP$ layers on this PP stage (both dense and MoE layers have attention blocks requiring these collectives).
- The second term accumulates **EP collectives** required only by the $L_{\text{moe}}/PP$ MoE layers on this PP stage. Dense layers do not require EP communication.
- The third term accounts for the **one PP hop** that forwards the token to the next stage.
- This combined expression represents the **total communication work** per token for the stage. Whether this communication becomes the latency bottleneck or is hidden by overlap is addressed in Section 4's roofline-style model and Section 6's end-to-end pipeline analysis.

### Mixed MoE/Dense Architectures

For architectures where only some layers are MoE (e.g., $L_{\text{moe}} < L$), the EP communication cost is proportionally reduced. This is particularly important for models like DeepSeek-V2 or Mixtral variants that alternate between dense and MoE layers.

For a **pure dense model**: $L_{\text{moe}} = 0$, so the EP term vanishes entirely.

For a **pure MoE model**: $L_{\text{moe}} = L$, recovering the original formula.

### Summary of Collective Types and Message Sizes

| Parallelism | Occurs in | Collective Type | Passes | Message Size (per device, per step) | Layer Types |
|-------------|-----------|------------------|---------|----------------------------|-------------|
| **PP** | between layers | point-to-point | 1 | $B\cdot(H/TP)\,b$ | All |
| **TP** | attn + FFN | all-reduce (ring/tree) | 2 | $B\cdot H\,b$ | All |
| **EP** | MoE FFN | all-to-all | 2 | $B\cdot kH\,b$ | MoE only |
| **SP** | attention | all-gather (ring) | 1 | $B\cdot(S/SP)\cdot (2H_{kv}/TP)\, b$ | All |

At $B=1$ these reduce to the classical single-token payloads. The B-factor reflects that a decode step processes $B$ activations concurrently, so each collective carries $B \times$ the per-token activation vector.

### Practical Guidance: Shipped Algorithm Selection

Each collective in this section uses the algorithm that is actually shipped on the target fabric; other algorithms (Bruck A2A, recursive-doubling AR, PAT AG) are reference-only and live in `explaining/collectives/01 App. B`. Selection rules:

- **TP All-Reduce:** NCCL ships both ring and double binary tree (DBT) on a star fabric; the choice is a manual tuner knob `tuner.ar_algorithm` (`collectives.md §3.1`), default `"ring"`. On torus fabrics (TPU / Trainium), only dim-decomposed ring ships — the knob is ignored. Empirical crossover: DBT wins at small $M$, ring wins at large $M$ ([DEMYST-NCCL]).

- **EP All-to-All:** NCCL ships pairwise direct-send on a star; TPU / Trainium ships the bisection-bound pairwise form on a torus (`collectives.md §5`). Log-hop (Bruck) A2A is **not** shipped and does not appear in this section's formulas.

- **SP All-Gather:** Ring AG is the only shipped form in production inference stacks — KV shards are large and must be processed in sequence order, so tree variants are impractical. This applies to both star (`collectives.md §4.1`) and torus (`collectives.md §4.2`).

See `collectives.md §3–§6` for the full shipped-primitive inventory and per-topology cost formulas (including hierarchical RS → sub-AR → AG and in-network reduction); `collectives.md §7` for the contention coefficients $(\eta_\alpha, \eta_\beta)$.

---

<div style="page-break-before: always;"></div>

# 6. End-to-End Latency, Throughput, and Partition Strategy

This section integrates memory limits, compute, communication, and pipeline behavior into a complete
end-to-end decoding performance model. We discuss:

1. Feasible model partitioning via **HBM limits**  
2. Local per-token latency via the **compute–memory roofline**  
3. Per-token communication latency from **TP/EP/SP/PP**  
4. Overlap strategies between compute and communication  
5. Throughput (**TPS**, **TTPS**) based on PP-stage bottlenecks  
6. **TTFT**, including prefill on a separate cluster and PP-stage latency summation  

---

## 6.1 Model Partition Strategy from HBM Constraints

A parallel configuration $(DP, PP, EP, TP, SP)$ is **feasible** only if each device can store:

- its **parameter shard**,
- its **KV-cache shard**, and
- the **activation workspace** needed for a single decoding token,

within the available HBM capacity $M_{\text{HBM}}$.

We define the total per-device static footprint as

$$
M_{\text{device}}^{\text{total}} =
M_{\theta,\text{device}}
+
M_{\text{act,device}}
+
M_{\text{KV,device}}
\;\le\;
M_{\text{HBM}}
$$

Using the per-device memory expressions derived in Section 1, the **fully expanded** form for uniform architectures is:

$$
\frac{L}{PP}\;
\left[
\frac{2H^2 + 2 H H_{kv}}{TP}
\;+\;
\frac{3 H I N_{\text{exp}}}{TP \cdot EP}
+
\frac{2 S H_{kv}}{TP \cdot SP}
\right] b
+\;
B(4H + 2H_{kv}) b
+
\frac{V H}{TP} b
\le\;
M_{\text{HBM}}
$$

where:

- the bracketed term is the **intermediate PP-stage** footprint (parameters, activations, KV cache).
- and the final $\frac{V H}{TP} b$ term models the **worst-case embedding / LM-head overhead** on boundary PP stages.
- for a **dense MLP model**: $I = I_{\text{dense}}$, and $N_{\text{exp}} = EP = 1$.
- for a **MoE model**: $I = I_{\text{moe}}$, with $N_{\text{exp}} > 1$ and $EP \ge 1$.

For **mixed MoE/dense architectures**, the memory constraint uses the split formula from Section 1.4, where dense and MoE layer contributions are computed separately.

### Calculating DP for a Fixed Total HBM Capacity

The total Data Parallelism degree ($DP$) is constrained by both the total cluster size ($N_{\text{GPUs}}$) and the memory headroom available on each device.

1. **Memory Headroom Requirement:** $DP$ scaling is only possible if $M_{\text{device}}^{\text{total}} \le M_{\text{HBM}}$ for the chosen inner sharding degrees ($PP, EP, TP, SP$).
2. **Replication Logic:** Each model replica requires a dedicated group of $PP \cdot EP \cdot TP \cdot SP$ devices.

Let $N_{\text{GPUs}}$ be the total number of devices in the cluster. The maximum achievable $DP$ count is:

$$
DP = \left\lfloor \frac{N_{\text{GPUs}}}{PP \cdot EP \cdot TP \cdot SP} \right\rfloor
$$

**Physical Interpretation:**

- **Scaling Limit:** To increase $DP$ for higher throughput ($TTPS$), one must either add more total GPUs to the cluster or increase inner sharding (e.g., higher $PP$ or $SP$) to reduce $M_{\text{device}}^{\text{total}}$, though the latter consumes more devices per replica.
- **Footprint vs. Replica Count:** There is a direct trade-off: higher sharding degrees "thin out" the memory footprint per device to fit large context $S$ or large models, but they simultaneously reduce the number of independent replicas that can fit in a fixed cluster.

---

## 6.2 Local and Networking Per-Token Latency

### Compute-bound latency

$$
t_{\text{compute}} =
\frac{F_{\text{token,device}}}{R_{\text{GPU}}}
$$

### Memory-bandwidth-bound latency

$$
t_{\text{mem}} =
\frac{T_{\text{token,device}}^{\text{eff}}}{BW_{\text{mem}}}
$$

### Roofline local latency

$$
t_{\text{local}} =
\max(t_{\text{compute}},\; t_{\text{mem}})
$$

### Collective communication latency

$$
t_{\text{comm}}
\approx
\frac{L}{PP}
(
n_{TP}\, t_{TP} +
n_{SP}\, t_{SP}
) +
\frac{L_{\text{moe}}}{PP}
(
n_{EP}\, t_{EP}
) +
t_{PP}
$$

Note: EP collectives only apply to MoE layers ($L_{\text{moe}}$), while TP and SP collectives apply to all layers.

### Unified Overlap Model

We introduce an overlap factor $\rho \in [0, 1]$ representing the fraction of local compute/memory time that is successfully utilized to hide communication.

The effective per-token latency is the local time plus any **unhidden** communication:

$$
t_{\text{step,user}} =
t_{\text{local}}
+
\max(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}})
$$

**Regimes:**

- **$\rho = 0$ (No Overlap):**
  $$t_{\text{step,user}} = t_{\text{local}} + t_{\text{comm}}$$
  Typical for naive implementations or strictly sequential dependencies.

- **$\rho = 1$ (Perfect Overlap Opportunity):**
  $$t_{\text{step,user}} = t_{\text{local}} + \max(0, t_{\text{comm}} - t_{\text{local}}) = \max(t_{\text{local}}, t_{\text{comm}})$$
  Achieved by highly optimized kernels (e.g., Ring Attention) where independent work exists.

- **$0 < \rho < 1$ (Partial Overlap):**
  Models real-world overheads (kernel launch latency, synchronization barriers) that prevent utilizing the full local duration for hiding comms.

---

## 6.3 Pipeline Bubble, TPS, and TTPS

### 6.3.1 Per-stage step time

The overlap-aware time derived in §6.2 is the **per-stage** step time — the wall-clock cost of one pipeline stage processing the current batch:

$$
t_{\text{stage}} =
t_{\text{local}}
+
\max(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}})
$$

Per stage $j$ this is $t_{\text{stage},j}$. The **throughput bottleneck** of a DP replica is the slowest stage $\max_j t_{\text{stage},j}$; the **pipeline traversal time** (first-token latency across PP) is the sum
$\sum_{j=1}^{PP} t_{\text{stage},j}$.

### 6.3.2 Pipeline bubble correction (user-observed step time)

During decoding, the user of sequence $s$ observes one new token **per decode step**, where a decode step ends when every PP stage has contributed to the current batch. When $B \ge PP$, the pipeline is full: consecutive microbatches pipeline across stages and each user sees a token every $\max_j t_{\text{stage},j}$ seconds. When $B < PP$, the pipeline is underfilled — the single microbatch traverses all $PP$ stages sequentially with no overlap, so the step cost grows by a factor $PP/B$.

A first-order correction captures both regimes:

$$
\gamma_{\text{pp}} = \max\left(1,\; \frac{PP}{B}\right)
$$

The user-observed step time also includes the per-round CPU dispatch budget $t_{\mathrm{SW}}$ — the cumulative kernel-launch latency on the host side (kernel_launch_overhead.md §5):

$$
t_{\mathrm{SW}} = L \cdot k \cdot \tau_{\mathrm{launch}} + PP \cdot k_{\mathrm{pp\_hop}} \cdot \tau_{\mathrm{launch}}, \qquad k = k_{\mathrm{compute}} + k_{\mathrm{collective}} \cdot (n_{\mathrm{TP}}^{\mathrm{calls}} + n_{\mathrm{EP}}^{\mathrm{calls}} + n_{\mathrm{SP}}^{\mathrm{calls}})
$$

where $n_{*}^{\mathrm{calls}}$ are the per-layer NCCL API call counts that fire for the current shape (zero when the corresponding axis is 1). For TP and SP, $n_{*}^{\mathrm{calls}} = n_{*\mathrm{\_collectives}}$ directly. For EP, $n_{\mathrm{EP}}^{\mathrm{calls}} = 2 \cdot n_{\mathrm{EP\_collectives}}$ — the cost-model convention is "1 collective per MoE layer = 1 round-trip" (the 2× factor is wrapped inside `_cost("moe_a2a", ...)`), but the launch counter must expand back to 2 actual NCCL API calls (dispatch + combine). The $PP \cdot k_{\mathrm{pp\_hop}}$ term counts inter-stage P2P send/recv launches: each stage handles $k_{\mathrm{pp\_hop}}$ P2P kernels per microbatch (default 2: 1 recv + 1 send) $\times$ $PP$ microbatches per round (inert when $PP = 1$). Composing GPU work and host dispatch via the SW overlap factor $\rho_{\mathrm{SW}}$:

$$
t_{\mathrm{step,user}} = \max\!\bigl(t_{\mathrm{stage}},\ \rho_{\mathrm{SW}} \cdot t_{\mathrm{stage}} + (1 - \rho_{\mathrm{SW}}) \cdot (t_{\mathrm{stage}} + t_{\mathrm{SW}}),\ t_{\mathrm{SW}}\bigr) \cdot \gamma_{\mathrm{pp}}
$$

**Regimes (assuming $\rho_{\mathrm{SW}} = 1$, the default):**

- $t_{\mathrm{stage}} \ge t_{\mathrm{SW}}$: $t_{\mathrm{step,user}} = t_{\mathrm{stage}} \cdot \gamma_{\mathrm{pp}}$ (GPU-bound — async dispatch fully hides kernel-launch overhead).
- $t_{\mathrm{stage}} < t_{\mathrm{SW}}$: $t_{\mathrm{step,user}} = t_{\mathrm{SW}} \cdot \gamma_{\mathrm{pp}}$ (SW-bound — CPU cannot feed the GPU fast enough; common at small microbatch on dense decode).

When $t_{\mathrm{SW}} = 0$ (SW modeling disabled by setting `kernel_launch_us = 0` in the tuner) the formula reduces to the legacy $t_{\mathrm{step,user}} = t_{\mathrm{stage}} \cdot \gamma_{\mathrm{pp}}$. At $B = 1, PP = 1$ and $t_{\mathrm{SW}} = 0$, this further reduces to $t_{\mathrm{step,user}} = t_{\mathrm{stage}}$ — backward compatible with the non-pipelined decode model.

A separate Tensor Core efficiency term $\eta_{\mathrm{TC}}(\mathrm{mb})$ derates the compute roofline at small microbatch — see [practical_pp_choice.md §3.3](../explaining/practical_pp_choice.md#33-microbatch-granularity-and-kernel-launch-overhead) and [kernel_launch_overhead.md](../explaining/kernel_launch_overhead.md) for the full derivation; with the default $\eta_{\mathrm{TC}} = 1$ (no curve set) this term is a no-op.

### 6.3.3 Throughput

A single DP replica emits one token per sequence per step, so it outputs $B$ tokens every $t_{\text{step,user}}$:

$$
TPS_{\text{single}} =
\frac{B}{t_{\text{step,user}}}
$$

Across $DP$ fully independent replicas (no cross-replica coupling), the total cluster throughput scales linearly:

$$
TTPS = DP \cdot TPS_{\text{single}} =
\frac{DP \cdot B}{t_{\text{step,user}}}
$$

When $B \ge PP$ the bubble factor is unity and $TPS_{\text{single}} = B /
\max_j t_{\text{stage},j}$, recovering the classical "throughput is gated by the slowest stage" result.

---

## 6.4 Batch-Size Scaling and Throughput–Latency Tradeoff

TTFT and prefill analysis are covered in [prefill.md](prefill.md). This section analyzes how decoding performance changes as the **static batch size** $B$ grows — where $B$ sequences are decoded together and each contributes one token per step. The central question is: how does batching shift the operational regime from memory-bound to compute-bound, and what is the resulting throughput–latency tradeoff?

---

### 6.4.1 Arithmetic Intensity as a Function of Batch Size $B$

For a static batch of $B$ sequences decoded together, the key observation is that **parameter weights are loaded once from HBM and reused across all $B$ tokens in the same step**. The FLOPs scale with $B$, while weight traffic remains independent of $B$:

- **Tokens per step:** $B$ (one per sequence)
- **FLOPs per step:** $B \times F_{\text{token,device}}$ (scales linearly with $B$)
- **Weight traffic per step:** $T_{\theta,\text{device}}$ (loaded once, shared across all $B$ tokens)
- **KV cache traffic per step:** $B \times T_{\text{KV,device}}$ (each sequence has its own KV history)

The **operational intensity** at batch size $B$ is therefore:

$$
\text{OI}(B) =
\frac{B \times F_{\text{token,device}}}
     {T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}
$$

#### Two limiting regimes

**Memory-bound limit** ($B \to 0$, or equivalently when weight traffic dominates KV traffic):

$$
\lim_{B \to 0} \text{OI}(B) =
\frac{B \times F_{\text{token,device}}}{T_{\theta,\text{device}}}
\;\to\; 0
$$

At small $B$, weights dominate the denominator, and the model is **weight-traffic-limited**. The $B=1$ case recovers the single-token OI from §4.1: $\text{OI}(1) = 2/b$ (exact, since every weight matrix contributes FLOPs $= 2 \times$ params with the $6HI$ convention), which lies far below the ridge point $R_{\text{GPU}} / BW_{\text{mem}}$ for all practical GPUs and precisions — confirming that single-request decode is always memory-bound.

**Compute-bound limit** ($B \to \infty$, or when KV traffic dominates weight traffic):

$$
\lim_{B \to \infty} \text{OI}(B) =
\frac{F_{\text{token,device}}}{T_{\text{KV,device}}}
$$

At large $B$, KV cache reads dominate the denominator and the intensity saturates. This ceiling is reached when the KV term $B \times T_{\text{KV,device}}$ overwhelms $T_{\theta,\text{device}}$.

#### Ridge-point crossover

The **crossover batch size** $B^*$ is the point at which the roofline transitions from memory-bound to compute-bound. Setting $\text{OI}(B^*) = R_{\text{GPU}} / BW_{\text{mem}}$ (the ridge point per [ROOFLINE]) and solving:

$$
\frac{B^*\times F_{\text{token,device}}}
     {T_{\theta,\text{device}} + B^* \times T_{\text{KV,device}}} =
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
$$

$$
B^* =
\frac{T_{\theta,\text{device}} \times R_{\text{GPU}}}
     {F_{\text{token,device}} \times BW_{\text{mem}} - T_{\text{KV,device}} \times R_{\text{GPU}}}
$$

**Existence condition.** The crossover $B^*$ is finite and positive iff the denominator is positive. Rearranging into ridge-point form:

$$
B^{\star} < \infty
\quad\Longleftrightarrow\quad
\frac{F_{\text{token,device}}}{T_{\text{KV,device}}} \;>\; \frac{R_{\text{GPU}}}{BW_{\text{mem}}} = R_{\text{ridge}}
$$

i.e., the **arithmetic intensity of KV traffic alone** (per-token FLOPs divided by per-sequence KV bytes) must exceed the device ridge point. The intuition: $\text{OI}(B)$ asymptotes to $F_{\text{token,device}} / T_{\text{KV,device}}$ as $B \to \infty$ (weight traffic becomes negligible relative to the $B$-scaled KV traffic); if this asymptotic ceiling itself sits below $R_{\text{ridge}}$, the roofline is never crossed regardless of batch size.

**No-crossover regime.** When the inequality is violated — typical of very long contexts on small models, where $T_{\text{KV,device}}$ grows linearly in $S$ while $F_{\text{token,device}}$ stays fixed — decode remains memory-bound at every $B$, and $B^{\star} \to \infty$. In this regime batching still amortizes *weight* traffic ($T_\theta / B$ per token), which continues to reduce per-sequence TPOT, but it **cannot** push the step into the compute-bound zone; adding more sequences only adds linear KV bandwidth pressure until HBM saturates. The practical implication is that the Pareto frontier (§6.5) has no Zone 3 plateau in this regime — both throughput and TPOT scale linearly with $B$ indefinitely, and the operating point is chosen by HBM capacity rather than by the compute ceiling.

When $T_{\text{KV,device}}$ is small relative to $T_{\theta,\text{device}} / B^*$ (i.e., short-context decode where weight traffic dominates), this simplifies to the weight-dominated approximation:

$$
B^*
\;\approx\;
\frac{T_{\theta,\text{device}}}{F_{\text{token,device}}}
\times
\frac{R_{\text{GPU}}}{BW_{\text{mem}}}
\qquad (\text{weight-dominated regime})
$$

This expression has an intuitive interpretation: $T_{\theta,\text{device}} / F_{\text{token,device}}$ is the inverse OI for a single token (bytes per FLOP), and $R_{\text{GPU}} / BW_{\text{mem}}$ is the ridge point (FLOPs per byte). Their product gives the batch size at which weight reuse tips the balance from memory-bound to compute-bound.

---

### 6.4.2 Batched TPOT and the Compute-Bound Crossover

With $B$ sequences batched together, the per-step local execution time becomes:

$$
t_{\text{local}}(B) =
\max\left(
\frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}},
\;\;
\frac{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}{BW_{\text{mem}}}
\right)
$$

The **user-observed** Time Per Output Token (TPOT) — the inter-token latency seen by any single sequence — is exactly the user-observed step time from §6.3.2:

$$
\text{TPOT}(B) =
t_{\text{step,user}}(B) =
t_{\text{stage}}(B) \cdot \max\left(1,\; \frac{PP}{B}\right)
$$

Each decode step produces exactly one output token per active sequence, so the streaming rate at the user is $1 / t_{\text{step,user}}$ — **not** $B / t_{\text{stage}}$. Dividing the step time by $B$ would be an *amortized-per-token* cost that double-counts the parallel tokens across sequences; we reserve that for throughput-per-GPU (§6.3.3).

#### Regime analysis for TPOT

All regimes below assume $B \ge PP$ (full pipeline, bubble factor $=1$); the $B < PP$ regime simply inflates TPOT by the pipeline depth.

**Memory-bound regime** ($B \ll B^*$, weight-dominated):

$$
t_{\text{stage}}(B)
\approx
\frac{T_{\theta,\text{device}}}{BW_{\text{mem}}}
\quad \Rightarrow \quad
\text{TPOT}(B)
\approx
\frac{T_{\theta,\text{device}}}{BW_{\text{mem}}}
$$

TPOT is approximately **flat in $B$** in this regime: weights are streamed once per step regardless of $B$, and the per-user step time stays pinned at the weight-streaming cost. What improves with $B$ is cluster throughput (more tokens per step), not user-observed latency.

**Compute-bound regime** ($B \gg B^*$, KV-dominated):

$$
t_{\text{stage}}(B)
\approx
\frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
\quad \Rightarrow \quad
\text{TPOT}(B)
\approx
\frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
$$

TPOT **grows linearly with $B$**: once the device saturates compute, every added sequence extends the step time, and every user sees the extension.

**Summary table:**

| Regime | Condition | $t_{\text{stage}}(B)$ | $\text{TPOT}(B)$ |
|--------|-----------|-----------------------|------------------|
| Memory-bound | $B \ll B^*$ | $\approx T_{\theta} / BW_{\text{mem}}$ (flat) | $\approx T_{\theta} / BW_{\text{mem}}$ (flat) |
| Crossover | $B = B^*$ | ridge point | knee of the Pareto curve |
| Compute-bound | $B \gg B^*$ | $\propto B$ (growing) | $\propto B$ (growing) |

---

### 6.4.3 Throughput–Latency Pareto Curve

Define the two key metrics as a function of $B$ (assuming $B \ge PP$ so the pipeline bubble factor is 1; for $B < PP$, multiply TPOT by $PP/B$):

- **Throughput** (tokens per second, all sequences in the batch):
  $$
  \text{Throughput}(B) =
  \frac{B}{t_{\text{step,user}}(B)} =
  \frac{B}{t_{\text{stage}}(B)}
  $$

- **TPOT** (user-observed inter-token latency, seconds per output token):
  $$
  \text{TPOT}(B) =
  t_{\text{step,user}}(B) =
  t_{\text{stage}}(B)
  $$

These two metrics are in tension in the compute-bound regime: past $B^*$, throughput plateaus while TPOT grows. Sweeping $B$ from $1$ to $\infty$ traces a **Pareto frontier** in the (Throughput, TPOT) plane.

#### Three zones of the Pareto curve

**Zone 1 — Memory-bound ($B < B^*$):**

Throughput grows approximately linearly with $B$ while TPOT stays approximately flat. Weight traffic is amortized across more sequences in the step without inflating the step time; each additional sequence is essentially "free" from a per-user latency perspective.

$$
\text{Throughput}(B) \approx \frac{B \times BW_{\text{mem}}}{T_{\theta,\text{device}}},
\qquad
\text{TPOT}(B) \approx \frac{T_{\theta,\text{device}}}{BW_{\text{mem}}}
$$

**Zone 2 — Crossover ($B \approx B^*$):**

The system reaches the ridge point — the best throughput-per-GPU achievable without raising TPOT. Further batching starts to cost latency per user.

**Zone 3 — Compute-bound ($B > B^*$):**

Throughput plateaus toward a ceiling set by compute capacity; TPOT grows linearly with $B$ because the added compute extends every user's step:

$$
\text{Throughput}*{\max} =
\frac{R*{\text{GPU}}}{F_{\text{token,device}}},
\qquad
\text{TPOT}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
$$

---
