# Framework Overhead Model: Per-Phase Latency Constants

**Author:** Yue Lu  
**Date:** April 2026

---

The hardware roofline model in `modeling.tpot.md` establishes a lower bound on per-token decode latency, governed by compute throughput and HBM bandwidth. Real serving systems, however, introduce additional latency above this bound: tokenization at request entry, CUDA kernel launch or graph replay per decode step, request scheduling and batch assembly, token sampling, and response streaming. Some of these overheads are analytically tractable — disaggregated KV transfer, for instance, follows the standard α–β model given network parameters — but most are empirical constants that require profiling on the target system. This document catalogs all such overhead terms, classifies them by phase and derivability, and formally defines the calibration constants $c_{\text{act}}$ and $c_{\text{norm}}$ that were removed from `modeling.tpot.md` as negligible in the dominant-term analysis. Together with `modeling.tpot.md` and `modeling.prefill.md`, this document provides the full per-request cost accounting assembled in `modeling.e2e.md`.

---

## Table of Contents

- [1. Overhead Classification](#1-overhead-classification)
- [2. Empirical Overhead Terms](#2-empirical-overhead-terms)
  - [2.1 Tokenization Latency ($t_{\text{tok}}$)](#21-tokenization-latency-t_texttok)
  - [2.2 CUDA Kernel Launch Overhead ($t_{\text{launch}}$)](#22-cuda-kernel-launch-overhead-t_textlaunch)
  - [2.3 CUDA Graph Replay ($t_{\text{graph}}$)](#23-cuda-graph-replay-t_textgraph)
  - [2.4 Request Scheduling / Batch Assembly ($t_{\text{sched}}$)](#24-request-scheduling--batch-assembly-t_textsched)
  - [2.5 Token Sampling ($t_{\text{sample}}$)](#25-token-sampling-t_textsample)
  - [2.6 Response Streaming / Detokenization ($t_{\text{detok}}$)](#26-response-streaming--detokenization-t_textdetok)
- [3. Analytical Overhead: Disaggregated KV Transfer](#3-analytical-overhead-disaggregated-kv-transfer)
- [4. Activation I/O and Norm Constants ($c_{\text{act}}$, $c_{\text{norm}}$)](#4-activation-io-and-norm-constants-c_textact-c_textnorm)
  - [4.1 Activation I/O Constant ($c_{\text{act}}$)](#41-activation-io-constant-c_textact)
  - [4.2 Norm FLOP Constant ($c_{\text{norm}}$)](#42-norm-flop-constant-c_textnorm)
- [5. Total Framework Overhead Per Request](#5-total-framework-overhead-per-request)
- [6. Symbol Summary](#6-symbol-summary)

---

<div style="page-break-before: always;"></div>

## 1. Overhead Classification

The table below catalogs all framework overhead terms, organized by the serving phase in which they occur, whether they are derivable from first principles or require empirical profiling, the symbol used throughout this suite, and representative ranges from published sources and hardware measurements.

| Overhead | Phase | Type | Symbol | Typical Range |
|----------|-------|------|--------|---------------|
| Tokenization | Prefill entry | Empirical | $t_{\text{tok}}$ | ~0.1–2 ms |
| CUDA kernel launch | Decode per-step | Empirical | $t_{\text{launch}}$ | ~5–50 µs/step |
| CUDA graph replay | Decode per-step | Empirical | $t_{\text{graph}}$ | ~10–100 µs/step |
| Request scheduling / batch assembly | Serving | Empirical | $t_{\text{sched}}$ | ~10–200 µs |
| Token sampling | Decode per-step | Empirical | $t_{\text{sample}}$ | ~20–200 µs |
| Response streaming / detokenization | Decode per-token | Empirical | $t_{\text{detok}}$ | ~1–10 µs/token |
| Disaggregated KV transfer | Prefill→decode | Analytical (α–β) | $t_{\text{KV\_transfer}}$ | depends on network |
| Activation I/O residual | Decode per-layer | Empirical | $c_{\text{act}}$ | ~8–12 (dimensionless) |
| Norm FLOP overhead | Decode per-layer | Empirical | $c_{\text{norm}}$ | ~5–20 (dimensionless) |

> **Note:** "Empirical" means the value requires profiling on the target system and cannot be derived from hardware specifications alone. "Analytical" means it can be computed from first principles given the network parameters $\alpha_{\text{inter}}$ and $B_{\text{eff,inter}}$.

---

## 2. Empirical Overhead Terms

Each subsection below describes what the term measures, why it exists at the systems level, how to calibrate it on a target deployment, and typical values from published sources where available.

---

### §2.1 Tokenization Latency ($t_{\text{tok}}$)

**What it measures.** $t_{\text{tok}}$ is the wall-clock time from receiving a raw text request to the point where token IDs are ready and the first CUDA kernel for the prefill pass is launched. It includes byte-pair encoding (BPE) or SentencePiece lookup, Unicode normalization, and any pre-processing applied by the serving framework.

**Why it exists.** Tokenization runs on CPU in most serving stacks (vLLM, TensorRT-LLM, TGI). While GPU-side tokenization libraries exist, they are not universally deployed. The CPU→GPU handoff itself adds a small scheduling gap.

**Scaling behavior.** $t_{\text{tok}}$ scales weakly with input length: roughly $\mathcal{O}(S_{\text{input}})$ character-level operations, but constant factors are small (BPE lookup is table-driven). For $S_{\text{input}} \leq 4096$ tokens the cost is typically 0.1–2 ms and is often pipelined with the previous request's decode tail.

**Calibration.** Measure the latency from HTTP request receipt to the timestamp of the first `cudaLaunchKernel` call in a profiler trace (e.g., Nsight Systems). Vary $S_{\text{input}}$ to isolate the length-dependent component.

**Typical values.** 0.1–0.5 ms for short prompts ($S_{\text{input}} \leq 512$); up to 2 ms for very long prompts ($S_{\text{input}} \approx 8192$) on a single CPU thread. Parallelizing tokenization across CPU threads or pipelining it with KV allocation reduces the effective contribution to near zero for batch serving.

---

### §2.2 CUDA Kernel Launch Overhead ($t_{\text{launch}}$)

**What it measures.** $t_{\text{launch}}$ is the cumulative CPU→GPU kernel submission latency incurred per decode step when CUDA graph capture is not used. Each GEMM, attention kernel, norm kernel, and elementwise activation contributes a fixed per-kernel launch cost.

**Why it exists.** CUDA kernel launch involves a driver API call (`cuLaunchKernel` or the runtime equivalent), which transfers grid/block parameters to the GPU command processor via a CPU-visible ring buffer. On H100, a single kernel launch costs approximately 5–20 µs of CPU time; the GPU itself sees the kernel only after this submission overhead. For a full transformer layer with attention, FFN, and norm kernels, $n_{\text{kernels}} \approx \mathcal{O}(10\text{–}30)$ per layer, giving:

$$
t_{\text{launch}} \approx n_{\text{kernels}} \times t_{\text{kernel\_launch}} \approx L \times 10\text{–}600\;\mu\text{s}
$$

for a model with $L$ layers without any kernel fusion. Fused implementations (e.g., FlashAttention fusing Q/K/V projection + attention + output projection) reduce $n_{\text{kernels}}$ substantially.

**Calibration.** Profile a decode step with Nsight Systems; count distinct kernel launches and measure the gap between consecutive kernel starts. The difference between `cuda:0` timeline utilization and 100% in the idle periods between kernels reflects launch overhead.

**Typical values.** On H100 SXM5: ~5–20 µs per kernel launch. For an unfused 80-layer model with ~20 kernels per layer: up to 16–32 ms per decode step, which would dominate over compute. In practice, kernel fusion and CUDA graphs (§2.3) reduce this to negligible levels.

---

### §2.3 CUDA Graph Replay ($t_{\text{graph}}$)

**What it measures.** $t_{\text{graph}}$ is the latency to replay a pre-captured CUDA graph representing the entire decode step. It replaces the per-kernel launch sequence with a single graph launch command, amortizing all submission overhead into one fixed cost.

**Why it exists.** CUDA graph capture records the exact sequence of GPU operations (kernels, memory copies, synchronization points) in a single graph object during a warm-up pass. On subsequent decode steps, the graph is replayed atomically from the GPU side, eliminating CPU involvement per kernel. This is the primary technique used by TensorRT-LLM [TENSORRT-LLM] and vLLM's graph mode [VLLM] to reduce kernel launch overhead to a single fixed cost per step.

**Limitations.** CUDA graph replay requires static input shapes (fixed batch size, fixed sequence configuration). This is incompatible with naive continuous batching (PagedAttention with variable batch sizes per step). Production systems work around this by bucketing batch sizes to a discrete set of graph captures, or by falling back to eager mode for irregular shapes.

**Calibration.** Capture graphs at each batch-size bucket; measure end-to-end decode step time with and without graph mode at matching batch sizes. The difference is $t_{\text{launch}}$ from §2.2. The residual with graph mode is $t_{\text{graph}}$.

**Typical values.** 10–100 µs per decode step on H100 SXM5 with graph mode, across a range of batch sizes.

---

### §2.4 Request Scheduling / Batch Assembly ($t_{\text{sched}}$)

**What it measures.** $t_{\text{sched}}$ is the time the serving scheduler spends at the start of each decode iteration selecting which requests to include in the next batch, allocating or reclaiming KV cache pages, and preparing the batch metadata tensors (token IDs, position IDs, block tables).

**Why it exists.** Continuous batching (the standard production serving mode) requires the scheduler to run between every decode step. The scheduler must:
1. Check the priority queue of waiting requests.
2. Estimate whether a new request's KV cache pages fit in available HBM.
3. Run the PagedAttention block allocator [VLLM] to assign physical KV blocks.
4. Optionally, perform chunked-prefill piggybacking [SARATHI] for waiting requests.
5. Construct the attention metadata (block tables, sequence lengths) needed by the kernel.

All of this runs on CPU and must complete before the GPU decode kernel can launch.

**Calibration.** Instrument the scheduler loop with Python-level timestamps (`time.perf_counter_ns()`). Vary the number of concurrently active requests and the scheduler policy (FCFS, preemptive, chunked prefill) to characterize the dependence on batch size.

**Typical values.** 10–50 µs for small batches (B ≤ 8) with simple FCFS scheduling; 50–200 µs for large batches (B ≥ 64) with full PagedAttention block allocation and chunked-prefill scheduling.

---

### §2.5 Token Sampling ($t_{\text{sample}}$)

**What it measures.** $t_{\text{sample}}$ is the time required to select the next token from the logit distribution output by the LM head, after the final linear projection and softmax.

**Why it exists.** Sampling involves operations on the vocabulary dimension $V$ (typically 32K–128K tokens). These are not GEMM-bound and do not benefit from tensor parallelism in the same way as weight projections. Depending on the sampling strategy:

- **Greedy decoding** (argmax over $V$): a reduction over $V$ values, roughly $\mathcal{O}(V)$ comparisons → ~5–20 µs on H100.
- **Temperature + top-$p$/top-$k$ sampling**: requires sorting or partial sort over $V$ to find the nucleus, followed by a multinomial draw → ~20–200 µs.
- **Nucleus sampling on large vocabularies**: without optimized sorted-topk kernels (e.g., those in TensorRT-LLM [TENSORRT-LLM]), can reach 100–500 µs per step.

**Calibration.** Profile the decode step with Nsight Systems or `torch.cuda.Event` timing. Isolate the sampling kernel (distinct from the LM head GEMM). Measure across vocabulary sizes and sampling strategies.

**Typical values.** Greedy: ~5–20 µs. Temperature + top-$p$: ~20–200 µs. Nucleus sampling with unoptimized kernels: ~100–500 µs. TensorRT-LLM's fused sampling kernels achieve ~10–30 µs across strategies [TENSORRT-LLM].

---

### §2.6 Response Streaming / Detokenization ($t_{\text{detok}}$)

**What it measures.** $t_{\text{detok}}$ is the per-token cost of converting the sampled token ID back to a UTF-8 string fragment and transmitting it to the client (via HTTP chunked transfer or server-sent events).

**Why it exists.** Detokenization is the inverse of tokenization: a table lookup from token ID to byte sequence. It runs on CPU and is followed by a streaming write to the network socket. For most vocabulary sizes, the lookup itself is negligible; the dominant cost is the system call for the socket write plus any Python overhead in the serving framework.

**Scaling behavior.** $t_{\text{detok}}$ applies once per generated token and is effectively constant in token output length (it does not grow with $T_{\text{out}}$, only accumulates). The cumulative detokenization cost is $T_{\text{out}} \times t_{\text{detok}}$, which is small relative to the total decode time for any non-trivially sized model.

**Calibration.** Measure round-trip latency from GPU token ID to client receipt at the application layer, minus network round-trip time (RTT). The residual is approximately $t_{\text{detok}} + t_{\text{sample}}$.

**Typical values.** ~1–10 µs per token for the detokenization lookup itself. Socket write latency varies with network configuration and is separate from this constant.

---

## 3. Analytical Overhead: Disaggregated KV Transfer

In disaggregated prefill–decode deployments [DISAGG-PREFILL], the prefill pass is executed on a dedicated set of "prefill" GPUs, and the resulting KV cache must be transferred over the inter-cluster interconnect to the "decode" GPUs before decoding can begin. This is the one framework overhead term that can be derived analytically rather than measured empirically.

Using the standard α–β latency model [ALPHA-BETA]:

$$
\boxed{
t_{\text{KV\_transfer}} = \alpha_{\text{inter}} + \frac{M_{\text{KV\_transfer}}}{B_{\text{eff,inter}}}
}
$$

where:

- $\alpha_{\text{inter}}$ — inter-cluster network startup latency (InfiniBand or NVLink-C2C round-trip setup)
- $B_{\text{eff,inter}}$ — effective inter-cluster bandwidth (bytes/s) accounting for protocol overhead
- $M_{\text{KV\_transfer}}$ — bytes of KV state to transfer for one sequence's shard assigned to this device

The KV transfer volume per device is:

$$
M_{\text{KV\_transfer}} = \frac{2 \cdot S_{\text{input}} \cdot H_{kv} \cdot b}{TP \cdot SP} \cdot \frac{L}{PP}
$$

Breaking this down:
- $2$ — one key tensor and one value tensor per layer
- $S_{\text{input}}$ — number of prefill tokens whose KV state must be transferred
- $H_{kv} = n_{kv} \cdot d_{\text{head}}$ — KV projection dimension (reduced under GQA/MQA)
- $b$ — bytes per element (e.g., $b=2$ for bf16, $b=1$ for fp8 KV cache)
- $TP \cdot SP$ — tensor and sequence parallel sharding of the KV cache per PP stage
- $L / PP$ — number of layers owned by this PP stage

**Co-located deployment.** When prefill and decode execute on the same cluster (same physical GPUs or via NVLink within a node), the KV transfer is an in-HBM copy and contributes negligible latency:

$$
t_{\text{KV\_transfer}} = 0 \quad \text{(co-located)}
$$

**Disaggregated deployment.** For production disaggregated serving (e.g., DistServe [DISAGG-PREFILL] using InfiniBand HDR 200 Gb/s = 25 GB/s per link):

$$
t_{\text{KV\_transfer}} \approx \alpha_{\text{inter}} + \frac{M_{\text{KV\_transfer}}}{25\;\text{GB/s}}
$$

For a long-context request ($S_{\text{input}} = 32768$, $H_{kv} = 1024$, $b = 2$, $L = 80$, $PP = 1$, $TP = 8$, $SP = 1$):

$$
M_{\text{KV\_transfer}} = \frac{2 \times 32768 \times 1024 \times 2}{8 \times 1} \times 80 = 1.34\;\text{GB}
$$

$$
t_{\text{KV\_transfer}} \approx 10\;\mu\text{s} + \frac{1.34\;\text{GB}}{25\;\text{GB/s}} \approx 53\;\text{ms}
$$

This 50+ ms transfer cost sets a hard lower bound on TTFT in disaggregated systems for long contexts, independent of prefill compute time.

---

## 4. Activation I/O and Norm Constants ($c_{\text{act}}$, $c_{\text{norm}}$)

These two constants capture per-layer overhead terms that were present in early drafts of `modeling.tpot.md` but were removed because they are negligible relative to the dominant weight traffic and FFN FLOPs at large model scale. They are defined here for completeness and for calibration in edge cases (small models, specialized hardware, or research into sub-dominant terms).

---

### §4.1 Activation I/O Constant ($c_{\text{act}}$)

**Definition.** The residual per-layer activation traffic not eliminated by FlashAttention-style kernel fusion is modeled as:

$$
T_{\text{act,layer}} \approx c_{\text{act}} \cdot H \cdot b \;\;\text{bytes}
$$

where $b$ is bytes per element and $H$ is the hidden size. The constant $c_{\text{act}}$ counts the number of unavoidable full-hidden-state tensor reads or writes per layer that remain after optimal kernel fusion.

**What contributes to $c_{\text{act}}$.** Even with fully fused FlashAttention kernels, the following inter-kernel boundaries require HBM I/O on the hidden state:
- Residual stream read before attention input projection: +1 load ($H$ elements)
- Attention output + residual add before layer norm 2: +1 store, +1 load ($2H$ elements)
- FFN output + residual add: +1 store ($H$ elements)
- Boundary between attention and FFN (hidden state passed between fused kernels): +1 load, +1 store ($2H$ elements)

This yields $c_{\text{act}} \approx 8$ unavoidable hidden-state transfers per layer under ideal fusion. In practice, non-ideal fusion boundaries in frameworks add further reads/writes, pushing the value toward 10–12.

**Calibration.** From kernel traces in TensorRT-LLM [TENSORRT-LLM] and xFormers [XFORMERS] on H100 with bf16 ($b=2$): $c_{\text{act}} \approx 8\text{–}12$.

**Significance.** For $H = 8192$, $b = 2$:

$$
T_{\text{act,layer}} \approx 10 \times 8192 \times 2 = 164\;\text{KB}
$$

Compare to per-layer weight traffic for a Llama-3-70B-scale model:

$$
T_{\theta,\text{layer}} \approx P_{\text{attn}} \cdot b + P_{\text{FFN}} \cdot b \approx 550\;\text{MB}
$$

The activation traffic is approximately three to four orders of magnitude smaller than weight traffic and is correctly omitted from the dominant-term roofline model in `modeling.tpot.md`.

---

### §4.2 Norm FLOP Constant ($c_{\text{norm}}$)

**Definition.** Per-layer normalization FLOPs are modeled as:

$$
F_{\text{norm}} \approx c_{\text{norm}} \cdot H
$$

where $c_{\text{norm}}$ is a small integer determined by the normalization variant.

**Estimates by normalization type:**

| Norm type | Operations | $c_{\text{norm}}$ estimate |
|-----------|-----------|--------------------------|
| RMSNorm | mean-square + rsqrt + element-wise scale | ~5 |
| LayerNorm | mean + variance + normalize + scale + bias | ~10 |
| Gated LayerNorm / QKNorm | additional gating multiplication | ~15–20 |

Most modern LLMs (LLaMA, Qwen, DeepSeek) use RMSNorm, giving $c_{\text{norm}} \approx 5$. Each transformer layer applies norm twice (pre-attention, pre-FFN), so the total per-layer norm FLOPs are $2 \times c_{\text{norm}} \times H$.

**Significance.** For $H = 8192$, RMSNorm:

$$
F_{\text{norm}} = 2 \times 5 \times 8192 = 81{,}920\;\text{FLOPs per layer}
$$

Compare to per-layer FFN FLOPs (dense, $I_{\text{dense}} = 4H$, SwiGLU):

$$
F_{\text{ffn,dense}} = 3 \times 2 H I_{\text{dense}} = 6 H \times 4H = 24 H^2 \approx 1.6 \times 10^9\;\text{FLOPs}
$$

Norm contributes approximately $5 \times 10^{-5}$ of total FLOPs per layer and is correctly dropped from the dominant-term compute model in `modeling.tpot.md`.

---

## 5. Total Framework Overhead Per Request

Combining all overhead terms, the total framework latency for a complete request consisting of a prefill pass followed by $T_{\text{out}}$ decode steps is:

$$
\boxed{
t_{\text{framework}} = \underbrace{t_{\text{tok}}}_{\text{once, at start}}
+ \underbrace{t_{\text{sched}}}_{\text{once per batch}}
+ \underbrace{T_{\text{out}} \cdot \bigl(t_{\text{graph}} + t_{\text{sample}} + t_{\text{detok}}\bigr)}_{\text{per decode step}}
+ \underbrace{t_{\text{KV\_transfer}}}_{\text{if disaggregated}}
}
$$

**Notes on this decomposition:**

1. $t_{\text{launch}}$ is replaced by $t_{\text{graph}}$ in graph-capture mode. If running in eager mode without CUDA graphs, replace $t_{\text{graph}}$ with $t_{\text{launch}}$ in the per-step term.

2. $t_{\text{sched}}$ is charged once per batch assembly event, not once per request. In continuous batching, a new batch is assembled every decode step, so $t_{\text{sched}}$ is effectively per-step for the active batch, not per-request. At the per-request level, it integrates to $T_{\text{out}} \times t_{\text{sched}}$ amortized over all requests in the batch.

3. $t_{\text{KV\_transfer}}$ is charged at the prefill→decode handoff boundary and is zero for co-located deployments.

4. $c_{\text{act}}$ and $c_{\text{norm}}$ do not appear in this total because they are sub-dominant at all practically relevant model scales (see §4.1 and §4.2). They are defined here for completeness only.

**Relationship to hardware model.** The full per-request latency is:

$$
t_{\text{total}} = t_{\text{TTFT}} + T_{\text{out}} \cdot t_{\text{token}} + t_{\text{framework}}
$$

where $t_{\text{TTFT}}$ and $t_{\text{token}}$ are defined in `modeling.prefill.md` and `modeling.tpot.md` respectively. End-to-end assembly is in `modeling.e2e.md`.

**Order-of-magnitude comparison.** For a representative decode-heavy workload ($T_{\text{out}} = 512$, graph mode, greedy sampling, co-located):

| Term | Per-step | Cumulative ($T_{\text{out}}=512$) |
|------|----------|----------------------------------|
| $t_{\text{token}}$ (hardware) | ~2–20 ms | 1–10 s |
| $t_{\text{graph}}$ | ~50 µs | ~26 ms |
| $t_{\text{sample}}$ | ~10–30 µs | ~5–15 ms |
| $t_{\text{detok}}$ | ~5 µs | ~2.5 ms |
| $t_{\text{tok}}$ (once) | — | ~0.5 ms |
| $t_{\text{sched}}$ (once, amortized) | — | ~0.1 ms |

Framework overhead is typically 1–5% of total request latency for well-optimized serving stacks on large models. It becomes more significant for small models (where $t_{\text{token}}$ is short) or for deployments without CUDA graph capture.

---

## 6. Symbol Summary

All symbols introduced in this document; these are consolidated into §13 of `modeling.notation.md`.

| Symbol | Definition | Units |
|--------|-----------|-------|
| $t_{\text{tok}}$ | Tokenization latency (CPU BPE/SP processing) | ms |
| $t_{\text{launch}}$ | CUDA kernel launch overhead per decode step (no graph) | µs/step |
| $t_{\text{graph}}$ | CUDA graph replay latency per decode step | µs/step |
| $t_{\text{sched}}$ | Request scheduling / batch assembly latency | µs |
| $t_{\text{sample}}$ | Token sampling latency (logits → token ID) | µs/step |
| $t_{\text{detok}}$ | Response streaming / detokenization latency per token | µs/token |
| $t_{\text{KV\_transfer}}$ | Disaggregated KV transfer latency (α–β model) | ms |
| $c_{\text{act}}$ | Activation I/O constant: unavoidable hidden-state HBM transfers per layer | dimensionless (~8–12) |
| $c_{\text{norm}}$ | Norm FLOP constant: per-element norm operation count | dimensionless (~5–20) |
| $B_{\text{eff,inter}}$ | Effective inter-cluster bandwidth (InfiniBand / NVLink-C2C) | GB/s |
| $\alpha_{\text{inter}}$ | Inter-cluster network startup latency | µs |
| $T_{\text{out}}$ | Number of output tokens generated per request | tokens |
| $M_{\text{KV\_transfer}}$ | KV bytes transferred per device in disaggregated serving | bytes |

---

## References

**[VLLM]**  
Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.  
→ Scheduling loop latency; PagedAttention block allocation; batch assembly cost.

**[TENSORRT-LLM]**  
NVIDIA Corporation (2023–2025). *TensorRT-LLM.* https://github.com/NVIDIA/TensorRT-LLM  
→ Fused kernel traces for $c_{\text{act}}$ calibration (~8–12 on H100 with bf16); optimized sampling kernels; CUDA graph integration.

**[XFORMERS]**  
Lefaudeux et al. (2022). *xFormers: A Modular and Hackable Transformer Modelling Library.* arXiv:2209.14970.  
→ Kernel profiling data for activation I/O ($c_{\text{act}}$) and norm operation counts ($c_{\text{norm}}$).

**[SARATHI]**  
Agrawal et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369.  
→ Chunked prefill scheduling overhead; $t_{\text{sched}}$ characterization under mixed prefill–decode batching.

**[DISAGG-PREFILL]**  
Zhong et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.* OSDI 2024. arXiv:2401.09670.  
→ Prefill–decode disaggregation architecture; KV transfer latency as a function of sequence length and network bandwidth.

**[ALPHA-BETA]**  
Hockney, R. (1994). *The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.* Parallel Computing 20(3).  
→ α–β latency model: $t = \alpha + n/\beta$; basis for $t_{\text{KV\_transfer}}$ derivation in §3.

**[H100-SPEC]**  
NVIDIA Corporation (2022). *NVIDIA H100 Tensor Core GPU Architecture.* NVIDIA Whitepaper WP-10792-001.  
→ Kernel launch latency bounds; NVLink 4.0 bandwidth for co-located KV transfer baseline.
