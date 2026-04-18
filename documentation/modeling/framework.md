# Framework Overhead Model: Per-Phase Latency Constants

**Author:** Yue Lu  
**Date:** April 2026

---

The hardware roofline model in `decode.md` establishes a lower bound on per-token decode latency, governed by compute throughput and HBM bandwidth. Real serving systems, however, introduce additional latency above this bound that lives entirely on the host CPU and serving-software path: tokenization at request entry, CUDA kernel launch or graph replay per decode step, request scheduling and batch assembly, token sampling, and response streaming. These are empirical constants that require profiling on the target system — they cannot be derived from hardware specifications alone. This document catalogs the CPU / software-stack overhead terms, explains where each comes from, and defines the per-request framework overhead $t_{\text{framework}}$ that enters end-to-end cost accounting in `e2e.md`. Hardware-level overheads tied to the network fabric (e.g., disaggregated KV transfer) are covered in `prefill.md §6`; memory-traffic calibration constants (e.g., the activation I/O residual) are covered in `decode.md`.

---

## Table of Contents

- [1. Overhead Classification](#1-overhead-classification)
- [2. CPU / Software-Stack Overhead Terms](#2-cpu--software-stack-overhead-terms)
  - [2.1 Tokenization Latency ($t_{\text{tok}}$)](#21-tokenization-latency-t_texttok)
  - [2.2 CUDA Kernel Launch Overhead ($t_{\text{launch}}$)](#22-cuda-kernel-launch-overhead-t_textlaunch)
  - [2.3 CUDA Graph Replay ($t_{\text{graph}}$)](#23-cuda-graph-replay-t_textgraph)
  - [2.4 Request Scheduling / Batch Assembly ($t_{\text{sched}}$)](#24-request-scheduling--batch-assembly-t_textsched)
  - [2.5 Token Sampling ($t_{\text{sample}}$)](#25-token-sampling-t_textsample)
  - [2.6 Response Streaming / Detokenization ($t_{\text{detok}}$)](#26-response-streaming--detokenization-t_textdetok)
- [3. Total Framework Overhead Per Request](#3-total-framework-overhead-per-request)
- [4. Symbol Summary](#4-symbol-summary)

---

<div style="page-break-before: always;"></div>

## 1. Overhead Classification

The table below catalogs the CPU / software-stack overhead terms, organized by the serving phase in which each occurs, the symbol used throughout this suite, and representative ranges from published sources and hardware measurements. All terms are empirical — they depend on the serving framework, OS, and CPU/tokenizer library path, and must be profiled on the target deployment.

| Overhead | Phase | Symbol | Typical Range |
|----------|-------|--------|---------------|
| Tokenization | Prefill entry | $t_{\text{tok}}$ | ~0.1–2 ms |
| CUDA kernel launch | Decode per-step | $t_{\text{launch}}$ | ~5–50 µs/step |
| CUDA graph replay | Decode per-step | $t_{\text{graph}}$ | ~10–100 µs/step |
| Request scheduling / batch assembly | Serving | $t_{\text{sched}}$ | ~10–200 µs |
| Token sampling | Decode per-step | $t_{\text{sample}}$ | ~20–200 µs |
| Response streaming / detokenization | Decode per-token | $t_{\text{detok}}$ | ~1–10 µs/token |

> **Out of scope for this document.** Disaggregated KV transfer latency (an α–β network-fabric term) is derived in `prefill.md §6.4`. Memory-traffic calibration constants used by the decode traffic model live in `decode.md` (implemented in `core/decode_model.py`).

---

## 2. CPU / Software-Stack Overhead Terms

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
t_{\text{launch}} \approx n_{\text{kernels}} \times t_{\text{kernel-launch}} \approx L \times 10\text{–}600\;\mu\text{s}
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

## 3. Total Framework Overhead Per Request

Combining all CPU / software-stack overhead terms, the total framework latency for a complete request consisting of a prefill pass followed by $T_{\text{out}}$ decode steps is:

$$
t_{\text{framework}} = t_{\text{tok}} + t_{\text{sched}} + T_{\text{out}} \cdot (t_{\text{graph}} + t_{\text{sample}} + t_{\text{detok}})
$$

where the three groups are: tokenization (once, at start), scheduling (once per batch), and per-decode-step overhead (repeated $T_{\text{out}}$ times).

**Notes on this decomposition:**

1. $t_{\text{launch}}$ is replaced by $t_{\text{graph}}$ in graph-capture mode. If running in eager mode without CUDA graphs, replace $t_{\text{graph}}$ with $t_{\text{launch}}$ in the per-step term.

2. $t_{\text{sched}}$ is charged once per batch assembly event, not once per request. In continuous batching, a new batch is assembled every decode step, so $t_{\text{sched}}$ is effectively per-step for the active batch, not per-request. At the per-request level, it integrates to $T_{\text{out}} \times t_{\text{sched}}$ amortized over all requests in the batch.

3. Disaggregated KV-transfer latency is **not** included here; it is a network-fabric term handled in `prefill.md §6.4` as part of the prefill→decode handoff.

**Relationship to hardware model.** The full per-request latency is:

$$
t_{\text{total}} = t_{\text{TTFT}} + T_{\text{out}} \cdot t_{\text{step,user}} + t_{\text{framework}}
$$

where $t_{\text{TTFT}}$ and $t_{\text{step,user}}$ are defined in `prefill.md` and `decode.md` respectively. End-to-end assembly is in `e2e.md`.

**Order-of-magnitude comparison.** For a representative decode-heavy workload ($T_{\text{out}} = 512$, graph mode, greedy sampling, co-located):

| Term | Per-step | Cumulative ($T_{\text{out}}=512$) |
|------|----------|----------------------------------|
| $t_{\text{step,user}}$ (hardware) | ~2–20 ms | 1–10 s |
| $t_{\text{graph}}$ | ~50 µs | ~26 ms |
| $t_{\text{sample}}$ | ~10–30 µs | ~5–15 ms |
| $t_{\text{detok}}$ | ~5 µs | ~2.5 ms |
| $t_{\text{tok}}$ (once) | — | ~0.5 ms |
| $t_{\text{sched}}$ (once, amortized) | — | ~0.1 ms |

Framework overhead is typically 1–5% of total request latency for well-optimized serving stacks on large models. It becomes more significant for small models (where $t_{\text{step,user}}$ is short) or for deployments without CUDA graph capture.

---

## 4. Symbol Summary

All symbols introduced in this document; these are consolidated into §13 of `notation.md`.

| Symbol | Definition | Units |
|--------|-----------|-------|
| $t_{\text{tok}}$ | Tokenization latency (CPU BPE/SP processing) | ms |
| $t_{\text{launch}}$ | CUDA kernel launch overhead per decode step (no graph) | µs/step |
| $t_{\text{graph}}$ | CUDA graph replay latency per decode step | µs/step |
| $t_{\text{sched}}$ | Request scheduling / batch assembly latency | µs |
| $t_{\text{sample}}$ | Token sampling latency (logits → token ID) | µs/step |
| $t_{\text{detok}}$ | Response streaming / detokenization latency per token | µs/token |
| $t_{\text{framework}}$ | Total CPU / software-stack overhead per request (§3) | ms |
| $T_{\text{out}}$ | Number of output tokens generated per request | tokens |

---

## References

**[VLLM]**  
Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.  
→ Scheduling loop latency; PagedAttention block allocation; batch assembly cost.

**[TENSORRT-LLM]**  
NVIDIA Corporation (2023–2025). *TensorRT-LLM.* <https://github.com/NVIDIA/TensorRT-LLM>  
→ Optimized sampling kernels; CUDA graph integration.

**[SARATHI]**  
Agrawal et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369.  
→ Chunked prefill scheduling overhead; $t_{\text{sched}}$ characterization under mixed prefill–decode batching.

**[H100-SPEC]**  
NVIDIA Corporation (2022). *NVIDIA H100 Tensor Core GPU Architecture.* NVIDIA Whitepaper WP-10792-001.  
→ Kernel launch latency bounds.
