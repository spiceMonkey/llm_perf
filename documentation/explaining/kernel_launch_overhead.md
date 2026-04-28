# Kernels, CUDA Graphs, and Per-Round Dispatch Overhead

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
GPU kernel, kernel launch, CUDA Graphs, dispatch overhead, kernel fusion, transformer inference, decode roofline, framework integration

---

# Table of Contents

- [1. Why This Matters for Inference](#1-why-this-matters-for-inference)
- [2. What Is a Kernel](#2-what-is-a-kernel)
- [3. The Launch Tax](#3-the-launch-tax)
- [4. CUDA Graphs](#4-cuda-graphs)
  - [4.1 Mechanism: Capture, Instantiate, Replay](#41-mechanism-capture-instantiate-replay)
  - [4.2 Fixed-Shape Constraint and Multi-Graph Deployment](#42-fixed-shape-constraint-and-multi-graph-deployment)
  - [4.3 Where the Graph Is Stored](#43-where-the-graph-is-stored)
  - [4.4 CUDA Graphs Among Other DAG Layers in Production Frameworks](#44-cuda-graphs-among-other-dag-layers-in-production-frameworks)
- [5. Mapping the Overhead to the llm_perf Framework](#5-mapping-the-overhead-to-the-llm_perf-framework)
  - [5.1 The Per-Round SW Formula](#51-the-per-round-sw-formula)
  - [5.2 Where It Plugs Into the Roofline](#52-where-it-plugs-into-the-roofline)
  - [5.3 Numerical Mapping for GPT-1.8T MoE on GB200](#53-numerical-mapping-for-gpt-18t-moe-on-gb200)
- [6. Practical Implications](#6-practical-implications)
- [References](#references)

---

# 1. Why This Matters for Inference

The decode roofline used by [decode.md](../modeling/decode.md) computes a per-stage step time $t_{\mathrm{stage}} = \max(t_{\mathrm{compute}}, t_{\mathrm{mem}}) + \max(0, t_{\mathrm{comm}} - \rho \cdot t_{\mathrm{local}})$. All three terms — compute, memory, and communication — are *GPU-side* costs. The framework implicitly assumes the GPU is fed efficiently from the CPU side, with no idle time waiting for instructions to arrive. For pure compute-bound prefill at large batch this is approximately correct. For memory-bound decode at small microbatch (mb), it is *not*: the CPU-side dispatch path becomes a first-order latency component, and the roofline silently over-states GPU utilization.

This document explains where that hidden cost comes from (what a kernel is, what a kernel launch costs, what CUDA Graphs do about it) and how to map it onto the framework's existing per-round time so partition sweeps can be corrected when needed. The accompanying [practical_pp_choice.md §3.3](practical_pp_choice.md#33-microbatch-granularity-and-kernel-launch-overhead) uses the formula derived here to show that mb=1 decode is launch-overhead-dominated even with CUDA Graphs.

---

# 2. What Is a Kernel

A **kernel** in CUDA is a function that runs on the GPU. The GPU is a coprocessor — it sits idle until the CPU asks it to do work. The unit of work the CPU asks for is a kernel: a piece of GPU code with its arguments (tensor pointers, sizes, scalar parameters) that the CUDA driver dispatches into a *stream* (a queue of work the GPU consumes in order).

In a transformer forward pass, each operator is typically one kernel. A dense layer's decode step sequence is:

1. Layer-norm
2. Q projection (matmul)
3. K projection (matmul)
4. V projection (matmul)
5. Attention (typically a single fused kernel: FlashAttention [FA2], PagedAttention [VLLM])
6. O projection (matmul)
7. Residual add
8. Layer-norm
9. Up projection (matmul)
10. Activation (GELU, SwiGLU)
11. Down projection (matmul)
12. Residual add

That is ~12 kernel launches per dense decode layer. **Kernel fusion** — combining adjacent operators into a single kernel — can reduce this: layer-norm + residual fuse trivially; QKV can be a single matmul if the projection weights are concatenated; activation + down-projection can fuse via a custom kernel. With aggressive fusion the count can drop to ~5-7 kernels per dense layer. Without fusion it is closer to 12-15.

Mixture-of-Experts (MoE) layers have additional kernels for routing: a router scoring kernel, an expert dispatch (often using all-to-all communication primitives), per-expert feed-forward network (FFN) kernels, and an expert combine. For the GPT-1.8T MoE used in our partition sweeps, MoE layers have ~15-20 kernels each at decode. We use $k = 12$ as a representative midpoint in [practical_pp_choice.md §3.3](practical_pp_choice.md#33-microbatch-granularity-and-kernel-launch-overhead).

Each kernel runs on the GPU's streaming multiprocessors (SMs); kernels on the same stream execute sequentially. Multiple streams can run concurrently if the workload permits, which is the basis for compute-communication overlap (see the framework's $\rho$ parameter).

---

# 3. The Launch Tax

The CPU pays an overhead each time it launches a kernel, regardless of how trivial the kernel itself is. The cost has several components:

1. **Python / C++ dispatch logic** — the framework (PyTorch, JAX, TensorRT-LLM) interprets the operator, dispatches to the right backend kernel, packs arguments into a launch descriptor.
2. **Driver call** — `cudaLaunchKernel` validates parameters, marshals into a hardware-specific format, queues into the stream.
3. **Hardware command-buffer dispatch** — the GPU's command processor pulls the launch descriptor from the queue, sets up the SM scheduling state, and starts the kernel.

Each step has its own latency. Empirical numbers on H100 / B200 with current CUDA toolkits ([CUDA-RUNTIME], [HOPPER-PROG-GUIDE]):

- **Without CUDA Graphs**: ~5-10 μs per kernel launch (driver + dispatch + framework overhead). The exact number depends on the launch path; PyTorch eager mode is at the higher end, custom C++ launchers at the lower end.
- **With CUDA Graphs (post-replay)**: ~1-2 μs per node (the dispatch logic is collapsed into the graph; only the GPU-side scheduling cost remains).

For a 120-layer model with 12 kernels per layer, one full forward pass is **120 × 12 = 1440 kernel launches**. Multiplied by per-launch overhead, that is a **dispatch budget of ~10 ms without CUDA Graphs or ~2.16 ms with**. On modern hardware where memory-bound decode reads 15-20 GB of weights at 8 TB/s — taking ~1.8-2.2 ms per microbatch per stage — this dispatch budget is the same order of magnitude as the actual GPU work.

This is not a small constant. At decode mb=1, the entire pipeline round is competing with the dispatch cost.

---

# 4. CUDA Graphs

## 4.1 Mechanism: Capture, Instantiate, Replay

A **CUDA Graph** is a directed acyclic graph (DAG) of kernel-launch nodes that CUDA captures from a recorded execution and can replay as a single dispatch unit. The mechanism, introduced in CUDA 10 ([CUDA-RUNTIME] §3.2.8), works in three phases:

1. **Capture** — execute the forward pass once in `cudaStreamBeginCapture` mode. CUDA records every kernel launch, with its arguments and stream dependencies, into an internal graph data structure. No work runs on the GPU during capture.
2. **Instantiate** — convert the captured graph into an executable form (`cudaGraphInstantiate`). The driver pre-validates parameters, allocates internal buffers, and builds a hardware-optimized launch sequence.
3. **Launch (replay)** — call `cudaGraphLaunch` to dispatch the entire kernel sequence. The driver issues all 1440 kernels with no per-kernel framework or validation overhead.

The key win: per-replay, the CPU pays *one* graph-launch API call instead of 1440 individual kernel-launch API calls. Per-kernel overhead drops from the 5-10 μs dispatch path down to the ~1-2 μs GPU-scheduling residual.

## 4.2 Fixed-Shape Constraint and Multi-Graph Deployment

**The catch is fixed shape.** The graph captures the exact tensor pointers, sizes, batch dimensions, and pointer addresses present at capture time. If the operating point changes — different batch size, different sequence length, different KV-cache page layout — the captured graph is no longer valid for replay. Production LLM serving frameworks work around this by capturing **many graphs**, one per supported operating point, and selecting the closest one at runtime:

- **vLLM** [VLLM] captures decode graphs at $B \in \{1, 2, 4, 8, 16, 32, ..., B_{\max}\}$; continuous batching rounds requests up to the nearest captured size.
- **TensorRT-LLM** [TENSORRT-LLM] uses CUDA Graphs at the engine level, with per-engine batch-size buckets.
- **SGLang** uses CUDA Graphs aggressively, with extensions to support more dynamic shapes via tensor padding.

Capture itself has a one-time cost (typically tens of milliseconds per shape) and graphs occupy GPU memory; serving systems amortize this over millions of replays. CUDA Graphs are not a panacea. They cannot capture host-side control flow (Python `if` / `for` that affects the kernel sequence), they cannot capture variable-shape kernels without padding, and they introduce engineering complexity. But for steady-state decode — the operating point this framework's roofline targets — they are the standard production choice.

## 4.3 Where the Graph Is Stored

Three things are kept distinct in CUDA Graph accounting:

1. **The kernel code itself** is *not* in the graph. Kernel binaries (CUBIN / PTX) are loaded into GPU memory once via `cuModuleLoad` and live in a separate code / instruction memory region. Both regular kernel launches and graph replays dispatch the *same* underlying kernel binaries; only the launch path differs.
2. **The captured graph (`cudaGraph_t`)** — the DAG of nodes produced by stream capture — is a host-side data structure (CPU RAM). It holds the topology, kernel function pointers, and copies of kernel-launch parameters. This is the editable, inspectable form. Once instantiated, it can be freed.
3. **The instantiated executable graph (`cudaGraphExec_t`)** — what `cudaGraphLaunch` actually runs — lives largely in **GPU memory**. It is a pre-formatted command buffer the GPU's command processor consumes directly. Concretely it contains pre-validated launch descriptors for each kernel node (grid/block dimensions, shared-memory size), pre-marshalled kernel-argument blobs (pointer values, scalars, sizes captured at record time), internal driver scheduling state used during replay, and embedded tensor pointer values — *not* the tensor data, which lives in its normal memory pool.

**Pointer semantics matter.** The graph captures *device pointers*, not data. Activation buffers, KV-cache pages, and weight tensors all live in their usual memory pools; the graph just records the addresses to read and write. This is why CUDA Graphs require *stable allocations*: vLLM, TensorRT-LLM, and SGLang all use fixed-size memory pools for activations so the same pointer values are valid across every replay.

**Memory footprint, empirically.** A single decode graph for a typical LLM (~1500 kernel nodes) is ~1-5 MB on device. vLLM captures ~30-40 graphs (one per supported batch-size bucket), giving a per-replica graph cache of ~50-200 MB of GPU memory. Negligible compared to weights (10-100s of GB) and KV cache (often 50%+ of HBM), but non-zero. The framework's `MemoryResults.fits_in_HBM` does not currently account for graph storage; for HBM-pressured deployments (very long context, KV-cache-dominated), folding ~150 MB per replica into the existing `system_overhead_GB` constant in [KVPagingConfig](../../llm_perf/core/kv_paging_model.py) is the simplest integration point.

## 4.4 CUDA Graphs Among Other DAG Layers in Production Frameworks

DAG-based execution shows up at *several* layers of an inference stack, not just at the kernel-launch level. Three distinct DAG abstractions are common:

| Layer | Node | Edge | Where it lives |
|---|---|---|---|
| **Model graph** | High-level operator (matmul, attention, layer-norm) | Tensor data flow | TensorRT engine, JAX / XLA HLO, ONNX graph, PyTorch FX |
| **Compilation IR graph** | Primitive op or fused cluster | Use-def chain | `torch.compile` / Inductor, XLA, TVM, MLIR |
| **Kernel-launch graph (CUDA Graphs)** | A single CUDA kernel launch | Stream / event dependency | CUDA driver |

These are stacked: a model graph is lowered through a compilation IR graph into a sequence of kernel launches, which CUDA Graphs then captures as a kernel-launch DAG. Different production frameworks own different layers:

- **vLLM** [VLLM] is a runtime / scheduler — it does *not* build its own model-level DAG. It runs PyTorch models more or less directly and leans on **CUDA Graphs** for the kernel-launch DAG and (in v1) `torch.compile` for the compilation IR DAG. The DAG that vLLM most directly owns is the captured CUDA Graph cache.
- **TensorRT-LLM** [TENSORRT-LLM] *does* build its own model-level DAG: a TensorRT engine is a serialized DAG of fused operators with shape and precision baked in. The engine is consumed by TensorRT's own runtime, which separately uses CUDA Graphs underneath. TensorRT-LLM therefore owns both the model-level DAG and the kernel-launch DAG explicitly.
- **JAX-based stacks** (Pathways, MaxText) use **XLA HLO** as the model-level DAG; XLA's runtime can lower to CUDA Graphs as one of its execution backends.

**For the overhead analysis in §5, the relevant DAG is the kernel-launch one (CUDA Graphs)** — it is the layer that determines per-launch dispatch cost. Whether the model-level DAG exists (TensorRT engine, XLA HLO) or not (PyTorch eager) changes *which* kernels get launched (more or less fusion, different kernel implementations), but the per-launch tax mechanism is the same once a kernel sequence is fixed. The formula $t_{\mathrm{SW}} = L \cdot k \cdot \tau_{\mathrm{launch}}$ in §5.1 does not care which framework you are on — it needs only $k$ (kernels per layer, after fusion) and $\tau_{\mathrm{launch}}$ (per-launch cost, with or without CUDA Graphs). Aggressive model-level DAG compilation (TensorRT engine) tends to lower $k$ via more fusion; CUDA Graphs lower $\tau_{\mathrm{launch}}$. The two are independent levers and production stacks generally use both.

---

# 5. Mapping the Overhead to the llm_perf Framework

## 5.1 The Per-Round SW Formula

In a steady-state inflight pipeline ([pipeline_bubble.md §5](pipeline_bubble.md#5-the-first-order-correction)) with $B \ge PP$, microbatch size $\mathrm{mb} = B / PP$, and $PP$ microbatches in flight:

- Each microbatch through one stage requires $(L/PP) \cdot k$ kernel launches, where $k$ is the number of kernels per layer.
- Each device sequentially processes all $PP$ microbatches per pipeline round.
- Per-stage launches per round = $PP \cdot (L/PP) \cdot k = L \cdot k$, **independent of $PP$**.

The per-round SW (software, host-side dispatch) overhead on each device is therefore:

$$
t_{\mathrm{SW}} = L \cdot k \cdot \tau_{\mathrm{launch}}
$$

where $\tau_{\mathrm{launch}}$ is per-kernel dispatch latency (~7 μs without CUDA Graphs, ~1.5 μs with). $t_{\mathrm{SW}}$ is the total CPU-side and dispatch budget consumed per pipeline round per device.

## 5.2 Where It Plugs Into the Roofline

The framework's GPU-side per-stage step time (from [decode_model.py](../../llm_perf/core/decode_model.py)) is:

$$
t_{\mathrm{stage}} = \max(t_{\mathrm{compute}}^{\mathrm{eff}}, t_{\mathrm{mem}}) + \max(0, t_{\mathrm{comm}} - \rho \cdot t_{\mathrm{local}})
$$

The CPU-side dispatch composes with $t_{\mathrm{stage}}$ via the SW-overlap factor $\rho_{\mathrm{SW}} \in [0, 1]$, then the user-observed step time becomes:

$$
t_{\mathrm{step,user}} = \max\!\bigl(t_{\mathrm{stage}},\ \rho_{\mathrm{SW}} \cdot t_{\mathrm{stage}} + (1 - \rho_{\mathrm{SW}}) \cdot (t_{\mathrm{stage}} + t_{\mathrm{SW}}),\ t_{\mathrm{SW}}\bigr) \cdot \gamma_{\mathrm{pp}}
$$

The two boundary cases:

- $\rho_{\mathrm{SW}} = 1$ (full overlap): $t_{\mathrm{step,user}} = \max(t_{\mathrm{stage}}, t_{\mathrm{SW}}) \cdot \gamma_{\mathrm{pp}}$. Async dispatch fully hides $t_{\mathrm{SW}}$ when it is shorter than the GPU's $t_{\mathrm{stage}}$; otherwise $t_{\mathrm{SW}}$ is a hard floor and the GPU starves.
- $\rho_{\mathrm{SW}} = 0$ (no overlap): $t_{\mathrm{step,user}} = (t_{\mathrm{stage}} + t_{\mathrm{SW}}) \cdot \gamma_{\mathrm{pp}}$. CPU dispatch and GPU work serialize; $t_{\mathrm{SW}}$ adds linearly.

The three knobs that drive $t_{\mathrm{SW}}$ are exposed on `TuningSpec`:

- `kernels_per_layer_compute` — compute kernels per layer (default 10, after typical fusion).
- `kernels_per_collective_call` — NCCL kernels per collective call (default 2).
- `kernel_launch_us` — $\tau_{\mathrm{launch}}$ (default 1.5 μs with CUDA Graphs; set to ~7 μs to model eager-mode without graphs; set to 0 to disable the SW term entirely).
- `sw_overlap_factor` — $\rho_{\mathrm{SW}}$ (default 1.0 — full overlap; see caveat below).

**Is $\rho_{\mathrm{SW}} = 1$ realistic?** It is the upper-end case, not the empirical average. The 1.0 default accurately models CUDA-Graphs-replayed steady-state on production stacks (TensorRT-LLM, vLLM, SGLang) where the CPU's cost per microbatch is one `cudaGraphLaunch` (~1.5 μs) while the GPU runs ms of kernels queued from that single API call — three orders of magnitude of slack, which is effectively perfect overlap. Empirically these stacks measure $\rho_{\mathrm{SW}} \approx 0.85$–$0.95$ in production. Eager-mode PyTorch / Python serving paths sit at $\rho_{\mathrm{SW}} \approx 0.3$–$0.6$ because Python interpreter overhead between kernel launches breaks the CPU-runs-ahead invariant. The framework's 1.0 default matches its roofline philosophy (give the optimistic upper bound; users dial down to model deployment imperfections). Crucially, $t_{\mathrm{SW}}$ is still a *hard floor* when $t_{\mathrm{SW}} > t_{\mathrm{stage}}$ regardless of $\rho_{\mathrm{SW}}$, so the optimistic default does not hide the dispatch tax in the SW-bound regime — it only matters in the GPU-bound regime where the launch budget *can* in principle be hidden.

For users wanting a more cautious default, $\rho_{\mathrm{SW}} = 0.85$ captures "production with CUDA Graphs but not perfectly tuned"; $\rho_{\mathrm{SW}} = 0.5$ models eager-mode Python serving; $\rho_{\mathrm{SW}} = 0$ is the strict additive bound. Override per-deployment via `TuningSpec.sw_overlap_factor` or by adding `sw_overlap_factor: <value>` to the tuner JSON.

## 5.3 Numerical Mapping for GPT-1.8T MoE on GB200

Plugging the values used elsewhere in this document set ($L = 120$, $k = 12$, $\tau_{\mathrm{launch}} \in \{7, 1.5\}$ μs):

$$
t_{\mathrm{SW,no\_graphs}} = 120 \cdot 12 \cdot 7\ \mathrm{\mu s} = 10080\ \mathrm{\mu s}
$$

$$
t_{\mathrm{SW,graphs}} = 120 \cdot 12 \cdot 1.5\ \mathrm{\mu s} = 2160\ \mathrm{\mu s}
$$

Comparing to the framework's $t_{\mathrm{stage}}$ at representative operating points:

| Shape | mb | $t_{\mathrm{stage}}$ | $t_{\mathrm{SW,no\_graphs}}/t_{\mathrm{stage}}$ | $t_{\mathrm{SW,graphs}}/t_{\mathrm{stage}}$ |
|---|---:|---:|---:|---:|
| PP=60 TP=1 | 1 | 2.2 ms | 459% | 98% |
| PP=60 TP=1 | 16 | 6.9 ms | 146% | 31% |
| PP=60 TP=1 | 64 | 22.0 ms | 46% | 10% |
| PP=8 TP=8 | 1 | 1.8 ms | 560% | 120% |
| PP=8 TP=8 | 64 | 4.3 ms | 236% | 50% |
| PP=1 TP=64 | 1 | 1.8 ms | 570% | 122% |

Reading the table left-to-right: at fixed PP, increasing mb is the only effective lever — $t_{\mathrm{SW}}$ is constant in mb while $t_{\mathrm{stage}}$ grows roughly linearly. At fixed mb=1 the launch budget exceeds the GPU's per-round work everywhere; at mb=64 with CUDA Graphs the overhead drops to a manageable 10-50%. Reading top-to-bottom: the launch overhead does *not* favor any particular partition shape — the formula cancels $PP$ — so the per-round overhead is shape-independent, and the column-to-column ratios reflect $t_{\mathrm{stage}}$ only.

The strong implication: *for any partition shape*, the framework's roofline is meaningful only when mb is large enough to amortize $t_{\mathrm{SW}}$. For decode this means CUDA Graphs *and* $B/PP \ge 16$ in practice. Sweeps that hold $B$ very low ($B < 8 \cdot PP$) are operating in a regime the framework does not currently price.

---

# 6. Practical Implications

For users running partition sweeps:

1. **CUDA Graphs are not optional.** Any production-realistic comparison should assume CUDA Graphs are enabled. The factor-5× difference between the no-graphs and graphs columns dwarfs most partition-shape differences and would otherwise dominate the result.

2. **Decode at mb=1 is launch-bound on this hardware.** The framework's $t_{\mathrm{stage}}$ at mb=1 is ~50% of the actual user-observed step time even with CUDA Graphs. Sweeps that report mb=1 frontiers should be read as upper bounds on throughput, not predicted throughput.

3. **Increasing $B$ amortizes the launch budget linearly.** If the deployment service-level objective (SLO) allows higher mb (e.g., mb=16 or 64), the framework's $t_{\mathrm{stage}}$ becomes a much closer approximation. This is consistent with production guidance to operate at the largest $B$ the latency budget tolerates.

4. **Partition shape is decoupled from launch overhead** — for the same mb, the per-round SW budget is the same across PP/TP/EP/SP shapes. So when comparing partitions, the framework's existing $t_{\mathrm{stage}}$ ordering is correct *up to a constant offset* that affects all shapes equally; the relative comparison is unchanged. The absolute throughput is what is over-stated.

5. **Adding $t_{\mathrm{SW}}$ as a TuningSpec knob** is the natural extension if the framework needs to predict absolute TPOT in production-realistic regimes. It is a one-line addition to the latency model (Section 5.2), gated on three new tuner fields.

For framework developers: incorporating $t_{\mathrm{SW}}$ would let the partition optimizer trade off $t_{\mathrm{stage}}$ against $t_{\mathrm{SW}}$ at small mb, which would in turn surface a real preference for moderate $B$ on the Pareto frontier. Currently, with $t_{\mathrm{SW}} = 0$, the optimizer sees no penalty for picking mb=1 partitions, which is not what production sees.

---

# References

**[CUDA-RUNTIME]**
NVIDIA Corporation. (2024).
*CUDA Runtime API Reference Manual.*
https://docs.nvidia.com/cuda/cuda-runtime-api/
→ Authoritative reference for `cudaLaunchKernel`, `cudaGraphInstantiate`, `cudaGraphLaunch`, and stream-capture semantics.

**[HOPPER-PROG-GUIDE]**
NVIDIA Corporation. (2024).
*Hopper Tuning Guide & CUDA C Programming Guide.*
https://docs.nvidia.com/cuda/hopper-tuning-guide/
→ Per-architecture launch overhead numbers; Tensor Core programming details.

**[VLLM]**
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*
SOSP 2023. arXiv:2309.06180.
→ Production CUDA-Graphs deployment for decode; per-batch-size graph capture.

**[TENSORRT-LLM]**
NVIDIA Corporation. (2023–2025).
*TensorRT-LLM: Open-Source Library for Optimized LLM Inference.*
https://github.com/NVIDIA/TensorRT-LLM
→ Engine-level CUDA Graphs with batch-size buckets.

**[FA2]**
Dao, T. (2023).
*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
arXiv:2307.08691.
→ Fused attention as a single kernel; canonical fusion example reducing the per-layer kernel count.

**[MEGATRON3]**
Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., & Zaharia, M. (2021).
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*
SC '21. arXiv:2104.04473.
→ Microbatch sizing tradeoffs; the analogous training-time argument for keeping launch overhead tractable.

---

## Cross-References

- [pipeline_bubble.md](pipeline_bubble.md) — The pipeline-round semantics underlying the per-round formula in §5.1.
- [practical_pp_choice.md §3.3](practical_pp_choice.md#33-microbatch-granularity-and-kernel-launch-overhead) — The production-side argument that uses the formula derived here.
- [batched_decode.md](batched_decode.md) — Why operating at large $B$ is the primary lever for amortizing fixed per-round overheads.
- [../modeling/decode.md §6.3](../modeling/decode.md#63-pipeline-bubble-tps-and-ttps) — The framework's current $t_{\mathrm{stage}}$ formulation, which §5.2 proposes extending.
- [../../llm_perf/core/decode_model.py](../../llm_perf/core/decode_model.py) — Code where the $t_{\mathrm{SW}}$ term would plug in.
