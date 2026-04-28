# Why Production Inference Prefers TP/EP over Deep PP — Practical Considerations

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, pipeline parallelism, tensor parallelism, expert parallelism, production serving, time-to-first-token (TTFT), continuous batching, KV cache, microbatching, deployment

---

# Table of Contents

- [1. The Apparent Paradox](#1-the-apparent-paradox)
- [2. The Steady-State Decode Lens (Where the Roofline Lives)](#2-the-steady-state-decode-lens-where-the-roofline-lives)
- [3. Where Production Departs from the Steady-State Decode Lens](#3-where-production-departs-from-the-steady-state-decode-lens)
  - [3.1 Maintaining $B \ge PP$ Is Brittle Under Bursty Traffic](#31-maintaining-b-ge-pp-is-brittle-under-bursty-traffic)
  - [3.2 Prefill / Time-To-First-Token Scales Linearly With PP](#32-prefill--time-to-first-token-scales-linearly-with-pp)
  - [3.3 Microbatch Granularity and Kernel-Launch Overhead](#33-microbatch-granularity-and-kernel-launch-overhead)
  - [3.4 KV Cache Sharding](#34-kv-cache-sharding)
  - [3.5 Stage Imbalance and Lock-Step Scheduling](#35-stage-imbalance-and-lock-step-scheduling)
  - [3.6 What Production Frameworks Actually Default To](#36-what-production-frameworks-actually-default-to)
- [4. When PP Is Genuinely the Right Call](#4-when-pp-is-genuinely-the-right-call)
- [References](#references)

---

# 1. The Apparent Paradox

The decode roofline used by this framework — and by any analytical model that respects the autoregressive bubble correction $\max(1, PP/B)$ from [pipeline_bubble.md §5](pipeline_bubble.md#5-the-first-order-correction) — predicts that pipeline parallelism (PP) at full pipeline ($B \ge PP$) is approximately as good as tensor parallelism (TP) for per-user time-per-output-token (TPOT). Expert parallelism (EP), the per-layer expert sharding used in Mixture-of-Experts (MoE) models, sits in the same cost class as TP — a per-layer collective with no pipeline-depth penalty. Both PP and TP shrink the per-device weight footprint by the same factor and amortize high-bandwidth memory (HBM) traffic across the same number of devices; PP additionally pays only a small point-to-point activation hop per stage boundary, which is cheap relative to TP's per-layer all-reduce.

When the partition optimizer sees this roofline, it greedily picks deep PP — `pareto_vs_io.ipynb` and `pareto_vs_cluster_size.ipynb` both show winning shapes like `PP=60, TP=1, EP=1` for GPT-1.8T on NVL72 (see also [pp_vs_tp_decode_scaling.md](pp_vs_tp_decode_scaling.md) for the same observation). Yet production serving stacks — vLLM [VLLM], TensorRT-LLM [TENSORRT-LLM], NVIDIA Dynamo [DYNAMO] — default to TP intra-node and use PP only when forced by capacity. The roofline is internally consistent; the production preference is empirically robust; resolving the apparent disagreement is the topic of this document.

The short answer: the steady-state decode roofline assumes a regime that is hard to *sustain* in production, and silently ignores several costs that production care about. Each of those costs is small or zero in steady-state pure decode but becomes load-bearing once you consider arrival burstiness, prefill, key-value (KV) cache footprint, kernel-tile granularity, and multi-tenant scheduling. This explainer walks through them.

---

# 2. The Steady-State Decode Lens (Where the Roofline Lives)

The decode roofline this framework implements (see [decode.md §7](../modeling/decode.md#7-pipeline-bubble-kernel-launch-overhead-and-throughput)) computes $t_{\mathrm{step,user}} = t_{\mathrm{stage}} \cdot \max(1, PP/B)$. At $B \ge PP$ the bubble factor is one, and $t_{\mathrm{stage}}$ scales as $1/PP$ because each stage holds only $L/PP$ layers' worth of weights. The roofline therefore rewards PP almost as effectively as TP for the inflight-batched, post-prefill, decode-only operating point — which is exactly the operating point the formula is *for*.

Three implicit assumptions underpin this favorable accounting:

1. **Inflight batching keeps the pipeline full.** $B \ge PP$ users are concurrently active and the scheduler can microbatch them across stages without forming a stall. This is the regime [VLLM] and [ORCA] introduced for transformer inference.
2. **Prefill is out of scope.** The formula models the steady decode loop after the prompt has already been processed. Prefill latency for any *new* request entering the system is not priced.
3. **Per-stage compute is on the GPU's efficient operating curve.** Microbatch sizes large enough for matmul kernels and quantization tiles to run at peak; kernel launch, dispatch, and synchronization overheads are negligible.

When all three hold simultaneously and indefinitely, the roofline is accurate and PP is in fact a fine choice. Each section below describes a regime where one or more assumptions break, and what that costs you.

---

# 3. Where Production Departs from the Steady-State Decode Lens

## 3.1 Maintaining $B \ge PP$ Is Brittle Under Bursty Traffic

The bubble factor $\max(1, PP/B)$ is a step function — it stays at one as long as the active-batch count is at least $PP$, then jumps the moment a request finishes and the count dips even momentarily below $PP$. Production traffic is rarely smooth: arrivals are bursty, sequence lengths vary, and individual requests complete asynchronously as their stop tokens fire. Continuous-batching schedulers ([ORCA], [VLLM]) raise the *average* utilization but do not eliminate transient dips. A deployment sized for $PP=60$ needs ~60 sustained concurrent decode sequences to avoid bubble penalties; a deployment sized for $PP=8$ needs only ~8. The penalty for falling below threshold is multiplicative and per-step, so even a single dip in a long-running session shows up directly in the user-observed tail TPOT.

The serving systems literature has documented this sensitivity: [SARATHI] and [DISAGG-PREFILL] both motivate their scheduling improvements by showing that naive request batching produces head-of-line blocking and tail-latency degradation; both recommend keeping the working set wide enough that one stalled request cannot starve the pipeline. Deeper PP makes the working-set requirement larger and the system more sensitive to traffic variance.

## 3.2 Prefill / Time-To-First-Token Scales Linearly With PP

For decode, the bubble factor is the right pricing of pipeline depth. For prefill, *there is no inflight batching* on a single request: a new user's prompt is one indivisible unit of work that must traverse all $PP$ stages sequentially before the first token can be sampled. Time-to-first-token (TTFT) for any individual request is therefore approximately $PP \cdot t_{\mathrm{stage,prefill}}$, plus the prompt's compute fraction. Doubling PP roughly doubles a user's TTFT — a metric production deployments take seriously enough that [SARATHI] and [DISAGG-PREFILL] redesign the scheduler around it. [DISAGG-PREFILL] (DistServe) goes further and physically separates the prefill cluster from the decode cluster precisely because their parallelism preferences disagree.

A pure decode roofline (this framework's `decode.md` and the partition sweep notebooks) does not surface this cost. A production deployment that runs the same model on the same hardware sees TTFT directly in its service-level objective (SLO) dashboard.

## 3.3 Microbatch Granularity and Kernel-Launch Overhead

Inflight batching in a $PP$-stage pipeline distributes $B$ in-flight requests across the stages. Steady-state utilization requires a microbatch at every stage; the natural microbatch size is $\mathrm{mb} = B/PP$. At $B = PP$ this is one — every stage processes a single sequence per tick. The per-microbatch compute is *not* the dominant cost at small $\mathrm{mb}$ for decode; two distinct overheads are, and they pull in different directions.

**Tensor Core tile floor (matters for compute-bound paths, not memory-bound decode).** GPU Tensor Core instructions have hardware-fixed minimum M-tile sizes: FP16/BF16 `wmma` and `mma.sync` instructions accept M=16 minimum; FP8 `wgmma` (Hopper) has a *fixed* M=64 ([H100-SPEC] §3.5; [PTX-ISA] §9.7.14.5); FP4 `tcgen05.mma` (Blackwell) is similar or stricter. Below the floor, the matrix multiply (GEMM) cannot use the Tensor Core path at all. Above the floor, throughput ramps gradually and only saturates at 4-8× the floor (M=128-256 for FP16, M=256-512 for FP8). DeepSeek-V3 [DEEPSEEK-V3] §3.3 explicitly motivates their FP8 microbatch sizing around the `wgmma` M-tile. **However**, this floor only bites in *compute-bound* regimes — primarily prefill and large-$B$ batched decode near the memory-compute crossover. Pure mb=1 decode is fully HBM-bandwidth-bound: per-stage compute is ~1-80 μs while per-stage memory traffic is ~1.8-2.2 ms (a ~30× gap). At that operating point, whether the GEMM uses the Tensor Core path is irrelevant — the matmul is waiting on HBM either way.

**Kernel-launch overhead (the dominant SW cost at small B).** Each microbatch through each stage requires its own ~12 kernel launches per layer (layer-norm, projections, attention, FFN, residuals). In steady-state inflight batching with $\mathrm{mb} = B/PP$, exactly $PP$ microbatches are in flight, and each device sequentially processes all $PP$ of them per pipeline round. Per-stage kernel launches per round therefore equal $L \cdot k$, where $k$ is kernels per layer — **independent of $PP$**:

$$
\mathrm{launches\_per\_round} = PP \cdot \frac{L}{PP} \cdot k = L \cdot k.
$$

Per-round SW overhead is $L \cdot k \cdot \tau_\mathrm{launch}$, where $\tau_\mathrm{launch}$ is per-kernel dispatch latency: ~5-10 μs without CUDA Graphs, ~1-2 μs with (see [kernel_launch_overhead.md](kernel_launch_overhead.md) for the full mechanism). For GPT-1.8T MoE ($L = 120$) at $k = 12$, this is **~10 ms per round without CUDA Graphs** or **~2.16 ms with**. Compared to the framework's per-round $t_{\mathrm{stage}}$:

| Shape | mb | $t_{\mathrm{stage}}$ | SW% (no graphs) | SW% (graphs) |
|---|---:|---:|---:|---:|
| PP=60 TP=1 | 1 | 2.2 ms | 459% | 98% |
| PP=60 TP=1 | 16 | 6.9 ms | 146% | 31% |
| PP=60 TP=1 | 64 | 22.0 ms | 46% | 10% |
| PP=8 TP=8 | 1 | 1.8 ms | 560% | 120% |
| PP=8 TP=8 | 64 | 4.3 ms | 236% | 50% |
| PP=1 TP=64 | 1 | 1.8 ms | 570% | 122% |
| PP=1 TP=64 | 64 | 2.1 ms | 485% | 104% |

Three findings stand out. **(a)** Decode at mb=1 is launch-overhead-dominated even with CUDA Graphs (~100% of $t_{\mathrm{stage}}$); without graphs it is ~5×$t_{\mathrm{stage}}$. CUDA Graphs are not optional for production decode — they are the difference between a tractable serving system and one bottlenecked on CPU-side dispatch. **(b)** The launch overhead is *roughly the same* across PP shapes at fixed mb, because the per-round formula is $L \cdot k$ independent of PP. The shape-to-shape differences in the table come from $t_{\mathrm{stage}}$ varying with the per-device weight footprint, not from the launch overhead itself. **(c)** The real lever is **mb (via large $B$)**, not the partition shape. To drive SW% below 50% with CUDA Graphs you need $\mathrm{mb} \ge 16$. This is the underlying mechanism behind production's preference for the largest $B$ the SLO allows.

The implication for the partition optimizer: the framework's roofline assumes the per-stage compute term runs at the GPU's peak curve. At $\mathrm{mb} = 1$ this is silently optimistic, by a factor of ~2 with CUDA Graphs and ~5 without. Deep PP at fixed $B$ does *not* worsen this — the launch budget is constant across PP — but small operating $B$ and missing CUDA Graphs *do*. [MEGATRON3] §4 covers the analogous training-time tradeoff and motivates the interleaved 1F1B schedule for the same reason: keeping kernel-launch overhead tractable when microbatches shrink.

## 3.4 KV Cache Sharding

Per-token decode HBM traffic comes from two places: weight reads ($T_\theta$) and KV-cache reads ($T_{\mathrm{KV}}$). Both PP and TP shard weights — PP across stages, TP within each layer — but their effect on KV is different.

- **TP shards the KV cache.** Each TP rank stores and reads only a $1/TP$ fraction of every layer's KV per active sequence ([MEGATRON]).
- **PP does not shard the KV cache.** Each stage holds the full KV for the $L/PP$ layers assigned to it, summed over every in-flight sequence.

For long-context decode, KV traffic per token grows with sequence length $S$ and dominates over weights ([VLLM] §3 documents this transition; see also [batched_decode.md](batched_decode.md) and [why_flops_doesnt_help_at_long_context.md](why_flops_doesnt_help_at_long_context.md)). At long context TP is the only axis that shrinks the KV term; PP does not help. This is one reason production deployments that target long-context use cases ([MOONCAKE], [DISAGG-PREFILL]) keep TP wide and PP shallow.

The decode roofline does account for KV traffic in $t_{\mathrm{mem}}$, but the partition optimizer often sees a dense-attention model with a manageable $S$ where the KV term is not the dominant bottleneck — so it tolerates PP-heavy shapes that would lose to TP at longer contexts. Re-running the same sweep at $S = 32{,}768$ or $S = 131{,}072$ flips the optimum back toward TP for exactly this reason.

## 3.5 Stage Imbalance and Lock-Step Scheduling

PP requires every stage to advance in lock-step — synchronous send/receive, with each tick gated by the slowest stage. Two structural sources of stage imbalance are well documented:

- **Embedding and language-model (LM) head.** The first stage typically holds the input embedding and the last stage holds the output projection. Both contribute work proportional to the vocabulary size, not the layer count, so they do not scale with $L/PP$. [MEGATRON] §3 acknowledges this and either pins them to dedicated devices or accepts the imbalance.
- **Microbatch warmup and drain.** Even with [MEGATRON3]'s interleaved 1F1B schedule, requests flowing through the pipeline encounter warmup and drain phases when the active batch turns over. In training these are amortized over millions of optimizer steps; in inference each request sees them.

Multi-tenancy on shared infrastructure adds a further constraint: different requests are at different decode steps, and the lock-step requirement makes it difficult to mix workloads with different PP depth or to dynamically resize the pipeline as load shifts. [DYNAMO] and [MOONCAKE] both invest substantial scheduling complexity in working around this; deeper PP makes the work harder.

## 3.6 What Production Frameworks Actually Default To

The empirical pattern across published production stacks aligns with this analysis:

- **vLLM** [VLLM] — TP-first; PP support added later and recommended primarily for fitting models that exceed single-node memory.
- **TensorRT-LLM** [TENSORRT-LLM] — supports PP and TP; deployment guidance leads with TP for low-latency single-node serving and reaches for PP only when the model spans nodes.
- **NVIDIA Dynamo** [DYNAMO] — disaggregates prefill from decode workers and uses PP across nodes for the largest models, but generally keeps intra-node parallelism dominated by TP and EP.
- **DeepSpeed-Inference** ([DEEPSPEED-MOE] for the MoE side; the broader DeepSpeed-Inference [AMINABADI22]) — TP-first inside a node; PP at coarse granularity only.

These defaults are not contradicting the roofline; they are pricing in §3.1–§3.5.

---

# 4. When PP Is Genuinely the Right Call

The same analysis flips when the structural constraints favor PP:

1. **Capacity (model does not fit otherwise).** Trillion-parameter models exceed a single node's HBM. PP is the only axis that further reduces per-device weight footprint after TP/EP have shared the within-node fabric. [NVIDIA-BLOG]'s GPT-1.8T MoE deployment falls in this category.
2. **Sustained large-batch throughput-oriented serving.** Offline batch scoring, dataset evaluation, or long-context generation jobs that maintain $B \gg PP$ continuously can fully amortize the bubble; $t_{\mathrm{stage}}$'s $1/PP$ scaling is then a real win, matching the roofline.
3. **Cross-node where TP bandwidth is the bottleneck.** Inter-node fabrics (Ethernet, InfiniBand) have far lower bandwidth than intra-node NVLink. TP across the node boundary is expensive; PP across it is cheap because PP pays only point-to-point activation hops, not all-reduces. [VALIANT81] and [SLINGSHOT] discuss the structural reason. The disaggregated-prefill literature ([DISAGG-PREFILL], [MOONCAKE]) takes advantage of this exact asymmetry.
4. **Latency targets that are loose enough to tolerate some TTFT inflation.** Asynchronous content generation, summarization pipelines, and other "best-effort" workloads can absorb the prefill-PP cost from §3.2.

The rule of thumb in [NVIDIA-BLOG] aligns with this: keep PP small enough to fit the model, and use TP/EP for the rest of the available scale-up width.

---

# References

**[VLLM]**
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*
SOSP 2023. arXiv:2309.06180.
→ Continuous batching for inference; KV-cache paging; the canonical reference for inflight batching.

**[ORCA]**
Yu, G-I., Jeong, J.S., Kim, G-W., Kim, S., & Chun, B-G. (2022).
*Orca: A Distributed Serving System for Transformer-Based Generative Models.*
OSDI 2022.
→ Iteration-level scheduling; introduced the continuous-batching idea that vLLM productized.

**[SARATHI]**
Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B.S., & Ramjee, R. (2023).
*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.*
arXiv:2308.16369.
→ Chunked prefill; head-of-line blocking; SLO-aware scheduling.

**[DISAGG-PREFILL]**
Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024).
*DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.*
OSDI 2024. arXiv:2401.09670.
→ Physical separation of prefill and decode workers; different parallelism per phase.

**[MOONCAKE]**
Qin, R., Li, Z., He, W., Zhang, M., Wu, Y., Zheng, W., & Xu, X. / Moonshot AI. (2024).
*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.*
arXiv:2407.00079.
→ KV-centric scheduling; long-context production deployment.

**[DYNAMO]**
NVIDIA Corporation. (2024–2025).
*NVIDIA Dynamo: Distributed Inference Serving Framework.*
→ Production serving stack; disaggregated prefill/decode; multi-node scheduling.

**[TENSORRT-LLM]**
NVIDIA Corporation. (2023–2025).
*TensorRT-LLM: Open-Source Library for Optimized LLM Inference.*
https://github.com/NVIDIA/TensorRT-LLM
→ Reference production stack; deployment guidance favors TP intra-node, PP for cross-node fit.

**[MEGATRON]**
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019).
*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*
arXiv:1909.08053.
→ Original TP/PP layout; embedding and LM-head pinning; stage imbalance discussion.

**[MEGATRON3]**
Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., & Zaharia, M. (2021).
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*
SC '21. arXiv:2104.04473.
→ Interleaved 1F1B schedule; microbatch tradeoff analysis; the canonical bubble formula at scale.

**[DEEPSPEED-MOE]**
Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R.Y., Awan, A.A., Rasley, J., & He, Y. (2022).
*DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.*
ICML 2022. arXiv:2201.05596.
→ Expert parallelism; all-to-all routing; production MoE deployment.

**[AMINABADI22]**
Aminabadi, R.Y., Rajbhandari, S., Awan, A.A., Li, C., Li, D., Zheng, E., Ruwase, O., Smith, S., Zhang, M., Rasley, J., & He, Y. (2022).
*DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale.*
SC '22. arXiv:2207.00032.
→ Production inference with TP-first parallelism; PP at coarse granularity for fit only.

**[POPE22]**
Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2022).
*Efficiently Scaling Transformer Inference.*
arXiv:2211.05102.
→ Comprehensive analysis of inference partitioning across prefill and decode; weight-stationary vs weight-gathered partitions; the canonical reference for phase-aware parallelism choice.

**[NVIDIA-BLOG]**
Mao, A., Yoon, B., et al. (2025).
*Demystifying AI Inference Deployments for Trillion-Parameter Large Language Models.*
NVIDIA Technical Blog.
→ Production deployment guidance for GPT-1.8T MoE on GB200; PP for fit, TP/EP for performance.

**[VALIANT81]**
Valiant, L.G. (1981).
*Universality Considerations in VLSI Circuits.*
IEEE Transactions on Computers C-30(2).
→ Cross-node-bandwidth-asymmetry as a structural constraint on partition choice.

**[SLINGSHOT]**
De Sensi, D., Di Girolamo, S., McMahon, K.H., Roweth, D., & Hoefler, T. (2020).
*An In-Depth Analysis of the Slingshot Interconnect.*
SC '20.
→ Quantitative comparison of NVLink-class vs Ethernet-class fabrics; motivates phase-aware partitioning.

**[H100-SPEC]**
NVIDIA Corporation. (2022).
*NVIDIA H100 Tensor Core GPU Architecture.*
NVIDIA Whitepaper WP-10792-001.
→ Hopper Tensor Core capabilities; `wgmma` instruction set; FP8 / FP16 / BF16 throughput per SM.

**[PTX-ISA]**
NVIDIA Corporation. (2024).
*Parallel Thread Execution ISA Application Guide.*
https://docs.nvidia.com/cuda/parallel-thread-execution/
→ Authoritative `wgmma` / `mma.sync` / `tcgen05.mma` instruction shapes per architecture; the M-tile floors used in §3.3.

**[CUTLASS]**
NVIDIA Corporation. (2017–2025).
*CUTLASS: CUDA Templates for Linear Algebra Subroutines.*
https://github.com/NVIDIA/cutlass
→ Reference open-source GEMM library; tile-shape templates expose the per-architecture Tensor Core M-tile, K-tile, and N-tile parameters.

**[DEEPSEEK-V3]**
DeepSeek-AI. (2024).
*DeepSeek-V3 Technical Report.*
arXiv:2412.19437.
→ Production FP8 training at scale; §3.3 motivates microbatch sizing around the Hopper `wgmma` M=64 tile.

---

## Cross-References

- [pipeline_bubble.md](pipeline_bubble.md) — The bubble correction the steady-state roofline does include; necessary background for §2.
- [pp_vs_tp_decode_scaling.md](pp_vs_tp_decode_scaling.md) — In-model PP-vs-TP comparison; this doc is the production-side complement.
- [batched_decode.md](batched_decode.md) — Why $B \ge PP$ is the primary escape from the bubble regime.
- [why_flops_doesnt_help_at_long_context.md](why_flops_doesnt_help_at_long_context.md) — The KV-traffic-dominance regime referenced in §3.4.
- [when_hierarchical_scale_up_matters.md](when_hierarchical_scale_up_matters.md) — Companion: another case where the optimizer routes around a cost the roofline doesn't price.
- [../modeling/decode.md §7](../modeling/decode.md#7-pipeline-bubble-kernel-launch-overhead-and-throughput) — Formal decode roofline.
- [../modeling/e2e.md](../modeling/e2e.md) — End-to-end metric that does include prefill TTFT.
