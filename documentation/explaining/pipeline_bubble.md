# Pipeline Parallelism and the Bubble: Why PP Does Not Help Single-User Decode

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, pipeline parallelism, pipeline bubble, microbatch, decode, TPOT, inflight batching, 1F1B schedule, GPipe, user-observed latency

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Motivation](#1-motivation)
- [2. What Pipeline Parallelism Does](#2-what-pipeline-parallelism-does)
- [3. The Bubble: Geometric Intuition](#3-the-bubble-geometric-intuition)
  - [3.1 Full Pipeline (B ≥ PP)](#31-full-pipeline-b--pp)
  - [3.2 Underfilled Pipeline (B < PP)](#32-underfilled-pipeline-b--pp)
- [4. Training vs Inference: Different Bubble Regimes](#4-training-vs-inference-different-bubble-regimes)
  - [4.5 Microbatching vs Inflight Batching — How the Pipeline Gets Filled](#45-microbatching-vs-inflight-batching--how-the-pipeline-gets-filled)
- [5. The First-Order Correction](#5-the-first-order-correction)
  - [5.1 The Formula](#51-the-formula)
  - [5.2 Why It Is Only First-Order](#52-why-it-is-only-first-order)
- [6. Why PP Does Not Help Single-User Decode](#6-why-pp-does-not-help-single-user-decode)
  - [6.1 The Cancellation Argument](#61-the-cancellation-argument)
  - [6.2 Numerical Illustration (NVIDIA Blog Reproduction)](#62-numerical-illustration-nvidia-blog-reproduction)
- [7. When PP *Does* Help](#7-when-pp-does-help)
- [8. Practical Implications for Partition Sweeps](#8-practical-implications-for-partition-sweeps)
- [9. FAQ](#9-faq)
- [References](#references)

---

<div style="page-break-before: always;"></div>

# 1. Motivation

Pipeline Parallelism (PP) is one of the four common parallelism axes in LLM inference (alongside DP, TP, EP, SP). It is the most effective way to fit a model that does not fit on a single device into HBM: by assigning distinct transformer layers to distinct devices, per-device weight memory scales as $1/PP$. For trillion-parameter models this is not optional — a 1.8T-parameter MoE model at FP4 occupies ~0.9 TB of weights, which cannot fit on any single GPU today.

However, **PP interacts badly with low batch sizes at decode time**. A naive roofline model predicts that PP reduces per-token latency by a factor $PP$ — because each stage sees only $L/PP$ layers of work. This is wrong. At batch size $B=1$, PP's latency benefit is exactly cancelled by a **bubble** penalty: every token must traverse all PP stages sequentially before emerging at the pipeline tail.

This document explains the bubble concept in detail: what it is, why it appears, how we model it, and why its presence dominates partition choice at low $B$.

---

# 2. What Pipeline Parallelism Does

A transformer with $L$ layers is split into $PP$ contiguous groups called **stages**. Stage $j$ lives on a dedicated device (or a dedicated TP/EP group) and owns exactly $L / PP$ layers of the model [MEGATRON]. A forward pass for one token flows through the stages in order:

$$
\text{input} \;\to\; \text{stage 1} \;\to\; \text{stage 2} \;\to\; \cdots \;\to\; \text{stage PP} \;\to\; \text{output token}
$$

Each stage hands its output activations to the next stage over a point-to-point link (the PP hop; see [decode.md §5.1](../modeling/decode.md#51-pipeline-parallel-pp-hop)).

The cost of one stage processing one microbatch is the **per-stage step time** $t_{\text{stage}}$, defined in [decode.md §6.3.1](../modeling/decode.md#631-per-stage-step-time). This is the overlap-aware sum of local compute/memory time and any unhidden communication inside that stage.

Because each stage only holds $L/PP$ layers, $t_{\text{stage}}$ scales as $1/PP$ (fewer layers per stage → less weight traffic and compute per stage). This is where the "PP makes things faster" intuition comes from.

---

# 3. The Bubble: Geometric Intuition

The key question is: **how many tokens emerge from the pipeline per unit time?**

## 3.1 Full Pipeline (B ≥ PP)

When there are enough microbatches in flight — specifically $B \ge PP$ — every stage can be kept busy on a different microbatch simultaneously. This is the steady-state illustrated by the canonical GPipe/1F1B diagrams [GPIPE, MEGATRON3]:

```
time →          t=0   t=1   t=2   t=3   t=4   t=5   t=6
stage 1 (l 0..L/PP):  [A]   [B]   [C]   [D]   [E]   [F]   [G]
stage 2           :         [A]   [B]   [C]   [D]   [E]   [F]
stage 3           :               [A]   [B]   [C]   [D]   [E]
stage 4           :                     [A]   [B]   [C]   [D]
                                        ↑ tok A emerges, then B, C, D, ... every t_stage
```

After the initial $PP-1$ warmup steps, one token emerges **every** $t_{\text{stage}}$ seconds. The pipeline is said to be "full" — all $PP$ stages are doing useful work in parallel on different tokens. A user whose sequence is one of the microbatches (say, $A$) observes a new token every $t_{\text{stage}}$ seconds.

**Key insight:** the latency for a single token to traverse the pipeline is still $PP \cdot t_{\text{stage}}$ — but because new tokens enter the pipeline in a rolling fashion, the **throughput** (and user-observed inter-token time) is $t_{\text{stage}}$, not $PP \cdot t_{\text{stage}}$. This is the standard pipelining payoff, and is why classical training analysis celebrates PP.

## 3.2 Underfilled Pipeline (B < PP)

When $B < PP$ — most pressingly at $B=1$, the single-user case targeted by interactive chat — there are fewer microbatches than stages. At most $B$ stages can be active simultaneously; the remaining $PP - B$ stages sit idle:

```
time →          t=0     t=1     t=2     t=3     t=4     t=5     t=6     t=7
                                                [tok A emerges]                [tok B emerges]
stage 1 :  [A]   idle   idle    idle    [B]    idle    idle    idle
stage 2 :  idle  [A]    idle    idle    idle   [B]     idle    idle
stage 3 :  idle  idle   [A]     idle    idle   idle    [B]     idle
stage 4 :  idle  idle   idle    [A]     idle   idle    idle    [B]
```

The shaded "idle" cells are the **bubble** — stages that have no work to do because no token is currently at them. Token $A$ still takes $PP$ stage-steps to emerge (unchanged), but crucially **token $B$ cannot start until token $A$ clears out** (or, more precisely, until $A$ is at least $B$-stages ahead, which at $B=1$ means entirely out of the pipeline).

So at $B=1$, user-observed inter-token time equals $PP \cdot t_{\text{stage}}$ — the full pipeline depth — not $t_{\text{stage}}$.

At intermediate $1 < B < PP$, the argument generalizes: $B$ microbatches can coexist in the pipeline, each taking $PP \cdot t_{\text{stage}}$ to emerge, but they emerge at a rate of $B$ tokens per $PP \cdot t_{\text{stage}}$. From a single user's perspective (who owns one of those microbatches), tokens arrive at rate $1 / (PP \cdot t_{\text{stage}} / B) = B / (PP \cdot t_{\text{stage}})$. Equivalently, the user-observed step time is $(PP/B) \cdot t_{\text{stage}}$.

---

# 4. Training vs Inference: Different Bubble Regimes

The classical pipeline bubble literature [GPIPE, MEGATRON3, PIPEDREAM] studies a different setting than inference decode:

| Aspect | Training | Inference decode |
|---|---|---|
| Work unit | Microbatch of $B_{\text{micro}}$ samples | One new token per sequence per step |
| In-flight quantity | $N_{\text{micro}}$ (chosen freely, often $\gg PP$) | Number of concurrent sequences $B$ (bounded by HBM, user traffic) |
| Bubble shape | Triangular warmup + cooldown at batch boundaries [GPIPE §3.2] | Steady-state underfill when $B < PP$ |
| Relative bubble cost | $\frac{PP-1}{N_{\text{micro}} + PP - 1}$ [GPIPE, MEGATRON3] | $\max(1, PP/B) - 1$ (our formula) |
| Primary mitigation | Large $N_{\text{micro}}$; interleaved 1F1B schedules [MEGATRON3] | Inflight batching that raises $B \ge PP$ [VLLM, ORCA] |

In training, the dominant bubble is the **warmup/cooldown** at the start and end of a minibatch (before the pipeline fills and after the last microbatch enters). Making the minibatch longer — more microbatches per minibatch — amortizes this fixed cost. Interleaved 1F1B [MEGATRON3] further reduces it by scheduling multiple forward and backward passes interleaved across stages.

In inference decode, the situation is fundamentally different:

- Each decode *step* produces exactly one token per sequence. There is no multi-token "minibatch" to fill the pipeline with in one shot.
- The "microbatches" that flow through the pipeline are **individual sequences' single-token forward passes**. The number of microbatches in flight at any time equals the number of concurrent sequences $B$.
- There is no training-style warmup/cooldown bubble (inference is steady-state, not bounded by a minibatch boundary), but there *is* a steady-state **underfill** bubble whenever $B < PP$.

This distinction is important because the training literature's "$PP - 1$ bubble at start of each minibatch" formula does not apply directly to decode — decode has a *continuous* underfill problem, not a per-minibatch transient.

## 4.5 Microbatching vs Inflight Batching — How the Pipeline Gets Filled

Both training and inference rely on *some* mechanism to keep $PP$ stages simultaneously busy. The mechanisms differ because the underlying work structure differs.

### Microbatching (training)

In training, a single optimizer step operates on a **global batch** (e.g., 1024 samples). Memory limits force the global batch to be sliced into smaller **microbatches** of size $B_{\text{micro}}$, so that $N_{\text{micro}} = B_{\text{global}} / B_{\text{micro}}$ microbatches flow through the pipeline in one step:

$$
\underbrace{B_{\text{global}}}_{\text{1024 samples}}
\;\;=\;\;
\underbrace{N_{\text{micro}}}_{32}
\;\times\;
\underbrace{B_{\text{micro}}}_{32}
$$

Gradients from each microbatch are **accumulated** (summed) before the optimizer takes one step. Mathematically the result is identical to processing all 1024 samples at once; physically the work is split into 32 pipeline-friendly chunks [GPIPE]. The bubble fraction then depends on how many microbatches are in flight relative to $PP$:

$$
\text{bubble fraction} \;\approx\;
\frac{PP - 1}{N_{\text{micro}} + PP - 1}
\qquad \text{[GPIPE, MEGATRON3]}
$$

Choosing $N_{\text{micro}} \gg PP$ amortizes the warmup/cooldown. The practitioner has full control over $N_{\text{micro}}$ — it is a logical slice of a single training step.

### Inflight batching (inference)

Inference has no "global batch" to slice. Each user request is independent and requires many decode steps (one per output token) spread over seconds of wall time. The naive alternative, **static batching**:

> *Wait until $B$ requests arrive, pad to the longest prompt, decode all $B$ sequences together until every sequence finishes, then return all $B$ responses.*

wastes the pipeline in two ways: (i) slow sequences block fast ones from being returned, and (ii) as sequences finish one by one, $B_{\text{active}}$ decays toward 1 and the pipeline underfills.

**Inflight batching** (also called *continuous batching* or *iteration-level scheduling*; introduced by [ORCA] and popularized by [VLLM]) reshuffles the batch **at every decode step**:

```
At step t:
  - Run one decode step on the current set of active sequences
  - Evict any sequence that just emitted EOS, free its KV cache
  - Admit waiting requests into the freed slots (plus optionally fit in a
    chunked prefill for a newly-arrived request)
At step t+1:
  - Repeat with whatever active set exists now
```

The batch identity is no longer fixed — it is a time-varying set $B_{\text{active}}(t)$. A given user's request experiences a different batch composition at every step of its lifetime. In steady state under sustained request arrivals, $B_{\text{active}}$ hovers near a value set by arrival rate × mean request length (Little's Law; see [e2e.md §3.2](../modeling/e2e.md#32-continuous-batching-tpot)).

### The analogy, and where it breaks

| Aspect | Microbatch (training) | Inflight batch (inference) |
|---|---|---|
| Work unit in the pipeline | Microbatch of $B_{\text{micro}}$ samples | One decode step for one sequence |
| Source of in-flight quantity | Logical slicing of one training step | Independent user requests arriving over time |
| Controllable by | Practitioner (chooses $N_{\text{micro}}$ freely) | Workload (depends on traffic) |
| Failure mode | Too-small $N_{\text{micro}}$ → bubble | Too-low $B_{\text{active}}$ → bubble |
| Bubble-killing regime | $N_{\text{micro}} \gg PP$ | $B_{\text{active}} \ge PP$ |

The deep similarity: both mechanisms place enough concurrent work units in the pipeline to keep all $PP$ stages busy simultaneously. The deep difference: in training, $N_{\text{micro}}$ is a free knob; in inference, $B_{\text{active}}$ is dictated by how many users show up. A deployment serving a single user at $B = 1$ cannot be rescued by any scheduling cleverness — the bubble is fundamental at that operating point.

This is exactly why the NVIDIA blog's "batch of 1" case [NVIDIA-BLOG] is the adversarial configuration we formalize in §6: no inflight-batching trick applies, and the bubble cancellation kicks in. At higher batch sizes (inflight batching keeping $B_{\text{active}} \ge PP$), PP returns to being latency-useful, exactly as it is in training with sufficient $N_{\text{micro}}$.

---

# 5. The First-Order Correction

## 5.1 The Formula

We model the user-observed decode step time as:

$$
\boxed{\;
t_{\text{step,user}} = t_{\text{stage}} \cdot \max\!\left(1,\; \frac{PP}{B}\right)
\;}
$$

where $\gamma_{\text{pp}} \equiv \max(1, PP/B)$. The two regimes are:

- **$B \ge PP$:** factor = 1, $t_{\text{step,user}} = t_{\text{stage}}$. The pipeline is full; the classical "throughput = 1/$t_{\text{stage}}$" result holds. Throughput scales linearly with DP replicas: $TPS_{\text{single}} = B / t_{\text{step,user}}$, $TTPS = DP \cdot B / t_{\text{step,user}}$.

- **$B < PP$:** factor = $PP/B$, $t_{\text{step,user}} = (PP/B) \cdot t_{\text{stage}}$. At $B=1$ this is $PP \cdot t_{\text{stage}}$ — the full pipeline depth per emitted token.

From a user's perspective, TPOT = $t_{\text{step,user}}$ (one token per decode step, per user — see [decode.md §6.3.2](../modeling/decode.md#632-pipeline-bubble-correction-user-observed-step-time)).

## 5.2 Why It Is Only First-Order

The formula is deliberately simple — it captures the dominant effect (step time scales as $1/B$ for $B < PP$) without modeling schedule-specific details. Real systems may diverge from this prediction in several ways:

1. **Interleaved 1F1B schedules [MEGATRON3]** partition each stage into multiple virtual pipelines, effectively raising the effective $PP$ denominator seen by the bubble. In inference, this has limited impact because the bubble is driven by $B$, not by warmup steps.

2. **Stage imbalance**: if stages are not exactly equal-cost (e.g., embedding + LM head on the end stages), the slowest stage dominates. Our formula uses a single $t_{\text{stage}}$, implicitly assuming balance.

3. **Asynchronous pipeline execution** (e.g., PipeDream [PIPEDREAM]) can partially hide the underfill bubble by relaxing strict in-order execution, but most production inference systems use synchronous PP because decoded tokens have hard data dependencies (token $t+1$ depends on token $t$ from all stages).

4. **Inflight batching with non-uniform sequence lengths** means $B$ varies across steps as some sequences complete and others enter. The instantaneous bubble factor varies; time-averaging gives the effective TPOT.

These refinements are second-order for most analyses — the dominant first-order correction is exactly $\max(1, PP/B)$.

---

# 6. Why PP Does Not Help Single-User Decode

## 6.1 The Cancellation Argument

Consider the effect on $t_{\text{step,user}}$ of doubling $PP$, holding $B=1$ and all other partition axes fixed:

- $t_{\text{stage}}$ halves — each stage has half the layers, so half the weight traffic and compute. Per-stage step time drops by 2×.
- $\gamma_{\text{pp}}$ doubles — at $B=1$, the bubble factor is $PP$, so doubling $PP$ doubles it.

The two effects cancel exactly:

$$
t_{\text{step,user}}(\text{2PP}, B{=}1)
= \frac{t_{\text{stage}}}{2} \cdot (2 \cdot PP)
= t_{\text{stage}} \cdot PP
= t_{\text{step,user}}(PP, B{=}1)
$$

**Corollary:** At $B=1$, scaling PP changes neither TPOT nor interactivity. It only changes per-device HBM occupancy. PP remains useful for fitting the model, but not for making the user wait less.

This is the formal statement of the NVIDIA blog's empirical observation that "PP yields only modest improvement" for decode at $B=1$ [NVIDIA-BLOG].

## 6.2 Numerical Illustration (NVIDIA Blog Reproduction)

From our GPT-1.8T MoE reproduction on 64 × B200 NVL72 at $B=1$, $TP=1$, $EP=1$, $SP=1$, varying PP:

| PP | $t_{\text{stage}}$ (ms) | bubble factor | $t_{\text{step,user}}$ (ms) | Interactivity (tok/s) |
|----|----|----|----|----|
| 1 | 112.9 | 1 | 112.9 | 8.9 |
| 2 | 56.5 | 2 | 112.9 | 8.9 |
| 4 | 28.2 | 4 | 112.9 | 8.9 |
| 8 | 14.1 | 8 | 112.9 | 8.9 |
| 16 | 7.1 | 16 | 112.9 | 8.9 |
| 32 | 3.5 | 32 | 112.9 | 8.9 |
| 64 | 1.8 | 64 | 112.9 | 8.9 |

All seven PP choices land on exactly the same interactivity. What *does* change is throughput-per-GPU: increasing PP reduces DP replicas (fewer full-model copies fit), so throughput-per-GPU falls linearly with PP. On the throughput–interactivity Pareto plot, the PP scan at $B=1$ therefore traces a **vertical line** in Interactivity × Tput/GPU space, not a sloped Pareto improvement.

---

# 7. When PP *Does* Help

The bubble cancellation applies at $B < PP$. PP becomes useful again in two regimes:

**(a) Throughput regime ($B \ge PP$, inflight batching).**
When the serving system can keep the pipeline full by concurrent sequences, bubble factor = 1, and increasing PP drops $t_{\text{stage}}$ (fewer layers per stage) while keeping the bubble factor pinned at 1. User-observed TPOT then *does* fall with PP. This regime matches the large-batch "inference throughput" scenario rather than the single-user latency scenario. The crossover happens at exactly $B = PP$.

**(b) Capacity regime (model does not fit without PP).**
PP can be a prerequisite: if the per-device HBM cannot hold even the minimum parameter shard after TP/EP sharding, PP is the only axis that further reduces weight footprint. In this case PP is chosen for fit, not for latency, and any bubble cost is accepted as the price of running at all.

For GPT-1.8T MoE on 64 × B200, approximately $\ge 8$-way parallel sharding is required for HBM fit, but any combination of TP/EP achieving that is latency-preferable to PP at low $B$.

---

# 8. Practical Implications for Partition Sweeps

When sweeping $(PP, TP, EP, SP)$ at low $B$:

1. **Do not compare PP points by $t_{\text{stage}}$ alone.** A PP=32 config with a tiny $t_{\text{stage}}$ is not faster for the user; it is identical in user-observed TPOT to PP=1 at the same local time.

2. **PP-dominated configs appear on the *same* interactivity line as PP=1** in Pareto plots at $B=1$. If the blog or a benchmark shows them as distinct points, suspect a bubble-correction bug.

3. **If fit requires PP, prefer PP with a compensating EP/TP shard** that reduces $t_{\text{stage}}$ *further*, not just enough to offset the bubble.

4. **Once inflight batching is enabled and $B \ge PP$ is achievable**, PP becomes latency-useful again — but at that point the analysis is in the throughput regime, where the key figure of merit is Tput/GPU at the crossover batch $B^*$ (see [batched_decode.md §4](batched_decode.md#4-the-crossover-batch-size-b)).

5. **The bubble correction is already applied in [core/decode_model.py](../../llm_perf/core/decode_model.py)** — no manual post-hoc fix is required in sweeps or notebooks.

---

# 9. FAQ

**Q: Why does the bubble exist at all? Can't we just schedule differently?**

The bubble at $B < PP$ is a fundamental consequence of data dependencies: token $t$ must finish at stage $PP$ before token $t+1$ can enter stage $1$, because the sampled token ID from step $t$ becomes the input embedding for step $t+1$. No schedule can work around this — the problem is in the autoregressive dependency itself, not in the scheduling policy. (Speculative decoding [SPEC-DECODE] relaxes this by generating candidate tokens ahead of time, and can partially pipeline the dependency chain — but it does so at the cost of correctness-checking overhead, and is out of scope here.)

**Q: What about TP? Does TP have a bubble too?**

No. TP splits *one* matrix multiply across devices and then all-reduces the result. All TP ranks do useful work simultaneously within every layer — there is no idle-stage analogue. TP pays an all-reduce communication cost (see [decode.md §5.3](../modeling/decode.md#53-tensor-parallel-tp-all-reduce)) but no geometric bubble.

**Q: What about EP?**

EP also has no bubble: during the dispatch-and-combine all-to-all, all EP ranks are active simultaneously on their assigned experts. However, EP has a different trap — load imbalance when token-to-expert routing is uneven — which is handled separately.

**Q: How does the bubble interact with TP-within-PP?**

The bubble is in the time dimension (step-to-step), while TP is in the spatial dimension (inside one layer's compute). They are orthogonal. $PP=4, TP=8$ runs $L/4$ layers per PP stage and splits each GEMM 8-way within the stage. The bubble analysis uses $t_{\text{stage}}$, which already accounts for TP's effect on per-stage time; then the bubble multiplier applies on top.

**Q: Is $\max(1, PP/B)$ ever an over-estimate?**

Yes, slightly, if the schedule overlaps PP hops with stage compute (e.g., using non-blocking NCCL sends). The formula assumes strict serialization between stages — a conservative upper bound. In practice, with well-tuned overlap, the effective bubble factor is ~$(PP-1)/B + 1$ at $B < PP$ rather than $PP/B$. The two agree at $B=1$ ($PP$ vs $PP$) and diverge slightly at $1 < B < PP$. For our modeling purposes the simpler $PP/B$ form is used; see [MEGATRON3 §4.1] for a fully detailed training-time analysis.

**Q: Why is the correction applied *after* the roofline, not inside it?**

The roofline bounds the throughput of a single stage in isolation. The bubble is a pipeline-level effect that emerges from how tokens flow through stages over time. Keeping them separated cleanly — $t_{\text{stage}}$ from the roofline, $t_{\text{step,user}}$ from the bubble correction — makes the model easier to extend (e.g., to add an interleaved-1F1B variant) and lets us verify each layer independently.

---

# References

**[GPIPE]**
Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, M.X., Chen, D., Lee, H., Ngiam, J., Le, Q.V., Wu, Y., & Chen, Z. (2019).
*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.*
NeurIPS 2019. arXiv:1811.06965.
→ Original pipeline parallelism; synchronous microbatching; $\frac{PP-1}{N_{\text{micro}}+PP-1}$ bubble fraction.

**[PIPEDREAM]**
Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V., Devanur, N.R., Ganger, G.R., Gibbons, P.B., & Zaharia, M. (2019).
*PipeDream: Generalized Pipeline Parallelism for DNN Training.*
SOSP 2019.
→ Asynchronous pipeline scheduling; 1F1B introduced; weight version tracking.

**[MEGATRON]**
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019).
*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*
arXiv:1909.08053.
→ Stage-based PP layout; column/row-parallel layers within each stage.

**[MEGATRON3]**
Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., & Zaharia, M. (2021).
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*
SC '21. arXiv:2104.04473.
→ Interleaved 1F1B schedule; detailed bubble analysis at scale; training-time formula.

**[VLLM]**
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*
SOSP 2023. arXiv:2309.06180.
→ Iteration-level (continuous) batching for inference; raises effective $B$ toward $PP$.

**[ORCA]**
Yu, G-I., Jeong, J.S., Kim, G-W., Kim, S., & Chun, B-G. (2022).
*Orca: A Distributed Serving System for Transformer-Based Generative Models.*
OSDI 2022.
→ Iteration-level scheduling for transformer inference; continuous batching precursor to vLLM.

**[SPEC-DECODE]**
Leviathan, Y., Kalman, M., & Matias, Y. (2023).
*Fast Inference from Transformers via Speculative Decoding.*
ICML 2023. arXiv:2211.17192.
→ Relaxing the autoregressive dependency by drafting-and-verifying; orthogonal to the bubble.

**[NVIDIA-BLOG]**
Mao, A., Yoon, B., et al. (2025).
*Demystifying AI Inference Deployments for Trillion-Parameter Large Language Models.*
NVIDIA Technical Blog.
→ Empirical PP scan on GPT-1.8T MoE showing "modest improvement" at $B=1$; the motivating observation this document formalizes.

---

## Cross-References

- [../modeling/decode.md §6.3](../modeling/decode.md#63-pipeline-bubble-tps-and-ttps) — Formal equations for $t_{\text{stage}}$, bubble factor, $t_{\text{step,user}}$.
- [../modeling/e2e.md §1.2](../modeling/e2e.md#12-time-per-output-token-tpot) — TPOT definition from user perspective.
- [../modeling/notation.md §9](../modeling/notation.md#9-decode-timing-and-throughput) — Symbol reference.
- [batched_decode.md](batched_decode.md) — Companion: why batching ($B \ge PP$) is the primary escape from the bubble regime.
- [../../llm_perf/core/decode_model.py](../../llm_perf/core/decode_model.py) — Code implementation of the $\max(1, PP/B)$ correction.
