# End-to-End LLM Inference Metrics

**Assembling TTFT, TPOT, E2E Latency, Throughput/GPU, Interactivity, and the Throughput–Latency Pareto Frontier**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
LLM inference, TTFT, time-to-first-token, TPOT, time-per-output-token, E2E latency,  
throughput per GPU, interactivity, continuous batching, chunked prefill, Pareto frontier,  
roofline model, InferenceX, throughput–latency tradeoff

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Metric Definitions](#1-metric-definitions)
  - [1.1 Time To First Token (TTFT)](#11-time-to-first-token-ttft)
  - [1.2 Time Per Output Token (TPOT)](#12-time-per-output-token-tpot)
  - [1.3 End-to-End Latency](#13-end-to-end-latency)
  - [1.4 Throughput per GPU](#14-throughput-per-gpu)
  - [1.5 Interactivity](#15-interactivity)
  - [1.6 Goodput (brief)](#16-goodput-brief)

- [2. TTFT Assembly](#2-ttft-assembly)
  - [2.1 Single-Request TTFT](#21-single-request-ttft)
  - [2.2 Batched-Prefill TTFT](#22-batched-prefill-ttft)
  - [2.3 Chunked-Prefill TTFT](#23-chunked-prefill-ttft)

- [3. TPOT Assembly](#3-tpot-assembly)
  - [3.1 Static Batching TPOT](#31-static-batching-tpot)
  - [3.2 Continuous Batching TPOT](#32-continuous-batching-tpot)

- [4. End-to-End Latency](#4-end-to-end-latency)
  - [4.1 General Formula](#41-general-formula)
  - [4.2 Regime Analysis](#42-regime-analysis)
  - [4.3 Numerical Example (H100)](#43-numerical-example-h100)

- [5. Throughput/GPU and Interactivity](#5-throughputgpu-and-interactivity)
  - [5.1 Throughput/GPU Derivation](#51-throughputgpu-derivation)
  - [5.2 Interactivity and the InferenceX Y-Axis](#52-interactivity-and-the-inferencex-y-axis)

- [6. Throughput–Latency Pareto Frontier](#6-throughputlatency-pareto-frontier)
  - [6.1 Levers Shaping the Pareto Curve](#61-levers-shaping-the-pareto-curve)
  - [6.2 Three Zones of the Pareto Frontier](#62-three-zones-of-the-pareto-frontier)
  - [6.3 Roofline Ceiling on Pareto Efficiency](#63-roofline-ceiling-on-pareto-efficiency)
  - [6.4 InferenceX Axis Mapping](#64-inferencex-axis-mapping)

- [Symbol Summary](#symbol-summary)

---

<div style="page-break-before: always;"></div>

# 1. Metric Definitions

This section defines all six end-to-end metrics precisely. They are the outputs users and system operators care about; all prior modeling documents (memory, FLOPs, traffic, communication, latency) contribute to computing them. The InferenceX benchmark [INFERENCEX] organizes production LLM inference evaluation around two of these six: **Throughput/GPU** (system efficiency) and **Interactivity** (user-perceived quality). The others provide essential context and diagnostic signal.

Symbols used throughout this document are defined in `modeling.notation.md`; see especially §§4, 9, 11, and 14. New symbols introduced here are summarized in the [Symbol Summary](#symbol-summary) at the end.

---

## 1.1 Time To First Token (TTFT)

**Definition.** $TTFT$ is the wall-clock elapsed time from the moment a request is received by the serving system to the moment the **first output token** is returned to the caller. It encompasses all latency incurred before the first token can be streamed: request scheduling, tokenization, the prefill forward pass, optional KV cache transfer (disaggregated architectures), and the first decode step that produces token 1.

$$
\boxed{TTFT = t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{KV\_transfer}} + t_{\text{token}}}
$$

where $t_{\text{KV\_transfer}} = 0$ for co-located prefill+decode. The prefill latency $t_{\text{prefill}}$ is derived in full in `modeling.prefill.md §3`; framework overhead terms $t_{\text{sched}}$ and $t_{\text{tok}}$ are defined in `modeling.framework.md §2`.

**Key property.** TTFT is a *latency-to-first-byte* (TTFB) metric. For interactive applications, a high TTFT produces a blank screen during prefill — the most perceptible user experience degradation for long prompts.

---

## 1.2 Time Per Output Token (TPOT)

**Definition.** $\text{TPOT}$ is the **average inter-token latency** for tokens 2 through $N_{\text{out}}$ of a single response (the decode phase). It equals the per-step decode time $t_{\text{token}}$ from `modeling.tpot.md §6.2`, measured per sequence:

$$
\boxed{\text{TPOT} = \frac{t_{\text{token}}(B)}{B}}
$$

where $B$ is the number of sequences decoded concurrently in the same step and $t_{\text{token}}(B)$ is the overlap-aware per-step wall-clock time (see `modeling.tpot.md §6.4.2`).

**Key property.** TPOT is the *streaming rate* perceived by the user. A TPOT of 50 ms means one new token appears every 50 ms — a rate of 20 tokens/s. Human reading comprehension speed is approximately 5–15 tokens/s; TPOT below 100 ms (>10 tokens/s) is a common production SLA threshold.

---

## 1.3 End-to-End Latency

**Definition.** The full wall-clock latency from request receipt to the **last output token** of a response of $N_{\text{out}}$ tokens:

$$
\boxed{\text{E2E}(N_{\text{out}}) = TTFT + (N_{\text{out}} - 1) \times \text{TPOT}}
$$

The factor $(N_{\text{out}} - 1)$ arises because the first output token is already produced by the end of TTFT; the decode phase produces tokens $2, 3, \ldots, N_{\text{out}}$, requiring $N_{\text{out}} - 1$ additional steps.

---

## 1.4 Throughput per GPU

**Definition.** The rate of output token generation per physical GPU, expressed in output tokens/second/GPU:

$$
\boxed{\text{Tput/GPU} = \frac{TTPS}{N_{\text{GPUs}}}}
$$

where $TTPS$ is the global cluster token throughput (tokens/s, all sequences) defined in `modeling.tpot.md §6.3`, and $N_{\text{GPUs}}$ is the total number of GPUs in the cluster. This is the X-axis of the InferenceX benchmark [INFERENCEX].

---

## 1.5 Interactivity

**Definition.** The rate at which a single user receives output tokens, expressed in output tokens/second per request:

$$
\boxed{\text{Interactivity} = \frac{1}{\text{TPOT}}}
$$

This is the Y-axis of the InferenceX benchmark [INFERENCEX]. Higher interactivity means faster streaming to the individual user. The reciprocal relationship makes clear that Interactivity and TPOT encode identical information in different units.

---

## 1.6 Goodput (brief)

**Goodput** is the fraction of GPU-time spent on *useful* token generation — i.e., generating tokens for requests that successfully complete, excluding time spent on preempted or aborted requests, speculative tokens that are rejected, and idle stalls waiting for new requests.

A formal goodput model requires a queuing-theoretic treatment and is out of scope for this document. Informally:

$$
\text{Goodput} = \frac{\text{tokens generated for completed requests}}{\text{max-theoretical tokens/s} \times T_{\text{wall}}}
$$

Goodput is the primary metric used in [DISAGG-PREFILL] to compare disaggregated vs. co-located serving architectures at varying request rates.

---

<div style="page-break-before: always;"></div>

# 2. TTFT Assembly

TTFT varies substantially depending on the serving architecture (co-located vs. disaggregated), the prefill batching policy (single request vs. batched vs. chunked), and whether the system is under load. This section assembles the full TTFT formula for each scenario, building on the per-phase models derived in `modeling.prefill.md` and `modeling.framework.md`.

---

## 2.1 Single-Request TTFT

### Phase decomposition

For a single request on a **co-located** prefill+decode cluster (no disaggregation), TTFT decomposes into four sequential phases:

1. **$t_{\text{sched}}$** — Request scheduling and batch assembly latency: the serving scheduler assigns KV memory pages, builds the batch metadata tensor, and triggers the first CUDA kernel (or graph replay). From `modeling.framework.md §2.4`, $t_{\text{sched}}$ is empirical and typically 10–200 µs depending on batch size and scheduler implementation.

2. **$t_{\text{tok}}$** — Tokenization latency: raw text is converted to token IDs on CPU. From `modeling.framework.md §2.1`, $t_{\text{tok}} \sim 0.1\text{–}2$ ms and is often pipelined with the previous request's decode tail, making it negligible in steady-state batch serving. We retain it for single-request analysis.

3. **$t_{\text{prefill}}$** — Prefill forward pass latency: the model processes all $S_{\text{input}}$ tokens in a single GEMM-dominated pass. The full derivation is in `modeling.prefill.md §3`; the boxed result is:

   $$
   t_{\text{prefill}} = \underbrace{t_{\text{prefill,local}}}_{\text{roofline}} + \max\!\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right) + t_{\text{pipeline,warmup}}
   $$

   where $t_{\text{prefill,local}} = \max(t_{\text{prefill,compute}},\; t_{\text{prefill,mem}})$ is the per-stage roofline time, $t_{\text{prefill,comm}}$ is the collective communication time during prefill (same TP/EP/SP structure as decode, scaled by $S_{\text{input}}$), $\rho$ is the overlap factor (same as in decode, `modeling.tpot.md §6.2`), and $t_{\text{pipeline,warmup}} = (PP - 1) \times t_{\text{stage,max}}$ is the time for the prefill pass to fill the pipeline (`modeling.prefill.md §3.3`).

4. **$t_{\text{token}}$** — First decode step: one forward pass of the decode kernel, generating token 1. From `modeling.tpot.md §6.2`:

   $$
   t_{\text{token}} = t_{\text{local}} + \max\!\left(0,\; t_{\text{comm}} - \rho\, t_{\text{local}}\right)
   $$

### Boxed result (co-located)

Combining all four phases, and absorbing $t_{\text{tok}}$ into $t_{\text{sched}}$ as a lumped scheduling overhead:

$$
\boxed{
TTFT_{\text{single}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{token}}
}
$$

### With disaggregated prefill

When prefill and decode run on **separate** GPU clusters [DISAGG-PREFILL], the KV cache generated by the prefill cluster must be transferred over the inter-cluster interconnect before decode can begin. Using the α–β latency model (`modeling.framework.md §3`):

$$
t_{\text{KV\_transfer}} = \alpha_{\text{inter}} + \frac{M_{\text{KV\_transfer}}}{B_{\text{eff,inter}}}
$$

where $M_{\text{KV\_transfer}} = \frac{2 \cdot S_{\text{input}} \cdot H_{kv} \cdot b}{TP \cdot SP} \cdot \frac{L}{PP}$ is the per-device KV transfer volume. TTFT becomes:

$$
\boxed{
TTFT_{\text{disagg}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{KV\_transfer}} + t_{\text{token}}
}
$$

The motivation for disaggregation is that the prefill and decode phases have fundamentally different computational characteristics (compute-bound vs. memory-bound) and therefore benefit from different hardware configurations. The cost is the added $t_{\text{KV\_transfer}}$ latency. For large KV caches and slow inter-cluster links, this can be the dominant TTFT term.

---

## 2.2 Batched-Prefill TTFT

When $B_{\text{prefill}} > 1$ requests are prefilled together in a single forward pass, the computational structure changes in an asymmetric way: **FLOPs scale with $B_{\text{prefill}}$, but weight traffic does not.**

### FLOPs under batched prefill

Each request contributes $S_{\text{input}}$ tokens to the joint prefill batch of size $B_{\text{prefill}} \times S_{\text{input}}$ (assuming equal-length prompts for simplicity). The projection and FFN GEMMs have shape $[B_{\text{prefill}} \cdot S_{\text{input}} \times H] \times [H \times H_{\text{out}}]$, so FLOPs scale linearly:

$$
F_{\text{prefill,batch}} = B_{\text{prefill}} \times F_{\text{prefill,single}}
$$

where $F_{\text{prefill,single}}$ is the per-request FLOPs from `modeling.prefill.md §1`.

### Weight traffic under batched prefill

The weight matrices are read **once** from HBM regardless of $B_{\text{prefill}}$, because all $B_{\text{prefill}}$ sequences share the same weights in the GEMM. Weight traffic is therefore invariant in $B_{\text{prefill}}$:

$$
T_{\theta,\text{device}}(B_{\text{prefill}}) = T_{\theta,\text{device}} \qquad (\text{independent of } B_{\text{prefill}})
$$

This is precisely the batching benefit: more FLOPs are executed per byte of weight loaded, improving arithmetic intensity and GPU utilization.

### Batched prefill local time

Generalizing `modeling.prefill.md §3.1` to batch size $B_{\text{prefill}}$:

$$
t_{\text{prefill,local}}(B_{\text{prefill}})
=
\max\!\left(
\frac{B_{\text{prefill}} \times F_{\text{prefill,device}}}{R_{\text{GPU}}},\;
\frac{T_{\theta,\text{device}} + B_{\text{prefill}} \times T_{\text{KV,write,device}}}{B_{\text{eff,mem}}}
\right)
$$

The KV write traffic $T_{\text{KV,write,device}}$ scales with $B_{\text{prefill}}$ because each request in the batch writes its own KV entries. For large $B_{\text{prefill}}$, the compute term dominates and the latency grows linearly with $B_{\text{prefill}}$; at small $B_{\text{prefill}}$ in the compute-bound prefill regime ($S_{\text{input}} \gg S_{\text{input}}^{\star}$, see `modeling.prefill.md §2.3`), compute is already the bottleneck even at $B_{\text{prefill}} = 1$, so the latency likewise scales linearly.

### TTFT for the last request in the batch

From a user's perspective, the worst-case TTFT applies to the **last request** admitted to the prefill batch — that request waits for the entire joint prefill to complete before its first token is produced:

$$
\boxed{
TTFT_{\text{batched}} = t_{\text{sched}} + t_{\text{prefill,local}}(B_{\text{prefill}}) + \max\!\left(0,\; t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}\right) + t_{\text{pipeline,warmup}} + t_{\text{token}}
}
$$

Batching prefill requests improves GPU utilization (higher arithmetic intensity → better hardware efficiency) at the cost of increased TTFT for late arrivals in the batch. The optimal $B_{\text{prefill}}$ balances throughput and tail TTFT latency; see `modeling.prefill.md §4.3` for the batch-size optimization.

---

## 2.3 Chunked-Prefill TTFT

Chunked prefill [SARATHI] splits the prefill of a single request into $N_{\text{chunks}}$ smaller chunks of $C$ tokens each, interleaving each chunk with one decode iteration. This technique reduces **head-of-line blocking**: long prefill passes in standard serving can stall the decode pipeline for tens to hundreds of milliseconds, inflating TPOT for all existing decode requests. Chunking limits the maximum stall duration per decode iteration to $t_{\text{chunk}}$, the time to process one $C$-token chunk.

### Chunk count

For a prompt of length $S_{\text{input}}$ with chunk size $C$:

$$
N_{\text{chunks}} = \left\lceil \frac{S_{\text{input}}}{C} \right\rceil
$$

### Per-chunk latency

Each chunk is a mini-prefill of $C$ tokens. The per-chunk local time follows the same roofline formula as a full prefill (`modeling.prefill.md §3.1`) with $S_{\text{input}}$ replaced by $C$:

$$
t_{\text{chunk}} = \max\!\left(\frac{F_{\text{prefill,device}}(C)}{R_{\text{GPU}}},\; \frac{T_{\theta,\text{device}} + T_{\text{KV,write,device}}(C)}{B_{\text{eff,mem}}}\right)
$$

For small $C$ (e.g., $C = 256$), the chunk is likely memory-bound (weight traffic dominates KV write traffic), meaning $t_{\text{chunk}} \approx T_{\theta,\text{device}} / B_{\text{eff,mem}}$ — identical to one decode step latency. This is the **chunked-prefill design point**: each chunk costs approximately the same as one decode step, which is the minimum possible disruption to the decode pipeline.

### TTFT under chunking

The new request's TTFT is the time to process all $N_{\text{chunks}}$ chunks sequentially (each occupying one decode slot), plus scheduling overhead:

$$
\boxed{
TTFT_{\text{chunked}} = t_{\text{sched}} + N_{\text{chunks}} \times t_{\text{chunk}} + t_{\text{token}}
\approx t_{\text{sched}} + \left\lceil \frac{S_{\text{input}}}{C} \right\rceil \times t_{\text{chunk}} + t_{\text{token}}
}
$$

The pipeline warmup $t_{\text{pipeline,warmup}}$ applies once across the entire prefill sequence rather than per chunk (the pipeline is kept warm by the ongoing decode traffic), so it does not multiply by $N_{\text{chunks}}$.

### Tradeoff

| Smaller $C$ | Larger $C$ |
|-------------|------------|
| Less stall per decode iteration → better TPOT for existing requests | More stall per iteration → worse TPOT for existing requests |
| More chunks required → worse TTFT for new request | Fewer chunks → better TTFT for new request |
| Approaches standard decode behavior | Approaches standard (non-chunked) prefill behavior |

The optimal $C$ depends on the SLA balance between new-request TTFT and existing-request TPOT. A common production heuristic is $C \approx B \cdot S_{\text{decode}}$, matching the chunk FLOPs to the concurrent decode work so that the GPU is equally utilized during chunk and decode steps [SARATHI].

---

<div style="page-break-before: always;"></div>

# 3. TPOT Assembly

TPOT is the per-sequence inter-token latency during the decode phase. Its derivation from the roofline model was developed in detail in `modeling.tpot.md §6.4.2`; this section assembles the result for both static and continuous batching, and explains the steady-state behavior under each scheduling policy.

---

## 3.1 Static Batching TPOT

In **static batching**, all $B$ requests in the batch start together, are padded to the same output length, and finish together. The batch composition is fixed for the entire decode session.

### Per-step wall-clock time

At each decode step, the model processes $B$ tokens simultaneously (one per sequence). The overlap-aware per-step time from `modeling.tpot.md §6.2` is:

$$
t_{\text{token}}(B) = t_{\text{local}}(B) + \max\!\left(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}}(B)\right)
$$

where the batched local time is (`modeling.tpot.md §6.4.2`):

$$
t_{\text{local}}(B)
=
\max\!\left(
\frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}},\;
\frac{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}{B_{\text{eff,mem}}}
\right)
$$

### TPOT definition

Each decode step emits exactly **one token per active sequence** and takes $t_{\text{token}}(B)$ wall-clock seconds. The system produces $B$ output tokens in that step — one per sequence — so the amortized cost per output token is $t_{\text{token}}(B) / B$. This is the per-sequence TPOT: how long the system "spends" per token of each sequence's output (`modeling.tpot.md §6.4.2`):

$$
\boxed{
\text{TPOT}_{\text{static}}(B) = \frac{t_{\text{token}}(B)}{B}
}
$$

This definition is consistent with §1.2 and with `modeling.tpot.md §6.4.2`. Note the distinction from the raw step time: $t_{\text{token}}(B)$ is the wall-clock duration of one decode step (all $B$ tokens processed in parallel), while $\text{TPOT}(B) = t_{\text{token}}(B)/B$ is the effective per-output-token latency that determines streaming Interactivity.

### Regime behavior

From the batch-size analysis in `modeling.tpot.md §6.4.2`:

**Memory-bound regime** ($B \ll B^*$, weight traffic dominates):
$$
t_{\text{local}}(B) \approx \frac{T_{\theta,\text{device}}}{B_{\text{eff,mem}}} \quad\Rightarrow\quad \text{TPOT}_{\text{static}}(B) \approx \frac{T_{\theta,\text{device}}}{B \times B_{\text{eff,mem}}}
$$

TPOT **improves** (decreases) with $B$: the fixed cost of loading weights is amortized over more sequences. Throughput grows linearly; TPOT falls proportionally.

**Compute-bound regime** ($B \gg B^*$, compute dominates):
$$
t_{\text{local}}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}} \quad\Rightarrow\quad \text{TPOT}_{\text{static}}(B) \approx \frac{F_{\text{token,device}}}{R_{\text{GPU}}}
$$

TPOT is approximately **constant** in $B$: the step time grows proportionally with $B$ (more compute), but dividing by $B$ cancels the growth. Per-sequence latency is determined solely by the compute rate.

**Summary:**

| Regime | Condition | $\text{TPOT}_{\text{static}}(B)$ |
|--------|-----------|----------------------------------|
| Memory-bound | $B \ll B^*$ | $\approx T_{\theta,\text{device}} / (B \times B_{\text{eff,mem}})$ (decreases with $B$) |
| Crossover | $B = B^*$ | transition to flat TPOT regime |
| Compute-bound | $B \gg B^*$ | $\approx F_{\text{token,device}} / R_{\text{GPU}}$ (constant in $B$) |

---

## 3.2 Continuous Batching TPOT

In **continuous batching** [VLLM], requests arrive and depart asynchronously. At each decode iteration, the scheduler assembles a new batch from active requests; requests that complete (reach their stop token or maximum length) are immediately removed and new requests admitted. The effective batch size $B_{\text{eff}}$ therefore varies from iteration to iteration.

### Per-request average TPOT

A request that requires $N_{\text{out}}$ decode steps experiences a different $B_{\text{eff},i}$ at each step $i$. Using $\text{TPOT}(B) = t_{\text{token}}(B)/B$ (§1.2, `modeling.tpot.md §6.4.2`), the average TPOT over the full response is:

$$
\boxed{
\overline{\text{TPOT}} = \frac{1}{N_{\text{out}}} \sum_{i=1}^{N_{\text{out}}} \frac{t_{\text{token}}(B_{\text{eff},i})}{B_{\text{eff},i}}
}
$$

The function $g(B) = t_{\text{token}}(B)/B$ is convex in $B$ (it is decreasing in Zone 1 and flat in Zone 3; its second derivative is non-negative throughout). By Jensen's inequality:

$$
\overline{\text{TPOT}} \geq \frac{t_{\text{token}}\!\left(\overline{B_{\text{eff}}}\right)}{\overline{B_{\text{eff}}}}
$$

In practice the inequality is loose when $B_{\text{eff}}$ is approximately stationary during the lifetime of a request.

### Steady-state effective batch size (Little's Law)

In steady state under Poisson request arrivals at rate $\lambda$ (requests/second), **Little's Law** relates the mean number of active requests $\overline{B_{\text{eff}}}$ to the mean sojourn time:

$$
\overline{B_{\text{eff}}} = \lambda \times \mathbb{E}[\text{sojourn time}]
$$

The sojourn time for a single request is the total wall-clock time it occupies a decode slot: $TTFT + \sum_{i=1}^{N_{\text{out}}} t_{\text{token}}(B_{\text{eff},i})$. Since each step takes $t_{\text{token}}(B_{\text{eff},i})$ wall-clock seconds (not $\text{TPOT} = t_{\text{token}}/B_{\text{eff}}$), the mean sojourn time is:

$$
\overline{B_{\text{eff}}} \approx \lambda \times \left(TTFT + \mathbb{E}[N_{\text{out}}] \times \overline{t_{\text{step}}}\right)
\qquad \text{where } \overline{t_{\text{step}}} = \mathbb{E}[t_{\text{token}}(B_{\text{eff}})]
$$

This is self-referential ($\overline{t_{\text{step}}}$ depends on $\overline{B_{\text{eff}}}$) and resolved numerically in practice:

- At **low load** ($\lambda$ small): $\overline{B_{\text{eff}}}$ is small → each step is memory-bound ($t_{\text{step}} \approx T_\theta/B_{\text{eff,mem}}$) → TPOT $\approx T_\theta / (B_{\text{eff}} \times B_{\text{eff,mem}})$ is high (inefficient — weight cost not amortized).
- At **higher load** ($\lambda$ growing): $\overline{B_{\text{eff}}}$ grows → TPOT improves (weight cost amortized across more sequences); throughput also grows.
- At **saturation** ($B_{\text{eff}} > B^*$): TPOT plateaus at $F_{\text{token,device}}/R_{\text{GPU}}$; throughput also plateaus.

### The efficiency–load relationship

The key insight is: with TPOT defined as the amortized per-output-token cost ($t_{\text{token}}/B$), **higher load improves both TPOT and throughput** up to the Zone 3 ceiling. There is no fundamental tension between Tput/GPU and Interactivity — they are constrained to lie on the same hyperbola $\text{Tput/GPU} \times \text{TPOT} = 1/N_{\text{GPUs,per-replica}}$ regardless of load. The practical tradeoff is between **operating close to or below $B^*$** (near-optimal for both metrics) versus **idling below Zone 1** (waste). This tradeoff is developed in full in §6.

---

<div style="page-break-before: always;"></div>

# 4. End-to-End Latency

## 4.1 General Formula

Combining the TTFT and TPOT definitions from §§1.1–1.2, the full wall-clock latency from request receipt to the final output token is:

$$
\boxed{
\text{E2E}(N_{\text{out}}) = TTFT + (N_{\text{out}} - 1) \times \text{TPOT}
}
$$

**Derivation.** Token 1 arrives at time $TTFT$ (the first-token latency). Tokens $2, 3, \ldots, N_{\text{out}}$ are produced in $N_{\text{out}} - 1$ subsequent decode steps, each taking $\text{TPOT}$ seconds. The wall-clock time of the last token is therefore $TTFT + (N_{\text{out}} - 1) \times \text{TPOT}$.

---

## 4.2 Regime Analysis

The relative contribution of prefill (captured in TTFT) and decode (captured in the TPOT term) depends on $N_{\text{out}}$:

**Short responses** ($N_{\text{out}}$ small): E2E $\approx TTFT$. The decode contribution $(N_{\text{out}} - 1) \times \text{TPOT}$ is small relative to TTFT. Latency is dominated by prefill; reducing $S_{\text{input}}$ or using chunked prefill has the largest impact.

**Long responses** ($N_{\text{out}}$ large): E2E $\approx N_{\text{out}} \times \text{TPOT}$. The prefill contribution is amortized over many decode steps; TTFT contributes negligibly. Latency is dominated by decode; increasing GPU count (DP replicas) or batch efficiency has the largest impact.

**Crossover point.** The decode term equals the TTFT term when:

$$
(N_{\text{out}} - 1) \times \text{TPOT} = TTFT
\quad\Longrightarrow\quad
N_{\text{out}}^{\star} \approx \frac{TTFT}{\text{TPOT}} + 1
$$

For $N_{\text{out}} \ll N_{\text{out}}^{\star}$, the response is "prefill-dominated"; for $N_{\text{out}} \gg N_{\text{out}}^{\star}$, it is "decode-dominated."

---

## 4.3 Numerical Example (H100)

We use representative H100 SXM5 parameters [H100-SPEC] for a 70B-class dense model with $S_{\text{input}} = 2048$ and batch size $B = 1$:

- $t_{\text{prefill}} \approx 100$ ms (compute-bound prefill at $S_{\text{input}} = 2048$; typical from `modeling.prefill.md §3`)
- $\text{TPOT} \approx 20$ ms (memory-bound decode at $B = 1$; from $T_{\theta} / B_{\text{eff,mem}}$ with 70B weights at bf16 and 3.35 TB/s HBM)
- $t_{\text{sched}} \approx 0.1$ ms (negligible in single-request scenario)
- $t_{\text{token}} \approx 20$ ms (first decode step ≈ TPOT at $B = 1$)

So $TTFT \approx 100 + 20 = 120$ ms, giving $N_{\text{out}}^{\star} = 120/20 + 1 = 7$ tokens.

| $N_{\text{out}}$ | Decode contribution | TTFT contribution | E2E latency | Prefill fraction |
|-----------------|---------------------|-------------------|-------------|-----------------|
| 7 | $(7-1) \times 20 = 120$ ms | 120 ms | ~240 ms | ~50% |
| 50 | $49 \times 20 = 980$ ms | 120 ms | ~1.1 s | ~11% |
| 250 | $249 \times 20 = 4.98$ s | 120 ms | ~5.1 s | ~2.4% |
| 1000 | $999 \times 20 = 19.98$ s | 120 ms | ~20.1 s | ~0.6% |

For responses in the range typical of conversational use ($N_{\text{out}} = 50$–$250$), TTFT contributes 2–11% of total E2E latency. At $N_{\text{out}} = 1000$ (long-form generation), TTFT is negligible. These numbers underscore that **TTFT optimization matters most for short, latency-sensitive responses**, while **TPOT optimization dominates for long-form generation**.

---

<div style="page-break-before: always;"></div>

# 5. Throughput/GPU and Interactivity

## 5.1 Throughput/GPU Derivation

From `modeling.tpot.md §6.3`, the global decode throughput across all DP replicas is:

$$
TTPS = DP \cdot TPS_{\text{single}} = DP \cdot \frac{1}{\max_j t_{\text{stage},j}}
$$

The total GPU count in the cluster is:

$$
N_{\text{GPUs}} = DP \cdot PP \cdot TP \cdot EP \cdot SP
$$

Substituting into the Tput/GPU definition:

$$
\text{Tput/GPU}
= \frac{TTPS}{N_{\text{GPUs}}}
= \frac{DP \cdot TPS_{\text{single}}}{DP \cdot PP \cdot TP \cdot EP \cdot SP}
$$

$$
\boxed{
\text{Tput/GPU} = \frac{TPS_{\text{single}}}{PP \cdot TP \cdot EP \cdot SP}
}
$$

### Insight: DP is the only "free" dimension

Data Parallelism ($DP$) cancels from both numerator and denominator: adding DP replicas multiplies $TTPS$ and $N_{\text{GPUs}}$ equally, leaving Tput/GPU unchanged. The DP dimension scales aggregate throughput without affecting per-GPU efficiency.

Pipeline ($PP$), Tensor ($TP$), Expert ($EP$), and Sequence ($SP$) parallelism all appear in the denominator only. They expand the denominator because each added GPU reduces per-device FLOPs and memory traffic but introduces communication overhead — the per-stage $t_{\text{stage},j}$ decreases with more GPUs, but communication overhead partially offsets this gain, and more total GPU-seconds are consumed per token generated. Therefore:

- **DP** is the preferred dimension for scaling cluster throughput without TPOT impact.
- **TP/PP/EP/SP** are used to fit large models into available HBM, but each carries a Tput/GPU cost (more GPUs per token, with communication overhead).

---

## 5.2 Interactivity and the InferenceX Y-Axis

From §1.5, Interactivity = $1/\text{TPOT}$. Expanding with the static batching result from §3.1 ($\text{TPOT}(B) = t_{\text{token}}(B)/B$):

$$
\text{Interactivity} = \frac{1}{\text{TPOT}_{\text{static}}(B)} = \frac{B}{t_{\text{token}}(B)}
$$

**At the memory-bound operating point** ($B \ll B^*$, $t_{\text{token}} \approx T_{\theta}/B_{\text{eff,mem}}$):
$$
\text{Interactivity} \approx \frac{B \times B_{\text{eff,mem}}}{T_{\theta,\text{device}}} \quad\propto\quad B
$$

Interactivity **grows** with $B$ in the memory-bound regime: amortizing the fixed weight-load cost over more sequences benefits each sequence's streaming rate. Both Tput/GPU and Interactivity improve together in this zone.

**At the compute-bound operating point** ($B \gg B^*$, $t_{\text{token}} \approx B \times F/R_{\text{GPU}}$):
$$
\text{Interactivity} \approx \frac{R_{\text{GPU}}}{F_{\text{token,device}}} \qquad (\text{constant in } B)
$$

Interactivity **plateaus** in the compute-bound regime: the step time grows proportionally with $B$, and dividing the $B$ tokens across that longer step leaves the per-sequence rate unchanged. This is the **maximum interactivity** achievable on the given hardware for this model, set entirely by the compute rate and per-token FLOPs.

**Human reading threshold.** For streaming LLM output to feel "live" to a human reader, interactivity above approximately 5–15 tokens/s (TPOT below 67–200 ms) is required. This maps to a specific constraint on $t_{\text{token}}$ — a maximum allowable step time — which, through the roofline model, constrains the maximum batch size $B$ that can be served while meeting the SLA.

---

<div style="page-break-before: always;"></div>

# 6. Throughput–Latency Pareto Frontier

The Pareto frontier is the set of (Tput/GPU, Interactivity) pairs that are achievable without waste: no point on the frontier can improve both metrics simultaneously. This section assembles the frontier from the roofline model, identifies the three operating zones, derives the roofline efficiency ceiling, and maps the result to the InferenceX benchmark axes [INFERENCEX].

---

## 6.1 Levers Shaping the Pareto Curve

Three principal levers shift the operating point along the Pareto frontier, or move the frontier itself:

1. **Batch size $B$ (or $B_{\text{eff}}$ under continuous batching).** Moving along the frontier. Increasing $B$ increases Tput/GPU (more tokens processed per step) but also increases TPOT (each step takes longer), reducing Interactivity. Decreasing $B$ improves Interactivity at the cost of lower Tput/GPU.

2. **Parallelism configuration (TP, PP, EP, SP; fixed DP).** Shifting the frontier. TP/PP/EP/SP change $t_{\text{token}}(B)$ by altering per-device FLOPs, traffic, and communication overhead. They also change $B^*$ (the crossover batch size from `modeling.tpot.md §6.4.1`). Better parallelism configurations can push the frontier outward (higher Tput/GPU for the same Interactivity).

3. **Context length $S$.** Shifting the frontier inward. Longer decode contexts $S$ increase KV cache traffic $T_{\text{KV,device}}$, which lowers $B^*$ (the system becomes compute-bound at smaller batches) and increases $t_{\text{token}}$ for all $B$, shrinking both Tput/GPU and Interactivity simultaneously.

---

## 6.2 Three Zones of the Pareto Frontier

The Pareto curve sweeps $B$ from $1$ to $\infty$. Its shape is inherited directly from `modeling.tpot.md §6.4.3`; we express it here in the (Tput/GPU, Interactivity) coordinate system of InferenceX [INFERENCEX].

Define Tput/GPU and Interactivity as explicit functions of $B$:

$$
\text{Tput/GPU}(B) = \frac{B}{t_{\text{token}}(B) \times N_{\text{GPUs,per-replica}}}
$$

$$
\text{Interactivity}(B) = \frac{1}{t_{\text{token}}(B)}
$$

where $N_{\text{GPUs,per-replica}} = PP \cdot TP \cdot EP \cdot SP$ is the number of GPUs per DP replica.

### Zone 1 — Memory-bound ($B < B^*$)

Weight traffic dominates. From `modeling.tpot.md §6.4.3`:

$$
t_{\text{token}}(B) \approx \frac{T_{\theta,\text{device}}}{B_{\text{eff,mem}}}
\qquad (B \ll B^*)
$$

The step time is approximately **constant** in $B$. Therefore:

$$
\text{Tput/GPU}(B) \approx \frac{B \cdot B_{\text{eff,mem}}}{T_{\theta,\text{device}} \cdot N_{\text{GPUs,per-replica}}} \quad\propto\quad B
$$

$$
\text{Interactivity}(B) = \frac{B}{t_{\text{token}}(B)} \approx \frac{B \times B_{\text{eff,mem}}}{T_{\theta,\text{device}}} \quad\propto\quad B
$$

**Zone 1 behavior:** Both Tput/GPU and Interactivity grow linearly with $B$. This is the ideal operating regime — increasing batch size improves *both* system efficiency *and* per-user streaming speed simultaneously. The system is underutilizing compute; adding more sequences amortizes the fixed weight-load cost over more outputs, benefiting every axis.

### Zone 2 — Crossover ($B \approx B^*$)

The system operates near the ridge point. Both throughput and TPOT transition. This is the "knee" of the Pareto curve — the operating point with the best throughput-per-unit-TPOT ratio, and the natural SLA operating point for production deployments balancing GPU utilization with latency SLAs.

### Zone 3 — Compute-bound ($B > B^*$)

KV cache traffic (or equivalently, compute) dominates. From `modeling.tpot.md §6.4.3`:

$$
t_{\text{token}}(B) \approx \frac{B \times F_{\text{token,device}}}{R_{\text{GPU}}}
\qquad (B \gg B^*)
$$

The step time grows linearly with $B$. Therefore:

$$
\text{Tput/GPU}(B) \approx \frac{R_{\text{GPU}}}{F_{\text{token,device}} \cdot N_{\text{GPUs,per-replica}}} \quad (\text{constant in } B)
$$

$$
\text{Interactivity}(B) = \frac{B}{t_{\text{token}}(B)} \approx \frac{R_{\text{GPU}}}{F_{\text{token,device}}} \quad (\text{constant in } B)
$$

**Zone 3 behavior:** Throughput plateaus at the compute ceiling; Interactivity also plateaus. Both metrics saturate — adding more sequences changes neither per-user streaming speed nor per-GPU token production rate. The system is compute-saturated; the Pareto frontier "flattens out" and further increases in $B$ yield no benefit on either axis.

### Summary table

| Zone | Condition | $\text{Tput/GPU}(B)$ | $\text{Interactivity}(B)$ |
|------|-----------|----------------------|---------------------------|
| 1 — Memory-bound | $B \ll B^*$ | $\propto B$ (growing) | $\propto B$ (growing) |
| 2 — Crossover | $B \approx B^*$ | near maximum | near maximum |
| 3 — Compute-bound | $B \gg B^*$ | $\approx R_{\text{GPU}} / (F \cdot N)$ (flat) | $\approx R_{\text{GPU}} / F$ (flat) |

---

## 6.3 Roofline Ceiling on Pareto Efficiency

A fundamental property of the Pareto frontier emerges when we compute the product $\text{Tput/GPU} \times \text{TPOT} = \text{Tput/GPU} / \text{Interactivity}$ at a given operating point. Using $\text{Tput/GPU}(B) = B / (t_{\text{token}}(B) \cdot N_{\text{GPUs,per-replica}})$ and $\text{TPOT}(B) = t_{\text{token}}(B) / B$:

$$
\text{Tput/GPU}(B) \times \text{TPOT}(B)
= \frac{B}{t_{\text{token}}(B) \cdot N_{\text{GPUs,per-replica}}} \times \frac{t_{\text{token}}(B)}{B}
= \frac{1}{N_{\text{GPUs,per-replica}}}
$$

This identity holds for **any $B$** and any regime — the $B$ and $t_{\text{token}}$ terms cancel exactly:

$$
\boxed{
\text{Tput/GPU} \times \text{TPOT} = \frac{1}{N_{\text{GPUs,per-replica}}}
}
$$

This is the **roofline ceiling** on Pareto efficiency: it is a constant determined entirely by the number of GPUs per DP replica ($N_{\text{GPUs,per-replica}} = PP \cdot TP \cdot EP \cdot SP$). It does not depend on batch size, hardware speed, model size, or serving policy.

**Implication.** No serving strategy can change the product $\text{Tput/GPU} \times \text{TPOT}$ — it is fixed by the parallelism configuration. The only way to improve this product is to use fewer GPUs per replica (tighter model sharding). This gives a clean hardware-model co-design objective: **minimize $N_{\text{GPUs,per-replica}}$** subject to fitting the model in HBM.

Equivalently, for a fixed hardware configuration: higher Tput/GPU always implies lower Interactivity (higher TPOT) in exact proportion, and vice versa. The serving policy (batch size, scheduling algorithm) only determines where on the hyperbola $\text{Tput/GPU} = 1 / (N_{\text{GPUs,per-replica}} \cdot \text{TPOT})$ the system operates.

---

## 6.4 InferenceX Axis Mapping

The InferenceX benchmark [INFERENCEX] organizes LLM inference performance on a two-axis scatter plot:

- **X-axis: Throughput/GPU** — output tokens per second per GPU
- **Y-axis: Interactivity** — output tokens per second per request ($= 1/\text{TPOT}$)

A system at a single operating point (fixed hardware, model, parallelism, and batch size) appears as one point on this plot. As batch size $B$ varies:

- **Zone 1** (increasing $B$ from 1 toward $B^*$): both Tput/GPU and Interactivity grow together — the operating point moves toward the **upper-right**. This is the free-lunch zone: weight traffic is amortized across more sequences, benefiting both axes simultaneously.
- **Zone 2** (near $B^*$): both metrics approach their combined maximum — the **knee** of the curve.
- **Zone 3** (beyond $B^*$): both Tput/GPU and Interactivity plateau — the operating point stops moving. Compute is saturated; further increases in $B$ neither help nor hurt either axis.

The Pareto frontier traced by sweeping $B$ from 1 to $\infty$ has the characteristic shape:

1. **Zone 1 segment**: diagonal — both axes increase together (moving upper-right).
2. **Zone 2 knee**: the curve flattens as both axes approach their ceiling.
3. **Zone 3 plateau**: the curve saturates — Tput/GPU and Interactivity are both approximately constant.

**The ideal operating point** is at or near $B^*$: the system is at the knee of the curve, maximizing both axes simultaneously.

Different hardware configurations (H100 vs. A100), model sizes (7B vs. 70B vs. 405B), and parallelism choices shift the frontier outward or inward. A 3D-stacked accelerator with higher HBM bandwidth [ACCELSTACK] increases $B_{\text{eff,mem}}$, which raises the Zone 1 slope and pushes $B^*$ higher — expanding the diagonal Zone 1 segment and increasing the Zone 3 plateau level.

The roofline ceiling from §6.3 — $\text{Tput/GPU} \times \text{TPOT} = 1 / N_{\text{GPUs,per-replica}}$ — constrains every operating point: all systems lie on the hyperbola $\text{Tput/GPU} = 1 / (N_{\text{GPUs,per-replica}} \cdot \text{TPOT})$ defined by their parallelism configuration, regardless of batch size or serving policy. Systems with smaller $N_{\text{GPUs,per-replica}}$ (more aggressive model sharding) operate on a higher hyperbola — a better hardware-efficiency frontier.

---

<div style="page-break-before: always;"></div>

# Symbol Summary

The following symbols are introduced in this document and are **not** defined in `modeling.notation.md §14` (which refers to this document for their definitions). All other symbols used here are defined in `modeling.notation.md` and cross-referenced to their home sections.

| Symbol | Definition | First used |
|--------|-----------|-----------|
| $N_{\text{out}}$ | Number of output tokens in a single response (decode length) | §1.3 |
| $N_{\text{out}}^{\star}$ | Crossover output length at which TTFT = decode contribution to E2E | §4.2 |
| $C$ | Chunked-prefill chunk size (tokens per chunk) | §2.3 |
| $N_{\text{chunks}}$ | Number of prefill chunks: $\lceil S_{\text{input}} / C \rceil$ | §2.3 |
| $t_{\text{chunk}}$ | Per-chunk prefill latency (roofline at $C$ tokens) | §2.3 |
| $\lambda$ | Request arrival rate (requests/second) for continuous batching analysis | §3.2 |
| $N_{\text{GPUs,per-replica}}$ | GPUs per DP replica: $PP \cdot TP \cdot EP \cdot SP$ | §5.1 |
| $\overline{B_{\text{eff}}}$ | Mean effective batch size in steady-state continuous batching | §3.2 |
| $\overline{\text{TPOT}}$ | Average TPOT over a request's decode lifetime under continuous batching | §3.2 |
| $\text{Tput/GPU}$ | System throughput per GPU: $TTPS / N_{\text{GPUs}}$ (tokens/s/GPU) | §1.4 |
| $\text{Interactivity}$ | Per-user output rate: $1/\text{TPOT}$ (tokens/s/request) | §1.5 |
| $\text{Goodput}$ | Fraction of GPU time spent on useful token generation | §1.6 |

The following existing symbols from `modeling.notation.md` are used extensively; they are listed here for reading convenience:

| Symbol | Defined in | Meaning |
|--------|-----------|---------|
| $TTFT$ | `notation.md §11` | Time To First Token |
| $t_{\text{prefill}}$ | `notation.md §11` | Full prefill latency (roofline + comm + pipeline warmup) |
| $t_{\text{prefill,local}}$ | `notation.md §11` | Per-stage prefill local roofline time |
| $t_{\text{prefill,comm}}$ | `notation.md §11` | Prefill communication time |
| $t_{\text{token}}$ | `notation.md §9` | Overlap-aware per-step decode time |
| $TPS_{\text{single}}$ | `notation.md §9` | Single-replica decode throughput (tokens/s) |
| $TTPS$ | `notation.md §9` | Global decode throughput (tokens/s) |
| $t_{\text{sched}}$ | `notation.md §13` | Request scheduling / batch assembly latency |
| $t_{\text{KV\_transfer}}$ | `notation.md §13` | Disaggregated KV cache transfer latency |
| $B_{\text{eff}}$ | `notation.md §4` | Effective batch size under continuous batching |
| $B^*$ | `tpot.md §6.4.1` | Crossover batch size (memory-bound → compute-bound) |
| $\rho$ | `notation.md §9` | Compute–communication overlap factor |

---

## References

- [INFERENCEX] SemiAnalysis (2024–2025). *InferenceX: LLM Inference Benchmark.* https://inferencex.semianalysis.com/inference — Throughput/GPU vs. Interactivity axes; TPOT and E2E latency definitions.
- [VLLM] Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180 — Continuous batching; PagedAttention.
- [SARATHI] Agrawal et al. (2023). *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* arXiv:2308.16369 — Chunked prefill scheduling; head-of-line blocking reduction.
- [DISAGG-PREFILL] Zhong et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.* OSDI 2024. arXiv:2401.09670 — Disaggregated prefill; KV transfer latency; goodput framework.
- [ROOFLINE] Williams, Waterman & Patterson (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 52(4) — Operational intensity; ridge point.
- [H100-SPEC] NVIDIA Corporation (2022). *NVIDIA H100 Tensor Core GPU Architecture.* WP-10792-001 — H100 SXM5 specs used in numerical examples.
- [ACCELSTACK] Bai et al. (2025). *AccelStack: A Co-Design Framework for 3D-Stacked Accelerators.* HKUST FACT Lab — 3D DRAM bandwidth modeling.
- Sibling documents: [modeling.tpot.md](modeling.tpot.md), [modeling.prefill.md](modeling.prefill.md), [modeling.kv.md](modeling.kv.md), [modeling.framework.md](modeling.framework.md), [modeling.notation.md](modeling.notation.md), [modeling.references.md](modeling.references.md).
