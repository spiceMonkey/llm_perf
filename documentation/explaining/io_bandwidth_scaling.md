# HBM Bandwidth, Network Bandwidth, and Latency: Closed-Form Sensitivity of TPOT and the B* Crossover

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, HBM bandwidth, interconnect latency, roofline, elasticity, TPOT sensitivity, KV wall, crossover batch, Pareto front, scaling laws

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Motivation](#1-motivation)
- [2. Setup and Notation](#2-setup-and-notation)
- [3. Regime 1 — Memory-Bound (B ≤ B\*)](#3-regime-1--memory-bound-b--b)
  - [3.1 Case 1a: Comm Hidden by Memory (ρ · t_mem ≥ t_comm)](#31-case-1a-comm-hidden-by-memory--t_mem--t_comm)
  - [3.2 Case 1b: Unhidden Comm on the Critical Path](#32-case-1b-unhidden-comm-on-the-critical-path)
- [4. Regime 2 — Compute-Bound (B ≫ B\*)](#4-regime-2--compute-bound-b--b)
- [5. Regime 3 — Crossover (B = B\*)](#5-regime-3--crossover-b--b)
  - [5.1 Derivation of B\* Elasticity](#51-derivation-of-b-elasticity)
  - [5.2 The KV Wall: κ > 1](#52-the-kv-wall---1)
  - [5.3 Throughput at the Knee is HBM-Invariant](#53-throughput-at-the-knee-is-hbm-invariant)
- [6. Two-Network Extension: Scale-Up vs Scale-Out](#6-two-network-extension-scale-up-vs-scale-out)
  - [6.1 Decomposed Comm Model](#61-decomposed-comm-model)
  - [6.2 Four-Way Elasticity](#62-four-way-elasticity)
  - [6.3 Message-Size Asymmetry](#63-message-size-asymmetry)
  - [6.4 PP Add/Remove Inequality](#64-pp-addremove-inequality)
  - [6.5 Provisioning Implications](#65-provisioning-implications)
  - [6.6 The Byte-Ratio Rule: $B_n^{up}$ as a Function of $B_m$](#66-the-byte-ratio-rule-b_nup-as-a-function-of-b_m)
- [7. Scaling-Law Cheat Sheet](#7-scaling-law-cheat-sheet)
- [8. Three Headline Takeaways](#8-three-headline-takeaways)
- [9. Numerical Validation](#9-numerical-validation)
- [10. FAQ](#10-faq)
- [References](#references)

---

<div style="page-break-before: always;"></div>

# 1. Motivation

Modern LLM inference system co-design trades three expensive knobs:

1. **HBM bandwidth** $B_m$ — dominates single-user decode latency
2. **Interconnect bandwidth** $B_n$ — dominates collective time at large batches and for large tensor/sequence shards
3. **Interconnect latency** $\alpha$ — dominates collective time at small batches and high collective radix

A common question is: *"If I double HBM bandwidth, how much does TPOT improve?"* The answer depends strongly on **where on the Pareto front the system is operating**. In the memory-bound regime, HBM bandwidth buys interactivity 1-for-1. In the compute-bound regime, HBM bandwidth does **nothing** — it is entirely off the critical path. Crucially, at the knee of the Pareto front ($B = B^*$), HBM bandwidth moves the knee but does **not** change the throughput ceiling.

This document derives the closed-form sensitivities — equivalently, log-log elasticities $\varepsilon_x = \partial \ln(\mathrm{TPOT}) / \partial \ln(x)$ — of user-observed TPOT and the crossover batch $B^*$ with respect to $(B_m, B_n, \alpha)$, at each of three characteristic operating points. The derivations are validated numerically against the `llm_perf` core code in §8.

---

# 2. Setup and Notation

The analytical decode model is developed in `documentation/modeling/decode.md`. We summarize the minimal set of identities used here.

**Per-stage step time** (`decode.md §6.2`):
$$
t_{\text{stage}}(B) = t_{\text{local}}(B) + \max\!\bigl(0,\; t_{\text{comm}}(B) - \rho \cdot t_{\text{local}}(B)\bigr)
$$

where the roofline $t_{\text{local}}$ is:
$$
t_{\text{local}}(B) = \max\!\bigl(t_{\text{compute}}(B),\; t_{\text{mem}}(B)\bigr)
\qquad
t_{\text{compute}} = \frac{B \cdot F}{R}, \quad t_{\text{mem}} = \frac{T_\theta + B \cdot T_{\text{kv}}}{B_m}
$$

and $\rho \in [0,1]$ is the overlap factor between compute/memory and communication.

**Collective time** ($\alpha$–$\beta$ model):
$$
t_{\text{comm}}(B) = \underbrace{\alpha \cdot N_{\text{hops}}}_{\text{latency term}} \;+\; \underbrace{\frac{m(B)}{B_n}}_{\text{bandwidth term}}
$$

For decode, message sizes $m(B)$ scale linearly with $B$ (see `decode.md §5`: $m_{TP}, m_{EP}, m_{PP}, m_{SP}$ all carry a $B$ factor).

**User-observed TPOT** (bubble-corrected; `decode.md §7.2`):
$$
\mathrm{TPOT}(B) = t_{\text{stage}}(B) \cdot \max\!\left(1, \frac{PP}{B}\right)
$$

**Crossover batch** $B^*$:
$$
B^* = \frac{T_\theta \cdot R}{F \cdot B_m - T_{\text{kv}} \cdot R}
\qquad (B^* = \infty \text{ when the denominator is } \le 0)
$$

We define the **KV-wall parameter**:
$$
\boxed{\;\kappa \;=\; \frac{T_{\text{kv}} \cdot R}{F \cdot B_m}\;}
$$

$\kappa < 1$ means the denominator is positive and a finite crossover exists. $\kappa \ge 1$ means per-batch KV traffic cannot be amortized by compute — no amount of batching will cross into compute-bound (the **KV wall**).

**Elasticity shorthand:**
$$
\varepsilon_x \;\equiv\; \frac{\partial \ln(\mathrm{TPOT})}{\partial \ln(x)}
$$

A 1% change in $x$ induces an $\varepsilon_x\%$ change in TPOT (to leading order). We treat $B \ge PP$ throughout so the bubble factor is unity and does not obscure the sensitivity to $(B_m, B_n, \alpha)$.

---

# 3. Regime 1 — Memory-Bound (B ≤ B\*)

In the memory-bound regime, $t_{\text{mem}} \ge t_{\text{compute}}$, so $t_{\text{local}} = t_{\text{mem}} = (T_\theta + B \cdot T_{\text{kv}})/B_m$. The sensitivities depend on whether communication is hidden or not.

## 3.1 Case 1a: Comm Hidden by Memory ($\rho \cdot t_{\text{mem}} \ge t_{\text{comm}}$)

With full overlap, $t_{\text{stage}} = t_{\text{mem}}$ and $\mathrm{TPOT} = t_{\text{mem}}$. Differentiating:

$$
\boxed{\;\varepsilon_{B_m} = -1, \qquad \varepsilon_{B_n} = 0, \qquad \varepsilon_{\alpha} = 0\;}
$$

**Interpretation.** HBM bandwidth buys interactivity 1-for-1. Network bandwidth and latency are invisible — they are entirely absorbed into the overlap window. This is the "clean" memory-bound regime that the single-user decode marketing literature implicitly assumes.

## 3.2 Case 1b: Unhidden Comm on the Critical Path

When $t_{\text{comm}} > \rho \cdot t_{\text{mem}}$, the stage time is:
$$
t_{\text{stage}} = t_{\text{mem}} + (t_{\text{comm}} - \rho \cdot t_{\text{mem}}) = (1 - \rho) \cdot t_{\text{mem}} + t_{\text{comm}}
$$

Let $f_m$, $f_A$, $f_C$ be the fractions of the critical path spent in (unhidden) memory, $\alpha$-latency comm, and bandwidth-term comm respectively, with $f_m + f_A + f_C = 1$. Then:

$$
\boxed{\;\varepsilon_{B_m} = -f_m, \qquad \varepsilon_{B_n} = -f_C, \qquad \varepsilon_{\alpha} = +f_A\;}
$$

**Interpretation.** The elasticity of each component equals its share of the critical path. If comm is $\alpha$-dominated (small $B$, high collective radix), $\alpha$ matters and $B_n$ does not. If comm is bandwidth-dominated (large messages, wide tensor shards), $B_n$ matters and $\alpha$ does not. HBM bandwidth still helps, but only by the fraction of time the critical path spends in unhidden memory.

**Special case $\rho = 0$ (no overlap).** $f_m = t_{\text{mem}} / t_{\text{stage}}$ directly; overlap just compresses $f_m$ toward zero without changing the structure.

---

# 4. Regime 2 — Compute-Bound (B ≫ B\*)

At high batch sizes, $t_{\text{compute}} \ge t_{\text{mem}}$, so $t_{\text{local}} = t_{\text{compute}} = B \cdot F / R$. HBM bandwidth is completely off the critical path.

**Case 2a (comm hidden by compute).** $t_{\text{stage}} = t_{\text{compute}}$:
$$
\boxed{\;\varepsilon_{B_m} = 0, \qquad \varepsilon_{B_n} = 0, \qquad \varepsilon_{\alpha} = 0\;}
$$

Interactive sensitivity is zero in all three I/O knobs. Only FLOPS $R$, the shard factors, and the model shape move TPOT here. **HBM bandwidth is wasted** — a higher-BW part gets the same TPOT as a lower-BW part, up to the point where it drops back into memory-bound.

**Case 2b (unhidden comm).** With $t_{\text{stage}} = (1-\rho) t_{\text{compute}} + t_{\text{comm}}$:
$$
\boxed{\;\varepsilon_{B_m} = 0, \qquad \varepsilon_{B_n} = -f_C, \qquad \varepsilon_{\alpha} = +f_A\;}
$$

HBM bandwidth is still zero; only the comm terms register. Since decode messages scale with $B$, the bandwidth term tends to dominate $\alpha$ at large $B$, making $\varepsilon_{B_n}$ the dominant elasticity in this regime.

---

# 5. Regime 3 — Crossover (B = B\*)

The crossover batch $B^*$ is defined by $t_{\text{compute}}(B^*) = t_{\text{mem}}(B^*)$, equivalently the value of $B$ at which HBM bandwidth stops helping. This is the knee of the Pareto front.

## 5.1 Derivation of B\* Elasticity

Starting from $B^* = T_\theta R / (F B_m - T_{\text{kv}} R)$ and recalling $\kappa = T_{\text{kv}} R / (F B_m)$:
$$
B^* = \frac{T_\theta R / (F B_m)}{1 - \kappa}
$$

Taking $\ln$ and differentiating with respect to $\ln B_m$ (noting $\ln(T_\theta R / (F B_m))$ has elasticity $-1$ and $\ln(1-\kappa)$ has elasticity $\kappa/(1-\kappa)$ in $B_m$ via $\kappa \propto 1/B_m$):

$$
\boxed{\;\varepsilon^{B^*}_{B_m} = -\frac{1}{1-\kappa}\;}
$$

**Interpretation.** $B^*$ is *more* sensitive to HBM bandwidth than a naive $-1$ scaling: a 1% bandwidth boost shrinks the crossover batch by $1/(1-\kappa)\%$. Close to the KV wall ($\kappa \to 1^-$), the elasticity diverges — small HBM-BW changes cause huge shifts in the knee location.

For TPOT **at** the crossover, $\mathrm{TPOT}(B^*) = B^* F / R$, so:
$$
\boxed{\;\varepsilon^{\mathrm{TPOT}@B^*}_{B_m} = -\frac{1}{1-\kappa}\;}
$$

## 5.2 The KV Wall: κ > 1

If $\kappa \ge 1$, the denominator of $B^*$ is non-positive. Physically, each additional request loads $T_{\text{kv}}$ bytes of KV cache from HBM that cannot be amortized by the matching compute even at infinite batch. The system is **always** memory-bound — there is no compute-bound regime to cross into.

This happens when context length $S$ is large (because $T_{\text{kv}} \propto S$), HBM is narrow, or compute is very fast relative to HBM. When $\kappa > 1$, the *only* way to reduce TPOT is to increase $B_m$ (or shrink $T_{\text{kv}}$ via shorter context, KV quantization, or more aggressive SP sharding).

## 5.3 Throughput at the Knee is HBM-Invariant

At $B = B^*$, $t_{\text{local}} = t_{\text{compute}} = B^* F / R$ and per-replica throughput is:
$$
\mathrm{TPS}_{\text{single}}(B^*) = \frac{B^*}{\mathrm{TPOT}(B^*)} = \frac{R}{F}
$$

This is independent of $B_m$, $B_n$, $\alpha$. In other words, **the Pareto front's throughput ceiling is set entirely by FLOPS and model shape** — the I/O knobs shift *where* on the x-axis the knee appears, but not its height. This is the mathematical statement of the cliché "HBM bandwidth buys interactivity, not throughput."

---

# 6. Two-Network Extension: Scale-Up vs Scale-Out

Real systems rarely have a single uniform fabric. A modern cluster typically has two distinct network tiers:

- **Scale-up** (intra-node / intra-rack): NVLink, NVSwitch, high-radix copper — used for TP, EP, SP collectives that must run at shard frequency. Characterized by $(B_n^{up},\, \alpha^{up})$ with $\alpha^{up}$ on the order of 1 μs and bandwidth in the multi-TB/s range.
- **Scale-out** (inter-node / inter-rack): InfiniBand, Ethernet+RoCE, photonics — used for PP hops between pipeline stages and for cross-rack DP gradient sync. Characterized by $(B_n^{out},\, \alpha^{out})$ with $\alpha^{out}$ on the order of 3–10 μs and bandwidth 10–100× lower than scale-up.

The single-network analysis of §3–5 lumps all comm into one $t_{\text{comm}}$ with one $(B_n, \alpha)$ pair. This section refines the model by splitting comm into its scale-up and scale-out components.

## 6.1 Decomposed Comm Model

Within one pipeline stage, the collective critical path is the sum of scale-up collectives (TP + EP + SP, which happen inside each MLP/attention block) and the outbound scale-out PP hop at the stage boundary:
$$
t_{\text{comm}}(B) = \underbrace{\alpha^{up} N_{\text{hops}}^{up} + \frac{m^{up}(B)}{B_n^{up}}}_{t_{\text{comm}}^{up}\;(\text{TP + EP + SP})} \;+\; \underbrace{\alpha^{out} + \frac{m_{PP}(B)}{B_n^{out}}}_{t_{\text{comm}}^{out}\;(\text{PP hop})}
$$

where $m^{up}(B) = m_{TP}(B) + m_{EP}(B) + m_{SP}(B)$ aggregates the scale-up message sizes (all $\propto B$; see `decode.md §5`) and $m_{PP}(B) = B \cdot H/TP \cdot b$ is the per-stage PP message. All other identities from §2 carry over unchanged.

The HBM-side model, the roofline $t_{\text{local}}$, the bubble correction, and the crossover parameter $\kappa = T_{\text{kv}} R / (F B_m)$ are all **unchanged** — the two-network split only touches the comm term. In particular, the ceiling $R/F$ at the knee and the elasticity $\varepsilon_{B_m}$ are identical to the single-network case.

## 6.2 Four-Way Elasticity

In the unhidden-comm regimes (1b and 2b), let $f_m$, $f_A^{up}$, $f_C^{up}$, $f_A^{out}$, $f_C^{out}$ be the fractions of the critical path spent in unhidden memory, scale-up $\alpha$, scale-up bandwidth, scale-out $\alpha$, and scale-out bandwidth respectively, with:
$$
f_m + f_A^{up} + f_C^{up} + f_A^{out} + f_C^{out} = 1
$$

Then the two-knob $(\alpha, B_n)$ elasticities of §3.2 and §4 split into four:
$$
\boxed{\;
\varepsilon_{B_n^{up}} = -f_C^{up}, \quad
\varepsilon_{\alpha^{up}} = +f_A^{up}, \quad
\varepsilon_{B_n^{out}} = -f_C^{out}, \quad
\varepsilon_{\alpha^{out}} = +f_A^{out}
\;}
$$

HBM elasticity $\varepsilon_{B_m} = -f_m$ (Regime 1b) or $0$ (Regime 2) is preserved. The crossover elasticity $\varepsilon^{B^*}_{B_m} = -1/(1-\kappa)$ is preserved. In the fully-hidden sub-cases (1a, 2a), all four network elasticities drop to zero just as in the single-network model.

## 6.3 Message-Size Asymmetry

The key structural fact about the split is that scale-up and scale-out messages are **not comparable in size**. From the decode message identities:
$$
\frac{m_{PP}(B)}{m_{TP}(B)} = \frac{B \cdot H/TP \cdot b}{B \cdot H \cdot b} = \frac{1}{TP}
$$

For typical $TP \in \{4, 8, 16\}$, the PP hop carries an order of magnitude less data than a single TP all-reduce. Consequently:

1. **Scale-out is nearly always $\alpha$-dominated.** The ratio $f_A^{out} / f_C^{out}$ scales as $\alpha^{out} B_n^{out} \cdot TP / (B \cdot H \cdot b)$, which for $\alpha^{out} = 5$ μs, $B_n^{out} = 50$ GB/s, $TP = 8$, $H = 8192$, $b = 2$ B gives $f_A^{out} / f_C^{out} \approx 120 / B$. Only at batch sizes above $\sim 100$ does scale-out bandwidth start to matter at all.

2. **Scale-up flips to bandwidth-dominated faster.** The same ratio for scale-up gives $f_A^{up} / f_C^{up} \approx \alpha^{up} B_n^{up} / (B \cdot H \cdot b)$ which for $\alpha^{up} = 1$ μs, $B_n^{up} = 900$ GB/s is $\sim 55 / B$ — scale-up becomes $B_n$-dominated by $B \approx 50$.

3. **At small $B$, $\alpha^{out}$ often dominates the entire comm budget.** Even though scale-out is only one hop per stage, its latency is 3–10× the scale-up latency, and the PP message is too small for bandwidth to matter. At $B = 1$ it is common for $f_A^{out}$ to rival or exceed $f_A^{up}$ despite TP/EP/SP collectives having many more hops.

## 6.4 PP Add/Remove Inequality

The single-network §3–5 analysis treats PP as a fixed part of the topology. With the scale-out term exposed, we can state a quantitative criterion for when adding a PP dimension helps or hurts user-observed TPOT.

Consider going from $PP \to PP + 1$ while holding the device budget fixed. The $t_{\text{mem}}$ term shrinks because weights-per-stage drop:
$$
\Delta t_{\text{mem}} \;=\; -\frac{T_\theta}{PP(PP+1) \cdot B_m} \;<\; 0 \quad (\text{gain})
$$

But the scale-out hop budget grows — each token now traverses one more PP stage, adding one scale-out hop to the critical path:
$$
\Delta t_{\text{comm}}^{out} \;=\; +\left(\alpha^{out} + \frac{B \cdot H / TP \cdot b}{B_n^{out}}\right) \quad (\text{cost})
$$

Under the overlap-aware stage model, and assuming we stay in the memory-bound regime (so $t_{\text{local}} = t_{\text{mem}}$), the PP upgrade helps if and only if:
$$
\boxed{\;
\frac{T_\theta}{PP(PP+1) \cdot B_m} \;>\; (1 - \rho)\cdot\!\left[\alpha^{out} + \frac{B \cdot H / TP \cdot b}{B_n^{out}}\right]
\;}
$$

plus the bubble penalty if $B < PP + 1$ (see `pipeline_bubble.md §5`). This is the scale-out-aware version of the decision rule implicit in `pipeline_bubble.md §6`.

**Interpretation.** The LHS is "memory-time per stage saved by the split". The RHS is "scale-out hop added per token traversal, minus whatever the overlap window absorbs". The inequality flips when any of: $B_m$ grows (memory-time cheaper), $\alpha^{out}$ grows (hops more expensive), $B$ grows at fixed $TP$ (PP hop less bandwidth-efficient), $\rho$ drops (less overlap). At $B = 1$ with small-to-medium models, the inequality is often violated — which is the mathematical statement of "PP hurts single-user decode on a cross-node fabric."

## 6.5 Provisioning Implications

The four-way decomposition gives a clean co-design recipe:

- **Scale-up bandwidth $B_n^{up}$ is the throughput lever.** Messages $m_{TP}, m_{EP}, m_{SP}$ all grow with $B$, so at the compute-bound end of the Pareto front, $\varepsilon_{B_n^{up}}$ is the dominant non-zero elasticity. Batch-heavy deployments (offline serving, RAG pipelines) should spend on scale-up bandwidth.

- **Scale-out latency $\alpha^{out}$ is the interactivity lever.** At $B = 1$ with non-trivial PP, $f_A^{out}$ often dominates the comm budget. Interactive chat deployments should spend on scale-out *latency* (hop count reduction, photonic/direct-attach fabrics) before spending on scale-out *bandwidth*.

- **Scale-out bandwidth $B_n^{out}$ is rarely the right lever.** The $1/TP$ message-size ratio means scale-out bandwidth only matters at unusually high $B$ or unusually low $TP$. For typical deployments, it can be undersized relative to scale-up by a factor of $TP$ without measurable TPOT impact.

- **Scale-up latency $\alpha^{up}$ matters at small $B$ with deep TP/EP trees.** The $N_{\text{hops}}^{up}$ multiplier means that at high TP radix, $\alpha^{up}$ can still register even with 1 μs per hop. This is where scale-up latency — the link design, not just its bandwidth — starts to matter for single-user decode.

The §7 cheat sheet below shows the single-network elasticities; the two-network extension just splits each $\varepsilon_{\alpha}$ and $\varepsilon_{B_n}$ row into an "up" and an "out" component, preserving the sign and summing to the original value.

## 6.6 The Byte-Ratio Rule: $B_n^{up}$ as a Function of $B_m$

The elasticities of §6.2 tell us *how sensitive* TPOT is to each knob, but they do not by themselves say *how much* scale-up bandwidth a given HBM bandwidth demands. This subsection closes that loop with a single closed-form rule.

### Balance condition

For scale-up communication to stay inside the overlap window — i.e., for comm to not land on the critical path — the time to move scale-up comm bytes must fit within $\rho$ times the time to move HBM bytes:
$$
\frac{M^{up}_{\text{comm}}}{B_n^{up}} \;\le\; \rho \cdot \frac{M_{\text{hbm}}}{B_m}
$$

Rearranging gives the **byte-ratio rule**:
$$
\boxed{\;B_n^{up} \;\ge\; \frac{M^{up}_{\text{comm}}}{\rho \cdot M_{\text{hbm}}} \cdot B_m\;}
$$

In words: the required scale-up bandwidth scales **linearly with HBM bandwidth**, with a slope set by the ratio of bytes moved over the scale-up network to bytes moved from HBM per decode step (divided by the overlap factor).

### Closed form for the KV-light regime

Per transformer layer per decode step, to leading order:

- **HBM reads** (dense weights): $M_{\text{hbm}}^{(\ell)} \approx \dfrac{2 H^2 b}{TP}$
- **Scale-up comm** (two TP all-reduces per block): $M^{up,(\ell)}_{\text{comm}} \approx 2 \cdot B H b$

Plugging in and canceling:
$$
\boxed{\;\frac{B_n^{up}}{B_m} \;\approx\; \frac{B \cdot TP}{\rho \cdot H} \quad (\text{KV-light regime})\;}
$$

The scale-up-to-HBM bandwidth ratio equals $(B \cdot TP) / (\rho \cdot H)$. Note that $L$ and $PP$ drop out — this is a *per-layer* ratio, so it is independent of how many layers there are or how they are distributed across pipeline stages.

### KV correction (long-context regime)

When KV cache traffic is non-negligible, HBM reads grow but scale-up comm does not (KV is not transmitted on the scale-up fabric). The corrected expression is:
$$
\frac{B_n^{up}}{B_m} \;\approx\; \frac{B \cdot TP}{\rho \cdot H} \cdot \frac{1}{\,1 + \dfrac{B \cdot S \cdot H_{kv}}{H^2 \cdot SP}\,}
$$

The extra denominator term is the KV share of HBM traffic. Longer contexts make scale-up BW **cheaper** relative to HBM BW — because each decode step reads many KV bytes that are not transmitted over the scale-up network.

### Sanity table

Fixing $\rho = 0.9$, $H = 8192$ (Llama-3-70B class), and using the KV-light closed form:

| $B$ | $TP$ | $B_n^{up}/B_m$ | $B_n^{up}$ at $B_m = 4$ TB/s |
|---:|---:|---:|---:|
| 1 | 8 | 0.0011 | 4.3 GB/s |
| 8 | 8 | 0.0087 | 35 GB/s |
| 64 | 8 | 0.070 | 278 GB/s |
| 256 | 16 | 0.56 | 2.2 TB/s |

At $B = 1$ the scale-up requirement is minuscule — a few GB/s of scale-up bandwidth suffices for overlap. Modern NVLink's TB/s-class bandwidth is vastly over-provisioned at this operating point (which is why the single-user regime is insensitive to scale-up bandwidth in the elasticities of §6.2). At $B \approx 256$ with $TP = 16$, the requirement saturates NVLink — which is why real NVLink domains are sized for the throughput regime, not the single-user regime.

### Three takeaways

1. **$B_n^{up} \propto B_m$ with a workload-dependent slope.** If you double HBM bandwidth and want to preserve overlap at the same operating point, you must double scale-up bandwidth. The two are coupled, not independent.

2. **The slope is set by workload + partition, not by silicon.** The ratio $B \cdot TP / (\rho H)$ involves no hardware parameters. Hardware designers can read this as: "given the target $(B, TP)$ for this deployment, this is the minimum scale-up-to-HBM bandwidth ratio I need."

3. **Long context shifts the balance toward HBM.** The KV correction term shows that long-context serving is HBM-bound (not network-bound), while batched short-prompt serving is the regime where scale-up networks pay for themselves. This is the mathematical statement of why very-long-context deployments tend to look HBM-limited.

---

# 7. Scaling-Law Cheat Sheet

| Regime | Condition | $\varepsilon_{B_m}$ | $\varepsilon_{B_n}$ | $\varepsilon_{\alpha}$ | Notes |
|---|---|---|---|---|---|
| **1a** Memory-bound, comm hidden | $B \le B^*$, $\rho t_{\text{mem}} \ge t_{\text{comm}}$ | $-1$ | $0$ | $0$ | Clean HBM-limited regime |
| **1b** Memory-bound, unhidden comm | $B \le B^*$, else | $-f_m$ | $-f_C$ | $+f_A$ | Fractions sum to 1 |
| **2a** Compute-bound, comm hidden | $B \gg B^*$, $\rho t_{\text{compute}} \ge t_{\text{comm}}$ | $0$ | $0$ | $0$ | HBM BW wasted |
| **2b** Compute-bound, unhidden comm | $B \gg B^*$, else | $0$ | $-f_C$ | $+f_A$ | Dominated by $B_n$ at large $B$ |
| **3** Crossover ($B = B^*$) | exact knee | $-\frac{1}{1-\kappa}$ | — | — | Diverges as $\kappa \to 1^-$ |

Pareto-front ceiling (throughput at $B^*$) $= R/F$ is **invariant** in all three I/O knobs.

---

# 8. Three Headline Takeaways

**① HBM bandwidth buys interactivity, not throughput.**
In memory-bound regime it gives a full $\varepsilon_{B_m} = -1$ payoff on TPOT. In compute-bound regime its payoff is zero. At the knee, it moves the knee but leaves the throughput ceiling $R/F$ unchanged. When provisioning for chat/interactive workloads (low $B$), invest in HBM BW. When provisioning for offline/throughput workloads (high $B$), invest in FLOPS.

**② $\alpha$ matters at small $B$; $B_n$ matters at large $B$.**
Decode collective messages carry a $B$ factor, so the bandwidth term $m(B)/B_n$ grows with $B$ while the latency term $\alpha \cdot N_{\text{hops}}$ is $B$-invariant. At $B=1$ with deep TP/EP trees, $\alpha$ is often > 90% of comm. At $B=100+$ with wide messages, $B_n$ dominates. Fabric design should target whichever regime the deployment lives in.

**③ $\kappa$ predicts HBM-BW return-on-investment.**
The single scalar $\kappa = T_{\text{kv}} R / (F B_m)$ summarizes the entire trade. $\kappa \ll 1$: HBM upgrades help only until $B^*$, then stop. $\kappa \to 1^-$: HBM upgrades have outsized impact on the knee. $\kappa > 1$: the KV wall — HBM is the **only** thing that moves TPOT, no matter the batch. Compute $\kappa$ first; it tells you whether HBM is the right investment.

---

# 9. Numerical Validation

Derivations were sanity-checked against the `llm_perf` core pipeline (`memory_model.py` → `decode_model.py` / `prefill_model.py` → `primitives/`) by sweeping each of $B_m$, $B_n$, $\alpha$ by ±1% and measuring the induced log-change in TPOT.

**Regime 1b validation** (TP=8, $\rho=0$, $B=1$, $S_{\text{decode}}=1024$, H100-class system). Configuration chosen to force unhidden comm on the critical path:

| Measured | Predicted ($f_A = 82.3\%$, $f_C = 0.2\%$) |
|---|---|
| $\varepsilon_{\alpha} = +0.823$ | $+f_A = +0.823$ ✓ |
| $\varepsilon_{B_n} = -0.002$ | $-f_C = -0.002$ ✓ |
| $\varepsilon_{B_m} = -0.175$ | $-f_m = -0.175$ ✓ |

Confirms that at $B=1$ with deep TP, comm is $\alpha$-dominated (82% of critical path), matching takeaway ②.

**Regime 2 validation** (reducing $S_{\text{decode}}$ to 256 to shift $\kappa < 1$, then sweeping to $B = 20 \cdot B^*$):

| Measured | Predicted |
|---|---|
| $\varepsilon_{B_m} = 0.000$ | $0$ ✓ (HBM wasted) |
| $\varepsilon_{B_n} = -0.831$ | $-f_C$ (bandwidth-dominated) ✓ |
| $\varepsilon_{\alpha} = +0.011$ | $+f_A$ (small, as expected) ✓ |

Confirms that at large $B$, comm is bandwidth-dominated, and HBM BW contributes zero to TPOT.

**KV wall validation.** The default example config has $\kappa = 2.76$ at $S_{\text{decode}} = 1024$, so $B^* = \infty$ — the prediction is that no finite batch crosses into compute-bound. Numerically, sweeping $B$ from $1$ to $10^4$ shows the system remains memory-bound throughout, and $\varepsilon_{B_m}$ stays negative (never reaches zero). This itself validates the $\kappa$ predictor.

---

# 10. FAQ

**Q: Why does $\varepsilon_{B_m}$ change sign or magnitude at the knee?**
It does not change sign — it goes from $-1$ (memory-bound) through $-1/(1-\kappa)$ at the knee to $0$ (compute-bound). The transition is continuous; the knee is just where the derivative bottoms out. At the knee, bandwidth sensitivity is *maximized* because the system is simultaneously memory-bound (so $B_m$ still matters) and on the verge of compute-bound (so the crossover location itself becomes hyper-sensitive).

**Q: Why does $\varepsilon_{\alpha}$ appear *positive*?**
Because we defined $\varepsilon_x = \partial \ln(\mathrm{TPOT}) / \partial \ln(x)$ — lower $\alpha$ (better) reduces TPOT (good). A positive elasticity in $\alpha$ means "TPOT grows with $\alpha$", which is the physically expected direction.

**Q: How do I use $\kappa$ in practice?**
Compute $T_\theta$, $T_{\text{kv}}$, $F$, $R$, $B_m$ from the model/system specs. If $\kappa > 1$, you are on the KV wall — only HBM-BW upgrades (or shorter $S$, or KV quantization) will help. If $\kappa \ll 1$, you have substantial compute headroom — batching will amortize HBM-BW cost, but eventually $B_n$ will dominate.

**Q: Does this ignore the pipeline bubble?**
Section 5 of `pipeline_bubble.md` shows that for $B \ge PP$ the bubble factor is unity, so the elasticities above apply directly. For $B < PP$ the bubble factor $PP/B$ is a multiplicative constant in TPOT and does not change the elasticities with respect to $(B_m, B_n, \alpha)$ — it only affects their $B$-dependence indirectly through which regime the system lives in.

**Q: What about DP?**
DP scales throughput (TTPS) but not per-replica TPOT. The elasticities here are for a single DP replica. TTPS elasticities are identical with the sign flipped (because TTPS $\propto 1/\mathrm{TPOT}$).

---

# References

- [GPIPE] Huang et al. *GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism.* NeurIPS 2019.
- [MEGATRON] Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* arXiv:1909.08053.
- [VLLM] Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- [ROOFLINE] Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* CACM 2009.
- [ALPHA-BETA] Hockney. *The communication challenge for MPP: Intel Paragon and Meiko CS-2.* Parallel Computing 1994.
- [decode.md] `documentation/modeling/decode.md` — full derivation of the decode model
- [pipeline_bubble.md] `documentation/explaining/pipeline_bubble.md` — PP bubble correction
- [notation.md] `documentation/modeling/notation.md` — symbol glossary
