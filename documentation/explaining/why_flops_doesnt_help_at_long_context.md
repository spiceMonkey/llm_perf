# Why Peak FLOPS Doesn't Help at Long Context (Even at High $B$)

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, decode, peak FLOPS, roofline, batch size, $B^*$ crossover, arithmetic intensity, long context, GQA, KV traffic, memory-bound asymptote

---

## 1. The puzzle

The `notebooks/pareto_vs_flops.ipynb` sweep sets GPU peak FLOPS to `[0.5×, 1×, 2×, 4×, ∞]` on a fixed (HBM BW, scale-up I/O) machine and runs GPT-1.8T MoE FP4 on 72 GB200 GPUs. Every curve — including the FLOPS → ∞ reference — **overlaps exactly**.

The textbook roofline intuition is that extra FLOPS should lift the high-throughput (high-$B$) corner, because decode becomes compute-bound once $B$ is large enough to amortize the per-step weight read. Here it doesn't — not for any $B$ the paged-KV cache allows, and **not even asymptotically as $B \to \infty$**.

This note explains why: for this workload, the $B^*$ crossover does not exist — the denominator in the $B^*$ formula is negative. Extra FLOPS is pushing against a bottleneck that doesn't move.

## 2. The roofline in $B$

For a single decode step on one device, with batch size $B$ at the pipeline stage:

$$t_{compute}(B) = \frac{F_{token}}{R_{gpu}} \cdot B$$

$$t_{mem}(B) = \frac{T_\theta}{B_{eff,mem}} + \frac{T_{kv}}{B_{eff,mem}} \cdot B$$

$$t_{token}(B) = \max\bigl(t_{compute}(B),\; t_{mem}(B)\bigr) + t_{comm}^{unhidden}$$

- $T_\theta$ is the *fixed* per-step weight read (loaded once, amortized across all $B$ sequences).
- $T_{kv}$ is the per-sequence KV read per step.
- $F_{token}$ is FLOPs per token per device.

Both $t_{compute}$ and $t_{mem}$ are linear in $B$. The question of whether $t_{compute}$ ever overtakes $t_{mem}$ reduces to comparing their **slopes**.

## 3. The slope condition

$t_{compute}$ slope in $B$: $\;F_{token} / R_{gpu}$

$t_{mem}$ slope in $B$: $\;T_{kv} / B_{eff,mem}$

Setting $t_{compute}(B^*) = t_{mem}(B^*)$ and solving:

$$B^* = \frac{T_\theta \cdot R_{gpu}}{F_{token} \cdot B_{eff,mem} \;-\; T_{kv} \cdot R_{gpu}}$$

The standard "bigger $B$ turns decode compute-bound" story assumes the denominator is positive — i.e., $t_{compute}$'s slope is larger than $t_{mem}$'s slope — so the two lines eventually cross at a finite $B^*$.

If the denominator is **negative** — $t_{mem}$'s slope exceeds $t_{compute}$'s slope — then the two lines are *diverging* as $B$ grows. $B^*$ formally goes to $-\infty$ and then wraps to $+\infty$: there is no finite crossover, and more $B$ makes the memory-bound gap **wider**, not narrower.

An equivalent framing in arithmetic intensity: let

$$\text{AI}_B \;\equiv\; \frac{F_{token}}{T_{kv}} \quad (\text{FLOPs per byte added per unit of }B)$$

$$\text{AI}_{machine} \;\equiv\; \frac{R_{gpu}}{B_{eff,mem}}$$

- $\text{AI}_B > \text{AI}_{machine}$ → finite $B^*$, compute-bound at high $B$.
- $\text{AI}_B < \text{AI}_{machine}$ → no $B^*$, permanently memory-bound.

## 4. Where GPT-1.8T MoE FP4 on GB200 lands

From the notebook's §8 diagnostic on the winning partition (PP=8, TP=8, EP=1, SP=1) at $S$=8192:

| quantity | slope per $B$ |
|---|---|
| $t_{compute}$: $F_{token}/R_{gpu}$ | 1.27 μs/B |
| $t_{mem}$: $T_{kv}/B_{eff,mem}$ | 4.9 μs/B |

The memory slope is ~4× larger. Converting back to per-$B$ work:

- Added compute per unit $B$: $1.27\,\mu s \times 9000\,\text{TF} \approx 11.4$ GFLOPs
- Added memory read per unit $B$: $4.9\,\mu s \times 8\,\text{TB/s} \approx 39$ MB

$$\text{AI}_B \approx \frac{11.4 \times 10^9}{39 \times 10^6} \approx 290 \;\text{FLOPs/byte}$$

The machine's balance point:

$$\text{AI}_{machine} \;=\; \frac{9000\,\text{TF/s}}{8\,\text{TB/s}} \;=\; 1125 \;\text{FLOPs/byte}$$

$\text{AI}_B = 290 < 1125 = \text{AI}_{machine}$ — the per-$B$ arithmetic intensity is well *below* the machine's balance point. The $B^*$ denominator is negative. $t_{compute}$ and $t_{mem}$ diverge as $B$ grows. No finite $B$ makes this decode step compute-bound.

## 5. Why this workload sits in that regime

What makes $\text{AI}_B$ so low? The per-$B$ KV read dominates.

At long context, each extra sequence drags a large KV-cache read per decode step:

$$T_{kv}^{(\text{per seq, per device})} \;=\; \frac{2 \cdot S \cdot H_{kv} \cdot b \cdot (L/PP)}{TP \cdot SP}$$

For this model: $S$=8192, GQA with $n_{kv}$=16 KV heads, FP4 ($b$=0.5 B), $L/PP$=15 layers per stage, $TP\cdot SP$=8. The KV read per added $B$ is ~39 MB (measured above). That is the *entire* per-$B$ memory footprint the hardware has to stream every single decode step, for every added sequence.

Meanwhile, the per-$B$ compute is dominated by FFN + QKV + O projections + attention matmul against the KV. Per token per device that is on the order of tens of GFLOPs. The *ratio* — FLOPs per byte — comes out near 290, which is the AI the machine sees at the boundary between a memory-bound and compute-bound regime.

GB200 (and generally modern Hopper/Blackwell-class accelerators) is a very high-AI machine: 1125 FLOPs/byte at FP4. A workload needs to push $\text{AI}_B$ above that line to see FLOPS as a bottleneck. At $S$=8192 with GQA this model doesn't — the machine has more FLOPs per byte than the workload needs per byte.

## 6. Why more FLOPS makes it worse, not better

Scaling $R_{gpu}$ up shifts the machine balance point:

$$\text{AI}_{machine}(4\times) \;=\; \frac{36000}{8} \;=\; 4500 \;\text{FLOPs/byte}$$

$\text{AI}_B$ is unchanged at ~290. The gap between workload AI and machine AI **widens** from 1125/290 ≈ 3.9× to 4500/290 ≈ 15.5×. Extra FLOPS moves the balance point further away from where the workload operates, so a larger fraction of the FLOPs budget goes unused.

This is why the 4× FLOPS curve and the FLOPS → ∞ curve both overlap the 1× baseline exactly — there is no slack for compute to absorb. The `max(t_compute, t_mem)` roofline pins to $t_{mem}$ at every $B$, and extra FLOPS only shrinks an already-hidden term.

## 7. When would FLOPS matter

Flip the denominator sign by reducing $T_{kv}$'s slope relative to $F_{token}$'s slope. Practical levers:

**Shorter context ($S$).** $T_{kv}$ scales linearly with $S$; halving $S$ halves the memory slope in $B$ while leaving compute slope roughly unchanged. At $S$=1024 this workload's $\text{AI}_B$ rises ~8× to ~2300 FLOPs/byte, comfortably above 1125 — extra FLOPS would then lift the compute-bound ceiling the way textbook roofline predicts.

**KV compression (MLA, grouped/latent KV).** MLA drops $H_{kv}$ by a factor typically 4×–16× vs GQA. That scales $T_{kv}$ down directly and raises $\text{AI}_B$ into the compute-bound regime at the same $S$.

**More aggressive $TP$ on the KV path.** Sharding KV heads further (when $n_{kv}$ allows) reduces $T_{kv}$ per device. Bounded by $TP \le n_{kv}$ for KV-pathsharding.

**Lower $B_{eff,mem}$.** If you cut HBM bandwidth (not something you'd do on purpose) or introduce effective bandwidth loss from contention, $\text{AI}_{machine}$ drops and the workload can become compute-bound at a lower FLOPS threshold. This is the same mechanism the `pareto_vs_mem` sweep exposes from the other direction.

**Higher precision.** Going FP4 → FP16 quadruples $b$, which quadruples both $T_\theta$ and $T_{kv}$ — so memory-bound regions get *more* memory-bound. The flip side is $F_{token}$ also rises (fewer ops on higher-bitwidth multipliers per tensor core cycle is hardware-dependent), so the net effect on $\text{AI}_B$ depends on the specific kernel throughput curve.

## 8. Why training and prefill always want more FLOPS

Decode's memory-boundedness is a property of processing **one new token per sequence per step** against a *fully materialized* KV cache. Prefill and training are structurally different workloads — both process **many new tokens per step**, which changes every term in the roofline.

### 8.1 Prefill

In prefill, a sequence of $S$ new tokens is processed together. Per layer per device:

**Attention compute** scales with $S^2$:
$$F_{attn}^{prefill} \;=\; 4 \cdot S^2 \cdot H \; / \; (TP \cdot SP)$$

Each of the $S$ new queries reads all $S$ keys. Softmax, plus attention-output, together give the $4 S^2 H$ factor.

**KV read** scales with $S$ (same as decode — the K, V tensors are $S \times H_{kv} \times b$ per layer):
$$T_{kv}^{prefill} \;=\; 2 \cdot S \cdot H_{kv} \cdot b \cdot (L/PP) \; / \; (TP \cdot SP)$$

The arithmetic intensity of the *attention block alone* therefore grows **linearly with $S$**:

$$\text{AI}_{attn}^{prefill} \;\propto\; \frac{S^2 \cdot H}{S \cdot H_{kv} \cdot b} \;=\; \frac{S \cdot H}{H_{kv} \cdot b}$$

At $S$=8192, $H$=20480, $H_{kv} \cdot b = 16 \cdot 128 \cdot 0.5 = 1024$ B-ish, that's roughly $S \cdot 20 \approx$ **160k FLOPs/byte** — two orders of magnitude above any near-term hardware balance point. Prefill attention at long context lives far in the compute-bound regime.

**Weight reads amortize even better.** In decode, $T_\theta$ is paid once per step for a single new token (batched across $B$ sequences). In prefill, $T_\theta$ is paid once per step for *$S$ new tokens per sequence* — amortized across $B \cdot S$ tokens instead of $B$. At $S$=8192 that is an additional 8192× amortization of the weight-read cost, so the fixed $T_\theta$ term is negligible compared to compute.

Net effect: every term in $\text{AI}^{prefill}$ is far above the machine balance point. More FLOPS lifts prefill throughput directly, and peak FLOPS is the **primary** knob for TTFT.

### 8.2 Training

Training is prefill with a backward pass:

- **Forward:** identical compute pattern to prefill.
- **Backward:** ~2× the forward FLOPs (activation and weight gradients), with gradient-traffic ~ forward-activation-traffic.
- **Weight updates:** $2 \cdot P / B_{eff,mem}$ extra traffic per step for read-modify-write; negligible when amortized across millions of tokens.
- **Optimizer state:** Adam keeps 2 moment tensors in HBM, unchanged per token.

Per-token FLOPs in training ≈ 3× the forward FLOPs. Per-token memory traffic is essentially the same weight + activation pattern. So $\text{AI}^{training}$ is dominated by the same $S^2 \cdot H$ attention compute that drives prefill — with a 3× larger numerator. Training is even more FLOPS-hungry than prefill.

This is why GPU datasheet FLOPS (matmul throughput) is the headline number for training silicon and for prefill throughput — both workloads live so far above the balance point that delivered FLOPS is essentially the binding constraint. Decode is the exceptional workload where that intuition fails.

### 8.3 Unifying the three

| Workload | Tokens processed per step | $T_\theta$ amortization | Dominant compute | Dominant memory | Bottleneck |
|---|---|---|---|---|---|
| Decode | $B$ (one per sequence) | $\div B$ | $B \cdot F_{token}$ (linear in $B$) | $T_\theta + B \cdot T_{kv}$ | HBM BW / KV traffic |
| Prefill | $B \cdot S$ | $\div (B \cdot S)$ | $B \cdot S^2 \cdot H$ (quadratic in $S$) | $B \cdot S \cdot H_{kv}$ (+ weights $\div S$) | FLOPS |
| Training (fwd+bwd) | $B \cdot S$ | $\div (B \cdot S)$ | $\sim 3\times$ prefill | similar to prefill | FLOPS |

The quadratic $S^2$ in compute is the key asymmetry. Decode processes 1 query against $S$ keys → linear in $S$. Prefill processes $S$ queries against $S$ keys → quadratic. On any hardware with a high FLOPS/BW ratio, that quadratic is exactly what turns attention from memory-bound to compute-bound.

## 9. Summary

| Condition | Slope relationship | Behavior as $B \to \infty$ |
|---|---|---|
| $\text{AI}_B > \text{AI}_{machine}$ | $t_{compute}$ slope > $t_{mem}$ slope | Finite $B^*$; compute-bound at large $B$ |
| $\text{AI}_B = \text{AI}_{machine}$ | slopes equal | $B^* = \infty$; boundary case |
| $\text{AI}_B < \text{AI}_{machine}$ | $t_{compute}$ slope < $t_{mem}$ slope | No $B^*$; **diverging** memory-bound asymptote |

**Why FLOPS doesn't help for decode at long $S$ here:** GPT-1.8T MoE at $S$=8192 with GQA on GB200 has $\text{AI}_B \approx 290$ FLOPs/byte, well below GB200's balance of 1125. The workload lives in row 3 of the table above. $t_{compute}$ slope is smaller than $t_{mem}$ slope, so extra $B$ never closes the gap — and extra FLOPS only moves the balance point further out of reach.

**Why it does help for decode at short $S$:** at $S$=1024 the $T_{kv}$ slope drops ~8×, pushing $\text{AI}_B$ above the 1125 threshold. The `notebooks/pareto_vs_flops.ipynb` §9 companion sweep confirms: at $S$=1024, $B$=8192, the step flips compute-bound on 1× FLOPS, and 4× FLOPS lifts high-$B$ throughput/GPU from ~7850 → ~9550.

**Why it always helps for prefill and training:** attention compute grows as $S^2$ per sequence while KV traffic grows as $S$, so $\text{AI}^{prefill} \propto S$. At realistic $S$ that number is orders of magnitude above any hardware's FLOPS/BW ratio. Weight reads additionally amortize over $B \cdot S$ tokens instead of $B$, making $T_\theta$ negligible. Both workloads are squarely compute-bound — buying FLOPS buys performance directly.

**Practical rule of thumb.** Before spending silicon on FLOPS for a *decode* workload, compare the per-$B$ arithmetic intensity $F_{token}/T_{kv}$ against the target hardware's FLOPS-to-BW ratio. If $F_{token}/T_{kv}$ is below that ratio, FLOPS will not move the Pareto frontier at any $B$ — invest in HBM bandwidth, KV sharding, or KV compression instead. For prefill/training, this check is a non-issue: those workloads are always far above the balance point.

## 10. Related reading

- `documentation/explaining/frontier_convergence_at_high_b.md` — the $B^*$ formula and the partition-invariant FLOPS ceiling (relevant when the denominator *is* positive).
- `documentation/modeling/decode.md` — derivation of $t_{compute}$, $t_{mem}$, $T_\theta$, $T_{kv}$.
- `notebooks/pareto_vs_flops.ipynb` — the sweep data this note explains.
- `notebooks/pareto_vs_mem.ipynb` — the mirror sweep where HBM BW *does* move the frontier on both corners.
