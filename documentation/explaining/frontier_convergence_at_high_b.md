# Why the Pareto Frontier Converges at High Throughput

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, Pareto frontier, batch size, roofline, compute-bound, memory-bound, $B^*$ crossover, throughput ceiling, FLOPS capacity

---

## 1. Observation

Across every `pareto_*` sweep in this repo — IO, HBM, overhead, cluster size — the frontier curves look remarkably similar in the high-throughput corner. Partitions that differ substantially in the high-interactivity corner (low $B$, small $\text{TPOT}$) bunch together as you slide toward large $B$. At the rightmost edge of the throughput axis, competing partitions are nearly indistinguishable in throughput/GPU even though they differ in $PP$, $TP$, and $EP$.

This is not a plotting artifact. It reflects a structural property of the roofline model: **at high $B$, every partition is compute-bound, and total FLOPS capacity is the same invariant across partition choices**.

## 2. The roofline crossover

For a single decode step, the per-device latency follows a roofline:

$$t_{token} = \max(t_{compute}, t_{mem}) + t_{comm}^{unhidden}$$

where (in the $B$-parameterized decode model):

$$t_{compute} = \frac{B \cdot F_{token}}{R_{gpu}}, \qquad t_{mem} = \frac{T_\theta + B \cdot T_{kv}}{B_{eff,mem}}$$

- $T_\theta$ is the fixed per-step weight read traffic (weights are loaded once per token regardless of $B$).
- $T_{kv}$ is per-sequence KV read traffic.
- $F_{token}$ is FLOPs per token per device.
- $R_{gpu}$ is peak per-device FLOPS.
- $B_{eff,mem}$ is effective HBM bandwidth per device.

At low $B$, $t_{mem}$ is dominated by $T_\theta / B_{eff,mem}$ — the fixed weight read — and $t_{compute}$ is negligible. The system is memory-bound, and throughput/GPU grows nearly linearly with $B$ (you're amortizing the weight read across more tokens for free).

The crossover happens at:

$$B^* = \frac{T_\theta \cdot R_{gpu}}{F_{token} \cdot B_{eff,mem} - T_{kv} \cdot R_{gpu}}$$

Beyond $B^*$, $t_{compute}$ overtakes $t_{mem}$ and the regime flips to compute-bound.

## 3. The throughput ceiling

Once compute-bound:

$$t_{token} \approx \frac{B \cdot F_{token}}{R_{gpu}}, \qquad \text{TPOT} = \frac{t_{token}}{B} \approx \frac{F_{token}}{R_{gpu}}$$

Both $\text{TPOT}$ and throughput/GPU become **asymptotically independent of $B$**:

$$\text{throughput/GPU} = \frac{B}{t_{token} \cdot N_{dev}} \longrightarrow \frac{R_{gpu}}{F_{token} \cdot N_{dev}}$$

Increasing $B$ past $B^*$ no longer buys you throughput per GPU — the GPU's FLOPS capacity is the bottleneck. This ceiling is a hardware property.

## 4. Why different partitions converge to the same ceiling

The critical observation is that **total model FLOPs per token is invariant under partition choice**. If the full model does $F_{total}$ FLOPs per token, then:

$$F_{token,device} \cdot N_{dev,replica} \approx F_{total}$$

Splitting the model across more devices (more $TP$, more $EP$) reduces $F_{token,device}$ proportionally — each device does less work, but there are more of them. The total FLOPS budget per token across a replica is the same.

So in the compute-bound regime, every partition sees the same ceiling:

$$\text{throughput/GPU}_{max} \approx \frac{\sum R_{gpu}}{F_{total} \cdot N_{dev,total}}$$

independent of how you sliced the model. Different partitions differ in the low-$B$ memory-bound regime (where they have different weight-read overheads, different KV sharding, different comm patterns) but **converge** at the compute-bound ceiling.

## 5. Why the frontier is not perfectly flat at high $B$

A few residual effects separate partitions even in the compute-bound regime:

**Unhidden TP/EP communication.** Activation sizes scale with $B$: all-reduce messages for a $TP$ layer carry $B \cdot H \cdot b$ bytes instead of $H \cdot b$. The bandwidth term of the collective grows linearly with $B$ and competes for HBM/fabric time. Partitions with heavy $TP$ or $EP$ get their effective $R_{gpu}$ eaten into, sitting slightly below the pure-FLOPS ceiling.

**KV reads at long context.** $B \cdot T_{kv}$ is a real memory-side cost that does not vanish. At long $S$ (large KV per sequence), the memory term has a linear-in-$B$ component, so the transition to fully compute-bound happens at a higher $B$, and a partition with better KV sharding (more $SP$, more $TP$ on KV) lands above one without.

**PP warmup.** Deeper $PP$ amortizes its bubble as $1/B$, so the bubble contribution fades at high $B$ but is never exactly zero. Shallow-$PP$ partitions retain a tiny advantage here.

**KV paging ceiling.** At very high $B$, the paged KV cache fills up $M_{HBM} - M_\theta - M_{act}$. Different partitions have different residual capacity for KV. A partition with small $M_\theta$ (aggressive $TP$ weight sharding) can support a larger $B$ before running out of paging space, extending its frontier further to the right.

The net picture: at high $B$ the frontier asymptotes toward a horizontal ceiling, with **winners separated by who pays the least unhidden comm**. That's why the high-throughput corner tends to be captured by **deep-$PP$ + moderate-$TP$** partitions — $PP$ uses cheap P2P sends that don't fight compute for bandwidth, and moderate $TP$ keeps weight sharding tight without paying a large all-reduce tax per token.

## 6. Summary

| Regime | What dominates | Throughput/GPU behavior | Partition spread |
|--------|---------------|--------------------------|------------------|
| $B \ll B^*$ (memory-bound) | weight reads $T_\theta$ | grows $\sim$linearly with $B$ | Large — sharding differences matter |
| $B \approx B^*$ (crossover) | compute $\approx$ memory | slope softens | Moderate |
| $B \gg B^*$ (compute-bound) | FLOPS capacity | asymptotes to ceiling | Small — mostly comm-overhead tax |

**Why the frontier converges at high $B$:** every partition hits the same $R_{gpu} / F_{total}$ ceiling because total FLOPs per token is partition-invariant. The remaining spread is the cost of communication that cannot be hidden behind compute.

**Practical implication:** if your workload lives at the high-throughput corner (batch serving, background inference), the choice of partition matters less than the hardware's FLOPS — and any gains come from minimizing unhidden comm rather than optimizing memory locality. If your workload lives at the high-interactivity corner (chat, agents), partition choice matters a lot because you're in the memory-bound regime where $T_\theta$ sharding dictates speed.
