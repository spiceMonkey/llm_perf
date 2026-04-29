# When Does More Scale-up I/O Bandwidth Help?

**Author:** Yue Lu  
**Date:** April 2026  

A closed-form derivation of the bandwidth and α thresholds above which extra scale-up input/output (I/O) stops moving the per-step decode time.

This explainer answers a recurring practical question: given a fixed model and partition, at what point does buying more NVLink (or InfiniBand / PCIe) bandwidth stop helping decode time-per-output-token (TPOT)? The same derivation gives the dual α threshold for latency-side improvements.

The setup assumes the workload is **memory-bound** at the chosen batch size — *t*<sub>local</sub> = *t*<sub>mem</sub> — which is the typical decode regime for long-context inference. It also uses the per-stage hardware step time and overlap factor *ρ* from `decode.md §6.2`.

---

## 1. The per-stage hardware time

From `decode.md §6.2`, the per-stage hardware step time composes the GPU-only roofline with the unhidden remainder of communication:

$$
t_\mathrm{stage,hw}(B) \;=\; t_\mathrm{local}(B) \;+\; \max\!\bigl(0,\; t_\mathrm{comm}(B) - \rho \cdot t_\mathrm{local}(B)\bigr)
$$

where *ρ* ∈ [0, 1] is the fraction of *t*<sub>local</sub> available to overlap with *t*<sub>comm</sub>. Two regimes:

- *t*<sub>comm</sub> ≤ *ρ* · *t*<sub>local</sub> → comm fully hidden, *t*<sub>stage,hw</sub> = *t*<sub>local</sub>.
- *t*<sub>comm</sub> > *ρ* · *t*<sub>local</sub> → unhidden remainder is paid: *t*<sub>stage,hw</sub> = (1 − *ρ*) · *t*<sub>local</sub> + *t*<sub>comm</sub>.

The first regime is the case where extra bandwidth or smaller α buys you nothing. The second is where bandwidth and α improvements help proportionally. Our goal is to find the bandwidth (or α) value that puts you exactly on the boundary.

---

## 2. Decomposing *t*<sub>comm</sub> into α-side and β-side

Per `decode.md §5.5`, the per-stage aggregated communication sums one term per parallelism axis. Each per-call cost follows the Hockney α–β model: *t*<sub>X</sub> = *n*<sub>α</sub> · α + *n*<sub>β</sub> · *M* / *BW*. Collecting all the α-side counts and β-side payloads yields:

$$
t_\mathrm{comm}(B) \;=\; \underbrace{a_p \cdot \alpha}_{\text{α-side, count-driven}} \;+\; \underbrace{\frac{b_p(B)}{BW}}_{\text{β-side, payload-driven}}
$$

The two **partition coefficients** are fixed once you pick `(L, PP, TP, EP, SP)` and the shipped algorithms (ring all-reduce, ring all-gather, pairwise all-to-all, point-to-point hop):

$$
a_p \;=\; \frac{L}{PP}\,n_\mathrm{TP}\,2(G_\mathrm{TP}{-}1)
\;+\; \frac{L_\mathrm{moe}}{PP}\,n_\mathrm{EP}\,2(G_\mathrm{EP}{-}1)
\;+\; \frac{L}{PP}\,n_\mathrm{SP}\,(G_\mathrm{SP}{-}1)
\;+\; 1
$$

$$
b_p(B) \;=\; \frac{L}{PP}\,n_\mathrm{TP}\frac{2(G_\mathrm{TP}{-}1)}{G_\mathrm{TP}}\,B H b
\;+\; \frac{L_\mathrm{moe}}{PP}\,n_\mathrm{EP}\frac{2(G_\mathrm{EP}{-}1)}{G_\mathrm{EP}}\,B k H b
\;+\; \frac{L}{PP}\,n_\mathrm{SP}\frac{G_\mathrm{SP}{-}1}{G_\mathrm{SP}}\,B S\,\frac{2H_\mathrm{kv}}{TP}\,b
\;+\; B\,\frac{H}{TP}\,b
$$

The four terms in each equation are: tensor-parallel (TP) all-reduce, expert-parallel (EP) all-to-all (Dispatch + Combine), sequence-parallel (SP) all-gather, and pipeline-parallel (PP) point-to-point hop. *G*<sub>X</sub> is the group size for each axis. For other algorithm choices (double binary tree all-reduce, in-network collective via NVLink SHARP / NVLS, hierarchical reduce-scatter → sub-AR → all-gather), substitute the corresponding (*n*<sub>α</sub>, *n*<sub>β</sub>) coefficients from `documentation/modeling/collectives/00_summary.md §4`.

Important properties:

- *a*<sub>p</sub> depends only on the partition shape — independent of *B*, *BW*, α, or the model's *H* / *I*<sub>moe</sub>.
- *b*<sub>p</sub>(*B*) is linear in *B* — every collective payload scales with the batch.
- Both coefficients vanish on parallelism axes that are 1 (no collective fires).

---

## 3. The memory-bound assumption

Long-context decode is essentially always memory-bandwidth-pinned, with *t*<sub>local</sub> set by high-bandwidth memory (HBM):

$$
t_\mathrm{local}(B) \;=\; t_\mathrm{mem}(B) \;=\; \frac{T_\theta + B \cdot T_\mathrm{KV}}{BW_\mathrm{mem}}
$$

where *T*<sub>θ</sub> is the per-device weight bytes (`decode.md §2.1`) and *T*<sub>KV</sub> is the per-request key-value (KV) cache bytes (`decode.md §2.3`). Expanding each term lets us see how the partition shape moves *t*<sub>mem</sub>.

### Per-device weight bytes

For a model with *L*<sub>dense</sub> dense layers and *L*<sub>moe</sub> MoE layers (*L*<sub>dense</sub> + *L*<sub>moe</sub> = *L*), the per-device weight read on one PP stage is:

$$
T_\theta \;=\; \frac{L_\mathrm{dense}}{PP}\!\left(\frac{2H^2 + 2H\,H_\mathrm{kv}}{TP} + \frac{3H\,I_\mathrm{dense}}{TP}\right) b
\;+\; \frac{L_\mathrm{moe}}{PP}\!\left(\frac{2H^2 + 2H\,H_\mathrm{kv}}{TP} + \frac{3H\,I_\mathrm{moe}\,N_\mathrm{exp}}{TP \cdot EP}\right) b
$$

The four pieces are: dense attention QKVO (2*H*² + 2*HH*<sub>kv</sub>), dense FFN gate+up+down (3*HI*<sub>dense</sub>), MoE attention (same shape as dense), and MoE FFN (3*HI*<sub>moe</sub> per expert × *N*<sub>exp</sub> total experts). Each term is sharded as follows:

- **PP** divides the total layer count into per-stage shares — every term gets a 1/*PP* factor.
- **TP** shards every per-layer matrix along the head or hidden dimension — every term gets a 1/*TP* factor.
- **EP** shards only the MoE FFN slice across expert groups — only the (3*HI*<sub>moe</sub>·*N*<sub>exp</sub>) term gets the 1/*EP* factor.

For a pure-MoE model (*L*<sub>dense</sub> = 0, *L*<sub>moe</sub> = *L*) this collapses to:

$$
T_\theta \;=\; \frac{L}{PP}\!\left(\frac{2H^2 + 2H\,H_\mathrm{kv}}{TP} + \frac{3H\,I_\mathrm{moe}\,N_\mathrm{exp}}{TP \cdot EP}\right) b
$$

### Per-request KV cache bytes

Each sequence stores its own *S*-token K and V history per layer per PP stage; SP shards the sequence dimension across SP ranks:

$$
T_\mathrm{KV} \;=\; \frac{L}{PP} \cdot \frac{2 S H_\mathrm{kv}}{TP \cdot SP} \cdot b
$$

The 2 covers K and V; *H*<sub>kv</sub> = (*H* / *n*<sub>q</sub>) · *n*<sub>kv</sub> is the per-token key/value width (smaller than *H* for grouped-query attention).

### Substituted memory time

Combining the two:

$$
t_\mathrm{mem}(B) \;=\; \frac{L}{PP \cdot BW_\mathrm{mem}} \cdot b \cdot \left[\frac{2H^2 + 2H\,H_\mathrm{kv}}{TP} + \frac{3H\,I_\mathrm{moe}\,N_\mathrm{exp}}{TP \cdot EP} + B \cdot \frac{2 S H_\mathrm{kv}}{TP \cdot SP}\right]
$$

Three knobs to read:

- **Doubling PP** halves *t*<sub>mem</sub> for free (no comm cost in this expression — comm appears separately as *t*<sub>comm</sub>).
- **Doubling TP** halves *t*<sub>mem</sub> too, but inflates *a*<sub>p</sub>, *b*<sub>p</sub> in the comm formula because TP groups grow.
- **Doubling EP** halves only the MoE FFN slice; the attention and KV terms are untouched. This is why EP only beats PP on workloads where the MoE FFN slice dominates total weight (see `pareto_basic.ipynb` discussion).

The break-even derivation in §4 uses the abstract *T*<sub>θ</sub>, *T*<sub>KV</sub>, *t*<sub>mem</sub> symbols; substitute the expressions above to get a fully-specified *BW**\** in model + partition coordinates.

Plugging into the unhidden-comm condition:

$$
t_\mathrm{comm}(B) \;>\; \rho \cdot t_\mathrm{mem}(B)
\;\;\;\Longleftrightarrow\;\;\;
a_p\,\alpha \;+\; \frac{b_p(B)}{BW} \;>\; \rho \cdot \frac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}}
$$

This is the condition for "bandwidth or α changes affect *t*<sub>stage,hw</sub>". When it holds, communication is paid in part; when it fails, communication is fully hidden and bandwidth and α are inert.

---

## 4. Solving for the bandwidth break-even

Treating *BW* as the free variable and solving the boundary case (*t*<sub>comm</sub> = *ρ* · *t*<sub>mem</sub>):

$$
a_p\,\alpha \;+\; \frac{b_p(B)}{BW} \;=\; \rho \cdot \frac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}}
$$

Move the α term across:

$$
\frac{b_p(B)}{BW} \;=\; \rho \cdot \frac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}} \;-\; a_p\,\alpha
$$

Invert:

$$
BW^* (B) \;=\;
\frac{b_p(B)}
     {\;\;\rho \cdot \dfrac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}} \;-\; a_p\,\alpha\;\;}
\qquad \text{(valid when }\rho\,t_\mathrm{mem} > a_p\,\alpha\text{)}
$$

Reading the threshold:

- *BW* ≥ *BW**\** → comm fits inside the *ρ* · *t*<sub>mem</sub> overlap budget → **extra bandwidth does nothing**.
- *BW* < *BW**\** → comm overflows the budget; the unhidden remainder shrinks with *BW* → **extra bandwidth helps proportionally**.

The denominator becoming non-positive — *ρ* · *t*<sub>mem</sub> ≤ *a*<sub>p</sub> · α — means α alone exhausts the overlap budget. In that regime the boundary is unreachable: communication is **always** unhidden no matter how big *BW* gets, and any *BW* improvement helps. This is what happens at very small *t*<sub>mem</sub> (small *B*, no weight-read amortization) or very high α (legacy fabrics, software collectives over slow links).

---

## 5. Solving for the α break-even

By symmetry, fix *BW* and treat α as the free variable. Setting *t*<sub>comm</sub> = *ρ* · *t*<sub>mem</sub> and solving for α:

$$
\alpha^* (B) \;=\;
\frac{1}{a_p}\!\left[\rho \cdot \frac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}} \;-\; \frac{b_p(B)}{BW}\right]
$$

α ≤ α*\** → α-side comm fits in the overlap budget → **smaller α does nothing**.

α > α*\** → comm overflows the budget; tightening α helps.

The two thresholds (*BW**\**, α*\**) trace out the same boundary in the (*BW*, α) plane — the curve separating the *fully-hidden* regime from the *unhidden* regime.

---

## 6. Design-goal target: *t*<sub>comm</sub> as a bounded fraction of *t*<sub>mem</sub>

The §4 break-even ties BW to the overlap factor ρ — the question "what BW puts comm exactly at the edge of the overlap budget?". A more useful design framing flips the question: pick a target fraction *f* (e.g., *f* = 0.1 = 10%) and ask "what BW keeps total comm at most *f* · *t*<sub>mem</sub>, regardless of overlap?".

Setting *t*<sub>comm</sub> ≤ *f* · *t*<sub>mem</sub> and solving for *BW*:

$$
a_p\,\alpha + \frac{b_p(B)}{BW} \;\le\; f \cdot \frac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}}
$$

$$
BW_\mathrm{target}(f, B) \;=\;
\frac{b_p(B)}
     {\;\;f \cdot \dfrac{T_\theta + B\,T_\mathrm{KV}}{BW_\mathrm{mem}} \;-\; a_p\,\alpha\;\;}
\qquad \text{(valid when } f\,t_\mathrm{mem} > a_p\,\alpha\text{)}
$$

Structurally identical to *BW**\** in §4 with *f* replacing ρ — but interpreted as a **design knob**, not a physical overlap rate.

### Reading *f*

*f* sets the maximum allowed comm cost as a fraction of the memory roofline:

- **f = 0.1**: aggressive, "comm should be at most 10% of *t*<sub>mem</sub>". Engineering choice that keeps the workload robustly memory-bound — even at ρ = 0 (no overlap), the visible TPOT inflation from comm is capped at 10%.
- **f = 0.25**: relaxed, "comm can be up to 25%". Acceptable if you're confident in ρ ≥ 0.5 production overlap (then visible comm is at most max(0, 0.25 − 0.5)·*t*<sub>mem</sub> = 0).
- **f = ρ**: recovers §4 — comm sits exactly at the edge of what overlap can hide.
- **f > ρ**: deliberately accepts some unhidden comm, bounded by (f − ρ)·*t*<sub>mem</sub>.

The design rule: pick *f* based on how memory-bound you want the workload to remain at worst case. *f* = 0.1 is a good default — it ensures comm overhead is below the typical run-to-run variation in HBM bandwidth itself, so comm becomes invisible to operational monitoring.

### Empirical sweep

Three models (GPT-1.8T MoE FP4 — attention-heavy 16 experts × *k* = 2; DeepSeek-R1 FP4 — MoE-FFN-heavy 256 experts × *k* = 9; Llama-3.1-70B FP4 — dense GQA reference) × six partition shapes (all capped at **PP ≤ 8** — the typical PP-capped operational regime where bubble cost / layer-count drives the latency budget; the unbounded-PP winner shapes from `pareto_basic.ipynb` would shift the *BW*<sub>target</sub> curves further down) × log-spaced *B* ∈ [1, 2048] × two design targets (*f* = 0.10 solid, *f* = 0.25 dashed). Defaults: α = 0.2 μs, *BW*<sub>mem</sub> = 8 TB/s, *S* = 8192, ring algorithms (no INC). Generated by `notebooks/scale_up_io_bw_target.ipynb`.

#### *BW*<sub>target</sub> vs batch size

![BW_target vs B, three models](../../assets/scale_up_io_bw_target_vs_B.png)

Reading the plot: each curve is one (model, partition) pair; horizontal dotted lines mark common scale-up fabrics (NVLink5 = 900 GB/s, NVLink4 = 450 GB/s, PCIe5 x16 = 64 GB/s). A curve sitting **below** a fabric line means that fabric meets the design target on this configuration → extra BW is wasted. A curve **above** the line means BW investment helps; the §7 sensitivity formula gives the per-doubling savings. Curves missing entirely at small *B* are "unreachable" — *a*<sub>p</sub> · α exceeds *f* · *t*<sub>mem</sub> and no BW value hits the target.

#### β-side comm share at current BW

![t_β share at current BW, three models](../../assets/scale_up_io_bw_target_t_beta_share.png)

The marginal-sensitivity view from §7: at the current 900 GB/s NVLink5 reference (ρ = 0, no overlap), what fraction of *t*<sub>step,user</sub> is β-side communication? Doubling BW shaves up to **half** of the displayed percentage off TPOT. The 10% goal line marks the design target — anything below is comfortably memory-bound; anything above is a candidate for BW investment.

Patterns the two plots expose together:

- **PP=8 small-TP shapes** (PP=8 TP=2 / PP=8 TP=8 EP=1 SP=1): *BW*<sub>target</sub> stays well below NVLink5's 900 GB/s line for all three models; β-share is in the 0.1–5% range. **BW improvements move TPOT by less than 2.5%** even at the aggressive *f* = 0.10 target. (The unbounded-PP winners — e.g. PP=32 TP=2 — sit even lower; the PP=8 cap chosen here is the harder regime.)
- **PP=4 TP=16** (max-TP single-tier): *a*<sub>p</sub> ~1800 → *t*<sub>α</sub> ≈ 360 μs alone exceeds 10% of *t*<sub>mem</sub> at small *B* → *f* = 0.10 is unreachable; even *f* = 0.25 needs hundreds of GB/s. Wide-TP is BW-friendly only after INC drops *n*<sub>α</sub>.
- **EP=4 / EP=16 MoE shapes**: *t*<sub>mem</sub> shrinks (sharded FFN), so the *f* · *t*<sub>mem</sub> budget shrinks proportionally — high-EP small-*B* lives in the unreachable zone for *f* = 0.10 even at modest *a*<sub>p</sub>. The dashed *f* = 0.25 curves cover more of these cases.
- **SP=4 ring-attention**: per-layer KV-shard payload (`B · S · 2H_kv / TP`) drives *b*<sub>p</sub> into the GB-per-step range → *BW*<sub>target</sub> reaches **TB/s territory**. The only configuration in the sweep where current scale-up BW is genuinely under-provisioned, on every model.

To re-run with INC modeling, edit `a_p_coef` in the notebook to substitute *n*<sub>α</sub> = 2*k* (k = number of switching tiers crossed) for the TP and SP axes — see "If I have INC, can I assume tiny *a*<sub>p</sub>?" above for the per-axis substitution rules. Knobs (α, *BW*<sub>mem</sub>, *S*, *f* targets, models, partitions) are at the top of the notebook.

---

## 7. Per-partition vs frontier: when does *BW*<sub>target</sub> move the Pareto curve?

The §4 / §5 / §6 thresholds are all **per-partition**: given a chosen partition shape X, derive what bandwidth (or α) makes X's communication cost trivial relative to its memory roofline. The Pareto frontier (`pareto_vs_io.ipynb`) is **cross-partition**: an upper-right envelope over all valid `(partition, B)` candidates, with the optimizer free to pick any shape. Procuring *BW*<sub>target</sub>(X) improves X's TPOT — but whether the **frontier** improves depends on whether X is on (or joins) the frontier under the new bandwidth.

Three operational regimes:

1. **X is already a frontier winner AND bandwidth-bottlenecked.** Procuring more BW directly moves the frontier corner outward. Verify by checking the `pareto_vs_io` winners table at current BW: if X appears, and *t*<sub>β</sub>(X) is well above your *f* target (read it from the §5 plot in `notebooks/scale_up_io_bw_target.ipynb`), this is the best case.
2. **X is dominated at current BW, but procuring *BW*<sub>target</sub> lifts it onto the frontier.** Classic example: wide-TP shapes (PP=4 TP=16) are dominated at finite BW because the all-reduce cost outweighs the per-device weight savings, but become optimal at the ideal-I/O reference (39 of 40 frontier corners in `pareto_vs_io`'s ideal panel). Procuring BW on a wide-TP partition shifts the frontier to use wider-TP winners — the frontier improves, but **with a different partition shape** than the operator might have planned.
3. **X is dominated AND remains dominated even with more BW.** Common when X is bandwidth-bottlenecked but also worse on other axes (compute, memory). Procuring BW on X then doesn't help the frontier — the optimizer keeps picking some other partition for that corner. Wasted spend.

The closed-form *BW*<sub>target</sub> answers a per-partition question (TPOT delta from BW improvement); the frontier sweep answers the cross-partition question (which partition wins after BW improvement). Both are useful but distinct.

**Frontier upper bound.** The `pareto_vs_io` notebook plots an "ideal-I/O reference" (BW → ∞, α → 0, ρ = 1) — that's the maximum frontier headroom any BW investment can realize. The gap between realistic-BW frontiers and the ideal reference is the absolute ceiling on what BW + α improvements can buy. If your chosen X is already close to that reference at current BW, BW investment helps; if X is far below for compute or memory reasons, BW investment doesn't close the gap and you should look at HBM, FLOPS, or partition shape instead.

**Operational rule.** Compute *BW*<sub>target</sub>(X) per §6. If current BW < *BW*<sub>target</sub>, X gets a predictable TPOT improvement at the targeted BW. **To check whether the frontier itself moves**, re-run a partition-optimal sweep (`pareto_vs_io`-style) at the new BW: if X appears among the winners at any frontier corner, the frontier moves through X; if not, the frontier may still move (other partitions take over) but X-specific BW investment was only justified as part of the broader fabric upgrade. The companion notebook at `notebooks/scale_up_io_bw_target.ipynb` includes a "frontier-shift check" cell that automates this comparison for a single chosen X.

#### Worked frontier-shift example

![Frontier shift after procuring BW_target](../../assets/scale_up_io_frontier_shift.png)

Choosing X = DeepSeek-R1 FP4 / `PP=8 TP=4 EP=4 SP=1` / *B*<sub>ref</sub> = 128 / *f* = 0.10 → *BW*<sub>target</sub>(X) ≈ **21 TB/s** (~24× the 900 GB/s current NVLink5 reference). Re-sweeping the partition pool at both BWs:

- **Current-BW frontier**: X appears at **2** corners (mid-frontier).
- **New-BW (21 TB/s) frontier**: X appears at **0** corners.

This is a clean **regime 3 / regime 2 hybrid**: at current BW, X was a borderline frontier winner because comm wasn't yet the dominant cost. Once BW grows 24×, comm becomes nearly free for *every* partition — and the optimizer reallocates the win to wider-TP / shallower-PP shapes that previously paid an unacceptable comm penalty. X's TPOT did improve (the *BW*<sub>target</sub> formula's per-partition prediction is correct), but X stopped being the *best* shape at the new BW.

The takeaway: per-partition BW improvements are real, but the cross-partition optimizer can re-route the win to a different shape under the new fabric. **Always check the frontier sweep before justifying BW investment by appeal to a single chosen partition's *BW*<sub>target</sub>**.

---

## 8. Marginal sensitivity in the unhidden regime

When the partition is in the unhidden regime, *t*<sub>stage,hw</sub> = (1 − *ρ*) · *t*<sub>mem</sub> + *t*<sub>comm</sub>. Taking partial derivatives:

$$
\frac{\partial \,t_\mathrm{stage,hw}}{\partial (1/BW)} \;=\; b_p(B), \qquad
\frac{\partial \,t_\mathrm{stage,hw}}{\partial \alpha} \;=\; a_p
$$

So **doubling *BW* shaves *b*<sub>p</sub>(*B*) / (2 · *BW*) = *t*<sub>β</sub> / 2 off *t*<sub>stage,hw</sub>**. As a fractional improvement of the user-observed step time *t*<sub>step,user</sub>:

$$
\frac{\Delta t_\mathrm{step,user}}{t_\mathrm{step,user}} \;\approx\; \frac{1}{2}\cdot\frac{t_\beta(B)}{t_\mathrm{step,user}}
\qquad \text{(BW doubling, in the unhidden regime)}
$$

The same form gives the α improvement: halving α saves *a*<sub>p</sub> · α / 2 = *t*<sub>α</sub> / 2.

---

## 9. The role of *ρ*

The break-even threshold scales with *ρ* in the denominator. Two limits:

**ρ = 0 (no overlap)**: *BW**\** = − *b*<sub>p</sub>(*B*) / (*a*<sub>p</sub> · α), which is negative → **comm is always unhidden** and bandwidth changes always matter (proportional to *t*<sub>β</sub> / *t*<sub>step</sub>).

**ρ = 1 (perfect overlap)**: *BW**\** = *b*<sub>p</sub>(*B*) / (*t*<sub>mem</sub> − *a*<sub>p</sub>·α), the largest possible threshold — communication gets the full *t*<sub>mem</sub> budget to hide under, so *BW* has to drop very low before it becomes the bottleneck.

The **ρ knob has more leverage than the BW knob** for typical decode workloads. Going from ρ = 0 to ρ = 0.5 can collapse a 30-μs unhidden comm to 0 at no hardware cost, while doubling bandwidth only halves the β-side payload. This is why production stacks invest heavily in CUDA-stream pipelining (raising effective *ρ*) before reaching for more bandwidth.

---

## 10. The decision rule, distilled

Given a (model, system, partition, *B*, *ρ*) operating point and a design target *f* (e.g., 0.1):

1. Compute *a*<sub>p</sub>, *b*<sub>p</sub>(*B*) from §2 — depends only on partition shape and ladder.
2. Compute *t*<sub>mem</sub>(*B*) = (*T*<sub>θ</sub> + *B* · *T*<sub>KV</sub>) / *BW*<sub>mem</sub>.
3. Compute *t*<sub>α</sub> = *a*<sub>p</sub> · α and *t*<sub>β</sub>(*B*) = *b*<sub>p</sub>(*B*) / *BW*.
4. Compute *BW*<sub>target</sub>(*f*, *B*) = *b*<sub>p</sub>(*B*) / (*f* · *t*<sub>mem</sub> − *a*<sub>p</sub> · α) (§6) and check current *BW* against it:
   - **Current *BW* ≥ *BW*<sub>target</sub>** → comm is below the *f* · *t*<sub>mem</sub> design cap; extra bandwidth is wasted. Use the slot elsewhere (HBM, FLOPS, more GPUs for data parallel replicas) or tighten α / *ρ*.
   - **Current *BW* < *BW*<sub>target</sub>** → comm exceeds the design target; bandwidth improvement saves up to *t*<sub>β</sub>/2 per doubling (§7). Worth the spend until you reach *BW*<sub>target</sub>.
   - **Denominator non-positive** → α alone exceeds *f* · *t*<sub>mem</sub>; bandwidth can't reach the target regardless. Tighten α or pick a larger *f*.

The same logic with the α<sup>★</sup> formula in §5 covers the dual question of when latency-side improvements matter.

---

## See also

- `decode.md §6.2` — the *t*<sub>stage,hw</sub> composition and *ρ* definition.
- `decode.md §5.5` — the per-stage *t*<sub>comm</sub> formula that *a*<sub>p</sub>, *b*<sub>p</sub>(*B*) collect from.
- `documentation/modeling/collectives/00_summary.md §4` — the per-algorithm (*n*<sub>α</sub>, *n*<sub>β</sub>) values to substitute into *a*<sub>p</sub> and *b*<sub>p</sub> for non-ring algorithms.
- `pareto_vs_io.ipynb` — empirical Pareto sweeps that show frontier saturation under exactly this break-even logic (when the optimizer routes around comm-heavy partitions, you sit far above *BW**\** and bandwidth improvements are invisible).
- `pareto_vs_scale_up_tier.ipynb` — the partition-FORCED lens, where *BW**\** is below current *BW* and the frontier *does* move with *BW*.
