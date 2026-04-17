# When Hierarchical Scale-Up Actually Bites

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
hierarchical scale-up, NVLink, NVL72, NVL576, tier radix, collective group size, TP, EP, MoE all-to-all, GQA, KV heads, fine-grained MoE, DeepSeek, rack boundary, decode Pareto

---

## 1. The puzzle from the last case study

`pareto_vs_scale_up_tier.ipynb` sweeps GPT-1.8T MoE FP4 decode on 576 × GB200-class GPUs under two system configurations: an imaginary flat NVL576 (one `nvlink5-flat` fabric, every pair of ranks intra-rack-class: 0.5 μs, 900 GB/s) and a realistic hierarchical NVL576 (two fabrics — `nvlink5` inside each of 8 NVL72 racks at 900 GB/s / 0.5 μs / 72 ports; `ib` stitching those racks at 400 GB/s / 2.5 μs / 8 ports — with every collective chained `["nvlink5", "ib"]`). The two Pareto frontiers overlap exactly — **all 10,156 evaluation points are bit-identical**. A forced TP=144 stress test does show the hierarchical cost (TP all-reduce jumps ~6×) but no valid partition in the sweep exercises that cliff.

This note explains *why* — and, more usefully, *which models and partitions would actually put the hierarchical fabric under stress*.

## 2. The mechanism — one number that controls everything

Each scale-up collective runs over a **group** of ranks specific to its role:

- **TP all-reduce:** group size = TP factor, one group per layer (two collectives: attn + FFN).
- **EP all-to-all:** group size = EP factor, one group per MoE layer (two passes: dispatch + combine).
- **SP all-gather:** group size = SP factor, one group per layer.
- **PP hop:** group size = 2 (point-to-point between adjacent stages).

In the hierarchical configuration, the scale-up fabric (`nvlink5`) has a **radix** $P_0 = 72$ (ports in one NVL72 rack); beyond that, a collective escalates into the scale-out fabric (`ib`, radix 8). A collective of group size $G$ crosses:

- **Only nvlink** if $G \le 72$ — pays $(\alpha_{\text{nvlink}}, BW_{\text{nvlink}}) = (0.5\ \mu s, 900\ \text{GB/s})$.
- **nvlink + ib** if $G > 72$ — pays $(\alpha_{\text{nvlink}} + \alpha_{\text{ib}}, \min(BW_{\text{nvlink}}, BW_{\text{ib}})) = (3.0\ \mu s, 400\ \text{GB/s})$.

The 72-rank boundary is the **only threshold that matters**. Everything below it is free; everything above it pays a fixed tax. The question is simply: *can any of my collectives exceed 72 ranks?*

## 3. What caps each role's group size

Every collective's group size is bounded by a model-architectural number. Cross-reference the model's arch against these caps to know whether the rack boundary is reachable.

| Role | Typical upper bound | Origin |
|---|---|---|
| TP | $n_q$ (attention heads) | Each TP rank owns at least one attention head; TP > $n_q$ either idles ranks or replicates heads. |
| TP w/ GQA correctness | $n_{kv}$ | Strict GQA keeps each KV head on one rank; real implementations often replicate KV and allow TP up to $n_q$. |
| EP | $n_{\text{experts}}$ | Each EP rank owns a contiguous expert shard; EP > $n_{\text{experts}}$ wastes ranks. |
| SP | small (≤ 8 common) | Bounded by practical numerics, not architecture. |
| PP | $L$ (layers) | Each PP rank holds a contiguous stage. |

So whether the 2-tier fabric bites reduces to three questions:

1. Is $n_q > 72$ (equivalently, can I run TP > 72)?
2. Is $n_{\text{experts}} > 72$ (equivalently, can I run EP > 72)?
3. Do I actually *want* to push those partitions to that scale?

## 4. Model walkthrough — who crosses the cliff?

| Model | $n_q$ | $n_{kv}$ | $n_{\text{experts}}$ | Escalates into scale-out? |
|---|---|---|---|---|
| GPT-1.8T MoE (our sweep) | — | 16 | 16 | **No** — both TP and EP capped at 16. |
| DeepSeek-R1 / V3 (MoE) | 128 | 5 (MLA) | **256** | **Yes, on EP.** EP=128 or 256 puts every MoE layer's all-to-all across the scale-out fabric. |
| Qwen3-VL 235B (MoE) | 64 | 4 | 128 | **Yes, on EP** — EP=128 spills into scale-out. TP is capped at 64, so TP never crosses. |
| Kimi-K2 (MoE, announced) | — | — | **384** | **Yes, on EP.** |
| Mixtral 8×22B (MoE) | 48 | 8 | 8 | **No** — way below the cliff on both axes. |

The pattern: **GQA-constrained dense models with large head count, and fine-grained MoE models with hundreds of experts, are the architectures that put pressure on the rack boundary.** Classic GQA dense models with $n_q \le 64$ and coarse MoE models with $\le 64$ experts never do.

## 5. How big is the penalty, concretely?

The ring all-reduce time for a TP collective is:

$$
t_{TP} = 2(TP-1) \cdot \alpha + 2 \cdot \frac{TP-1}{TP} \cdot \frac{\text{msg}}{BW}
$$

where $\text{msg} = B \cdot H \cdot b$. Below 72 ranks the collective stays on `nvlink5` $(\alpha=0.5\ \mu s,\ BW=900\ \text{GB/s})$; above 72 it escalates into `ib` and pays $(\alpha=3.0\ \mu s,\ BW=400\ \text{GB/s})$.

**Measured on GPT-1.8T MoE FP4 at $PP=4$, $B=64$** (notebook §7):

| TP | $t_{TP}$ ideal (μs) | $t_{TP}$ hier (μs) | ratio | TPOT ideal (ms) | TPOT hier (ms) |
|---:|---:|---:|---:|---:|---:|
|  72 |  72.4 |  72.4 | 1.00× | 4.81 | 4.81 |
|  80 |  80.4 | 477.2 | **5.93×** | 5.24 | 29.05 |
| 128 | 128.4 | 765.3 | **5.96×** | 7.97 | 46.18 |

The step is discontinuous at the 72-rank radix. Below, the two systems are bit-identical; above, the hierarchical $t_{TP}$ jumps ~6× (the collective now spills from `nvlink5` into `ib`) and TPOT follows at ~5.8× because TP comm then dominates the step.

**For an all-to-all on DeepSeek-R1 at $EP=256$** (notebook §10.1), the same pattern holds: $t_{EP}$ goes 264 → 1551 μs (5.87×), and TPOT 19.3 → 96.5 ms (5.00×) at $PP=1, TP=1, B=64$. Both passes of the MoE all-to-all pay the scale-out cost on every MoE layer.

## 6. Why prefill and training are even more exposed

The caps in §3 are architectural and identical for decode, prefill, and training. But the *message sizes* scale differently:

- **Decode** messages are $O(B \cdot H)$ (one token's activation per sample).
- **Prefill / training** messages are $O(S \cdot H)$ per sample (whole sequence). For $S = 8192$, prefill messages are **~8000× larger** than decode.

When the message is small, the $\alpha$ term dominates; when it's large, the BW floor dominates. Prefill at long context hits the 400 GB/s BW floor hard on every TP/EP collective that crosses tier 1 — and those collectives are paid once per step instead of once per token, but the step is also hundreds of times heavier. The upshot: **the hierarchical fabric's BW floor penalizes prefill and training roughly proportional to $S$**, while decode sees a mix of $\alpha$ and BW effects that stay modest until $B$ grows large enough to saturate the bulk term.

This is consistent with the existing `pareto_vs_flops` finding that long-context decode is memory-bound end-to-end — the inter-rack BW floor lives in the same compute-vs-memory budget.

## 7. The phantom cliff — why decode often *doesn't* pay the penalty

An earlier draft of this note stopped after §6 with the expectation that fine-grained MoE and large-TP dense inference would show clear Pareto-frontier penalties on the 2-tier fabric. `pareto_vs_scale_up_tier.ipynb` §7 confirms the per-collective penalty (TP=144 all-reduce jumps 5.9×, TPOT jumps 5.8× at B=64) but §8's full-submanifold panel, §9's tier-0 radix sweep, **and §10's DeepSeek-R1 cross-model check** all find that **the Pareto frontier still overlaps almost exactly** between ideal and hierarchical fabrics — even after shrinking the rack all the way down to NVL8, and even on a model whose architecture (`n_experts = 256`) lets EP run straight through the cliff.

Why? The optimizer avoids the cliff by **picking partitions that fit**. For GQA-constrained decode (TP ≤ $n_{kv}$) and moderate MoE (EP ≤ $n_{\text{experts}}$), the Pareto-optimal winner uses **maximum pipeline parallelism** with small TP and EP — PP is a 2-rank P2P hop, which stays on tier 0 at any PP factor. The frontier-defining configurations never cross the rack boundary in the first place, so the scale-out tax is never paid. Even when we shrink tier 0 to 8 ranks (NVL8), the optimizer just picks TP ≤ 8, EP = 1 and ignores the fabric penalty — the Pareto frontier matches the ideal. DeepSeek (§10) reinforces the point through the opposite axis: EP is architecturally free to reach 256, but the full-set optimizer still picks EP ≤ 16 because lower-EP + higher-$B$ dominates on Pareto.

Frontier separation only emerges when we **force the crossing**: restrict the sweep to partitions where a group must exceed the rack radix (`pareto_vs_scale_up_tier.ipynb` §8 right panel — GPT TP ≥ 80; §9 right panel — NVL8 with TP or EP = 16; §10 right panel — DeepSeek EP ≥ 128). On those subsets the hierarchical envelope sits ~5× to the left of the ideal one (5.9× for GPT TP≥80, 5.45× for DeepSeek EP≥128) — the cliff is real, it just guards a region the optimizer was already planning to avoid.

The cliff is real, but it guards an empty region. Wide-TP / wide-EP partitions were already dominated on the ideal fabric; the hierarchical fabric just makes them lose by a wider margin. The architectural preference for depth-first parallelism does the work that fabric-awareness would do.

This *does not* make the hierarchical fabric free. It makes decode insensitive to the scale-out penalty for deep-$L$ models. Three places where the penalty still matters:

1. **Shallow models** ($L \lesssim 30$) force the cluster to use wide TP or EP to consume all ranks. Then every decode step pays the scale-out cost.
2. **Workloads where PP becomes the bottleneck.** PP-bubble overhead at small $B$, limited micro-batch budgets, or scheduler bugs can push the optimal partition away from max-PP and into the scale-out region.
3. **Prefill and training.** Collective messages scale with $S$ instead of $B$, so the BW floor taxes every long-context prefill step proportionally. Training additionally tends to prefer wider TP for numerical stability. Both are likely to show the frontier divergence that decode does not.

## 8. Four design rules

**1. Size the rack to the largest per-role group you actually intend to use, not to the architectural cap.** Architectural caps ($n_q$, $n_{\text{experts}}$) are an *upper bound* on what the deployment *could* run. The cliff bites only at the group sizes the deployment *does* run. If your model has $n_q \le 72$ *and* $n_{\text{experts}} \le 72$, no scale-up fabric beyond NVL72 helps — but even models that exceed those caps (DeepSeek-R1 at $n_{\text{experts}} = 256$) may never reach them on the decode Pareto frontier because lower-EP + higher-$B$ dominates (§7, §10). **Ask whether the deployment has a reason to force the large group** (expert-load balance at scale, numerical stability in training, long-$S$ prefill). If not, an NVL16 or NVL72 rack stitched into a larger domain for DP replication is often enough.

**2. For decode, depth saves you from the cliff.** Deep-$L$ models ($L \gtrsim 60$) have enough pipeline depth to fill a 576-GPU cluster with PP alone — the Pareto-optimal partition stays entirely inside the scale-up fabric regardless of how fine-grained the MoE or how large the dense head count is. Budget the stitched fabric for *DP capacity* (more concurrent sequences) rather than wider *intra-replica* parallelism. See §7.

**3. Fine-grained MoE is the strongest argument for stitched scale-up when max-PP isn't available.** A 256- or 384-expert shallow model genuinely needs EP > 72 to keep routing balanced; that's where extending the scale-up domain past NVL72 earns its place *and* pays the scale-out tax. Budget for the BW floor on all-to-all, which is the most expensive collective even before the scale-out hit.

**4. Large-TP dense inference is the second strongest — but only for prefill/training.** Dense models with $n_q \gtrsim 128$ *can* run TP past the rack radix and will pay the ~5.9× TP penalty demonstrated in §5 — but for decode, depth wins. The TP-wide case becomes load-bearing in prefill (where $S$-scaled messages hit the BW floor hard) and training (where TP is often chosen for numerical stability, not throughput). Whether the trade is worth it depends on the workload's $(TPS, TPOT)$ target and the specific $\alpha$ / BW values of the inter-rack tier in your fabric.

## 9. Follow-up case studies

- **`pareto_vs_scale_up_tier.ipynb`** covers the full story end-to-end: §3–§5 the original GPT-1.8T puzzle (frontiers bit-identical on the GQA-legal partition set), §6 the TP=144 stress test, §7 the per-collective cliff sweep (5.9× at TP > 72), §8 the phantom-cliff Pareto submanifold + forced-crossing panel, §9 the tier-0 radix sweep (NVL8 → NVL576) proving the main frontier is preserved even when the rack shrinks to 8 ranks, and §10 a DeepSeek-R1 cross-model check that confirms the phantom cliff through EP instead of TP (full frontier overlaps; forced EP ≥ 128 subset separates by 5.45×).
- **Open:** prefill and training sweeps. Expected to show actual frontier divergence because $S$-scaled messages stress the BW floor harder, and training tends to prefer wider TP.

## 10. Related reading

- `documentation/modeling/switching.md` §7 — Multi-tier extension, including the fabric-chain model (`collective_fabrics`) and the $\alpha_{\text{span}}$ / $BW_{\text{span}}$ derivation.
- `documentation/modeling/notation.md` §7 — Networking symbols for fabrics and tiers.
- `documentation/explaining/frontier_convergence_at_high_b.md` — why partitions converge at high B; helps interpret why the EP-heavy corner is most sensitive to scale-out BW.
- `documentation/explaining/pp_vs_tp_decode_scaling.md` — the compute/communication trade-off when choosing TP vs. PP; directly relevant to the §7 design rule on large-TP dense inference.
