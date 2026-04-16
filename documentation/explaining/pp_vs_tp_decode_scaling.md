# Why PP Dominates TP for Decode Interactivity at Scale

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, pipeline parallelism, tensor parallelism, decode, TPOT, interactivity, Pareto frontier, cluster scaling, communication overhead

---

## 1. Observation

In the `pareto_vs_cluster_size` sweep (GPT-1.8T MoE FP4 on GB200-class devices, N = 64 to 1024), the optimizer consistently maxes out PP before expanding TP:

| N    | winning partition     | PP  | TP |
|------|-----------------------|-----|----|
| 64   | PP=60 TP=1            | 60  | 1  |
| 128  | PP=120 TP=1           | 120 | 1  |
| 256  | PP=120 TP=2           | 120 | 2  |
| 512  | PP=120 TP=4           | 120 | 4  |
| 1024 | PP=120 TP=8           | 120 | 8  |

TP only grows after PP saturates at L = 120 (one layer per rank). This is counterintuitive — TP is usually presented as the primary intra-node parallelism strategy. Why does PP win for decode interactivity?

## 2. Communication cost comparison

Both PP and TP reduce per-stage compute and memory traffic by spreading work across more devices. The difference is **communication overhead per token**.

**PP (pipeline parallelism):**
- One point-to-point send per stage boundary: activation vector of size $H \cdot b$ bytes.
- Cost per boundary: $\alpha_{PP} + H \cdot b / BW_{PP}$.
- Total: $(PP - 1)$ sends, but they are pipelined across stages — the per-token overhead is one stage's P2P latency (the bubble is a separate concern handled by inflight batching at B > 1).

**TP (tensor parallelism):**
- Two all-reduces per layer (one after attention projection, one after FFN).
- Each all-reduce on a ring of TP ranks costs: $2 \cdot \frac{TP-1}{TP} \cdot \frac{msg}{BW_{TP}} + 2 \cdot \alpha_{TP} \cdot (TP-1)$
- Total per token: $2 \cdot (L / PP)$ all-reduces — paid every layer in every stage.

For single-token decode, the activation message ($H \cdot b$) is small (e.g., 20480 × 0.5B = 10 KB for this model). The all-reduce bandwidth term is tiny, but the **latency term** $\alpha$ is paid per layer per token and scales with TP. PP's point-to-point has no ring latency — it is a single hop.

## 3. Why PP's bubble doesn't kill interactivity

The classic argument against deep PP is the pipeline bubble: at B = 1, a PP-stage pipeline has $(PP - 1)$ idle stage-slots during warmup and drain. For decode, each stage processes one token through $L / PP$ layers — this is microseconds of compute. The bubble cost in absolute time is:

$$t_{bubble} = (PP - 1) \cdot t_{stage}$$

But $t_{stage}$ shrinks as $1/PP$ (fewer layers per stage), so:

$$t_{bubble} = (PP - 1) \cdot \frac{t_{total}}{PP} \approx t_{total} \quad \text{(for large PP)}$$

This looks bad — but $t_{total}$ itself is the single-device decode time divided by PP (less work per stage). The net effect is that deep PP at B = 1 roughly doubles the single-device time (bubble ≈ compute), while TP at the same device count would add $2 \cdot L \cdot \alpha_{TP} \cdot (TP - 1)$ of pure communication latency.

At B > 1 (inflight batching), the bubble amortizes as $1/B$ while TP's per-layer all-reduce cost stays fixed per token. This is why the Pareto frontier at the high-interactivity corner (moderate B) favors deep PP.

## 4. When TP wins instead

TP becomes the better lever when:

- **PP is maxed out** (PP = L): no more layers to split. This is exactly what the sweep shows — TP grows only after PP = 120.
- **Prefill (large S):** per-stage compute is $O(S)$ or $O(S^2)$, making the all-reduce bandwidth term worthwhile to overlap with large matmuls. The bubble cost also grows with S (more compute per stage to drain).
- **High α environments:** if $\alpha_{PP}$ is large (e.g., cross-node IB with 5 μs latency), the $(PP-1)$ serial hops accumulate significant latency, tipping the balance toward fewer PP stages and more TP.
- **Very large TP within NVLink domain:** when TP ≤ 8 on NVLink (α ≈ 0.5–1 μs, BW = 450–900 GB/s), the all-reduce overhead is small enough that TP competes with PP on latency while avoiding the bubble entirely.

## 5. Summary

| Dimension | Comm per token | Scales with | Bubble | Best regime |
|-----------|---------------|-------------|--------|-------------|
| PP | $(PP-1)$ P2P hops | stages | Yes, amortized by B | Decode, B ≥ 1 |
| TP | $2 \cdot (L/PP)$ all-reduces | layers × TP ranks | None | Prefill, or after PP maxed |

For decode interactivity, **PP is communication-cheaper per token than TP** because point-to-point sends have lower latency than all-reduces, and the pipeline bubble is amortized by inflight batching. The optimizer exhausts PP depth before turning to TP — this is a robust result across all cluster sizes in the sweep.

The η sensitivity analysis reinforces this: when fabric efficiency drops at large N (η < 1), all communication gets more expensive, which makes the PP-first strategy even more dominant since PP's communication cost grows slower than TP's.
