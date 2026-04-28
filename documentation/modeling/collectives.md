# Collective Communication Cost Model

**Author:** Yue Lu  
**Date:** April 2026  

Operational lookup table for the collective communication costs consumed by `decode.md` and `prefill.md`. For every shipped primitive on every modeled topology — star (NCCL), torus (TPU XLA / Trainium NeuronX CCL), hierarchical Clos / fat-tree (multi-pod), and in-network reduction (NVLink SHARP / NVLS, Quantum SHARP, Spectrum-X SHARP, Tomahawk Ultra) — this document states the closed-form $(\alpha, \mathrm{BW})$ cost, the $(n_\alpha, n_\beta)$ coefficients, and where the cost is incurred during inference. Contention is layered on top through scalar coefficients $(\eta_\alpha, \eta_\beta)$ per fabric tier (§7). Derivations, worked examples, and algorithm trade-offs live in the explainer series `documentation/modeling/collectives/`; this doc is the lookup table.

---

## 1. Scope and Pointer to the Explainer Series

This document costs the six collective primitives that appear in the inference-cluster decode and prefill pipelines: all-reduce (AR), all-gather (AG), reduce-scatter (RS), all-to-all (A2A), broadcast (BC), and reduce. AR / AG / RS / A2A are the load-bearing primitives — see §3.5, §4.5, §5.5 for where each appears. BC and reduce are documented in §6 mainly because they are the building blocks of hierarchical AR (inner RS → outer AR → inner AG) and of the in-network reduction (INC) family (switch ALU + multicast crossbar). Point-to-point (P2P) is the pipeline-parallel hop and is priced by `decode.md §5.5`.

Four topology classes are modeled: a **star** (single-tier crossbar or single-switch fabric, e.g. NVSwitch within a node or pod), a **torus** (dim-decomposed ring fabric, $k \in \{1, 2, 3\}$, shipped on TPU and Trainium), a **hierarchical Clos / fat-tree** (multi-tier composition such as NVL72 + InfiniBand SuperPOD, where an inner scale-up fabric attaches to an outer scale-out tier), and **in-network collectives (INC)** — switch-ASIC reduction and multicast that collapses the latency term on switched fabrics (NVLS within an NVSwitch domain, Quantum SHARP / Spectrum-X SHARP across IB / Ethernet fat-trees, Tomahawk Ultra on commodity Ethernet, plus emerging hardware A2A on Tomahawk Ultra and Rubin-generation NVSwitches). Dragonfly and mesh (k-D mesh without wraparound) are not modeled here because no inference cluster in the modeled set uses them.

Throughout, $M$ is the per-collective message volume in bytes and $N$ is the group size (the number of ranks participating). Derivation of each cost formula lives in the explainer series:

| Topic | Explainer section |
|---|---|
| α-β model and seven-primitive family (BC / reduce / AR / RS / AG / A2A / P2P) | `01_collective_algorithms.md` |
| Star vs torus per-topology derivations and worked examples | `02_topology_mapping.md` |
| Hierarchical composition rules (multi-tier Clos / fat-tree, NVL72 SuperPOD case study) | `03_hierarchical_topologies.md` |
| In-network reduction (NVLS / Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra / HW A2A) | `04_in_network_collectives.md` |
| Contention coefficients $(\eta_\alpha, \eta_\beta)$ derivation and per-tier calibration | `05_contention_and_congestion.md` |
| One-page cheatsheet (symbol table, per-algorithm $(n_\alpha, n_\beta)$, per-topology specializations) | `00_summary.md` |

"NCCL" in §3–§6 refers to NVIDIA NCCL's shipped defaults on switched fabrics (NVLink + NVSwitch or Ethernet / InfiniBand). "TPU / Trainium" refers to the XLA and NeuronX collective kernels shipped on Google TPU and AWS Trainium torus fabrics. Algorithms discussed in the explainer but not shipped in any production stack covered here (Bruck, simple recursive-doubling, Rabenseifner halving-doubling on torus) are excluded; they appear only in the explainer's appendices for reference.

---

## 2. Notation

This section re-lists only the symbols that are new or load-bearing for this doc. Complete definitions are in `notation.md §7` (networking) and `notation.md §9` (decode timing).

- $\alpha$ — Per-traversal startup latency of the crossed fabric tier (μs).
- $\alpha_\mathrm{switch}$ — Switch cut-through latency for an in-network operation (200–400 ns on shipping hardware). Endpoint software α-floor is $\sim 1\,\mu$s; INC pays $\alpha_\mathrm{switch}$ instead, and only once per collective rather than once per algorithmic step.
- $\mathrm{BW}$ — Effective per-port single-direction bandwidth of the crossed tier (bytes/s).
- $M$ — Message size per collective in bytes. Conventions: for AR and A2A, $M$ is the **total per-rank input volume**; for AG and RS, $M$ is the **per-rank shard size** (so the full gathered tensor is $N \cdot M$); for BC and reduce, $M$ is the broadcast/reduced payload.
- $N$ — Group size; number of ranks participating in the collective.
- $k$ — Torus dimensionality (1, 2, or 3 in practice); also reused as aggregation-tree depth $k = \lceil \log_r N \rceil$ for INC on multi-tier fabrics (§3.4, §6).
- $(D_1, \ldots, D_k)$ — Per-axis extents of a torus tier; $N = \prod_i D_i$.
- $D_\mathrm{max} = \max_i D_i$ — Longest axis; sets the A2A bisection floor.
- $\mathrm{diam} = \sum_i \lfloor D_i / 2 \rfloor$ — Wraparound torus diameter.
- $L$ — Outer-tier rank count in a 2-tier hierarchy: $N = L \cdot N_\mathrm{inner}$ where $N_\mathrm{inner} = N / L$ is the per-pod scale-up rank count.
- $\alpha_\mathrm{inner}$, $\alpha_\mathrm{outer}$, $\mathrm{BW}_\mathrm{inner}$, $\mathrm{BW}_\mathrm{outer}$ — Inner / outer tier α and BW for a hierarchical schedule. The inner tier is the scale-up fabric (NVSwitch within a pod); the outer tier is the scale-out fabric (InfiniBand or Ethernet across pods).
- $s \geq 1$ — Oversubscription ratio at a fat-tree / Clos tier boundary. $s = 1$ is non-blocking; $s > 1$ caps that tier's $\eta_\beta$ at $1/s$ (§7.3).
- $n_\alpha$ — Coefficient on $\alpha$ in the cost formula (number of startup traversals on the algorithmic critical path).
- $n_\beta$ — Coefficient on $M / \mathrm{BW}$ (per-rank byte movement in units of $M$).
- $\mathrm{BW_{eff}} = \mathrm{BW} / n_\beta$ — Effective per-rank bandwidth seen by the collective. AR alone has $\mathrm{BW_{eff}} = \mathrm{BW}/2$ in software (every byte is touched twice — RS in, AG out) and $\mathrm{BW_{eff}} = \mathrm{BW}$ under INC (switch ALU + multicast crossbar fuses the two halves).
- $\eta_\alpha \geq 1$, $\eta_\beta \in (0, 1]$ — Contention coefficients per fabric tier (§7).
- $\mathrm{ar\_algorithm}$ — Tuning-knob symbol selecting star AR algorithm. Admissible values: $\{\mathrm{ring}, \mathrm{DBT}\}$; default $\mathrm{ring}$.

---

## 3. All-Reduce

### 3.1 Star — Ring and Double Binary Tree (shipped by NCCL)

NCCL ships two AR algorithms on a star fabric. Both are pipelined and bandwidth-optimal per [PY09] and [SST09]; the difference is only in $n_\alpha$.

**Ring AR:**

$$t_\mathrm{AR}^\mathrm{star,ring}(M, N) \;=\; 2(N - 1)\,\alpha \;+\; 2 \cdot \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

**Double binary tree (DBT) AR:**

$$t_\mathrm{AR}^\mathrm{star,DBT}(M, N) \;=\; 2\,\lceil \log_2 N \rceil \cdot \alpha \;+\; 2 \cdot \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

| Algorithm | $n_\alpha$ | $n_\beta$ | Shipped by |
|---|---|---|---|
| Ring | $2(N - 1)$ | $2(N - 1)/N$ | NCCL (large $M$) |
| DBT | $2\,\lceil \log_2 N \rceil$ | $2(N - 1)/N$ | NCCL (small $M$) |

Both share the same BW-optimal bandwidth coefficient; DBT wins at small $M$ where $n_\alpha$ dominates, ring wins at large $M$ where implementation overhead in the tree kernel matters more than the $\log_2 N$ saving [DEMYST-NCCL]. Derivation: `01_collective_algorithms.md §5.1` (ring) and §5.2 (DBT).

### 3.2 Torus — Dim-Decomposed Ring (shipped by TPU XLA / Trainium NeuronX CCL)

On a $k$-D torus with extents $(D_1, \ldots, D_k)$ and $N = \prod_i D_i$, AR runs a ring independently along each axis, chained in sequence [CHPV07, PY09]:

$$t_\mathrm{AR}^\mathrm{torus}(M, N, \mathrm{dims}) \;=\; 2 \sum_{i=1}^{k} (D_i - 1)\,\alpha \;+\; 2 \cdot \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

| Primitive | $n_\alpha$ | $n_\beta$ | Shipped by |
|---|---|---|---|
| Dim-decomposed ring | $2 \sum_i (D_i - 1)$ | $2(N - 1)/N$ | TPU XLA, Trainium NeuronX CCL |

The bandwidth term telescopes to the flat-ring bound (identical BW coefficient to a star ring of size $N$); only the latency term compresses from $O(N)$ to $O(\sum_i (D_i - 1))$. Derivation: `02_topology_mapping.md §3.4` with $2 \times 2$ worked example. Dim-alignment is assumed; off-prefix group layouts degrade to the flat-ring fallback (see §7.2 below).

### 3.3 Hierarchical — RS → sub-AR → AG composition (multi-tier Clos / fat-tree)

When the deployment spans multiple scale-up pods connected by a scale-out tier (e.g. NVL72 pods over InfiniBand), AR decomposes into three phases that exploit reduction's associativity to push most algorithmic hops onto the fast inner tier and shrink the cross-tier payload [PY09, SHARP-IB]:

$$t_\mathrm{AR}^\mathrm{hier}(M, N, L) \;=\; t_\mathrm{RS}^\mathrm{inner}\!\left(M,\, \tfrac{N}{L}\right) \;+\; t_\mathrm{AR}^\mathrm{outer}\!\left(\tfrac{ML}{N},\, L\right) \;+\; t_\mathrm{AG}^\mathrm{inner}\!\left(M,\, \tfrac{N}{L}\right)$$

Each term instantiates the matching primitive on the matching tier — inner RS and AG run on the scale-up fabric (star or torus, §3.1 / §3.2 / §4.1 / §4.2) at $(\alpha_\mathrm{inner}, \mathrm{BW}_\mathrm{inner})$; the outer sub-AR runs on the cross-pod fabric at $(\alpha_\mathrm{outer}, \mathrm{BW}_\mathrm{outer})$ on the **telescoped payload** $ML/N$. For ring-on-ring with star fabrics at both tiers:

$$t_\mathrm{AR}^\mathrm{2\text{-}tier} \;\approx\; 2\!\left(\tfrac{N}{L} - 1\right)\alpha_\mathrm{inner} \;+\; 2(L - 1)\,\alpha_\mathrm{outer} \;+\; \frac{2(N - 1)}{N} \cdot \frac{M}{\mathrm{BW}_\mathrm{bottleneck}}$$

with $\mathrm{BW}_\mathrm{bottleneck} \approx \mathrm{BW}_\mathrm{outer}$ once any payload crosses the slower outer tier. For $k > 2$ tiers, the outer term recurses. Oversubscription $s > 1$ at the outer tier multiplies that tier's BW term by $s$ (equivalent to $\eta_\beta = 1/s$ on cross-tier traffic, §7.3). Derivation and the NVL72 + IB SuperPOD worked example: `03_hierarchical_topologies.md §2.1`.

### 3.4 In-Network Reduction (NVLS / Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra)

When the fabric supports switch-resident ALUs plus multicast crossbar (the SHARP-class capability), AR runs the entire reduce-and-broadcast inside the switch ASIC. AR is the **unique** primitive that gets both a structural $\alpha$-collapse and a BW-eff doubling: every byte that software AR touches twice (RS in, AG out, $\mathrm{BW_{eff}} = \mathrm{BW}/2$) is touched only once per endpoint under INC, because the switch ALU and multicast crossbar fuse the two halves [NVLINK-SHARP, SHARP-IB].

**Single-switch domain (scale-up INC, e.g. NVLS within an NVL72 pod):**

$$t_\mathrm{AR}^\mathrm{INC,scale\text{-}up}(M) \;\approx\; 2\,\alpha_\mathrm{switch} \;+\; \frac{M}{\mathrm{BW}}$$

**Multi-tier aggregation tree (scale-out INC, e.g. Quantum SHARP across an IB Clos):**

$$t_\mathrm{AR}^\mathrm{INC,scale\text{-}out}(M, k) \;\approx\; 2k \cdot \alpha_\mathrm{switch} \;+\; \frac{M}{\mathrm{BW}}$$

where $k = \lceil \log_r N \rceil$ is the number of switch tiers an $M$-byte flit traverses from an endpoint to the aggregation-tree root, with per-switch radix $r$ (e.g. $r = 64$ ports per switch covers $N = 4096$ in $k = 2$ tiers).

| Capability | $n_\alpha$ | $n_\beta$ | Hardware |
|---|---|---|---|
| Scale-up INC (single switch) | $\sim 2$ | $1$ | NVLS (NVSwitch Gen3 / Gen4), Tomahawk Ultra |
| Scale-out INC (aggregation tree, depth $k$) | $\sim 2k$ | $1$ | Quantum SHARP (IB Quantum-2 / X800), Spectrum-X SHARP (Ethernet) |

Realized $\mathrm{BW_{eff}}$ on shipping hardware is below the $1$ ceiling — measured NVLS lift is $\sim 1.3\times$ vs DBT, not the $2\times$ algorithmic ceiling — and is priced via $\eta_\beta^\mathrm{INC} \approx 0.52$ (§7.4). INC requires hardware support at every switch in the path; cross-domain traffic (e.g. a SHARP group spanning an INC-enabled and a non-INC fabric) falls back to software DBT or ring. Reducible types are limited to sum / max / min / bit-ops over BF16 / FP16 / FP32; compound reductions (softmax, top-$k$) and FP64 fall back to software. Derivation: `04_in_network_collectives.md §1.1`, §1.2 (mechanism), §2 (scale-up vs scale-out), §3 (worked $N = 512$ ladder).

### 3.5 Where AR Appears in the Inference Pipeline

- **TP AR** — `decode.md §5.3`, `prefill.md §3.2`. Row-parallel output reduction; two per layer (attention + FFN). Message size: decode $H \cdot b$, prefill $H \cdot S_\mathrm{input} \cdot b$.
- **Embedding AR** — not separately costed; folded into the TP AR count $n_\mathrm{TP}$ of the relevant layer.

---

## 4. All-Gather and Reduce-Scatter

### 4.1 Star — Ring (shipped by NCCL)

AG and RS share the same star-ring cost (duality: RS is the time-reverse of AG), with $M$ denoting the per-rank shard:

$$t_\mathrm{AG}^\mathrm{star,ring}(M, N) \;=\; t_\mathrm{RS}^\mathrm{star,ring}(M, N) \;=\; (N - 1)\,\alpha \;+\; (N - 1) \cdot \frac{M}{\mathrm{BW}}$$

| Primitive | $n_\alpha$ | $n_\beta$ | $M$ convention | Shipped by |
|---|---|---|---|---|
| Ring AG / RS | $N - 1$ | $N - 1$ | per-rank shard | NCCL |

PAT (NCCL 2.23+) ships only in the scale-out, 1-rank-per-node regime and is not a default for the in-pod modeling pass; see `01_collective_algorithms.md App. A`. A recursive-doubling AG / recursive-halving RS variant ($n_\alpha = \lceil \log_2 N \rceil$, same $n_\beta$) is available in MPI implementations and can be substituted on power-of-2 $N$ at small $M$, but NCCL ships ring as the production default. Derivation: `01_collective_algorithms.md §6`.

### 4.2 Torus — Dim-Decomposed Ring (shipped by TPU XLA / Trainium NeuronX CCL)

Same dim-decomposition structure as torus AR, but one-way traversal (AR does forward + backward, AG / RS does one direction):

$$t_\mathrm{AG}^\mathrm{torus}(M, N, \mathrm{dims}) \;=\; t_\mathrm{RS}^\mathrm{torus}(M, N, \mathrm{dims}) \;=\; \sum_{i=1}^{k} (D_i - 1)\,\alpha \;+\; (N - 1) \cdot \frac{M}{\mathrm{BW}}$$

| Primitive | $n_\alpha$ | $n_\beta$ | $M$ convention | Shipped by |
|---|---|---|---|---|
| Dim-decomposed ring AG / RS | $\sum_i (D_i - 1)$ | $N - 1$ | per-rank shard | TPU XLA, Trainium NeuronX CCL |

Derivation: `02_topology_mapping.md §3.5`.

### 4.3 Hierarchical — Inner + Outer Cascade

AG and RS cascade trivially across tiers (no reduction to amortize, no payload telescoping — the per-rank shard moves through both tiers verbatim):

$$t_\mathrm{AG}^\mathrm{hier}(M, N, L) \;=\; t_\mathrm{AG}^\mathrm{inner}\!\left(M,\, \tfrac{N}{L}\right) \;+\; t_\mathrm{AG}^\mathrm{outer}\!\left(\tfrac{NM}{L},\, L\right)$$

with $t_\mathrm{RS}^\mathrm{hier}$ symmetric (outer RS first, then inner RS). Each tier instantiates §4.1 or §4.2. AG / RS cost half of AR on both α (one pass, not two) and BW (one telescoping pass, not the doubled AR round-trip); see `03_hierarchical_topologies.md §2.1`.

### 4.4 In-Network Multicast (NVLS / Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra)

AG runs on the switch's multicast crossbar ($N$ concurrent broadcasts, one per source slice); RS runs on the switch ALU ($N$ ranks push, switch sums and scatters slices). Both collapse $n_\alpha$ but **do not lift $\mathrm{BW_{eff}}$** — full-duplex software ring AG / RS already saturates the link (forward + receive concurrently), so there is no factor-of-two slack to recover. INC's win on AG / RS is α-only [NVLINK-SHARP]:

$$t_\mathrm{AG}^\mathrm{INC,scale\text{-}up}(M, N) \;\approx\; 2\,\alpha_\mathrm{switch} \;+\; \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

with the multi-tier form replacing $2\,\alpha_\mathrm{switch}$ by $2k\,\alpha_\mathrm{switch}$. Same expression for RS. Derivation: `04_in_network_collectives.md §1.1`, §1.2, §1.4.

### 4.5 Where AG / RS Appear in the Inference Pipeline

- **SP AG (ring attention)** — `decode.md §5.4`, `prefill.md §3.2`. Per-layer KV shard circulation; $(SP - 1)$ rotation steps for pass-KV ring attention. Message size: $(S / SP) \cdot (2 H_{kv} / TP) \cdot b$ per step.
- **TP-AG / RS on sequence-parallel layouts** — folded into the TP AR count when sequence parallelism within TP is in use (`decode.md §5.3`).
- **MoE AG (capacity repack)** — not a separate line item; folded into the A2A Dispatch + Combine path in §5.

---

## 5. All-to-All

### 5.1 Star — Pairwise Direct-Send (shipped by NCCL)

NCCL's shipped A2A on a star fabric is pairwise direct-send — every rank exchanges data with every other rank in parallel. Bruck's log-hop A2A is reference-only (`01_collective_algorithms.md App. B.5`) and is **not** shipped; it does not appear in modeling equations.

$$t_\mathrm{A2A}^\mathrm{star,pairwise}(M, N) \;=\; (N - 1)\,\alpha \;+\; \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

| Primitive | $n_\alpha$ | $n_\beta$ | $M$ convention | Shipped by |
|---|---|---|---|---|
| Pairwise direct-send | $N - 1$ | $(N - 1)/N$ | total per-rank output | NCCL |

Derivation: `01_collective_algorithms.md §7.2`.

### 5.2 Torus — Bisection-Bound Pairwise (shipped by TPU XLA / Trainium NeuronX CCL)

On a torus, A2A cannot exceed the bisection capacity. The shipped pattern on TPU / Trainium is pairwise direct-send routed over the torus dimensions, with the per-rank BW floor set by the longest axis $D_\mathrm{max}$:

$$t_\mathrm{A2A}^\mathrm{torus}(M, N, \mathrm{dims}) \;\approx\; \mathrm{diam} \cdot \alpha \;+\; \frac{D_\mathrm{max}}{8} \cdot \frac{M}{\mathrm{BW}}$$

where $\mathrm{diam} = \sum_i \lfloor D_i / 2 \rfloor$ and $D_\mathrm{max} = \max_i D_i$.

| Primitive | $n_\alpha$ | $n_\beta$ | $M$ convention | Shipped by |
|---|---|---|---|---|
| Bisection-bound pairwise | $\mathrm{diam}$ | $D_\mathrm{max} / 8$ | total per-rank output | TPU XLA, Trainium NeuronX CCL |

Asymmetric layouts (e.g. $16 \times 16 \times 4$ vs $8 \times 8 \times 16$) pay disproportionately through $D_\mathrm{max}$; twisted-torus reshaping [TPU-V4] is the shipped mitigation (observed $1.63\times$ A2A improvement on TPU v4) but is modeled by the operator supplying a post-reshape $(D_1, \ldots, D_k)$ tuple rather than a separate twist parameter. Derivation: `02_topology_mapping.md §3.6` with $4 \times 4$ worked example.

### 5.3 Hierarchical — Per-Class Destination Accounting (the outlier)

A2A is the one primitive that does **not** decompose hierarchically. Every source-destination pair carries a distinct payload — no aggregation to amortize, no replication to telescope — so the cross-tier permutation traffic equals the full $(N - 1) \cdot (M/N)$ bytes per rank regardless of schedule. The implemented pattern is pairwise direct-send with each destination paying its own path cost. For an inner-pod-of-$N_\mathrm{inner}$ on an outer-of-$L$-pods deployment with $p$ pods sharing a rail-ToR (`03_hierarchical_topologies.md §2.2`):

$$t_\alpha^\mathrm{A2A,hier} \;=\; (N_\mathrm{inner} - 1)\,\alpha_\mathrm{inner} \;+\; (p - 1)\,N_\mathrm{inner}\,\alpha_\mathrm{leaf} \;+\; (L - p)\,N_\mathrm{inner}\,\alpha_\mathrm{spine}$$

$$t_\mathrm{BW}^\mathrm{A2A,hier} \;=\; (N_\mathrm{inner} - 1)\,\frac{M/N}{\mathrm{BW}_\mathrm{inner}} \;+\; (N - N_\mathrm{inner})\,\frac{M/N}{\mathrm{BW}_\mathrm{outer}} \cdot s$$

with $\alpha_\mathrm{leaf}$ and $\alpha_\mathrm{spine}$ the same-leaf and cross-leaf outer-tier hop costs and $s$ the outer oversubscription. Per-rank BW is bisection-bound at the **outermost** crossed tier; no schedule reduces this. The production rule is therefore "keep MoE expert parallelism (EP) inside the inner tier whenever it fits" — pulling A2A across an outer tier roughly quadruples its cost on shipping fabrics.

### 5.4 In-Network HW A2A (Tomahawk Ultra, Rubin-generation NVSwitches)

A2A's INC story is structurally separate from §3.4 and §4.4: there is no aggregation or replication semantic to exploit, so the SHARP-class switch ALU + multicast crossbar pair gives A2A nothing. A different primitive — **hardware crossbar scatter-gather** — accelerates A2A by collapsing $N - 1$ endpoint-driven scheduling rounds into one switch-driven transaction (per-chunk descriptor parsing + parallel routing within the switch crossbar). The α-side collapse is structural; the BW term stays bisection-bound (the switch routes bytes verbatim, no reduction):

$$t_\mathrm{A2A}^\mathrm{INC}(M, N) \;\approx\; \alpha_\mathrm{switch} \;+\; \frac{N - 1}{N} \cdot \frac{M}{\mathrm{BW}}$$

Shipping today on **Broadcom Tomahawk Ultra** (Ethernet, 2025) [TH-ULTRA]; planned for **Rubin-generation NVSwitches** as an extension to NVLS. **Not** available on current GB200 / NVL72 NVSwitches (Gen4) or Quantum-X800 — A2A on those platforms uses §5.1's software pairwise. Derivation: `04_in_network_collectives.md §1.3`.

### 5.5 Where A2A Appears in the Inference Pipeline

- **EP A2A (Dispatch + Combine)** — `decode.md §5.2`, `prefill.md §3.2`. Once per MoE FFN layer; $n_\mathrm{EP} = 1$ since Dispatch and Combine are folded into a single primitive call of cost $2 \cdot t_\mathrm{A2A}$ with $M = k \cdot H \cdot b$ for decode and $M = k \cdot H \cdot S_\mathrm{input} \cdot b$ for prefill. The underlying $(n_\alpha, n_\beta)$ comes from §5.1 (star), §5.2 (torus), §5.3 (hierarchical), or §5.4 (HW A2A).
- **KV repack A2A on partition change** — covered by `prefill.md §6` (KV handoff) rather than this doc.

---

## 6. Broadcast and Reduce

BC and reduce do not appear as standalone primitives in the inference pipeline — they are documented here mainly because they are the building blocks of hierarchical AR (inner RS uses reduce-style telescoping; inner AG uses BC-style replication) and of INC (switch ALU executes in-fabric reduce; multicast crossbar executes in-fabric BC). Cost formulas are summarized below for completeness; see `01_collective_algorithms.md §3` (BC), §4 (reduce) and `02_topology_mapping.md §5.1` for derivations.

| Topology | Primitive | Algorithm | $n_\alpha$ | $n_\beta$ | Shipped by |
|---|---|---|---|---|---|
| Star | BC | Pipelined binomial tree | $\lceil \log_2 N \rceil$ | $1$ | NCCL |
| Star | Reduce | Pipelined binomial tree | $\lceil \log_2 N \rceil$ | $1$ | NCCL |
| Star | BC / Reduce | INC (multicast / ALU) | $\sim \alpha_\mathrm{switch}$ | $1$ | NVLS, Quantum SHARP, Tomahawk Ultra |
| Torus | BC / Reduce | Dim-decomp bidirectional | $\sum_i \lfloor D_i / 2 \rfloor$ | $1$ | TPU XLA, Trainium NeuronX CCL |
| Hierarchical | BC | Outer BC → inner BC | $\lceil \log_2 L \rceil\,\alpha_\mathrm{outer} + \lceil \log_2 (N/L) \rceil\,\alpha_\mathrm{inner}$ | $1$ | composition |
| Hierarchical | Reduce | Inner reduce → outer reduce | $\lceil \log_2 (N/L) \rceil\,\alpha_\mathrm{inner} + \lceil \log_2 L \rceil\,\alpha_\mathrm{outer}$ | $1$ | composition |

Both primitives hit the $M / \mathrm{BW}$ ceiling under pipelining ($\mathrm{BW_{eff}} = \mathrm{BW}$) on every topology, so INC's win on BC / reduce is α-only at large $M$ and converges to $1\times$ at the BW-bound limit.

---

## 7. Contention Coefficients

The §3–§6 cost formulas describe an isolated collective on an uncongested fabric. Production workloads sit below those upper bounds — concurrent groups share links, off-prefix layouts force flat-ring fallbacks, MoE routing skews bisection traffic. Two scalar coefficients $(\eta_\alpha, \eta_\beta)$ per fabric tier absorb all of this without rewriting the primitives. Derivations and the full catalog of effects are in `05_contention_and_congestion.md`.

### 7.1 Single-Tier Form

For any tier with calibrated $(\alpha, \mathrm{BW})$, the contention-adjusted cost is

$$t \;=\; n_\alpha \cdot \alpha \cdot \eta_\alpha \;+\; n_\beta \cdot \frac{M}{\mathrm{BW} \cdot \eta_\beta}, \qquad \eta_\alpha \;\geq\; 1, \;\; \eta_\beta \;\in\; (0, 1]$$

$\eta_\alpha$ inflates startup cost (queueing, scheduler serialization, layout-driven extra hops); $\eta_\beta$ deflates effective bandwidth (link utilization ceiling, bisection saturation, cross-tier oversubscription). Both default to $1$, which recovers the Hockney form [ALPHA-BETA].

### 7.2 Hierarchical Form

Each tier carries its own $(\eta_\alpha^\mathrm{tier}, \eta_\beta^\mathrm{tier})$ pair. The §3.3 RS → sub-AR → AG decomposition applies the inner tier's coefficients to the inner phases and the outer tier's coefficients to the sub-AR:

$$t_\mathrm{AR,realistic}^\mathrm{hier} \;=\; \underbrace{t_\mathrm{RS}^\mathrm{inner}\!\left(M,\, \tfrac{N}{L}\right)}_{(\eta_\alpha^\mathrm{inner},\, \eta_\beta^\mathrm{inner})} \;+\; \underbrace{t_\mathrm{AR}^\mathrm{outer}\!\left(\tfrac{ML}{N},\, L\right)}_{(\eta_\alpha^\mathrm{outer},\, \eta_\beta^\mathrm{outer})} \;+\; \underbrace{t_\mathrm{AG}^\mathrm{inner}\!\left(M,\, \tfrac{N}{L}\right)}_{(\eta_\alpha^\mathrm{inner},\, \eta_\beta^\mathrm{inner})}$$

**Oversubscription cap.** A tier oversubscribed at ratio $s \geq 1$ has its $\eta_\beta$ structurally capped:

$$\eta_\beta^\mathrm{upper\text{-}tier} \;\leq\; \min\!\left(\eta_\beta^\mathrm{hw\,floor},\; \frac{1}{s}\right)$$

The cap binds at $1/s$ whenever oversubscription drives below the hardware floor (e.g. $s = 2$ caps a $0.80$ floor at $0.50$). $\eta_\alpha$ inflates from cross-tier queueing under load (typically $1.10$–$1.30$ at moderate load, climbing to $2$–$3\times$ near saturation per §7.3) but oversubscription is a BW lever, not an α lever — there is no structural $f(s)$ form for $\eta_\alpha$.

### 7.3 Calibrated Defaults

Starting values from public benchmarks; operators should override per measured fabric.

| Fabric / regime | $\eta_\alpha$ | $\eta_\beta$ | Source |
|---|---|---|---|
| Star (NVLink + NVSwitch, no SHARP) | $1.00$ | $0.80$ | [NCCL-TESTS] H100 AR busbw $360 / 450\,\mathrm{GB/s}$ |
| Star + NVLS (in-network reduction) | $1.00$ | $0.52$ | [NVLINK-SHARP] busbw lift back-solved under $n_\beta^\mathrm{INC} = 1$ |
| Torus, dim-aligned | $1.00$ | $0.85$ | [TPU-V4] twisted-gain context |
| Torus, off-prefix / concurrent groups | $1.20$ | $0.60$ | [TPU-V4] twisted-vs-untwisted $1.63\times$ upper bound |
| Fat-tree / Clos leaf | $1.00$ | $0.80$ | matches single-tier crossbar floor |
| Fat-tree / Clos spine at $s$ | $1.10$–$1.30$ | $\min(0.80,\, 1/s)$ | $1/s$ cap from §7.2 |

**Calibration recipe.** Run nccl-tests (or equivalent) under the intended concurrent-group pattern; back out $\eta$ from the ratio of measured `busbw` to the theoretical algbw peak. Full per-tier calibration profile, double-counting discipline (calibrated vs catalog BW), INC-as-algorithm-substitution caveat, and the limits of the scalar model (layout dependence, payload-size sensitivity, tail latency) are in `05_contention_and_congestion.md §4`–§7.

---

## 8. References

All citations in §3–§7 resolve to entries in `references.md`:

- [PY09] — Ring AR bandwidth-optimality $2(N-1)/N \cdot M/\mathrm{BW}$.
- [SST09] — Double binary tree AR, pipelined BW-optimal form.
- [CHPV07] — Dim-decomposed collective framework on torus.
- [DEMYST-NCCL] — NCCL tuner empirical ring-vs-DBT crossover.
- [TPU-V4] — 3D torus with optical reconfiguration; twisted-torus A2A gain.
- [TRN2-ARCH] — Trainium2 torus architecture.
- [NEURON-CC] — AWS Neuron CCL dim-decomposed ring AR.
- [ALPHA-BETA] — Hockney α-β cost model.
- [NCCL-TESTS] — Canonical busbw / algbw methodology; H100 AR busbw $\approx 360 / 450\,\mathrm{GB/s}$ baseline for $\eta_\beta = 0.80$.
- [NVLINK-SHARP] — NVSwitch in-network reduction; AR busbw $470$ vs DBT $360\,\mathrm{GB/s}$ lift back-solved as $\eta_\beta^\mathrm{INC} = 0.52$.
- [SHARP-IB] — Original SHARP specification on InfiniBand; hardware-offloaded reduction tree.
- [TH-ULTRA] — Broadcom Tomahawk Ultra (2025); first commodity-Ethernet INC ASIC and HW A2A.

Explainer cross-references used throughout:

- `documentation/modeling/collectives/00_summary.md` — symbol table and per-algorithm $(n_\alpha, n_\beta)$ cheatsheet.
- `documentation/modeling/collectives/01_collective_algorithms.md` — topology-free α-β derivations for the seven primitives.
- `documentation/modeling/collectives/02_topology_mapping.md` — star / torus / mesh per-topology derivations and $N = 512$ ideal comparison.
- `documentation/modeling/collectives/03_hierarchical_topologies.md` — multi-tier Clos / fat-tree composition, NVL72 SuperPOD case study.
- `documentation/modeling/collectives/04_in_network_collectives.md` — NVLS / Quantum SHARP / Spectrum-X SHARP / Tomahawk Ultra mechanics, scale-up vs scale-out, AR-only BW-eff doubling.
- `documentation/modeling/collectives/05_contention_and_congestion.md` — $(\eta_\alpha, \eta_\beta)$ derivation, per-tier calibration, realistic $N = 512$ re-run.
