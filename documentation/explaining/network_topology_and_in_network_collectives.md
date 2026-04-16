# Network Topology and In-Network Collectives: How the Fabric Shapes α, $N_{\text{hops}}$, and the Collective Cost Model

**Author:** Yue Lu
**Date:** April 2026

**Keywords:**
LLM inference, NVLink, NVSwitch, SHARP, in-network reduction, collective communication, ring all-reduce, tree all-reduce, fat-tree, dragonfly, wormhole routing, cut-through, α-β model, topology

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Motivation](#1-motivation)
- [2. The α-β Model, Rigorously](#2-the---model-rigorously)
- [3. Topology Primer](#3-topology-primer)
  - [3.1 Point-to-Point (Direct)](#31-point-to-point-direct)
  - [3.2 Ring](#32-ring)
  - [3.3 Star (Single Switch)](#33-star-single-switch)
  - [3.4 Fat-Tree](#34-fat-tree)
  - [3.5 Dragonfly](#35-dragonfly)
  - [3.6 Torus / Mesh](#36-torus--mesh)
- [4. Decomposing α: Endpoint vs Switch](#4-decomposing---endpoint-vs-switch)
- [5. Why β is Preserved by Wormhole/Cut-Through Routing](#5-why--is-preserved-by-wormholecut-through-routing)
- [6. Collective Algorithms and $N_{\text{hops}}$](#6-collective-algorithms-and-n_texthops)
  - [6.1 Ring All-Reduce](#61-ring-all-reduce)
  - [6.2 Tree All-Reduce](#62-tree-all-reduce)
  - [6.3 Recursive Halving-Doubling](#63-recursive-halving-doubling)
  - [6.4 All-to-All (for EP)](#64-all-to-all-for-ep)
- [7. In-Network Collectives](#7-in-network-collectives)
  - [7.1 What They Are](#71-what-they-are)
  - [7.2 NVLink SHARP (SHARPv3) and NVSwitch Multicast/Reduce](#72-nvlink-sharp-sharpv3-and-nvswitch-multicastreduce)
  - [7.3 Mellanox/Quantum SHARP for InfiniBand](#73-mellanoxquantum-sharp-for-infiniband)
  - [7.4 The $N_{\text{hops}}$ Collapse](#74-the-n_texthops-collapse)
  - [7.5 Limits and Caveats](#75-limits-and-caveats)
- [8. Implications for the TPOT Model](#8-implications-for-the-tpot-model)
- [9. Open Questions / Follow-Ups](#9-open-questions--follow-ups)
- [References](#references)

---

<div style="page-break-before: always;"></div>

# 1. Motivation

The TPOT model developed in `tpot.md` and the sensitivity analysis in `io_bandwidth_scaling.md` treat each parallelism domain (TP, EP, SP, PP) as characterized by a single $(\alpha, B_n)$ pair plus a collective algorithm multiplier $N_{\text{hops}}$. This lumped representation is correct to first order but hides two architecturally important facts:

1. **α is not one number** — it decomposes into an endpoint-side cost (software, driver, NIC engine) and a fabric-side cost (switch cut-through, link propagation). The two scale very differently with topology and with collective algorithm choice.

2. **$N_{\text{hops}}$ is not a property of the topology alone** — it is a property of the collective *algorithm*, which the topology enables or constrains. In-network collectives (SHARP, NVLink multicast/reduce) can collapse $N_{\text{hops}}$ from $O(P)$ or $O(\log P)$ to $O(1)$, with dramatic effect on the $\alpha$-term contribution.

This note unpacks both and gives a vocabulary for reasoning about topology choices in a way that plugs back into the TPOT sensitivity framework.

---

# 2. The α-β Model, Rigorously

The Hockney α-β model gives the time to deliver a point-to-point message of $m$ bytes across the fabric as:
$$
t_{p2p}(m) \;=\; \alpha \;+\; \beta \cdot m \;=\; \alpha \;+\; \frac{m}{B_n}
$$

The cost of a **collective** operation on $P$ endpoints is the sum over algorithmic steps $N_{\text{hops}}$, each of which is one point-to-point transfer of some (typically sub-message) payload:
$$
t_{\text{coll}}(m, P) \;=\; N_{\text{hops}}(P) \cdot \alpha \;+\; \gamma(P) \cdot \beta \cdot m
$$

where $\gamma(P)$ is an algorithm-dependent bandwidth coefficient (e.g., $2(P-1)/P$ for ring all-reduce, $\log P$ for tree all-reduce on the bandwidth term in some implementations). The α contribution scales with the **number of logical rounds**, not the number of bytes.

The key observation: for *small messages* (decode, $B=1$), the $\alpha N_{\text{hops}}$ term dominates. For *large messages* (prefill, high $B$), the $\gamma \beta m$ term dominates. This is the same message-size crossover we saw in `io_bandwidth_scaling.md §8`, now traceable to the collective algorithm.

---

# 3. Topology Primer

Topology is the graph structure of the fabric — how endpoints are physically wired. Four properties matter:

- **Diameter** $D$: longest shortest-path between any two endpoints (in hops)
- **Bisection bandwidth**: aggregate BW across a worst-case partition cut
- **Radix**: ports per switch
- **Scalability**: how cost scales with $P$

## 3.1 Point-to-Point (Direct)

$P = 2$, $D = 0$ switch hops. No switch, no arbitration. Baseline $\alpha = \alpha_{\text{endpoint}} + \alpha_{\text{link}}$. Not useful beyond 2 endpoints.

## 3.2 Ring

Each endpoint connects to two neighbors; $D = P/2$. No switches (or trivial 1-port switches). Common in older NVLink generations and as a logical overlay on other fabrics.

- **Collective behavior**: naturally suits ring algorithms with $N_{\text{hops}} = 2(P-1)$ for all-reduce.
- **α-contribution**: scales linearly with $P$. Bad for small-message collectives at large $P$.

## 3.3 Star (Single Switch)

All $P$ endpoints connect to a central high-radix switch; $D = 2$ link hops (endpoint → switch → endpoint). This is the NVLink+NVSwitch topology.

- **α-contribution per point-to-point hop**: $\alpha_{\text{endpoint}} + 2 \alpha_{\text{link}} + \alpha_{\text{switch}}$. The switch adds one cut-through delay (~100 ns) and one extra link propagation — typically 15–30% on top of a ~1 μs endpoint-dominated α.
- **Logical topology flexibility**: a star can emulate any logical topology (ring, tree, all-to-all) in hardware, because every endpoint is one switch hop from every other. $N_{\text{hops}}$ is then set by the algorithm, not the wiring.
- **Scale limit**: fundamentally bounded by switch radix. NVSwitch Gen4 gives 72 GPUs in a single scale-up domain; beyond that you need a multi-tier fabric.

## 3.4 Fat-Tree

Hierarchical two-level (or three-level) tree with bandwidth doubled at each level toward the root to achieve full bisection. Standard InfiniBand/Ethernet scale-out topology.

- **α-contribution**: $D = 2 \log_k P$ switch hops for a $k$-ary tree. Each hop adds a switch cut-through. For a 2-level fat-tree, $D = 3$ switch hops; for 3-level, $D = 5$.
- **Collective behavior**: supports tree-based collectives natively; also supports in-network reduction (see §7) because the root or spine switch sees all traffic.

## 3.5 Dragonfly

Groups of routers fully connected within a group, with sparse global links between groups. Used in Slingshot (HPE), Rockport, some Cray systems.

- **α-contribution**: $D = 3$ on average (local-global-local), but with adaptive routing the effective hop count varies.
- **Bandwidth asymmetry**: intra-group full bandwidth; inter-group ~1/10 the bandwidth. Looks two-tier from the TPOT model's perspective — maps naturally onto the scale-up/scale-out split in `io_bandwidth_scaling.md §6`.

## 3.6 Torus / Mesh

$k$-dimensional wrap-around lattice. $D = k \cdot P^{1/k} / 2$. Used in TPU pods (2D/3D torus) and older supercomputers.

- **α-contribution**: $D$ grows as $P^{1/k}$ — sub-linear but not logarithmic.
- **Collective behavior**: excellent for nearest-neighbor; mediocre for all-to-all without sophisticated routing. TPU pods rely on this heavily for DP gradient reductions.

---

# 4. Decomposing α: Endpoint vs Switch

Write $\alpha$ as a sum:
$$
\alpha \;=\; \alpha_{\text{endpoint}} \;+\; N_{\text{sw}} \cdot \alpha_{\text{switch}} \;+\; N_{\text{link}} \cdot t_{\text{prop}}
$$

where $N_{\text{sw}}$ is the number of switch cut-throughs on the path and $N_{\text{link}}$ is the number of physical link traversals. Typical values for NVLink Gen5 / NVSwitch Gen4:

| Component | Typical value | Notes |
|---|---|---|
| $\alpha_{\text{endpoint}}$ | ~800 ns | CUDA launch, NCCL scheduling, NIC engine, dominant for software collectives |
| $\alpha_{\text{switch}}$ | ~100 ns | Per NVSwitch cut-through |
| $t_{\text{prop}}$ (copper, 1 m) | ~5 ns/m | Speed of light in copper ≈ 2/3 $c$ |
| $t_{\text{prop}}$ (photonic) | ~5 ns/m | Similar — photonics doesn't save propagation time, it saves energy and distance scaling |

For InfiniBand / Ethernet RoCE:

| Component | Typical value | Notes |
|---|---|---|
| $\alpha_{\text{endpoint}}$ | ~1.5–3 μs | Kernel bypass RDMA; without kernel bypass, ~10–50 μs |
| $\alpha_{\text{switch}}$ | ~200–400 ns | Per IB switch cut-through |
| $t_{\text{prop}}$ | ~5 ns/m | Same physics |

**Key insight.** For short-distance scale-up fabrics, $\alpha_{\text{endpoint}}$ dominates — you cannot make α much smaller by eliminating switches, because software is already the floor. For long-distance scale-out, switch hops and propagation both matter at the 100s-of-ns to μs scale. The TPOT model's single α per fabric is correct only if the deployment operates consistently on the same hop pattern; multi-tier fabrics with variable hop counts require a path-distribution-aware α.

---

# 5. Why β is Preserved by Wormhole/Cut-Through Routing

Under **store-and-forward** routing (old Ethernet), a switch receives the entire packet before forwarding. Total transfer time through $N$ switches is $(N+1) \cdot (m / B_n)$ — bandwidth effectively divided by $N+1$.

Under **wormhole** or **cut-through** routing (modern NVSwitch, IB, Ethernet), the switch starts forwarding as soon as the header is decoded (typically ~10–30 ns after the first flit arrives). Flits pipeline through the fabric. Total transfer time is $\alpha + m / B_n$ where α includes the per-switch cut-through delay and β is set by the slowest link on the path.

**Consequence.** As long as every link on the path has the same per-port bandwidth $B_n$, the β term is independent of $N_{\text{sw}}$. The only way β degrades is:

1. **Rate mismatch** (e.g., GPU-switch link at 900 GB/s, switch-switch spine at 600 GB/s) — effective β set by the slowest link.
2. **Head-of-line blocking** or **queue contention** under load — effective β reduced by utilization-dependent factor. This is a congestion effect, usually modeled as a separate multiplier on β.
3. **Adaptive routing on dragonfly/torus** — messages may take longer detours under congestion, but β per-path remains fixed.

For the TPOT model, the clean interpretation is: **β = 1/B_n of the narrowest link on the expected path**, regardless of hop count.

---

# 6. Collective Algorithms and $N_{\text{hops}}$

## 6.1 Ring All-Reduce

Standard for bandwidth-optimal all-reduce. $P$ endpoints in a logical ring; each sends $1/P$ of its data per step.

- $N_{\text{hops}} = 2(P - 1)$ (half for reduce-scatter, half for all-gather)
- Bandwidth term: $\gamma = 2(P-1)/P \to 2$ as $P \to \infty$
- Optimal for **large messages** — the bandwidth term dominates and is near-optimal
- Bad for **small messages** — the α term scales linearly with $P$

## 6.2 Tree All-Reduce

Logical binary tree; data flows to root, gets reduced, flows back.

- $N_{\text{hops}} = 2 \log_2 P$
- Bandwidth term: $\gamma = \log P$ — suboptimal for large messages (each byte traverses $\log P$ links)
- Optimal for **small messages** — the α term scales logarithmically with $P$
- This is why NCCL switches between ring and tree based on message size

## 6.3 Recursive Halving-Doubling

$N_{\text{hops}} = 2 \log_2 P$ for all-reduce; bandwidth term approaches optimal for power-of-2 $P$. The algorithm of choice when $P$ is a power of 2 and the topology supports it (butterfly, hypercube, or any fully-connected fabric).

## 6.4 All-to-All (for EP)

Every endpoint sends a distinct message to every other endpoint. Used in MoE expert-parallelism routing.

- Direct algorithm: $N_{\text{hops}} = P - 1$, each a point-to-point
- On a star topology with a single switch: all $P$ messages can be serialized through the switch, bounded by switch bisection bandwidth; effectively $N_{\text{hops}} = 1$ wall-clock round if the switch has enough ports. This is why NVSwitch is a big deal for MoE.
- On fat-tree: $N_{\text{hops}}$ depends on the pair-matching but scales as $O(\log P)$ rounds with optimal scheduling.

---

# 7. In-Network Collectives

## 7.1 What They Are

Conventional collectives are software algorithms running on endpoints — switches are passive forwarding elements. **In-network collectives** move the reduction (sum, max, min, etc.) into the switch fabric itself: as data flits pass through a switch, the switch reduces them on the fly, emitting one output to the next level. The endpoint never sees intermediate values.

This turns a collective into approximately a single "collective hop" — regardless of how many endpoints participate.

## 7.2 NVLink SHARP (SHARPv3) and NVSwitch Multicast/Reduce

NVSwitch Gen3+ supports hardware multicast and hardware all-reduce. Key features:

- **Multicast**: a single write from one GPU is replicated by the switch to many destination GPUs in one switch-local operation. Reduces broadcast from $O(P)$ to $O(1)$ wall-clock rounds.
- **All-reduce (SHARPv3, NVSwitch Gen4)**: the switch performs in-fabric reduction of contributions from all GPUs and multicasts the result back. One round-trip total: $N_{\text{hops}} \approx 2$ regardless of $P$ (up to switch radix, typically 72 GPUs per NVSwitch domain today).

**Implication for α**: the α contribution of a TP all-reduce within a scale-up domain goes from $2(TP-1) \cdot \alpha$ (ring) or $2 \log_2(TP) \cdot \alpha$ (tree) to $\approx 2\alpha$. At $TP = 8$: ring α term is $14\alpha$, SHARP is $2\alpha$ — a 7× reduction in the latency term.

## 7.3 Mellanox/Quantum SHARP for InfiniBand

Analogous idea at the scale-out layer. IB switches (Quantum-2, Quantum-X800) support hardware reduction trees ("aggregation trees") across the fabric. A tree with $k$ levels does all-reduce in $2k$ logical rounds, but each round is a hardware switch operation rather than a round-trip between endpoints.

**Scaling**: SHARP scales collectives to thousands of endpoints at approximately constant latency (a few μs), by moving the reduction into the switch silicon.

## 7.4 The $N_{\text{hops}}$ Collapse

In the TPOT model:

| Algorithm | $N_{\text{hops}}$ at $P = 8$ | $N_{\text{hops}}$ at $P = 64$ | $N_{\text{hops}}$ at $P = 512$ |
|---|---|---|---|
| Ring all-reduce | 14 | 126 | 1022 |
| Tree all-reduce | 6 | 12 | 18 |
| SHARP / in-network | 2 | 2 | 2 (within one fabric tier) |

Plugging into $t_{\text{coll}} = N_{\text{hops}} \alpha + \gamma \beta m$ shows the α-term cost of a collective drops from $O(P) \alpha$ or $O(\log P) \alpha$ to $O(1) \alpha$. For small decode messages where α dominates, **this is effectively a collective-time reduction proportional to $N_{\text{hops}}^{\text{software}} / N_{\text{hops}}^{\text{SHARP}}$.**

## 7.5 Limits and Caveats

- **Scope**: in-network collectives work only within a single switch domain (or hierarchically-connected SHARP-enabled switches). Cross-domain collectives fall back to software.
- **Reducible types**: hardware reductions are limited to simple operations (sum, max, min, bit-and/or). Complex reductions (e.g., topk, softmax normalization) must still be software.
- **Precision**: hardware reducers may operate at lower precision than software (often BF16 or FP16 with accumulators). User must check that fits the model's numerical tolerance.
- **Message alignment**: SHARP operations typically require messages aligned to specific sizes; very small messages may fall through to software.
- **Bandwidth-limited regime**: SHARP collapses the α term, but the β term is still bounded by switch bisection bandwidth. For large messages, SHARP's benefit over a well-tuned ring all-reduce shrinks.

---

# 8. Implications for the TPOT Model

The decomposition suggests several refinements to the sensitivity analysis in `io_bandwidth_scaling.md`:

1. **α becomes (at minimum) a two-component vector**: $\alpha_{\text{endpoint}}$ (set by software, NIC, driver) and $\alpha_{\text{fabric}} = N_{\text{sw}} \alpha_{\text{switch}} + N_{\text{link}} t_{\text{prop}}$ (set by topology and distance). Provisioning decisions often trade these off — e.g., replacing copper with photonics lowers $t_{\text{prop}}$ at long distances but doesn't help $\alpha_{\text{endpoint}}$.

2. **$N_{\text{hops}}$ is an algorithmic choice, not a fabric property**. Moving from ring to SHARP changes $N_{\text{hops}}$ from $O(TP)$ to $O(1)$. In the four-way elasticity of §6.2 of the other note, this shows up as the $f_A^{up}$ share dropping substantially — shifting the system away from α-dominated to $B_n$-dominated (or $B_m$-dominated) regime.

3. **Star topologies with in-network reduce can make the scale-up fabric "feel like" a single giant GPU** from the collective-cost perspective, within the switch radix. Beyond the radix, you pay a second α for the inter-domain hop — which is effectively the scale-up-to-scale-out boundary discussed in §6 of `io_bandwidth_scaling.md`.

4. **Dragonfly and other two-tier topologies naturally map to the scale-up/scale-out split**, but with an intermediate "inter-group" tier that may not fit cleanly into either bucket. A three-tier α decomposition ($\alpha^{local}, \alpha^{group}, \alpha^{global}$) might be needed for large deployments.

A natural next step would be to extend the TPOT model to expose $N_{\text{hops}}^{up}$ as an algorithm parameter (ring / tree / SHARP) and $\alpha^{up}$ as the endpoint+switch decomposition, then re-run the elasticity analysis under each combination.

---

# 9. Open Questions / Follow-Ups

- **Calibrating $\alpha_{\text{endpoint}}$ vs $\alpha_{\text{switch}}$ from public specs**: what NCCL / NVIDIA documentation gives the software overhead floor vs switch cut-through? A calibration micro-benchmark plan would be useful.
- **When does SHARP actually pay off in decode?** For prefill, messages are large and the bandwidth term dominates — SHARP's α collapse is less impactful. For decode at $B=1$, the α collapse is maximal. Quantifying the SHARP-on vs SHARP-off TPOT delta as a function of $B$ would be a good follow-up.
- **Multi-tier α model**: does the dragonfly intra-group / inter-group split warrant a three-tier $(\alpha^{local}, \alpha^{group}, \alpha^{global})$ in the TPOT framework, or is a two-tier scale-up/scale-out adequate?
- **Congestion-aware β**: the wormhole-preserves-β argument in §5 breaks under load. How much does peak β degrade under representative LLM inference collective patterns? Is there a tractable closed-form model?
- **Photonic fabrics**: optical circuit-switched scale-up (Tesla Dojo, Lightmatter) changes the α vs β trade-off by making link bandwidth scale with wavelength rather than pins. How should the TPOT model represent that?
- **Co-design of TP rank and fabric radix**: given SHARP collapses $N_{\text{hops}}^{up}$ to $O(1)$ within a switch domain, the TP choice becomes partly a fabric-scope choice. When does TP = radix minimize TPOT, and when is sub-radix TP (with multiple TP groups per switch) better?

---

# References

- [HOCKNEY] Hockney. *The communication challenge for MPP: Intel Paragon and Meiko CS-2.* Parallel Computing, 1994.
- [NCCL] NVIDIA Collective Communications Library. https://docs.nvidia.com/deeplearning/nccl
- [NVSWITCH] NVIDIA NVSwitch Architecture Whitepapers (Gen3, Gen4).
- [SHARP] Graham et al. *Scalable Hierarchical Aggregation Protocol (SHARP): A Hardware Architecture for Efficient Data Reduction.* COMHPC 2016.
- [DRAGONFLY] Kim, Dally, Scott, Abts. *Technology-Driven, Highly-Scalable Dragonfly Topology.* ISCA 2008.
- [FAT-TREE] Leiserson. *Fat-trees: Universal Networks for Hardware-Efficient Supercomputing.* IEEE Trans. Computers, 1985.
- [WORMHOLE] Dally, Seitz. *Deadlock-free message routing in multiprocessor interconnection networks.* IEEE Trans. Computers, 1987.
- [RING-ALLREDUCE] Patarasuk, Yuan. *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations.* JPDC 2009.
- [io_bandwidth_scaling.md] `documentation/explaining/io_bandwidth_scaling.md` — sensitivity analysis this note extends
- [pipeline_bubble.md] `documentation/explaining/pipeline_bubble.md` — PP bubble and scale-out motivation
- [tpot.md] `documentation/modeling/tpot.md` — full TPOT derivation
