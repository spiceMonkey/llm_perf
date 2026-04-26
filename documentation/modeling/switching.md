# Scale-Up Network Switch Model

**Modeling Effective Bandwidth and Latency for Scale-Up Fabrics**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
scale-up network, NVSwitch, NVLink, UALink, ESUN, crossbar, switch fabric, aggregate bandwidth, radix, head-of-line blocking, VOQ, efficiency factor, single-direction bandwidth, Pareto frontier, collective communication, multi-tier fabric, fabric chain, hierarchical scale-up

---

A scale-up network feeds TP, EP, SP, and rack-local PP collectives. The decode and prefill models treat the per-role bandwidth $BW_{role}$ and startup latency $\alpha_{role}$ as constants. That abstraction holds only while each switching tier along the collective's path has enough silicon bandwidth budget to feed every port at its nominal rate, and while the collective's group size is known so its fabric chain can be costed. This document makes that abstraction honest in three stages. §2–6 develop the single-tier crossbar building block: as port count $P$ at a tier grows past the single-chip capacity frontier, the effective per-port bandwidth must decrease, and an efficiency factor $\eta$ captures the protocol/fabric losses that show up even in the ideal case. §7 composes tiers into named fabrics and fabric chains under a topology-agnostic α-sum / BW-min rule — a collective that outgrows its innermost tier escalates into outer tiers (or across fabric boundaries into a scale-out fabric). §8 and §9 specialize the per-tier cost for k-D torus and canonical dragonfly tiers, replacing the flat $(\alpha, BW)$ pair with topology-structured formulas; §10 shows how the three topologies compose under a single fabric-chain walk.

**Bandwidth convention.** All bandwidth quantities in this document — $B_{\text{agg}}$, $BW_{\text{nominal}}$, $BW_{\text{eff}}$, $BW_{\text{port}}$ — are **single-direction** (unidirectional) unless explicitly labeled otherwise. This matches the convention in the decode and prefill models and NCCL busbw measurements. Industry datasheets often quote full-duplex (bidirectional) aggregate capacity; we halve those figures.

**Goal:** produce a drop-in replacement for the constant $BW_{role}$ of the form

$$
BW_{\text{eff}}(P) \;=\; \eta \cdot \min\!\bigl(BW_{\text{nominal}},\; B_{\text{agg}} / P\bigr)
$$

that replaces the constant $BW_{role}$ without changing the α–β collective structure. The rest of the doc is: (1) the physics that gives the $B_{\text{agg}} / P$ term, (2) the decomposition of $\eta$ into citable components, and (3) calibration against published benchmarks.

---

<div style="page-break-after: always;"></div>

# Table of Contents

- [1. Scope](#1-scope)
- [2. Physical Primitives](#2-physical-primitives)
  - [2.1 Switch Chip Aggregate Bandwidth](#21-switch-chip-aggregate-bandwidth-b_textagg)
  - [2.2 Radix–Bandwidth Tradeoff](#22-radixbandwidth-tradeoff)
  - [2.3 Single-Hop Transit Latency](#23-single-hop-transit-latency-alpha_textbase)
- [3. Ideal First-Order Model](#3-ideal-first-order-model)
- [4. The Efficiency Factor η](#4-the-efficiency-factor-eta)
  - [4.1 Head-of-Line (HoL) Blocking](#41-head-of-line-hol-blocking--eta_texthol)
  - [4.2 Buffering](#42-buffering--eta_textbuf)
  - [4.3 Protocol Overhead](#43-protocol-overhead--eta_textproto)
  - [4.4 Composite Defaults](#44-composite-defaults)
- [5. Applying the Model](#5-applying-the-model)
  - [5.1 Input Parameters](#51-input-parameters)
  - [5.2 Effective Bandwidth and Latency](#52-effective-bandwidth-and-latency)
  - [5.3 Worked Examples](#53-worked-examples)
- [6. Open Questions](#6-open-questions)
- [7. Fabric Chains](#7-fabric-chains)
  - [7.1 Fabrics and Tier Descriptors](#71-fabrics-and-tier-descriptors)
  - [7.2 Spanning a Collective Across Tiers](#72-spanning-a-collective-across-tiers)
  - [7.3 Example: NVL576 as Ideal vs. Hierarchical](#73-example-nvl576-as-ideal-vs-hierarchical)
- [8. Torus Topology](#8-torus-topology)
  - [8.1 Physical Primitives](#81-physical-primitives)
  - [8.2 1D Reduction to Ring](#82-1d-reduction-to-ring)
  - [8.3 Dim-by-Dim Ring All-Reduce](#83-dim-by-dim-ring-all-reduce)
  - [8.4 All-Gather and Reduce-Scatter](#84-all-gather-and-reduce-scatter)
  - [8.5 All-to-All — Bisection-Limited](#85-all-to-all--bisection-limited)
  - [8.6 Worked Example — TPU v5p pod](#86-worked-example--tpu-v5p-pod)
  - [8.7 Alternative Algorithms — Swing and HammingMesh](#87-alternative-algorithms--swing-and-hammingmesh)
  - [8.8 Efficiency Note](#88-efficiency-note)
- [9. Dragonfly Topology](#9-dragonfly-topology)
  - [9.1 Parameters](#91-parameters)
  - [9.2 Three-Tier α-β Decomposition](#92-three-tier-α-β-decomposition)
  - [9.3 Adaptive Routing and the Valiant Fallback](#93-adaptive-routing-and-the-valiant-fallback)
  - [9.4 Hierarchical Collective Cost](#94-hierarchical-collective-cost)
  - [9.5 Worked Example — HPE Slingshot 11](#95-worked-example--hpe-slingshot-11)
  - [9.6 Open Questions](#96-open-questions)
- [10. Unified Fabric-Chain Dispatch](#10-unified-fabric-chain-dispatch)
- [Appendix A. Collective Bandwidth Multiplier](#appendix-a-collective-bandwidth-multiplier)
- [Appendix B. Symbol Glossary](#appendix-b-symbol-glossary)
- [References](#references)

---

## 1. Scope

**In scope.** Scale-up fabrics composed of one or more switching tiers. §2–6 model a single non-blocking tier — one monolithic switching chip between every pair of participating GPUs (NVLink/NVSwitch within a rack, UALink-class leaf, Tomahawk Ultra scale-up Ethernet). §7 extends to multi-tier fabrics and fabric chains: a collective's group size dictates how many tiers it must cross, each contributing its α to the startup cost and capping the sustained BW. Scale-out (inter-rack InfiniBand / Ethernet) appears as a second named fabric in the chain rather than a tier of the first, so the same α-sum / BW-min rule handles NVL72, NVL576-hierarchical, and scale-out-only deployments uniformly.

**Out of scope.**

- Topology-explicit collective formulas (e.g. ring vs. tree vs. halving-doubling). The α–β abstraction in this doc is topology-agnostic; the collective's rank-count dependence is already captured in the communication model's ring/tree variants (decode.md §5).
- Adversarial traffic patterns (targeted incast, pathological many-to-one). LLM collectives are structured (ring all-reduce, balanced all-to-all); we assume non-adversarial mixing throughout.

**Why this abstraction is enough for inference.** LLM inference collectives are:
- Structured (ring/tree all-reduce for TP/EP-aggregate, balanced all-to-all for MoE token routing).
- Known message sizes (set by $H$, batch, sequence).
- Mostly rack-local in the partitions we care about (TP ≤ 16, EP ≤ 16, PP stays inside the scale-up domain unless deliberately tiered).

So "one efficiency factor times the α–β model, with BW decreasing as the switch exhausts its aggregate budget" captures the first-order behavior without committing to a specific topology.

---

## 2. Physical primitives

### 2.1 Switch chip aggregate bandwidth ($B_{\text{agg}}$)

A switching chip has a fixed aggregate I/O capacity set by its SerDes complement: (number of lanes) × (per-lane rate). Industry datasheets quote this as a full-duplex (bidirectional) figure; we define $B_{\text{agg}}$ as the **single-direction** half. This is a **hard silicon budget** — you cannot get more total one-way I/O out of one chip than half its SerDes can drive.

For multi-chip fabrics (e.g., NVL72 uses 9 NVSwitch chips), the system-level single-direction aggregate is $N_{\text{chips}} \times B_{\text{agg,chip}}$.

Calibration points (per chip, single-direction):

| Generation | $B_{\text{agg}}$ (single-dir) | Full-duplex spec | Radix × per-port (single-dir) | Source |
|---|---|---|---|---|
| NVSwitch Gen3 (Blackwell, 2024) | 7.2 TB/s (57.6 Tbps) | 14.4 TB/s | 72 ports × 100 GB/s | [GB200-NV] |
| NVSwitch Gen4 (Rubin, 2026+) | 14.4 TB/s (115.2 Tbps) | 28.8 TB/s | 72 ports × 200 GB/s | [RUBIN-SA] |
| Broadcom Tomahawk Ultra (2025) | 25.6 Tbps | 51.2 Tbps | 64 × 800G or 128 × 400G | [TH-ULTRA] |
| UALink 1.0 switch *(spec, 2025)* | switch-dependent | — | up to 1,024 ports × 100 GB/s | [UALINK-SPEC] |
| Upscale AI SkyHammer *(announced, Q4 2026)* | TBD | — | 1,024 ports (UALink 1.0) | [SKYHAMMER] |

**Broadcom Tomahawk Ultra** is notable as the first Ethernet switch ASIC explicitly targeting scale-up: it adds in-network collectives (INC), credit-based flow control (CBFC), and link-layer retry (LLR) — features previously exclusive to NVSwitch and IB. At 25.6 Tbps single-direction, it matches NVSwitch Gen4 in raw silicon budget but with higher radix options (up to 128 × 400G).

**UALink 1.0** (released April 2025) defines an open scale-up interconnect standard: 200 GT/s per lane, 4-lane "stations" at 800 Gbps bidirectional (100 GB/s single-direction) per port, with UALink Switches (ULS) supporting up to 1,024 accelerator endpoints. Consortium members include AMD, Broadcom, Cisco, Google, HPE, Intel, Meta, and Microsoft. **Upscale AI** is building the first UALink switch ASIC ("SkyHammer"), targeting 1,024-port radix in Q4 2026 [SKYHAMMER].

**ESUN** (Ethernet for Scale-Up Networking) is a complementary OCP protocol workstream (2025, 175+ member companies) defining a compact 4-byte header that replaces the traditional IP/UDP stack, plus CBFC and LLR for lossless delivery [ESUN-OCP]. ESUN is a framing/protocol standard, not a chip — Tomahawk Ultra and future UALink switches are expected to implement it.

**Trend.** $B_{\text{agg}}$ roughly doubles per SerDes generation (~2–3 years). The landscape is broadening: NVSwitch remains the highest per-port BW (proprietary), while UALink/ESUN-based switches trade per-port rate for much higher radix (1,024 vs. 72) in an open ecosystem.

### 2.2 Radix–bandwidth tradeoff

For a single chip with single-direction aggregate budget $B_{\text{agg}}$, the single-direction per-port rate is

$$
BW_{\text{port}}(P) \;=\; B_{\text{agg}} / P
$$

where $P$ is the number of ports. This is a 1:1 tradeoff: **double the radix, halve the per-port rate, on the same silicon**. NVSwitch Gen4 illustrates this: the same chip (14.4 TB/s single-direction) could configure as 72 × 200 GB/s or, hypothetically, 144 × 100 GB/s if the radix were extended.

The useful regime for a scale-up fabric is where the target per-port rate ($BW_{\text{nominal}}$, single-direction) is achievable with the available radix:

$$
P \;\le\; P_{\max}(B_{\text{agg}},\, BW_{\text{nominal}}) \;=\; B_{\text{agg}} / BW_{\text{nominal}}
$$

**NVL72 worked example.** Target per-GPU bandwidth: $BW_{\text{nominal}} = 900$ GB/s (single-direction). Each NVSwitch Gen3 chip provides $B_{\text{agg}} = 7.2$ TB/s (single-direction) across 72 ports → 100 GB/s per port per direction. A single GPU aggregates 9 NVSwitch chips: $9 \times 100 = 900$ GB/s, exactly meeting $BW_{\text{nominal}}$. The system-level single-direction aggregate is $9 \times 7.2 = 64.8$ TB/s, giving $P^* = 64{,}800 / 900 = 72$ — exactly the NVL72 GPU count. Beyond $P = 72$ the per-GPU rate falls as $64.8\text{ TB/s} / P$, entering the capacity-limited regime.

### 2.3 Single-hop transit latency ($\alpha_{\text{base}}$)

A non-blocking crossbar hop has **rank-count-independent** latency:

$$
\alpha_{\text{port}}(P) \;\approx\; \alpha_{\text{base}}
$$

This is because traffic hops the switching silicon once regardless of $P$. Reported values:

| Fabric | $\alpha_{\text{base}}$ | Source |
|---|---|---|
| NVLink/NVSwitch (port-to-port) | undisclosed; quoted alongside ~240 ns reference for NDR IB | [NVS-FM] |
| UALink 1.0 (scale-up, Ethernet-class) | 100–150 ns | [UALINK-SPEC] |
| Broadcom Tomahawk Ultra (scale-up Ethernet) | 250 ns | [TH-ULTRA] |
| InfiniBand NDR 400G *(scale-out reference)* | 240 ns port-to-port hop | [GTC25-IB] |

A true crossbar has one hop; "internally-structured" switch silicon (Clos, banyan) adds log-P transit stages inside the chip but is still reported as single-hop to the user. Treat as a constant within the scope of this doc.

---

## 3. Ideal first-order model

Combining §2.2 and §2.3:

$$
\boxed{\; BW_{\text{eff}}(P) \;=\; \eta(P,\text{collective}) \cdot \min\!\bigl(BW_{\text{nominal}},\; B_{\text{agg}} / P\bigr) \;}
$$

$$
\alpha_{\text{eff}}(P) \;\approx\; \alpha_{\text{base}}
$$

All quantities are single-direction. Two regimes:

- **Radix-limited** ($P \le B_{\text{agg}} / BW_{\text{nominal}}$): the fabric can feed every port at its nominal single-direction rate. $BW_{\text{eff}} = \eta \cdot BW_{\text{nominal}}$, flat in $P$.
- **Capacity-limited** ($P > B_{\text{agg}} / BW_{\text{nominal}}$): aggregate budget is the bottleneck. $BW_{\text{eff}} = \eta \cdot B_{\text{agg}} / P$, falls as $1/P$.

The crossover $P^* = B_{\text{agg}} / BW_{\text{nominal}}$ is the **port count at which single-layer scale-up exhausts its silicon budget**. Past $P^*$, the only ways to grow the collective are (a) accept per-port rate degradation, or (b) tier the network — the fabric-chain extension in §7.

---

## 4. The efficiency factor $\eta$

$\eta$ absorbs everything between the physical chip budget and the bandwidth actually delivered to a collective. It decomposes as a product of independent (to first order) terms:

$$
\eta \;=\; \eta_{\text{HoL}} \cdot \eta_{\text{buf}} \cdot \eta_{\text{proto}}
$$

The collective algorithmic bandwidth multiplier (e.g., ring all-reduce's $2(P-1)/P$) is intentionally excluded — it is already accounted for in the communication model's message-size and step-count formulas (see Appendix A).

### 4.1 Head-of-line (HoL) blocking — $\eta_{\text{HoL}}$

**Classical bound [KAROL87]:** for an input-queued crossbar with a single FIFO queue per input and uniform random destinations, throughput saturates at

$$
\eta_{\text{HoL}}^{\text{FIFO}} \;=\; 2 - \sqrt{2} \;\approx\; 0.586 \quad \text{as } P \to \infty
$$

This is the famous 58.6% result. A packet at the head of an input queue blocks all behind it whenever its destination output port is busy, even when the trailing packets target idle outputs.

**Modern mitigation — VOQ.** Virtual output queueing maintains one per-destination queue at each input, eliminating HoL blocking under a suitable scheduling algorithm (iSLIP, Maximum Weight Matching). With VOQ, $\eta_{\text{HoL}}^{\text{VOQ}} \to 1.0$ for uniform admissible traffic [MCKEOWN99], [VOQ-WIKI].

**Modern switch silicon (NVSwitch, IB switches)** uses VOQ + shared-buffer architectures + credit-based flow control. In the idealized model, $\eta_{\text{HoL}} = 1.0$ for well-behaved traffic. In practice, LLM collectives (ring, all-to-all balanced) are admissible, so VOQ delivers near-100% throughput.

**Residual loss.** Measured degradation from imperfect scheduling / incomplete VOQ coverage: typically 2–5% for modern switches under non-adversarial traffic. **Default $\eta_{\text{HoL}} \approx 0.97$**.

### 4.2 Buffering — $\eta_{\text{buf}}$

Shared-buffer switch architectures (NVSwitch, IB switches) pool a global on-chip buffer across ports. Under burst, some ports transiently undersubscribe while others overflow their fair share; with credit-based flow control (NVLink credits, IB CBFC), drops are avoided at the cost of brief BW underutilization at the overflowing port.

No clean analytical bound exists for this; empirically it's 2–5% for structured collective traffic under modern shared-buffer switches. **Default $\eta_{\text{buf}} \approx 0.97$** for sustained collectives.

### 4.3 Protocol overhead — $\eta_{\text{proto}}$

Wire-level framing, credit/ACK messaging, and host-side software path (RDMA verbs, NCCL chunking):

- **Framing.** InfiniBand ~2% overhead; Ethernet/RoCE ~3–5% including preamble + IFG.
- **Credits/ACKs.** Link-level credit-based flow control (NVLink, IB) consumes ~1% of link BW for control traffic.
- **NCCL chunking.** `NCCL_P2P_NET_CHUNKSIZE` — the library breaks large messages into chunks and pipelines them; the interstitial per-chunk bookkeeping costs 1–3% depending on chunk size vs. latency-bandwidth product [NCCLX-META].

Total wire-to-useful-payload overhead: **$\eta_{\text{proto}} \approx 0.93–0.97$**. For NVLink (most efficient framing, hardware credit): ~0.97. For RoCE/IB: ~0.93.

### 4.4 Composite defaults

Multiplying $\eta_{\text{HoL}} \cdot \eta_{\text{buf}} \cdot \eta_{\text{proto}}$:

| Collective / regime | Recommended $\eta$ | Notes |
|---|---|---|
| Ring all-reduce over NVLink/NVSwitch, structured | **0.90 ± 0.03** | Matches A800 measured 92–97% busbw of 600 GB/s peak [NCCL-ISSUE-149] and H200 355/450 ≈ 79% at full scale [NCCL-ISSUE-272]. |
| All-to-all (MoE token routing) over NVSwitch | **0.80 ± 0.05** | Balanced-but-wide destination spread stresses buffering + bisection; NCCLX recent gains target this specifically [NCCLX-META]. |
| Ring all-reduce over RoCE/IB 400G | **0.85 ± 0.05** | IB NDR target is ">90% busbw"; real at-scale numbers land around 85–90% for large messages [NVIDIA-SLURM]. |
| All-reduce with SHARP (in-network reduction) | **effective 1.5–2.0×** | Not an efficiency *loss*; SHARP halves the traffic because reduction happens in the switch. Model by replacing $BW_{\text{nominal}} \to 2 BW_{\text{nominal}}$ rather than by $\eta > 1$. [SHARP-NV] reports 1.7× over non-SHARP NCCL at 400 Gb/s. |

**Practical default for scale-up LLM inference modeling:** $\eta = 0.90$ for TP/EP ring collectives, $\eta = 0.80$ for EP all-to-all (MoE gate), $\eta = 1.0$ for SHARP-accelerated all-reduce (with the $2\times$ already absorbed into `BW_nominal`).

**Calibration note.** $\eta$ in our model corresponds to $\text{busbw} / BW_{\text{link}}$ from NCCL benchmarks [NCCL-PERF], with the collective algorithmic multiplier already factored out (see Appendix A). The defaults above are calibrated against published measurements at rack scale ($P \le 72$). At larger scale, Meta's NCCLX paper [NCCLX-META] shows $\eta$ stays roughly flat for large messages but degrades for latency-sensitive small-message regimes — for LLM inference, TP/EP messages are sized $O(H \cdot B)$ which is typically well above the bandwidth-delay product, so the large-message assumption holds.

**Caveat: no empirical $\eta$ vs. $P$ data exists.** Published busbw measurements come from different hardware generations (A800, H100, H200) at small $P$ (typically 8 GPUs), not from the same fabric swept across varying port counts. Within a single non-blocking NVSwitch domain ($P \le 72$), $\eta$ is expected to be roughly $P$-independent since every GPU pair has equal-bandwidth paths. However, at $P \gg 72$ — where single-layer fabrics enter the capacity-limited regime or require higher-radix switch silicon — $\eta$ may degrade due to increased contention, deeper buffering, and longer scheduling paths. The first controlled $\eta$-vs-$P$ measurements at scale are expected from UALink switches (e.g., Upscale AI SkyHammer, 1,024 ports, Q4 2026). Until then, treat these defaults as constant in $P$ with the understanding that they carry increasing uncertainty beyond $P \approx 72$.

---

## 5. Applying the model

### 5.1 Input parameters

The switch model requires four parameters alongside the existing per-domain $BW_{\text{nominal}}$ and $\alpha_{\text{base}}$:

| Parameter | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Aggregate fabric bandwidth | $B_{\text{agg}}$ | TB/s | Single-direction aggregate. For multi-chip fabrics: $N_{\text{chips}} \times B_{\text{agg,chip}}$. |
| Port count | $P$ | — | Number of accelerator ports the switch fabric serves. |
| Efficiency coefficient | $\eta$ | — | Lumped fabric efficiency $\in (0, 1]$. Absorbs HoL residual, buffering, protocol overhead (§4.1–4.3). |
| Latency adder | $\Delta\alpha$ | μs | Additional per-hop latency beyond the base link $\alpha$. Zero for NVLink/NVSwitch; non-zero for fabrics with measurable extra switch-hop latency. |

### 5.2 Effective bandwidth and latency

The effective per-port bandwidth decomposes into two independent factors:

$$
BW_{\text{eff}} \;=\; \underbrace{\eta}_{\text{efficiency}} \;\times\; \underbrace{\min\!\bigl(BW_{\text{nominal}},\; B_{\text{agg}} / P\bigr)}_{\text{capacity cap}}
$$

$$
\alpha_{\text{eff}} \;=\; \alpha_{\text{base}} + \Delta\alpha
$$

The **capacity cap** $\min(BW_{\text{nominal}},\, B_{\text{agg}}/P)$ is the hard physical limit from the switch silicon budget — deterministic, no tuning knobs. The **efficiency** $\eta$ captures the protocol/fabric losses that reduce delivered bandwidth below the physical limit. The existing α–β collective formulas (ring α scales as $P_{\text{role}}-1$, ring BW $= 2(P_{\text{role}}-1)/P_{\text{role}} \cdot BW_{\text{eff}}$) stack cleanly on top.

### 5.3 Worked examples

**NVL72** — 9 NVSwitch Gen3 chips, $BW_{\text{nominal}} = 900$ GB/s, $\eta = 0.90$:

$$
B_{\text{agg}} = 9 \times 7.2 = 64.8 \text{ TB/s}, \quad P = 72
$$
$$
BW_{\text{cap}} = \min(900,\; 64800/72) = 900 \text{ GB/s} \quad \Rightarrow \quad BW_{\text{eff}} = 0.90 \times 900 = 810 \text{ GB/s}
$$

At $P = 72 = P^*$, the capacity cap is not binding; the 10% discount is purely from $\eta$.

**Hypothetical 128-GPU single-layer** on the same Gen3 NVSwitch aggregate:

$$
BW_{\text{cap}} = \min(900,\; 64800/128) = 506.25 \text{ GB/s} \quad \Rightarrow \quad BW_{\text{eff}} = 0.90 \times 506.25 = 455.6 \text{ GB/s}
$$

Now both factors bite: per-port rate dropped to 506 GB/s from the silicon budget, then $\eta$ takes another 10%.

---

## 6. Open questions

1. **Empirical calibration at N > 72.** We rely on published busbw numbers at rack scale (8–72 GPUs). UALink switches (Upscale AI SkyHammer) targeting 1,024-port radix will provide the first calibration data for single-layer fabrics at this scale — expected Q4 2026. Until then, the closest proxy is the Meta 100k-GPU paper, which is tier-aware rather than single-layer.
2. **MoE all-to-all with imbalanced expert loads.** Our $\eta_{\text{alltoall}} = 0.80$ assumes balanced routing. Expert-skew workloads can drop this to 0.5–0.6. Requires a separate extension (routing-aware efficiency) — out of scope here.
3. **Mixed traffic (concurrent TP + EP).** Two collectives sharing a switch contend for aggregate chip BW. Our model treats each collective in isolation; a proper treatment would allocate $B_{\text{agg}}$ across concurrent collectives weighted by message size. For inference this is usually not the bottleneck (TP and EP phases are typically serialized within a layer), but worth flagging.
4. **SHARP-beyond-AR.** NVLS SHARP currently covers all-reduce; all-to-all SHARP is announced for Rubin-gen. When it ships, update §4.5 to include SHARP-accelerated all-to-all with a similar $\eta = 1.0$, $2\times$ capacity treatment.
5. **NVLink vs. UALink/ESUN η divergence.** NVLink uses proprietary credit-based flow control and tight software integration (NCCL), while UALink/ESUN fabrics use Ethernet-derived framing (ESUN 4-byte header) with CBFC/LLR. Protocol overhead ($\eta_{\text{proto}}$) may differ: NVLink ~0.97 vs. UALink/ESUN ~0.93–0.95 (speculative — no measured data yet). Tomahawk Ultra's in-network collectives (INC) could also change the SHARP-equivalent treatment for Ethernet-based scale-up.
6. **Photonic interconnects.** Lightmatter's Passage M1000 claims 114 Tbps total optical bandwidth (57 Tbps single-direction) in a photonic interposer — an order of magnitude above electrical switch chips. If validated, this changes the $B_{\text{agg}}$ ceiling fundamentally and could push $P^*$ into the thousands. Not modeled here; watch for measured $\eta$ data.

---

## 7. Fabric chains

Sections 1–6 assume a single crossbar tier. Real systems beyond one rack add a second (or higher) tier stitching racks together — e.g. a hypothetical NVL576 built from 8× NVL72 racks plus an inter-rack scale-out layer. This section defines the topology-agnostic walk that composes tiers into named fabrics, and named fabrics into a per-collective chain. Tier-level cost is pluggable: crossbar tiers use §3's flat $(\alpha, BW)$ pair, torus tiers use §8's dim-by-dim formulas, and dragonfly tiers use §9's three-tier decomposition. §10 shows how the three compose under one walk. A one-tier, one-fabric crossbar collective is a trivial case of the general chain and produces identical numbers to §3.

**Downstream consumers.** `decode.md §5` and `prefill.md §3.2` do not re-derive collective costs; they call the shipped-primitive formulas catalogued in `collectives.md §3–§6` (ring / DBT AR, ring AG / RS, pairwise A2A on star; dim-decomposed ring and bisection-bound A2A on torus; hierarchical RS → sub-AR → AG; in-network reduction via NVLS / Quantum SHARP / Tomahawk Ultra) with the fabric-chain span quantities from this section, and apply per-tier $(\eta_\alpha, \eta_\beta)$ contention coefficients per `collectives.md §7`. The manual `tuner.ar_algorithm` knob selects star AR between ring and DBT (`collectives.md §3.1`); torus AR ships only as dim-decomposed ring (§8).

### 7.1 Fabrics and tier descriptors

The system model names physical networks as **fabrics**. Each `FabricSpec` is an ordered list of switching tiers, innermost first. A tier $i \in \{0, 1, \ldots\}$ inside a fabric is a triple:

| Symbol | Description | Units |
|---|---|---|
| $P_i$ | Reach at tier $i$ — ranks reachable within this tier from any single rank | — |
| $BW_i$ | Effective per-port bandwidth at tier $i$ (single-direction, post-$\eta$) | GB/s |
| $\alpha_i$ | Per-traversal latency of tier $i$'s switching silicon | μs |

Tier 0 is the innermost (e.g. intra-rack NVSwitch). The cumulative reach at tier $k$ is $\prod_{i=0}^{k} P_i$ ranks. A fabric with $n$ tiers can host a collective reaching up to $\prod_{i=0}^{n-1} P_i$ ranks on its own. For a crossbar tier, $P_i$ is the switch radix; for a torus tier with dims $(D_1, \ldots, D_k)$ it is $\prod_j D_j$; for a dragonfly tier parameterized by $(p, a, h, g)$ it is $p \cdot a \cdot g$. The reach symbol $P_i$ is topology-agnostic — only the per-tier $\alpha_i$ / $BW_i$ interpretation, and the cost formula applied inside the tier, differ (§8–§9).

Each collective (TP, EP, SP, PP) declares an ordered **fabric chain** via `SystemSpec.collective_fabrics[collective] = [fabric_name_0, fabric_name_1, ...]`. A collective first escalates through the tiers of its innermost fabric; when the collective's group size exceeds that fabric's reach, the walk continues into the next fabric in the chain. This lets scale-up (e.g. NVLink) and scale-out (e.g. InfiniBand / Ethernet) be modeled as distinct physical networks with their own $(P_i, BW_i, \alpha_i)$ triples, rather than as tiers of one synthetic fabric.

The single-tier, single-fabric case — one entry in `collective_fabrics`, one tier inside that fabric — reduces to §3's flat $(\alpha, BW)$ pair with $P_0 = N_{\text{devices}}$.

### 7.2 Spanning a collective across tiers

Flatten the collective's fabric chain into a single tier list by concatenating tiers innermost-first across every fabric in the chain (the same ordering `SystemSpec.get_tier_chain` returns). Then, for a collective of group size $G$ (e.g. TP=16 → $G$=16; EP=$N_{\text{exp}}$ → $G$=$N_{\text{exp}}$), identify the smallest tier index $k$ in that flattened list such that

$$
\prod_{i=0}^{k} P_i \;\ge\; G .
$$

The collective crosses tiers $0..k$. Its effective α and BW become

$$
\alpha_{\text{span}}(G) \;=\; \sum_{i=0}^{k} \alpha_i, \qquad
BW_{\text{span}}(G) \;=\; \min_{i \le k} BW_i .
$$

The intuition:

- **α sums** because every tier crossed contributes its switching latency to the per-message startup cost; the critical path touches each tier in series.
- **BW is floored at the narrowest crossed tier** because the inter-tier link is the bottleneck — faster innermost tiers can't speed up traffic that has to funnel through a slower outer tier. When the span crosses from one fabric into the next (e.g. NVLink → IB), the same rule applies at the fabric boundary.

Collectives that fit entirely within the innermost fabric's tier 0 (e.g. TP=8 on an 8-GPU ring inside an NVL72) see only $(\alpha_0, BW_0)$ — identical to today's single-tier model. Collectives that spill into an outer tier — whether in the same fabric or in the next fabric of the chain — pay every tier's α and are bandwidth-limited by the narrowest tier along the walk.

**Topology note.** The α-sum / BW-min rule above is exact for **all-crossbar chains**. When the chain contains torus or dragonfly tiers, each tier's contribution is computed by its own topology-specific primitive (§8 for torus, §9 for dragonfly) rather than by flattening to a single $(\alpha, BW)$ pair. A pure-crossbar chain, a pure-torus chain, and a single-dragonfly-tier chain are each handled exactly by their respective primitives. Genuinely mixed chains (e.g. torus scale-up + dragonfly scale-out) currently fall back to the crossbar-flatten bound with a warning; §10 documents the dispatch table in full, and the precise hierarchical cost for mixed chains is deferred to a follow-up pass.

### 7.3 Example: NVL576 as ideal vs. hierarchical

Two ways to imagine a 576-GPU scale-up domain:

**Ideal (single fabric, single tier):** one monolithic NVLink fabric where every pair of ranks is intra-rack-class. See `database/system/gb200.nvl576.ideal.json`.

Fabrics: `{"nvlink5-flat": [tier_0]}`. Every collective chains `["nvlink5-flat"]`.

| Fabric | Tier | $P_0$ | $BW_0$ | $\alpha_0$ |
|---|---|---|---|---|
| nvlink5-flat | 0 (only) | 576 | 900 GB/s | 0.5 μs |

Every collective — TP=8, TP=72, TP=576 — pays $(0.5\ \mu s, 900\ \text{GB/s})$. Physically unrealizable at this radix with current silicon (see §2.2), but it bounds "how fast could the model run if topology were free."

**Hierarchical (two fabrics: scale-up + scale-out):** 8× NVL72 racks stitched by an inter-rack scale-out fabric. Scale-up and scale-out are distinct physical technologies — NVLink inside the rack, InfiniBand (or Ethernet) between racks — so they appear as two named fabrics rather than two tiers of one fabric. See `database/system/gb200.nvl576.hierarchical.json`.

Fabrics: `{"nvlink5": [tier_0], "ib": [tier_0]}`. Each collective chains `["nvlink5", "ib"]`, escalating into `ib` once the group exceeds what `nvlink5`'s single tier can reach.

| Fabric | Tier | $P_i$ | $BW_i$ | $\alpha_i$ |
|---|---|---|---|---|
| nvlink5 (intra-rack NVSwitch) | 0 | 72 | 900 GB/s | 0.5 μs |
| ib (inter-rack scale-out) | 0 | 8 | 400 GB/s | 2.5 μs |

Collective size $G$ | Chain walked | $\alpha_{\text{span}}$ | $BW_{\text{span}}$
---|---|---|---
$G \le 72$ | nvlink5 only | 0.5 μs | 900 GB/s
$72 < G \le 576$ | nvlink5 then ib | 3.0 μs | 400 GB/s

The cliff at $G = 73$ — where the collective first needs to reach outside one rack (equivalently, where the walk crosses from `nvlink5` into `ib`) — is the single most consequential number in any hierarchical scale-up analysis. TP/EP partitions that stay ≤ 72 cost the same as NVL72; anything larger pays the scale-out bandwidth floor on every collective step.

---

## 8. Torus topology

A torus fabric replaces a single crossbar radix with a $k$-dimensional wraparound mesh. Every rank has $2k$ neighbor links (one each way along each of $k$ dimensions), so bandwidth scales with $k$ while the switch-silicon budget per node is small and fixed — the hallmark of dense supercomputer fabrics (Cray Gemini, IBM BG/Q, Google TPU v3/v4/v5p). §7's α-sum / BW-min flattening is wrong for a torus: its latency term compresses from the flat-ring $2(N-1)\alpha$ to $2\sum_i(D_i - 1)\alpha$ under the dim-by-dim schedule, while the bandwidth term stays Patarasuk-Yuan optimal [PY09]. This section makes that substitution precise.

### 8.1 Physical primitives

A torus tier is declared by its dimension tuple $(D_1, \ldots, D_k)$; reach is $N = \prod_i D_i$. Per-link single-direction bandwidth is $BW_\mathrm{link}$ and per-hop latency is $\alpha$. Key derived quantities:

| Quantity | Definition | Notes |
|---|---|---|
| Reach | $N = \prod_{i=1}^{k} D_i$ | Generalizes tier radix $P_i$ for crossbar tiers |
| Diameter | $\mathrm{diam} = \sum_{i=1}^{k} \lfloor D_i / 2 \rfloor$ | Worst-case rank-to-rank hop count (wraparound) |
| Per-node links | $2k$ | One each way along each of $k$ dims |
| Bisection width | $BW_\mathrm{bisect} = 2 \cdot (N / D_j) \cdot BW_\mathrm{link}$ | Cutting $\perp$ dim $j$ severs $2 (N/D_j)$ links |
| Min bisection | $BW_\mathrm{bisect}^\mathrm{min} = 2 N BW_\mathrm{link} / D_\mathrm{max}$ | Binds the largest dim — asymmetric layouts lose |
| Max dim | $D_\mathrm{max} = \max_i D_i$ | Sets the A2A bandwidth floor (§8.5) |

**Calibration points (production deployments):**

| System | Year | Dims | Per-link BW (single-dir) | α per hop | Source |
|---|---|---|---|---|---|
| IBM BG/Q | 2012 | 5D (4×4×4×4×2) | ~2 GB/s | ~0.5 μs | public docs |
| Cray Gemini (XE6) | 2010 | 3D (varies) | ~5 GB/s | ~1.5 μs | public docs |
| Google TPU v3 pod | 2018 | 2D (16×32) | ~82 GB/s | ~1 μs | [TPU-V4] §II refs |
| Google TPU v4 pod | 2022 | 3D (OCS-reconfigurable) | ~300 GB/s | ~1 μs | [TPU-V4] |
| Google TPU v5p pod | 2023 | 3D (16×16×16) | ~150 GB/s | ~1 μs | `tpu.v5p.pod.json` |

Per-hop $\alpha$ is roughly flat across generations (~0.5–1.5 μs) because SerDes traversal and router pipeline depth have not changed fundamentally — what grows is per-link BW and, with optical circuit switching on TPU v4, the set of dims a user can choose for their slice.

### 8.2 1D reduction to ring

A torus with $k = 1$, $D_1 = N$ reduces to a bidirectional ring of $N$ nodes. The dim-by-dim schedule collapses to one dim, hop count is $N - 1$, and the AR cost

$$t_{AR}^\mathrm{torus}(M, (N,), \alpha, BW) = 2(N-1)\alpha + 2\,\frac{N-1}{N}\,\frac{M}{BW}$$

matches §3's flat ring AR bit-identically. This is the continuity check that protects against silent divergence between the torus and crossbar code paths — the regression suite's T5 test asserts it per input, and `torus_all_reduce(dims=(N,)) == ring_all_reduce(N)` holds as a unit invariant.

### 8.3 Dim-by-dim ring all-reduce

The load-bearing formula:

$$\boxed{\; t_{AR}^\mathrm{torus}(M, (D_1, \ldots, D_k), \alpha, BW) \;=\; 2\sum_{i=1}^{k}(D_i - 1)\,\alpha \;+\; 2\,\frac{N-1}{N}\,\frac{M}{BW}, \quad N = \prod_{i=1}^{k} D_i \;}$$

**Derivation.** Run reduce-scatter along each dim innermost-first, then all-gather in reverse. The $i$-th reduce-scatter step operates on a payload that has already been scattered by the previous $i-1$ dims: it moves $(D_i - 1) M / \prod_{j \le i} D_j$ bytes per rank per link. Summing the BW term over $i$ gives a telescoping series that collapses to $(N-1)/N \cdot M / BW$ [PY09, CHPV07]. The AG pass is symmetric, so both legs together contribute $2(N-1)/N \cdot M/BW$ — identical to the flat 1D ring on $N$ nodes. The latency term, however, accumulates $2(D_i - 1)$ hops per dim rather than $2(N-1)$ for the flat ring.

**Numerical illustration ($N = 4096$).** Only the latency term compresses:

| Layout | $\sum_i(D_i - 1)$ | Latency ratio vs flat | Comment |
|---|---|---|---|
| Flat ring ($k=1$, $D=4096$) | 4095 | 1.00× | baseline |
| 2D 64×64 | 126 | 0.031× | |
| 3D 16×16×16 | 45 | 0.011× | TPU v5p pod layout |
| 4D 8×8×8×8 | 28 | 0.0068× | |
| 5D 4×4×4×8×8 | 23 | 0.0056× | diminishing returns |

The bandwidth term is identical across all rows — it is Patarasuk-Yuan optimal on any tree-connected fabric [PY09], and the dim-by-dim torus schedule is one such connected decomposition.

**Dim alignment.** The formula assumes the collective group of size $G$ aligns cleanly with a prefix-product of `dims` — i.e. $G = \prod_{i \le m} D_i$ for some $m \le k$. When this holds, the collective walks exactly $m$ ring dims (or fewer, if the caller requests a strict subtorus). When $G$ has no such prefix factoring (e.g. $G = 12$ on `dims = (8, 8, 8)`), the dispatcher emits a `UserWarning` and falls back to the conservative flat-ring bound $2(G-1)\alpha + 2(G-1)/G \cdot M/BW$. Avoiding this fallback is a partition-design concern — pick $G$ that matches a dim-prefix product of the torus slice.

### 8.4 All-gather and reduce-scatter

By symmetry with §8.3's AG leg:

$$t_{AG}^\mathrm{torus}(M, (D_1, \ldots, D_k), \alpha, BW) \;=\; \sum_{i=1}^{k}(D_i - 1)\,\alpha \;+\; (N - 1)\,\frac{M}{BW}$$

$M$ is the per-rank shard; the gathered volume at each rank is $N \cdot M$. This reduces to the flat ring AG formula for $k = 1$. Reduce-scatter has identical cost by time-reversal symmetry.

### 8.5 All-to-all — bisection-limited

Track per-link bytes in one direction directly. With $N$ ranks split by the min-bisection cut into halves $L$ and $R$ ($|L| = |R| = N/2$), each rank in $L$ holds $N$ chunks of size $M/N$ and sends $|R| = N/2$ of them across the cut. So $L \to R$ traffic is $(N/2) \cdot (N/2) \cdot (M/N) = NM/4$ bytes in one direction. The min-bisection severs $2N / D_\mathrm{max}$ links (wraparound torus), so each cut link carries $(NM/4) / (2N/D_\mathrm{max}) = D_\mathrm{max} M / 8$ bytes in one direction. Dividing by per-link single-direction $BW_\mathrm{link}$:

$$t_{A2A}^\mathrm{torus}(M, (D_1, \ldots, D_k), \alpha, BW) \;\approx\; \mathrm{diam}\cdot\alpha \;+\; \frac{D_\mathrm{max}\,M}{8\,BW_\mathrm{link}}$$

The bandwidth term depends on $D_\mathrm{max}$ alone, not on $N$. Two layouts with the same node count can differ by orders of magnitude in A2A cost — this is the core argument for cubic pod layouts, and the reason TPU v4's optical circuit switch carves balanced 3D slices rather than arbitrary rectangles [TPU-V4 §V]. At a cubic shape ($D_i = N^{1/k}$) the BW term equals $M / \mathrm{BW}_\mathrm{link}$ — torus matches star pairwise A2A's per-rank BW exactly; asymmetric layouts pay a $D_\mathrm{max}/N^{1/k}$ multiplier on top.

**Layout sensitivity at $N = 512$:**

| Layout | $D_\mathrm{max}$ | $BW_\mathrm{bisect}^\mathrm{min}$ | A2A BW term relative |
|---|---|---|---|
| 8×8×8 (balanced cubic) | 8 | $128 \cdot BW_\mathrm{link}$ | 1.00× |
| 4×4×32 (asymmetric) | 32 | $32 \cdot BW_\mathrm{link}$ | 4.00× |
| 2×2×128 (extreme pencil) | 128 | $8 \cdot BW_\mathrm{link}$ | 16.00× |

For MoE-heavy inference where EP collectives are A2A-dominated, a 4× BW hit from an asymmetric torus slice reads directly onto TPOT. The model picks $D_\mathrm{max}$ automatically from the tier's `dims` tuple; there is no layout-selection knob. For MoE A2A specifically, bake Dispatch+Combine into the caller's $M$ (matching the ring A2A convention in §5). Derivation matches `documentation/explaining/collectives/02_topology_mapping.md` Appendix D.3.

### 8.6 Worked example — TPU v5p pod

`database/system/tpu.v5p.pod.json`: 16×16×16 ICI torus, 4096 TPU v5p chips, $BW_\mathrm{link} = 150$ GB/s per link (single-direction), $\alpha = 1.0$ μs per hop. Consider AR for a full-pod TP or EP collective at $G = 4096$.

| Schedule | α-hops | Latency term | BW term ($M = 1$ GB) | Total ($M = 1$ GB) |
|---|---|---|---|---|
| Flat ring (hypothetical) | $2 \cdot 4095 = 8190$ | $8.19$ ms | $13.32$ ms | $21.5$ ms |
| Dim-by-dim torus (actual) | $2 \cdot 45 = 90$ | $90$ μs | $13.32$ ms | $13.4$ ms |

Latency savings: **91× on the α term**, which is a 1.6× speedup at $M = 1$ GB and dominates below the $M \approx (\text{hop-count-gap}) \cdot \alpha \cdot BW$ crossover. For a smaller message typical of per-layer TP AR ($M \sim 32$ MB), the torus schedule wins by ≈ 15× — the α term falls to 90 μs while the flat ring still pays 8.19 ms.

For comparison to a hypothetical flat NVL-class crossbar at the same node count: a single 4096-port 150 GB/s crossbar is not silicon-realizable in this generation (well past the NVSwitch radix/BW frontier per §2.2). The torus achieves what no single-chip fabric can: uniform 150 GB/s bandwidth between any pair of the 4096 chips with only 6 links per chip.

### 8.7 Alternative algorithms — Swing and HammingMesh

Dim-by-dim ring is the canonical torus AR schedule and the only one implemented in this pass. Two alternatives exist:

- **Swing AR [SWING].** Short-cuts rings by combining non-adjacent rank pairs, claiming higher effective bandwidth on narrow dims. Requires the dims to support the shortcut pattern; not universally applicable. `TuningSpec.torus_algorithm = "swing"` is reserved as a forward-compat flag and currently raises `NotImplementedError`.
- **HammingMesh [HAMMESH].** An irregular topology in the torus family with improved bisection at fixed link count; targeted at DL clusters. Orthogonal to the algorithm choice — would require a new tier type (`HammingMeshTier`) rather than a new primitive.

Both are flagged for follow-up study once their papers are digested against the code-level primitives. Neither is load-bearing for current production TPU deployments, which all use dim-by-dim ring.

### 8.8 Efficiency note

Torus tiers run with **$\eta \approx 0.95$** by default — slightly higher than the crossbar $\eta = 0.90$ from §4.4. The reason is structural: a torus hop traverses a single point-to-point link with no shared scheduler, no VOQ, and no shared output buffer. The HoL + buf + proto decomposition from §4 still applies, but $\eta_\mathrm{HoL}$ is essentially 1.0 (no crossbar scheduling loss) and $\eta_\mathrm{buf}$ is smaller because link-level flow control is simpler. Adversarial A2A is bounded by the bisection capacity calculated in §8.5, not by $\eta$ — the model already accounts for the worst-case traffic pattern in the $D_\mathrm{max}$ scaling, so applying $\eta < 1$ on top would double-count.

---

## 9. Dragonfly topology

A dragonfly fabric [KDSA08] replaces the single monolithic switch with a hierarchical two-level structure: routers are grouped into tightly coupled cliques (each clique fully connected internally), and cliques are sparsely connected via direct global links. The design targets the cost-performance frontier of long-reach scale-out networks — global fiber is expensive, so the topology minimizes global-hop counts (diameter 3 under minimal routing) while offering enough aggregate global bandwidth that a full bisection workload only rarely bottlenecks. It underlies HPE Slingshot (the Cray EX / Frontier interconnect) and Google Jupiter. This section describes the three-tier α-β decomposition used by the model's dragonfly primitives.

### 9.1 Parameters

A canonical balanced dragonfly is parameterized by a tuple $(p, a, h, g)$:

| Symbol | Meaning | Typical values |
|---|---|---|
| $p$ | Endpoints per router | 8–32 |
| $a$ | Routers per group (all-to-all within group) | 8–32 |
| $h$ | Global links per router (each to a different distant group) | 4–16 |
| $g$ | Number of groups | up to $a h + 1$ |

Total endpoints: $N = p \cdot a \cdot g$. Each group provides $a \cdot h$ global links, which is just enough to direct-connect $a h$ other groups — leaving $a h + 1$ as the maximum group count under the **canonical balanced** construction $g = a h + 1$ (one such link from the group to each other group, every one direct). Smaller $g$ under-utilizes global capacity but is sometimes chosen for cost reasons.

**Diameters.** Under **minimal adaptive routing**, any src→dst path walks at most three routers: source router → another router in the source group (to reach a global link aimed at dst group) → destination router in dst group. Counting the two endpoint-to-router hops, an end-to-end packet traverses five link-level hops in the worst case. Conventional practice reports **diameter 3** (router hops). Under **Valiant routing** [KDSA08 §4], the packet first hops to a *random intermediate group* before heading to dst, doubling the long-haul route: diameter 5 (router hops) / up to 7 link hops, with effective L2 bandwidth halved because every inter-group message now uses two global links.

### 9.2 Three-tier α-β decomposition

The dragonfly primitives decompose every collective into three independently costed tiers, each run as a classical α-β ring:

| Tier | Name | Group size | $\alpha$ | $BW$ |
|---|---|---|---|---|
| L0 | Intra-router | $p$ endpoints | $\alpha_\mathrm{r}$ | $BW_\mathrm{r}$ |
| L1 | Intra-group (router-router) | $a$ routers | $\alpha_\mathrm{l}$ | $BW_\mathrm{l}$ |
| L2 | Inter-group (global link) | $g$ groups | $\alpha_\mathrm{g}$ | $BW_\mathrm{g}$ |

Why three independent $(\alpha, BW)$ pairs rather than one per link tier: each tier uses a different link technology in deployed dragonflies. Slingshot 11 runs the same 25 GB/s port rate at every tier but calibrates distinct $\alpha$s (0.3 μs intra-router → 0.8 μs intra-group → 1.5 μs inter-group) because the switching pipeline depth differs by tier. Generic dragonflies can differ in BW too — e.g. if global links are the same technology but spread across long fiber with lower per-link utilization. The model consumes six independent parameters per dragonfly tier to avoid flattening this structure.

### 9.3 Adaptive routing and the Valiant fallback

Under **adaptive minimal routing** with uniform admissible traffic (LLM ring-AR, balanced A2A, random permutations), every global link carries approximately equal load and the three-tier ring schedule is bandwidth-optimal per tier [KDSA08 §3-4, JAIN22 §4]. This is the `worst_case=False` default.

Under **adversarial traffic** (pathological permutations, expert-skewed MoE routing that concentrates flows on a small subset of global links), adaptive min routing collapses — some global links saturate while others idle. The standard fix is **Valiant routing** [KDSA08 §4, VALIANT81]: each inter-group message is first routed to a random intermediate group, which by randomization re-balances global link load at the cost of two global hops per message. Our `worst_case=True` flag applies both effects to the L2 term: the $\alpha$-hops double (from $(g-1)$ to $2(g-1)$) *and* the effective BW halves (handled as a factor-2 on the L2 sub-cost). LLM ring AR and most structured workloads should stay at `worst_case=False`; MoE A2A under expert skew is the primary use case for turning it on.

### 9.4 Hierarchical collective cost

The three-tier ring AR schedule: L0 reduce-scatter → L1 reduce-scatter → L2 all-reduce → L1 all-gather → L0 all-gather. At each descent step the payload shrinks; at each ascent step it grows back.

**All-reduce.** With $c = 2$ under `worst_case=True` and $c = 1$ otherwise:

$$\boxed{\;\begin{aligned}
t_{AR}^\mathrm{df}(M) \;=\;\; & \underbrace{2(p{-}1)\alpha_\mathrm{r} + 2\,\tfrac{p{-}1}{p} \cdot \tfrac{M}{BW_\mathrm{r}}}_{\text{L0 intra-router}} \\
+\; & \underbrace{2(a{-}1)\alpha_\mathrm{l} + 2\,\tfrac{a{-}1}{a} \cdot \tfrac{M/p}{BW_\mathrm{l}}}_{\text{L1 intra-group}} \\
+\; & c \cdot \underbrace{\left[\, 2(g{-}1)\alpha_\mathrm{g} + 2\,\tfrac{g{-}1}{g} \cdot \tfrac{M/(p a)}{BW_\mathrm{g}} \,\right]}_{\text{L2 inter-group}}
\end{aligned}\;}$$

Payload shrinks from $M$ at L0 to $M/p$ at L1 to $M/(pa)$ at L2 — the L1 ring sees $M/p$ per rank because L0 has already reduce-scattered across $p$ endpoints; similarly L2 sees $M/(pa)$. The L2 bandwidth term is small relative to L0 for typical dragonflies with $p \cdot a \gg 1$ (e.g. Slingshot's $p \cdot a = 512$), which is why L0 tends to dominate at large $M$ even though L2 has the most participants.

Trivial-tier degeneracy: the primitive zeros out any tier with size $\le 1$, so the formula reduces to `ring_all_reduce(M, p, α_r, BW_r)` when $a = g = 1$ and to `ring_all_reduce(M, a, α_l, BW_l)` when $p = g = 1$, preserving continuity with the 1D ring.

**All-gather.** Same three-tier structure, payload *grows* down the tiers (each tier's AG fans out by its own factor):

$$t_{AG}^\mathrm{df}(M) \;=\; (p{-}1)\alpha_\mathrm{r} + (p{-}1)\tfrac{M}{BW_\mathrm{r}} \;+\; (a{-}1)\alpha_\mathrm{l} + (a{-}1)\tfrac{p\,M}{BW_\mathrm{l}} \;+\; c\left[(g{-}1)\alpha_\mathrm{g} + (g{-}1)\tfrac{p\,a\,M}{BW_\mathrm{g}}\right]$$

**All-to-all.** MoE A2A uses an identical decomposition to AR, with Dispatch+Combine baked into $M$ by the caller (matching the `ring_moe_all_to_all ≡ ring_all_reduce` identity from §5 of decode.md). L2 tends to dominate for MoE A2A because expert routing is often adversarial relative to the admissible-traffic assumption, so `worst_case=True` is more frequently justified for A2A than for AR.

### 9.5 Worked example — HPE Slingshot 11

`database/system/slingshot11.dragonfly.json`: $(p, a, h, g) = (32, 16, 16, 257)$ — canonical balanced with $g = a h + 1 = 257$. Port rate 25 GB/s at all tiers; per-tier α $(\alpha_\mathrm{r}, \alpha_\mathrm{l}, \alpha_\mathrm{g}) = (0.3, 0.8, 1.5)$ μs calibrated to Frontier/Slingshot published traces [SLINGSHOT]. Total reach $N = p \cdot a \cdot g = 131{,}584$ endpoints (the JSON declares `num_devices = 131{,}072` — under-populated by 512 endpoints, reflecting a realistic sub-full pod).

**Ring AR for $M = 128$ MB across the full fabric ($G = N = 131{,}584$):**

| Tier | Group size | α term | BW term | Sub-total |
|---|---|---|---|---|
| L0 intra-router | 32 | $2 \cdot 31 \cdot 0.3$ μs = 18.6 μs | $2 \cdot (31/32) \cdot 128\text{ MB} / 25\text{ GB/s}$ ≈ 9.92 ms | 9.94 ms |
| L1 intra-group | 16 | $2 \cdot 15 \cdot 0.8$ μs = 24 μs | $2 \cdot (15/16) \cdot 4\text{ MB} / 25\text{ GB/s}$ ≈ 0.30 ms | 0.32 ms |
| L2 inter-group | 257 | $2 \cdot 256 \cdot 1.5$ μs = 0.77 ms | $2 \cdot (256/257) \cdot 250\text{ KB} / 25\text{ GB/s}$ ≈ 0.020 ms | 0.79 ms |
| **Total (best case)** | | 0.81 ms | 10.2 ms | **11.0 ms** |

The L0 BW term dominates because the 32 endpoints per router must sequentially share the router's 25 GB/s upstream link during the intra-router ring. L2 is the lightest contributor despite having 257 ring-participants — the per-rank payload has shrunk by the factor $p \cdot a = 512$ by the time the collective reaches that tier. Under `worst_case=True` (Valiant routing), the L2 sub-cost doubles to 1.58 ms and total AR rises to 11.8 ms — a 7% increase on the total, confirming L2 is not the bottleneck in the best case.

**Comparison to crossbar flatten** (what §7's α-sum / BW-min rule would give if applied naively):

$$t_{AR}^\mathrm{flatten} \;=\; 2(N-1)(\alpha_\mathrm{r} + \alpha_\mathrm{l} + \alpha_\mathrm{g}) + 2\,\tfrac{N-1}{N}\,\tfrac{M}{\min(BW_\mathrm{r}, BW_\mathrm{l}, BW_\mathrm{g})}$$

which evaluates to $2 \cdot 131{,}583 \cdot 2.6\text{ μs} + 2 \cdot 128\text{ MB} / 25\text{ GB/s} \approx 684 + 10.2 = 694$ ms — a **63× overestimate**. The flatten rule is wrong because it treats every one of the $N$ ranks as participating in a single long α-hop chain, when in reality the three-tier ring has at most $p + a + g - 3 = 302$ sequential hops.

### 9.6 Open questions

1. **Congestion-control impact on $\eta$.** [SLINGSHOT] measures Rosetta's credit-based flow control under adversarial load patterns; transient congestion can drop effective BW on specific links. Our default $\eta$ stays flat in $P$ (§4.4 caveat); a dragonfly-specific $\eta$ calibrated against Slingshot traces is future work.
2. **Multi-job interference.** Dragonfly global links are shared across tenants; [PAARD] analyzes interference patterns for co-scheduled jobs. Current model assumes job-exclusive fabric access — accurate for dedicated inference clusters, inaccurate for shared HPC leadership systems.
3. **Dragonfly+ variant.** Shpiner et al. 2017 generalize dragonfly to Dragonfly+ (adds a second level of hierarchical all-to-all within groups). Flagged but not modeled here; would add another tier-type and a different L1 decomposition.
4. **Valiant auto-detection.** Currently `worst_case=True` is a manual lever; no heuristic to auto-flip it based on the collective workload. For MoE A2A under heavy expert skew, a routing-aware auto-default would be helpful but requires per-workload traffic telemetry the model doesn't currently consume.
5. **Empirical $\eta$ for dragonfly.** Default $\eta$ (§4.4) is NVSwitch-calibrated. Slingshot/Rosetta dragonfly likely has a different residual η — probably slightly lower because of the Ethernet-derived framing overhead (§4 open question 5). No measured value exists in our calibration set.

---

## 10. Unified fabric-chain dispatch

The fabric-chain walk (§7.2) is topology-agnostic: every tier contributes its cost to the chain by the rule "apply the tier's native primitive over the shard of the collective that stays within that tier." Three topologies are live today:

| Tier topology | Primitive family | Cost formula root |
|---|---|---|
| Crossbar | `ring_*` / `tree_*` from §3 + §5 | flat α-β |
| Torus | `torus_*` from §8 | dim-by-dim ring |
| Dragonfly | `dragonfly_*` from §9 | three-tier ring |

The dispatcher (`core/primitives/dispatch.py`) resolves the chain's topology set at call time:

1. **Pure crossbar chain.** Delegates to the legacy `span_tiers` flatten (α-sum + BW-min) and calls the crossbar primitive. Bit-identical to the pre-refactor path — the regression gate's `REL_TOL=0` guarantees this (see `scratch/switching_upgrade.md` §10.4).
2. **Pure torus chain.** Concatenates every tier's `dims` into one global dims tuple, aligns the collective's $G$ to a prefix-product, and calls the torus primitive with that sub-dims tuple. Misalignment emits a `UserWarning` and falls back to the flat-ring conservative bound.
3. **Single-dragonfly-tier chain.** Calls the dragonfly primitive with the tier's $(p, a, g)$ parameters, truncating when $G < p \cdot a \cdot g$: for $G \le p$ the collective stays intra-router (triple $(G, 1, 1)$); for $p < G \le p \cdot a$ it spans one group's routers $(p, \lceil G/p \rceil, 1)$; otherwise it spans groups $(p, a, \lceil G/(p a) \rceil)$, clamped to $(p, a, g)$.
4. **Genuinely mixed chains** (crossbar + torus, torus + dragonfly, or multi-dragonfly). The precise hierarchical cost is deferred to a follow-up pass. The dispatcher emits a `UserWarning` and falls back to the crossbar-flatten bound as a numerical upper envelope — safe to evaluate, not bit-accurate.

**Canonical motivating example — torus scale-up + dragonfly scale-out.** `database/system/hybrid.torus_scaleup_df_scaleout.json` declares two fabrics: `ici` (3D 16×16×16 torus, 4096 endpoints, 150 GB/s, 1 μs α) and `slingshot` (1-tier dragonfly with $(p, a, h, g) = (2, 2, 2, 2)$, 8 endpoints, 25 GB/s, α 0.3/0.8/1.5 μs — a deliberately small example to exercise dispatch rather than a production config). Each collective chains `["ici", "slingshot"]`, reach $4096 \times 8 = 32{,}768$ matching `num_devices`.

| Collective size $G$ | Chain walked | Dispatch path | Notes |
|---|---|---|---|
| $G \le 4096$ | `ici` only (pure torus) | torus primitive, dims aligned to prefix of (16,16,16) | dim-aligned $G \in \{16, 256, 4096\}$ are exact; other $G$ warn and flat-fall-back |
| $G > 4096$ | `ici` + `slingshot` (mixed) | crossbar-flatten fallback + `UserWarning` | precise hybrid cost deferred |

A collective that stays inside one TPU pod ($G \le 4096$) pays only the torus cost — usually the entire TP, EP, or SP collective for LLM inference, since partition factors rarely exceed the pod reach. Escalation into the scale-out dragonfly is the failure mode the model flags; future extensions will replace the fallback with a per-tier chained dragonfly cost.

**Reading the dispatcher.** The innermost-first walk matches §7.2 exactly; topology-structured primitives replace the flat α-β call for non-crossbar tiers. The chain boundary between fabrics is treated identically to the tier boundary within a fabric — both are cumulative-reach escalations. This keeps all three composition patterns (single-topology crossbar, single-topology torus, single-topology dragonfly, hybrid chains) on one conceptual walk; only the per-tier cost call differs.

---

## Appendix A. Collective bandwidth multiplier

The NCCL "busbw" correction [NCCL-PERF] converts algorithm-level throughput ($\text{algbw} = \text{data\_size} / \text{time}$) to actual per-link bandwidth utilization. This is **not** an efficiency factor in $\eta$ — it is a conversion multiplier that accounts for how much link bandwidth the collective algorithm consumes per unit of useful data:

| Collective | Bandwidth multiplier | Limit ($P \to \infty$) | Examples |
|---|---|---|---|
| All-reduce (ring) | $2(P-1)/P$ | **2.0** | $P=3$: 1.33; $P=8$: 1.75 |
| All-to-all | $(P-1)/P$ | **1.0** | $P=8$: 0.875; $P=16$: 0.9375 |
| Broadcast / reduce | $1$ | 1.0 | — |

For ring all-reduce, the factor approaches 2 because the algorithm has two phases (reduce-scatter + all-gather), each moving $(P-1)/P$ of the data through every link. A perfect ring all-reduce achieves per-link utilization equal to the link bandwidth.

**Why this is excluded from $\eta$:** the collective communication model (decode.md §5) already accounts for the correct message sizes and step counts for each algorithm (ring, tree, all-to-all). The $2(P-1)/P$ bandwidth scaling is baked into those formulas. Including it in $\eta$ would double-count.

When calibrating $\eta$ against published benchmarks, compare $\text{busbw} / BW_{\text{link}}$ — this ratio reflects the physical fabric efficiency ($\eta_{\text{HoL}} \cdot \eta_{\text{buf}} \cdot \eta_{\text{proto}}$) with the algorithmic multiplier already factored out.

---

## Appendix B. Symbol glossary

Symbols introduced across §§2–10, grouped by topology. Section references point to the first definition site.

**Crossbar tier (§2–§3, §7).**

| Symbol | Units | Meaning |
|---|---|---|
| $B_\mathrm{agg}$ | GB/s | Switch-chip aggregate single-direction bandwidth (silicon budget) |
| $P$ | — | Port count / radix at a tier |
| $BW_\mathrm{nominal}$ | GB/s | Target per-port single-direction rate |
| $BW_\mathrm{port}(P)$ | GB/s | Radix-bandwidth tradeoff: $B_\mathrm{agg} / P$ |
| $BW_\mathrm{eff}(P)$ | GB/s | Effective per-port BW after $\eta$ and capacity cap |
| $\alpha_\mathrm{base}$ | μs | Single-hop transit latency (rank-count-independent) |
| $\eta$ | — | Lumped fabric efficiency $\in (0, 1]$; product of $\eta_\mathrm{HoL} \cdot \eta_\mathrm{buf} \cdot \eta_\mathrm{proto}$ |
| $P^*$ | — | Capacity crossover: $B_\mathrm{agg} / BW_\mathrm{nominal}$ |

**Fabric chain (§7).**

| Symbol | Units | Meaning |
|---|---|---|
| $P_i$ | — | Reach at tier $i$ (crossbar: radix; torus: $\prod_j D_j$; dragonfly: $p a g$) |
| $\alpha_i$ | μs | Per-traversal latency at tier $i$ |
| $BW_i$ | GB/s | Per-port BW at tier $i$ (post-$\eta$, single-direction) |
| $\alpha_\mathrm{span}(G)$ | μs | $\sum_{i \le k} \alpha_i$ over tiers crossed by a collective of size $G$ |
| $BW_\mathrm{span}(G)$ | GB/s | $\min_{i \le k} BW_i$ over tiers crossed (crossbar-only rule) |

**Torus (§8).**

| Symbol | Units | Meaning |
|---|---|---|
| $k$ | — | Torus dimensionality (number of axes) |
| $(D_1, \ldots, D_k)$ | — | Per-dim extent; reach $N = \prod_i D_i$ |
| $N$ | — | Torus endpoint count |
| $D_\mathrm{max}$ | — | $\max_i D_i$; sets A2A bisection floor |
| $\mathrm{diam}$ | — | Wraparound diameter $\sum_i \lfloor D_i / 2 \rfloor$ |
| $BW_\mathrm{link}$ | GB/s | Per-link single-direction BW (same as tier's $BW_i$) |
| $BW_\mathrm{bisect}$ | GB/s | Bisection capacity perpendicular to a dim |
| $BW_\mathrm{bisect}^\mathrm{min}$ | GB/s | $2 N BW_\mathrm{link} / D_\mathrm{max}$ |

**Dragonfly (§9).**

| Symbol | Units | Meaning |
|---|---|---|
| $p$ | — | Endpoints per router |
| $a$ | — | Routers per group |
| $h$ | — | Global links per router |
| $g$ | — | Number of groups (canonical balanced: $g = a h + 1$) |
| $\alpha_\mathrm{r}, \alpha_\mathrm{l}, \alpha_\mathrm{g}$ | μs | Per-tier latencies: intra-router / intra-group / inter-group |
| $BW_\mathrm{r}, BW_\mathrm{l}, BW_\mathrm{g}$ | GB/s | Per-tier single-direction BW at each of the three tiers |
| $c$ | — | Valiant-routing multiplier on L2 sub-cost ($c = 2$ if `worst_case=True`, else $c = 1$) |

---

## References

Tagged citations resolve in `references.md`. New entries required for this doc:

- **[KAROL87]** M. J. Karol, M. G. Hluchyj, S. P. Morgan. "Input versus output queuing on a space-division packet switch." *IEEE Trans. Communications*, COM-35(12):1347–1356, Dec. 1987. The 58.6% FIFO HoL bound.
- **[MCKEOWN99]** N. McKeown. "The iSLIP scheduling algorithm for input-queued switches." *IEEE/ACM Trans. Networking*, 7(2):188–201, Apr. 1999. VOQ + scheduling → 100% throughput.
- **[VOQ-WIKI]** Virtual output queueing, Wikipedia. <https://en.wikipedia.org/wiki/Virtual_output_queueing>
- **[NCCL-PERF]** NVIDIA NCCL-Tests, `doc/PERFORMANCE.md`. <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>
- **[GB200-NV]** NVIDIA. "GB200 NVL72 delivers trillion-parameter LLM training and real-time inference." 2024. <https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/>
- **[RUBIN-SA]** D. Patel, et al. "Vera Rubin — extreme co-design." *SemiAnalysis*, 2025. NVLink 6 switch SerDes doubling.
- **[NVS-FM]** FiberMall. "In-depth analysis of NV Switch." <https://www.fibermall.com/blog/analysis-nv-switch.htm>
- **[UALINK-SPEC]** UALink Consortium. "UALink 200G 1.0 Specification Overview." Apr 2025. 200 GT/s per lane, 4-lane stations at 800 Gbps bidi (100 GB/s single-dir), up to 1,024 endpoints. <https://ualinkconsortium.org/blog/ualink-200g-1-0-specification-overview-802/>
- **[TH-ULTRA]** Broadcom. "Broadcom Ships Tomahawk Ultra: Reimagining the Ethernet Switch for HPC and AI Scale-up." Jul 2025. 51.2 Tbps full-duplex (25.6 Tbps single-dir), 250 ns switch latency, in-network collectives, CBFC/LLR. <https://investors.broadcom.com/news-releases/news-release-details/broadcom-ships-tomahawk-ultra-reimagining-ethernet-switch-hpc>
- **[SKYHAMMER]** Upscale AI. "SkyHammer Architecture." 2025–2026. First UALink switch ASIC, 1,024-port radix, Q4 2026 availability. <https://upscaleai.com/upscale-ai-unveils-skyhammer-architecture/>
- **[ESUN-OCP]** OCP. "Introducing ESUN: Advancing Ethernet for Scale-Up AI Infrastructure." 2025. 4-byte compact header, CBFC, LLR, 175+ member companies. <https://www.opencompute.org/blog/introducing-esun-advancing-ethernet-for-scale-up-ai-infrastructure-at-ocp>
- **[GTC25-IB]** NVIDIA GTC 2025. Port-to-port hop latency 240 ns for NDR 400G / XDR 800G IB devices.
- **[NCCL-ISSUE-149]** A800 NVLink busbw benchmark thread. <https://github.com/NVIDIA/nccl-tests/issues/149>
- **[NCCL-ISSUE-272]** H200 all-reduce exceeds NVL4 spec thread. <https://github.com/NVIDIA/nccl-tests/issues/272>
- **[NCCL-TUNING-NV]** NVIDIA GB200 NVL Multi-Node Tuning Guide — NCCL. <https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html>
- **[NVIDIA-SLURM]** NCCL all-reduce on Slurm/IB NDR. 400G target ">90% busbw, <2.5 μs for 512 GPUs."
- **[NCCLX-META]** M. Si et al. "NCCLX: Collective Communication for 100k+ GPUs." arXiv:2510.20171, 2025.
- **[SHARP-NV]** NVIDIA NCCL SHARP all-reduce 1.7× improvement over non-SHARP on 400 Gb/s IB.
