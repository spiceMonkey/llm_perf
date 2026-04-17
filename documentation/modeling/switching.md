# Scale-Up Network Switch Model

**Modeling Effective Bandwidth and Latency for Single-Layer Scale-Up Fabrics**

<br/>

**Author:** Yue Lu  
**Date:** April 2026

**Keywords:**  
scale-up network, NVSwitch, NVLink, UALink, ESUN, crossbar, switch fabric, aggregate bandwidth, radix, head-of-line blocking, VOQ, efficiency factor, single-direction bandwidth, Pareto frontier, collective communication

---

A single-layer scale-up network (crossbar-class switch) feeds TP, EP, SP, and rack-local PP collectives. The decode and prefill models treat the per-role bandwidth $BW_{role}$ and startup latency $\alpha_{role}$ as constants. That abstraction holds only while all ranks in a collective sit under a **single non-blocking switching layer** with enough silicon bandwidth budget to feed every port at its nominal rate. This document makes that abstraction honest: as port count $P$ grows past the single-chip capacity frontier, the effective per-port bandwidth must decrease, and an efficiency factor $\eta$ captures the protocol/fabric losses that show up even in the ideal case.

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
- [7. Multi-Tier Extension](#7-multi-tier-extension)
  - [7.1 Fabrics and Tier Descriptors](#71-fabrics-and-tier-descriptors)
  - [7.2 Spanning a Collective Across Tiers](#72-spanning-a-collective-across-tiers)
  - [7.3 Example: NVL576 as Ideal vs. Hierarchical](#73-example-nvl576-as-ideal-vs-hierarchical)
- [Appendix A. Collective Bandwidth Multiplier](#appendix-a-collective-bandwidth-multiplier)
- [References](#references)

---

## 1. Scope

**In scope.** Single-layer scale-up fabric: one monolithic switching tier between every pair of GPUs participating in a collective. Includes NVLink/NVSwitch within a rack (NVL72), a hypothetical larger single-switch pod, or a UALink-class Ethernet scale-up leaf. All ports connect to the same switching silicon; traffic hops the switch exactly once.

**Out of scope (for now).**

- Multi-tier fabrics (spine-leaf, Clos, dragonfly). These need a separate tier-aware model — ranks traversing the fabric incur per-tier α and oversubscription-dependent BW. Leave for a future `scale_out_network.md`.
- Topology-explicit collective formulas (e.g. ring vs. tree vs. halving-doubling). The α–β abstraction in this doc is topology-agnostic; the collective's rank-count dependence is already captured in the communication model's ring/tree variants (tpot.md §5).
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

The crossover $P^* = B_{\text{agg}} / BW_{\text{nominal}}$ is the **port count at which single-layer scale-up exhausts its silicon budget**. Past $P^*$, the only ways to grow the collective are (a) accept per-port rate degradation, or (b) tier the network (outside this doc's scope).

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

## 7. Multi-tier extension

Sections 1–6 assume a single switching tier. Real systems beyond one rack add a second (or higher) tier stitching racks together — e.g. a hypothetical NVL576 built from 8× NVL72 racks plus an inter-rack scale-out layer. This section extends the model to capture tier-aware α and bandwidth. A one-tier, one-fabric collective is a trivial case of the general chain and produces identical numbers to §3.

### 7.1 Fabrics and tier descriptors

The system model names physical networks as **fabrics**. Each `FabricSpec` is an ordered list of switching tiers, innermost first. A tier $i \in \{0, 1, \ldots\}$ inside a fabric is a triple:

| Symbol | Description | Units |
|---|---|---|
| $P_i$ | Radix at tier $i$ — ranks reachable within this tier from any single rank | — |
| $BW_i$ | Effective per-port bandwidth at tier $i$ (single-direction, post-$\eta$) | GB/s |
| $\alpha_i$ | Per-traversal latency of tier $i$'s switching silicon | μs |

Tier 0 is the innermost (e.g. intra-rack NVSwitch). The cumulative reach at tier $k$ is $\prod_{i=0}^{k} P_i$ ranks. A fabric with $n$ tiers can host a collective reaching up to $\prod_{i=0}^{n-1} P_i$ ranks on its own.

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

## Appendix A. Collective bandwidth multiplier

The NCCL "busbw" correction [NCCL-PERF] converts algorithm-level throughput ($\text{algbw} = \text{data\_size} / \text{time}$) to actual per-link bandwidth utilization. This is **not** an efficiency factor in $\eta$ — it is a conversion multiplier that accounts for how much link bandwidth the collective algorithm consumes per unit of useful data:

| Collective | Bandwidth multiplier | Limit ($P \to \infty$) | Examples |
|---|---|---|---|
| All-reduce (ring) | $2(P-1)/P$ | **2.0** | $P=3$: 1.33; $P=8$: 1.75 |
| All-to-all | $(P-1)/P$ | **1.0** | $P=8$: 0.875; $P=16$: 0.9375 |
| Broadcast / reduce | $1$ | 1.0 | — |

For ring all-reduce, the factor approaches 2 because the algorithm has two phases (reduce-scatter + all-gather), each moving $(P-1)/P$ of the data through every link. A perfect ring all-reduce achieves per-link utilization equal to the link bandwidth.

**Why this is excluded from $\eta$:** the collective communication model (tpot.md §5) already accounts for the correct message sizes and step counts for each algorithm (ring, tree, all-to-all). The $2(P-1)/P$ bandwidth scaling is baked into those formulas. Including it in $\eta$ would double-count.

When calibrating $\eta$ against published benchmarks, compare $\text{busbw} / BW_{\text{link}}$ — this ratio reflects the physical fabric efficiency ($\eta_{\text{HoL}} \cdot \eta_{\text{buf}} \cdot \eta_{\text{proto}}$) with the algorithmic multiplier already factored out.

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
