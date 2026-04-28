# NVL72: Topology and NVLink Bandwidth

**Author:** Yue Lu  
**Date:** April 2026  

The NVIDIA GB200 NVL72 packs 72 Blackwell B200 graphics processing units (GPUs) into a single rack-scale NVLink domain, wired together by a small set of switch chips. This note explains the wiring (how 72 GPUs share 18 NVSwitch application-specific integrated circuits (ASICs) housed in 9 switch trays) and the bandwidth (where the headline "900 GB/s GPU↔switch" and "1.8 TB/s per GPU" numbers come from). Both questions reduce to the same per-link primitive — once you fix how many NVLink ports a GPU has and how fast each one runs, every other figure in the system follows by multiplication.

## 1. Rack physical layout

A GB200 NVL72 rack stacks two kinds of trays vertically — compute trays that hold the GPUs, and switch trays that hold the NVLink fabric. The counts are fixed by the spec:

```
GB200 NVL72 rack (front view, 18 compute + 9 NVSwitch trays)
┌──────────────────────────────────────────────────┐
│ Compute tray 18  │  2× GB200  →  4 B200 GPUs    │ ┐
│ Compute tray 17  │  2× GB200  →  4 B200 GPUs    │ │
│ Compute tray 16  │  2× GB200  →  4 B200 GPUs    │ │
│      ...         │            ...               │ │ 18 trays
│ Compute tray 11  │  2× GB200  →  4 B200 GPUs    │ │ × 4 GPUs
│ Compute tray 10  │  2× GB200  →  4 B200 GPUs    │ │ = 72 GPUs
├──────────────────────────────────────────────────┤ │
│ NVSwitch tray 9  │  2× NVSwitch ASICs           │ ┐
│ NVSwitch tray 8  │  2× NVSwitch ASICs           │ │
│ NVSwitch tray 7  │  2× NVSwitch ASICs           │ │
│      ...         │            ...               │ │ 9 trays
│ NVSwitch tray 2  │  2× NVSwitch ASICs           │ │ × 2 ASICs
│ NVSwitch tray 1  │  2× NVSwitch ASICs           │ │ = 18 switches
├──────────────────────────────────────────────────┤ │
│ Compute tray 9   │  2× GB200  →  4 B200 GPUs    │ │
│      ...         │            ...               │ │
│ Compute tray 1   │  2× GB200  →  4 B200 GPUs    │ ┘
└──────────────────────────────────────────────────┘
```

That gives $18 \times 4 = 72$ GPUs and $9 \times 2 = 18$ NVSwitch ASICs in a single rack. The switch trays sit in the middle of the rack (between two banks of compute trays) so that the NVLink cabling lengths to every compute tray are similar, which keeps the per-link signaling budget uniform.

## 2. Logical wiring — one link per GPU per switch

Each B200 has 18 NVLink 5 ports, and each NVSwitch ASIC has 72 ports. The wiring that makes both port counts add up is unique:

- Every GPU sends exactly **one** NVLink to **each** of the 18 NVSwitch ASICs.
- Every NVSwitch ASIC has exactly **one** port reaching **each** of the 72 GPUs.

```
                                  18 NVSwitch ASICs
            S0   S1   S2   S3   ...   S14  S15  S16  S17
            ▲    ▲    ▲    ▲          ▲    ▲    ▲    ▲
            │    │    │    │  (72 ports per switch,
            │    │    │    │   one to every GPU)
   ┌────────┼────┼────┼────┼──── ... ─┼────┼────┼────┼─┐
   │        │    │    │    │          │    │    │    │ │
   │  G0 ───┴────┴────┴────┴── ... ───┴────┴────┴────┘ │  ← 18 links
   │  G1 ───┴────┴────┴────┴── ... ───┴────┴────┴────┘ │  ← 18 links
   │  G2 ───┴────┴────┴────┴── ... ───┴────┴────┴────┘ │  ← 18 links
   │   ...                                              │
   │  G70 ──┴────┴────┴────┴── ... ───┴────┴────┴────┘ │
   │  G71 ──┴────┴────┴────┴── ... ───┴────┴────┴────┘ │
   │                                                    │
   │              72 B200 GPUs                          │
   └────────────────────────────────────────────────────┘
```

The port-count balance is the easiest sanity check: GPU-side $72 \times 18 = 1{,}296$ ports, switch-side $18 \times 72 = 1{,}296$ ports — they match exactly, with no spare ports on either side.

Two consequences fall out of this wiring:

1. **Single-hop, non-blocking domain.** Any GPU-to-GPU path is exactly one switch hop; there is no oversubscription, no second tier, no in-rack blocking. The NVL72 NVLink domain behaves as a flat, fully-connected switch from the application's point of view.
2. **18-way striping for any pair.** Traffic between any two GPUs can be spread across all 18 switches in parallel, since each side has 18 independent links into the fabric. This is what lets a single pair sustain the full per-GPU bandwidth even while the other 70 GPUs are also communicating.

## 3. The 18 switches are NOT connected to each other

A natural follow-up question is: how are the 18 NVSwitch ASICs interconnected? The answer is that **they aren't**. Inside one NVL72 there are zero NVSwitch-to-NVSwitch links — every port on every switch is consumed by a GPU link, and the switches form 18 parallel, independent planes:

```
   PLANE 0              PLANE 1              ...        PLANE 17
   ┌────────┐           ┌────────┐                      ┌────────┐
   │ NVSw_0 │           │ NVSw_1 │                      │NVSw_17 │
   │72 ports│           │72 ports│                      │72 ports│
   └─┬─┬─┬──┘           └─┬─┬─┬──┘                      └─┬─┬─┬──┘
     │ │ │                │ │ │                           │ │ │
    G0 G1 ... G71        G0 G1 ... G71                  G0 G1 ... G71

           ✗ NO link between NVSw_i and NVSw_j (any i ≠ j) ✗
```

Each plane is a self-contained 72-endpoint switch. The planes never need to talk to each other because *every GPU has one foot in every plane*. When GPU 5 wants to send a tensor to GPU 42, the NVLink controller stripes the payload across all 18 ports in parallel:

```
GPU_5  →  GPU_42      (single point-to-point send, striped 18 ways)

   chunk 0   → GPU_5 port 0   → NVSw_0   → GPU_42 port 0
   chunk 1   → GPU_5 port 1   → NVSw_1   → GPU_42 port 1
   chunk 2   → GPU_5 port 2   → NVSw_2   → GPU_42 port 2
      ...
   chunk 17  → GPU_5 port 17  → NVSw_17  → GPU_42 port 17

   18 chunks × 50 GB/s per port  =  900 GB/s sustained, one direction.
```

GPU 42 reassembles the chunks. The 18 planes operate in lockstep — no one switch sees the whole message, and no switch ever has to forward to another switch. That is why the per-link bandwidth (50 GB/s) and the per-GPU bandwidth (900 GB/s) differ by exactly the port count (18).

The elegance of this design is that it eliminates the spine tier of a traditional fat-tree network. A leaf-and-spine fabric would treat the 18 switches as leaves and add a second tier of spine switches to connect them — costing extra latency, extra bandwidth, and extra hardware. NVIDIA skips that tier entirely by making the leaf radix (72 ports per ASIC) equal the endpoint count (72 GPUs). The trade-off is that the topology has no headroom: every port is already spoken for, so growing past 72 GPUs in a single domain (e.g., NVL576 = 8 NVL72 racks) requires either a separate scale-out fabric like InfiniBand or a second NVSwitch tier sitting *outside* the rack.

## 4. One GPU's view

Zooming in on a single GPU and tracing all 18 of its NVLink ports:

```
                       B200 GPU_0
                  ┌────────────────────┐
                  │   18 NVLink5 ports │
                  └─┬─┬─┬─┬──...──┬─┬─┬┘
                    │ │ │ │       │ │ │
   link 0  ─────────┘ │ │ │       │ │ │   ─→ NVSwitch_0   port 0
   link 1  ───────────┘ │ │       │ │ │   ─→ NVSwitch_1   port 0
   link 2  ─────────────┘ │       │ │ │   ─→ NVSwitch_2   port 0
   link 3  ───────────────┘       │ │ │   ─→ NVSwitch_3   port 0
        ...                       │ │ │
   link 15 ──────────────────────-┘ │ │   ─→ NVSwitch_15  port 0
   link 16 ────────────────────────-┘ │   ─→ NVSwitch_16  port 0
   link 17 ──────────────────────────-┘   ─→ NVSwitch_17  port 0
```

Every other GPU has the same fan-out, just landing on a different port number on each switch — GPU $k$ uses port $k$ on every NVSwitch ASIC. The wiring is symmetric across all 72 GPUs.

## 5. Where the "900 GB/s" comes from

The "900 GB/s" figure is the **single-direction** injection bandwidth from one GPU into the NVLink fabric (and equivalently the rate at which the fabric delivers traffic into one GPU). It is just NVLink 5's per-link primitive multiplied up by the 18 ports per GPU:

| Layer | Rate per direction | Notes |
|---|---|---|
| One signaling lane (one differential pair) | 200 Gbps PAM4 (100 GBd) | 4-level pulse amplitude modulation, NVLink 5 SerDes |
| One NVLink 5 **port**, single-direction | 2 lanes × 200 Gbps = 400 Gbps = **50 GB/s** | 8 bits per byte |
| One NVLink 5 **port**, bidirectional sum | 100 GB/s | transmit (TX) + receive (RX) |
| One GPU, single-direction (18 ports) | $18 \times 50\ \mathrm{GB/s} = \mathbf{900\ GB/s}$ | injection / egress bandwidth |
| One GPU, bidirectional sum (18 ports) | $18 \times 100\ \mathrm{GB/s} = \mathbf{1.8\ TB/s}$ | the headline NVLink 5 figure |

So whenever you see "900 GB/s NVLink 5 per GPU" in a spec or a paper, it means single-direction; "1.8 TB/s NVLink 5 per GPU" means TX + RX summed. NVIDIA marketing usually quotes the bidirectional figure, while cost models and analytical roofline calculations usually want the single-direction figure (because a collective's payload moves in one direction at a time across any given link).

## 6. Switch-side and system aggregates

Climbing back up to the switch and the whole rack:

| Scope | Per direction | Bidirectional sum |
|---|---|---|
| One NVSwitch ASIC (72 ports) | $72 \times 50\ \mathrm{GB/s} = 3.6\ \mathrm{TB/s}$ | 7.2 TB/s |
| Whole NVL72 fabric (18 ASICs) | $18 \times 3.6\ \mathrm{TB/s} = 64.8\ \mathrm{TB/s}$ | 129.6 TB/s |

The "130 TB/s NVLink fabric" headline number is the bidirectional rack aggregate. The "64.8 TB/s" figure that shows up in the system spec (as `B_agg_TBps`) is the same fabric measured single-direction — also equal to $72 \times 900\ \mathrm{GB/s}$, the simultaneous one-way injection of all 72 GPUs.

## 7. How this maps into llm_perf

The system spec at `llm_perf/database/system/gb200.72gpu.json` models the NVL72 NVLink domain as a single-tier `nvlink5` fabric:

```json
"tiers": [
  {
    "name": "intra-rack-nvswitch",
    "ports": 72,
    "bw_per_port_GBps": 900.0,
    "alpha_us": 0.5
  }
]
```

The 18-way striping across NVSwitch ASICs is *transparent* to the cost model: from a GPU's standpoint the whole NVLink domain looks like a flat, non-blocking switch with 72 endpoints, where each endpoint has 900 GB/s of single-direction injection bandwidth and a 0.5 μs per-hop latency floor. That is exactly what the spec encodes — `bw_per_port_GBps` is the per-direction GPU↔fabric figure, not the bidirectional sum, and `ports = 72` is the rack radix.

## 8. What it means for inference partitioning

A flat 72-port NVLink domain means any tensor parallelism (TP), expert parallelism (EP), or sequence parallelism (SP) collective with group size $G \le 72$ stays inside one rack and pays a single $(\alpha, BW) = (0.5\ \mu s, 900\ \mathrm{GB/s})$ per direction, regardless of which 72 ranks are selected. Larger group sizes — for instance, TP across two NVL72 racks — must escalate onto a slower scale-out fabric (typically InfiniBand or Ethernet), and the transition cost is the subject of the companion explainer [`when_hierarchical_scale_up_matters.md`](./when_hierarchical_scale_up_matters.md). As long as the partition keeps every collective within one NVL72, the cost model collapses to one fabric tier and 900 GB/s is the single load-bearing per-link constant.
