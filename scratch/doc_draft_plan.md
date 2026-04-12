# Documentation Draft Plan
_Branch: `doc/modeling-extensions`_  
_Status: In progress — not merged to main_

---

## Overview

Extending the existing `modeling.methodology.md` with dedicated documents covering
prefill, KV management, batching, framework overhead, end-to-end metrics, and 3D DRAM
performance modeling.

---

## Document List

### Existing (to be updated)

**`documentation/modeling.methodology.md`**  
Core decode model — covers single-request AND static-batch decode.  
Sections to keep (unchanged):
- §0 Introduction and Notation
- §1 Memory Footprint
- §2 Memory Traffic During Decoding
- §3 Compute (FLOPs) per Token
- §4 Compute vs. Memory Bound (Roofline Model)
- §5 Communication Time During Decoding
- §6 End-to-End Latency, Throughput, and Partition Strategy

Sections to modify:
- §3.6 Prefill FLOPs → **remove**, move to `modeling.prefill.md`
- §6.4 TTFT → **remove**, move to `modeling.prefill.md`
- §0 Notation → **trim** symbols only used for prefill: `B`, `F_prefill`,
  `t_prefill_local`, `t_prefill_comm`, `t_startup`

Section to add:
- §6.4 Batch-Size Scaling and Throughput–Latency Tradeoff _(new)_
  - §6.4.1 Arithmetic intensity as a function of batch size B
  - §6.4.2 Batched TPOT and the compute-bound crossover
  - §6.4.3 Throughput vs. interactivity Pareto curve

Rationale: batching is a natural parametric extension of the existing
roofline model (B=1 is the degenerate case). The reader naturally asks
"what about B>1?" immediately after reading §4. Belongs in the main doc.

---

### New Documents

**`documentation/modeling.prefill.md`**  
TTFT and prefill latency model.
- Prefill FLOPs (migrated from §3.6)
- Compute-bound regime — contrast with memory-bound decode
- Prefill communication (message sizes scale with B×S, unlike decode)
- Disaggregated prefill clusters (migrated from §6.4)
- KV cache transfer latency between prefill and decode clusters

---

**`documentation/modeling.kv.md`**  
KV cache management overhead — analytical model only.
- PagedAttention block structure and capacity waste
- Fragmentation factor on effective HBM capacity
- Block allocation traffic overhead per token
- Interaction with SP and TP sharding

---

**`documentation/modeling.batching.md`**  
Dynamic serving and continuous batching complexity.
- Continuous batching: requests arriving and leaving mid-batch
- Iteration-level scheduling overhead
- Prefill–decode token mixing within a batch
- Interaction with KV paging (references `modeling.kv.md`)

---

**`documentation/modeling.framework.md`**  
Cross-cutting framework latency terms — applies across all phases.  
Organized as: (a) analytical terms with formulas, (b) empirical calibration constants.

| Overhead | Phase(s) | Type |
|---|---|---|
| Tokenization | Prefill entry | empirical |
| CUDA kernel launch overhead | Decode per-step | empirical |
| CUDA graph replay | Decode per-step | empirical |
| Request scheduling / batch assembly | Serving | empirical |
| Token sampling latency | Decode per-step | empirical |
| Response streaming / de-tokenization | Decode per-token | empirical |
| Disaggregated KV transfer | Prefill→decode boundary | analytical (α–β model) |

Referenced by: `modeling.prefill.md`, `modeling.batching.md`, `modeling.e2e.md`

---

**`documentation/modeling.e2e.md`**  
End-to-end metric assembly — capstone document.
- TTFT = prefill (`modeling.prefill.md`) + KV transfer + framework terms (`modeling.framework.md`)
- TPOT = batched decode (§6.4) + KV overhead (`modeling.kv.md`) + framework (`modeling.framework.md`)
- E2E latency = TTFT + TPOT × output_tokens
- Throughput/GPU and Interactivity (maps to InferenceX benchmark axes)
- Throughput–latency Pareto frontier under continuous batching (`modeling.batching.md`)

---

**`documentation/modeling.dram3d.md`**  
3D stacked DRAM performance model — derives equivalent memory bandwidth
from physical interface parameters for use in `SystemSpec`.

Input parameters:

| Symbol | Description | Example |
|---|---|---|
| $A_\text{die}$ (mm²) | DRAM die area | 100 mm² |
| $p_\text{HB}$ (µm) | Hybrid bonding pitch | 0.9 µm |
| $\eta_\text{data}$ | Data pin fraction (vs. power/ctrl/ECC) | 0.4 |
| $f_\text{data}$ (Gbps) | Data rate per pin | 8 Gbps |
| $N_\text{dies}$ | Number of DRAM dies stacked | 4 |

Core bandwidth derivation:
$$N_\text{pins} = \lfloor A_\text{die} / p_\text{HB}^2 \rfloor \cdot \eta_\text{data}$$
$$\text{BW}_\text{eff} = N_\text{pins} \cdot f_\text{data} / 8 \quad \text{[GB/s]}$$

Latency model (from AccelStack §III-C2):
$$\text{lat}_\text{3D} = \text{lat}_\text{HBM} / k_\text{interconnect}$$
where $k_\text{interconnect} > 2$ reflects shorter vertical interconnect vs. HBM.

Open question: multi-die stacking bandwidth aggregation — does each
die-to-die interface contribute independently, or only the bottom
logic-facing interface? (HBM model vs. true 3D model — TBD.)

Reference: AccelStack (Bai et al., 2025), HKUST FACT Lab.  
Eq. 2 in AccelStack is the GEMM compute latency (two-level tiling);
3D DRAM bandwidth is described in §III-C2 without a dedicated equation.
The derivation above is a first-principles extension of that section.

Output feeds into: `SystemSpec.device.hbm_bandwidth_GBps`  
Referenced by: `modeling.methodology.md` §4, `modeling.e2e.md`

---

## Dependency Chain

```
modeling.methodology.md ──────────────────────────────► modeling.e2e.md
modeling.prefill.md ──────────────────────────────────►      ▲
modeling.kv.md ───────────────────────────────────────►      │
modeling.batching.md  (depends on modeling.kv.md) ────►      │
modeling.framework.md  (cross-cutting) ───────────────►      │
modeling.dram3d.md  ──► feeds SystemSpec ─────────────────────┘
                        referenced by §4 roofline + e2e
```

---

## File Summary

```
documentation/
├── codebase.structure.md          existing, unchanged
├── equations.cheatsheet.md        existing, unchanged
├── modeling.methodology.md        existing, updated (§3.6 + §6.4 removed, §6.4 batching added)
├── modeling.prefill.md            new
├── modeling.kv.md                 new
├── modeling.batching.md           new
├── modeling.framework.md          new
├── modeling.e2e.md                new
└── modeling.dram3d.md             new
```

---

## Open Questions

1. **`modeling.dram3d.md`**: How to handle multi-die stacking BW aggregation?
   Each interface independently or only logic-facing interface counts?
2. **`modeling.batching.md`**: Scope of continuous batching model —
   iteration-level scheduling detail vs. high-level throughput approximation?
3. **`modeling.framework.md`**: Should empirical constants be left as
   calibration placeholders, or populated with measured vLLM/SGLang values?
