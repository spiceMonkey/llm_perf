# Notation Reference

Shared symbol definitions and architectural conventions for all documents under
`documentation/modeling/`. Each section notes which document first uses or extends
the symbols. Tagged citations (e.g. `[MEGATRON]`, `[FA2]`) resolve to entries in
`references.md`.

---

## 1. Parallelism Architecture
_(→ tpot.md)_

All documents in this suite assume a fixed nesting order for parallelism dimensions:

$$
\text{DP} \;\rightarrow\; \text{PP} \;\rightarrow\; \text{EP} \;\rightarrow\; \text{TP} \;\rightarrow\; \text{SP}
$$

This order reflects how model state is partitioned and reused during inference. Each level depends on all outer levels having already determined weight placement, token routing, or tensor partitioning.

| Level | What it partitions | Why this ordering is required |
|-------|--------------------|------------------------------|
| **DP** | Entire model replica | Must wrap all state; inner groups cannot cross DP boundaries. |
| **PP** | Layers | Layer ownership must be decided before experts/tensor shards are assigned. |
| **EP** | Experts | Expert placement must be fixed before tensor sharding splits expert matrices. |
| **TP** | Weight matrices | TP defines weight shards used identically across all SP ranks. |
| **SP** | KV cache sequence dimension | KV is activation state only; must be sharded after all weight placement. |

### DP — outermost (replicated model weights)

DP creates fully independent model replicas for throughput scaling. No weight partitioning happens inside DP groups; all inner dimensions (PP, EP, TP, SP) apply **within** each DP replica.

### PP — inside DP (layers assigned before experts/tensor sharding)

PP determines **which layers live on which devices**. Only after PP is fixed can experts (EP), tensor dimensions (TP), and KV partitions (SP) be assigned. PP stages own their local weights and KV cache.

### EP — inside PP (expert groups belong to specific layers)

EP distributes MoE experts within the layers assigned by PP. Expert weights must be placed (EP) before tensor-parallel shards apply (TP). The expert-parallel degree must satisfy:
$$EP \le N_{\text{exp}}$$
In practice EP usually divides $N_{\text{exp}}$, but the only hard constraint is $EP \le N_{\text{exp}}$.

### TP — inside EP (tensor sharding within a defined expert/layer partition)

TP splits matrices within each expert or dense block. After TP, each rank holds a fraction of $H$ or $H_{kv}$. SP requires all SP ranks to share identical TP-sharded weights, so TP must precede SP.

### SP — innermost (KV sharding after all weights are fixed)

SP shards the **KV cache** (activation state), not model parameters. Only after DP/PP/EP/TP are fixed can the KV sequence dimension be partitioned.

---

## 2. Parallelism Dimensions
_(→ tpot.md)_

- $DP$ — Data Parallelism. Number of full model replicas; each handles disjoint input batches.
  $$DP = \left\lfloor \frac{N_{\text{GPUs}}}{PP \cdot EP \cdot TP \cdot SP} \right\rfloor$$
- $PP$ — Pipeline Parallelism. Layers split into stages; each stage holds $L_{\text{stage}} = L / PP$ layers.
- $TP$ — Tensor Parallelism. Splits matrix multiplies across devices (Megatron-LM column/row parallel).
- $EP$ — Expert Parallelism. For MoE: experts partitioned across devices via all-to-all routing.
- $SP$ — Sequence Parallelism. Ring-attention-style KV sharding for inference; sequence dimension
  partitioned across devices for KV storage.

---

## 3. Model Dimensions
_(→ tpot.md)_

- $L$ — Number of transformer layers.
- $L_{\text{moe}}$ — Number of MoE layers (defaults to $L$ if all layers are MoE, or $0$ for dense).
- $L_{\text{dense}} = L - L_{\text{moe}}$ — Number of dense layers.
- $V$ — Vocabulary size.
- $H$ — Hidden size (model dimension); applies to embeddings, LM head, FFN, and attention projections.
- $n_q$ — Number of query heads.
- $d_{\text{head}} = H / n_q$ — Head dimension.
- $n_{kv}$ — Number of KV heads (in GQA, $n_{kv} < n_q$).
- $H_{kv} = n_{kv} \cdot d_{\text{head}}$ — Total KV projection dimension.
- $I_{\text{dense}}$ — FFN intermediate dimension for dense layers.
- $I_{\text{moe}}$ — FFN intermediate dimension per MoE expert layer.
- $I_{\text{eff}}$ — Unified FFN intermediate dimension for FLOPs:
  $I_{\text{eff}} = I_{\text{dense}}$ (dense) or $k \cdot I_{\text{moe}}$ (MoE).
- $N_{\text{exp}}$ — Number of experts per MoE layer.
- $N_{\text{eff}}$ — Unified expert count for FLOPs: $0$ (dense) or $N_{\text{exp}}$ (MoE).
- $k$ — Number of experts selected per token (top-$k$ routing).

---

## 4. Sequence, Batch, and Precision
_(→ tpot.md §6.4 for batch scaling; → prefill.md for prefill batch)_

- $S$ — Decode context length (tokens in KV cache during decoding).
- $S_{\text{input}}$ — Input sequence length for prefill.
- $B$ — Decode batch size: number of **independent user requests** decoded in the same step, each carrying its own KV cache. Weights are loaded once and shared across all $B$ requests; KV reads scale linearly with $B$. $B=1$ for single-request decode.
- $B_{\text{eff}}$ — **Per-step** realized decode batch size under continuous batching: the number of user requests contributing a decode token in a single iteration. Unlike the static $B$ (configured or peak admissible), $B_{\text{eff}}$ fluctuates step to step as requests finish (EOS) or new ones are admitted, and may be smaller than $B$ if prefill slots displace decode slots in the same step (e.g. chunked-prefill). The steady-state mean is $\overline{B_{\text{eff}}}$ (§14).
- $B_{\text{prefill}}$ — Number of independent user requests batched together in a single prefill pass.
- $b$ — Bytes per parameter/activation element (e.g., bf16 → $b=2$, fp8 → $b=1$).

---

## 5. Memory
_(→ tpot.md; → kv.md for paging extensions)_

Parameter sizes:
- $P_{\text{attn}}$ — Attention parameter count.
- $P_{\text{FFN}}$ — Unified FFN/MoE parameter count.
- $P_{\text{emb}}$ — Embedding parameter count.
- $P_{\text{lm}}$ — LM head parameter count (0 if weight-tied).

Memory capacity (bytes **stored** in HBM):
- $M_{\theta,\text{device}}$ — Parameter memory on this device.
- $M_{\text{KV,device}}$ — KV cache storage (keys + values).
- $M_{\text{act,device}}$ — Activation working memory per token during decoding.
- $M_{\text{HBM}}$ — Available HBM capacity per device.

Memory traffic (bytes **moved** between HBM and compute per token):
- $T_{\theta,\text{device}}$ — Parameter traffic (weights read per token).
- $T_{\text{KV,device}}$ — KV traffic (read + write per new token).
- $T_{\text{act,device}}$ — Activation traffic (intermediate reads/writes).
- $T_{\text{token,device}}$ — Total per-token traffic on this device.
- $T_{\text{token,device}}^{\text{eff}}$ — Effective traffic after FlashAttention-style optimizations.

---

## 6. Device Compute and Bandwidth
_(→ tpot.md; → dram3d.md for 3D DRAM extensions)_

- $N_{\text{GPUs}}$ — Total devices in the cluster.
- $R_{\text{GPU}}$ — Peak compute throughput (FLOPs/s).
- $BW_{\text{mem}}$ — Effective HBM bandwidth (bytes/s).

---

## 7. Networking
_(→ tpot.md; → switching.md §1 for fabric scope)_

- $\alpha_{TP}, \alpha_{EP}, \alpha_{SP}, \alpha_{PP}$ — **Effective end-to-end** per-collective startup latency (α–β model), measured across the scale-up fabric that carries that domain's traffic. Assumes a **single switching tier** — see *Scope note* below.
- $BW_{\text{TP}}, BW_{\text{EP}}, BW_{\text{SP}}, BW_{\text{PP}}$ — **Effective end-to-end** interconnect bandwidths (single-direction, GB/s) across that same single-tier fabric. All bandwidth quantities in this suite are single-direction unless explicitly labeled bidirectional.
- $n_{TP}$ — Number of TP collective iterations per layer per token.
- $n_{EP}$ — Number of EP collective iterations per layer per token.
- $n_{SP}$ — Number of SP collective iterations per layer per token.

**Scope note — single-tier fabric.** Every $\alpha_{role}$ / $BW_{role}$ above represents the total cost of a collective that traverses exactly one switching tier (one NVSwitch fabric, one UALink leaf, one monolithic crossbar). When hierarchical multi-tier support lands, these symbols will generalize to per-tier values $\alpha_{role,i}$ / $BW_{role,i}$ for tier $i$, and a collective whose rank-set spans $k$ tiers pays $\sum_{i \le k} \alpha_{role,i}$ with the bandwidth floor of the narrowest tier it crosses. See `switching.md` §1 for the current scope boundary.

---

## 8. FLOPs
_(→ tpot.md)_

Attention:
- $F_Q, F_K, F_V, F_O$ — FLOPs for Q, K, V, output projections.
- $F_{\text{proj}}$ — Combined Q/K/V/O projection FLOPs.
- $F_{\text{score}}$ — Attention score FLOPs ($QK^\top$).
- $F_{\text{value}}$ — Value application FLOPs (Attn·V).
- $F_{\text{attn,KV}}$ — Score + value FLOPs combined.
- $F_{\text{attn}}$ — Total attention FLOPs per layer.

FFN and MoE:
- $F_{\text{ffn,dense}}$ — Dense FFN FLOPs per layer.
- $F_{\text{router}}$ — Router FLOPs per token (MoE).
- $F_{\text{expert}}$ — FLOPs per expert MLP per token.
- $F_{\text{ffn,moe}}$ — MoE FFN FLOPs per layer.
- $F_{\text{ffn}}$ — Unified FFN FLOPs (dense or MoE).

Layer and token:
- $F_{\text{layer}}$ — Total FLOPs per layer (attention + FFN; norm dropped as negligible).
- $F_{\text{layer,device}}$ — FLOPs per layer per device after sharding.
- $F_{\text{token,device}}$ — Total FLOPs per generated token on this device (decode).

---

## 9. Decode Timing and Throughput
_(→ tpot.md)_

- $t_{\text{compute}}$ — Per-token compute time: $F_{\text{token,device}} / R_{\text{GPU}}$.
- $t_{\text{mem}}$ — Per-token memory time: $T_{\text{token,device}}^{\text{eff}} / BW_{\text{mem}}$.
- $t_{\text{local}}$ — Roofline local time: $\max(t_{\text{compute}}, t_{\text{mem}})$.
- $t_{TP}, t_{EP}, t_{SP}, t_{PP}$ — Communication time per step per parallelism type (message sizes scale with $B$; see tpot.md §5).
- $t_{\text{comm}}$ — Combined communication time per decode step per PP stage.
- $t_{\text{stage}}$ — Per-PP-stage step time (overlap-aware, pre-bubble):
  $$t_{\text{stage}} = t_{\text{local}} + \max(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}})$$
- $\gamma_{\text{pp}}$ — Pipeline-bubble multiplier:
  $$\gamma_{\text{pp}} = \max\left(1,\; \frac{PP}{B}\right)$$
  Equal to 1 when the pipeline is kept full ($B \ge PP$); greater than 1 when a single microbatch must traverse all PP stages sequentially ($B < PP$).
- $t_{\text{step,user}}$ — User-observed per-step decode time:
  $$t_{\text{step,user}} = t_{\text{stage}} \cdot \gamma_{\text{pp}}$$
- $\rho$ — Overlap factor $\in [0,1]$: fraction of $t_{\text{local}}$ that hides $t_{\text{comm}}$.
- $TPS_{\text{single}}$ — Per-DP-replica decode throughput: $B / t_{\text{step,user}}$ (tokens/s).
- $TTPS$ — Global decode throughput across all DP replicas: $DP \cdot B / t_{\text{step,user}}$ (tokens/s).

---

## 10. Batch Scaling
_(→ tpot.md §6.4)_

- $OI(B)$ — Operational intensity as a function of batch size $B$:
  $$OI(B) = \frac{B \times F_{\text{token,device}}}{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}$$
- $B^*$ — Crossover batch size where the roofline transitions from memory-bound to compute-bound:
  $$B^* = \frac{T_{\theta,\text{device}} \times R_{\text{GPU}}}{F_{\text{token,device}} \times BW_{\text{mem}} - T_{\text{KV,device}} \times R_{\text{GPU}}}$$
  **Existence:** finite and positive iff $F_{\text{token,device}} / T_{\text{KV,device}} > R_{\text{ridge}}$ (asymptotic OI ceiling exceeds the ridge point). When violated — e.g., very long contexts on small models — decode stays memory-bound at every $B$ and $B^{\star} \to \infty$ (tpot.md §6.4.1).
- $\text{TPOT}(B)$ — Batched Time Per Output Token (user-observed): $t_{\text{step,user}}(B)$.
  Memory-bound ($B \ll B^*$): $\approx T_{\theta,\text{device}} / BW_{\text{mem}}$ (flat in $B$).
  Compute-bound ($B \gg B^*$): $\approx B \cdot F_{\text{token,device}} / R_{\text{GPU}}$ (linear in $B$).

---

## 11. Prefill and TTFT
_(→ prefill.md)_

FLOPs:
- $F_{\text{proj,prefill}}$ — Q/K/V/O projection FLOPs for prefill: $(4H^2 + 4HH_{kv}) S_{\text{input}}$.
- $F_{\text{score,prefill}}$ — Attention score FLOPs: $2 S_{\text{input}}^2 H$.
- $F_{\text{value,prefill}}$ — Value application FLOPs: $2 S_{\text{input}}^2 H$.
- $F_{\text{ffn,prefill}}$ — FFN FLOPs for prefill: $6 H I_{\text{eff}} S_{\text{input}}$.
- $F_{\text{layer,prefill}}$ — Per-layer prefill FLOPs (projections + $S^2$ attention).
- $F_{\text{prefill,device}}$ — Total prefill FLOPs per device across all layers on this PP stage.

Timing:
- $t_{\text{prefill,compute}}$ — Prefill compute time: $F_{\text{prefill,device}} / R_{\text{GPU}}$.
- $t_{\text{prefill,mem}}$ — Prefill memory time: $T_{\text{prefill,device}} / BW_{\text{mem}}$.
- $t_{\text{prefill,local}}$ — Prefill roofline local time: $\max(t_{\text{prefill,compute}}, t_{\text{prefill,mem}})$.
- $t_{\text{prefill,comm}}$ — Total prefill communication time (TP/EP/SP/PP collectives).
- $t_{\text{chunk}}$ — Latency of one chunked-prefill iteration.
- $t_{\text{pipeline,warmup}}$ — Pipeline fill time: $(PP-1) \times t_{\text{stage}}$.
- $t_{\text{prefill}}$ — Hardware prefill latency for one request: $\max(t_{\text{prefill,local}},\; t_{\text{prefill,comm}}) + t_{\text{pipeline,warmup}}$ (derived in prefill.md §3).
- $t_{\text{prefill,total}}$ — End-to-end hardware prefill latency including handoff: $t_{\text{prefill}} + t_{\text{handoff}} + t_{\text{pipeline,warmup,dec}}$ (prefill.md §6.5).
- $TTFT$ — Time To First Token (assembled from the terms above plus framework overheads in §13):
  $$TTFT = t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{handoff}} + t_{\text{step,user}}$$
  where $t_{\text{sched}}, t_{\text{tok}}$ are defined in §13 and $t_{\text{handoff}}$ is below (0 only when prefill and decode share an identical partition).
- $TTFT_{\text{disagg}}$ — TTFT for disaggregated prefill architecture (uses the refined $t_{\text{KV-transfer}}^{\text{eff}}$ for $t_{\text{handoff}}$).

Chunked prefill:
- $C$ — Chunk size in tokens.
- $N_{\text{chunks}}$ — Number of chunks: $\lceil S_{\text{input}} / C \rceil$.

Crossover:
- $S_{\text{input}}^{\star}$ — Prefill compute-bound crossover: $S_{\text{input}}^{\star} = (b/2) \times R_{\text{ridge}}$.
- $R_{\text{ridge}}$ — Device ridge point: $R_{\text{GPU}} / BW_{\text{mem}}$ (FLOPs/byte).

KV handoff — volumes (cluster-aggregate and per-device views):
- $M_{\text{KV,total}}$ — Total KV bytes from one prefill pass (cluster-aggregate): $2 L S_{\text{input}} H_{kv} b$.
- $M_{\text{KV,shard,p}}$ — Per-prefill-device KV shard (all layers that device holds): $M_{\text{KV,total}} / (TP_p \cdot SP_p)$ for one PP stage; used by prefill.md §6.
- $M_{\text{KV-transfer}}$ — Per-device KV bytes to transfer (sharded across $TP \cdot SP$, one PP stage's worth of layers): $2 S_{\text{input}} H_{kv} b \times L / (PP \times TP \times SP)$. Used by e2e.md in the simple α–β model.

KV handoff — latency model (prefill.md §6):
- $t_{\text{handoff}}$ — KV handoff time from prefill to decode; co-located or disaggregated branch:
  $$t_{\text{handoff}} = \begin{cases} t_{\text{handoff,colo}} & \text{(co-located, §6.3)}\\ t_{\text{KV-transfer}}^{\text{eff}} & \text{(disaggregated, §6.4)} \end{cases}$$
- $t_{\text{handoff,colo}}$ — Co-located KV layout-transition latency (scale-up collective):
  $$t_{\text{handoff,colo}} = \alpha_{\text{intra}} + \frac{M_{\text{KV,total}}}{BW_{\text{intra}}} \cdot \eta_{\text{repack}}$$
  Equals 0 only when prefill and decode partitions match exactly.
- $t_{\text{KV-transfer}}^{\text{bulk}}$ — Textbook α–β disaggregated transfer (no overlap, no overheads): $\alpha_{\text{inter}} + M_{\text{KV,total}} / BW_{\text{inter}}$.
- $t_{\text{KV-transfer}}^{\text{eff}}$ — Refined disaggregated transfer (overheads + layer-wise streaming):
  $$t_{\text{KV-transfer}}^{\text{eff}} = \max\!\left(0,\; \alpha_{\text{inter}}^{\text{eff}} + \frac{M_{\text{KV,total}}}{BW_{\text{inter}}} + t_{\text{repack}} - \rho_{KV}\cdot t_{\text{prefill}} \right)$$
- $t_{\text{KV-transfer}}$ — Generic KV transfer latency in the simple α–β model (used by e2e.md): $\alpha_{\text{inter}} + M_{\text{KV-transfer}} / BW_{\text{inter}}$ (0 for co-located prefill+decode in the simple model).
- $t_{\text{repack}}$ — Layout repack on decode side (scale-up all-gather): $M_{\text{KV,total}} / BW_{\text{intra,d}} \cdot \eta_{\text{repack}}$.

KV handoff — bandwidth and startup parameters:
- $BW_{\text{inter}}$ — *Effective, delivered* end-to-end per-GPU inter-cluster bandwidth (calibration knob). Absorbs PCIe egress, NIC sharing, and HBM-write inefficiencies; is **not** the NIC catalog line rate. See prefill.md §6.4.
- $BW_{\text{intra}}$, $BW_{\text{intra,d}}$ — Scale-up fabric bandwidth (NVLink / NVSwitch); decode-side variant for repack cost.
- $\alpha_{\text{inter}}$ — Inter-cluster link startup latency (single round-trip).
- $\alpha_{\text{inter}}^{\text{eff}}$ — Effective startup including RDMA WR posting: $\alpha_{\text{inter}} + N_{\text{WR}} \cdot \tau_{\text{WR}}$.
- $\alpha_{\text{intra}}$ — Scale-up collective startup (≈1–5 µs over NVLink/NVSwitch).
- $\eta_{\text{repack}}$ — Layout-repack inefficiency factor ($\in [1, 2]$); covers non-contiguous gather + paged-block writes.
- $\rho_{KV}$ — Layer-wise streaming overlap factor for disaggregated KV transfer ($\in [0, 1]$); fraction of $t_{\text{prefill}}$ that hides KV transfer (MoonCake / NVIDIA Dynamo pattern).
- $N_{\text{WR}}$ — Number of RDMA work requests posted in one handoff: $\approx L \cdot TP_p \cdot SP_p$.
- $\tau_{\text{WR}}$ — Per-RDMA-WR posting latency (≈1 µs).
- $t_{\text{pipeline,warmup,dec}}$ — Pipeline warmup on decode cluster after handoff: $(PP_{\text{dec}} - 1) \cdot t_{\text{stage,dec}}$.

---

## 12. KV Cache Management
_(→ kv.md)_

- $\text{BLK}_{KV}$ — KV block size in tokens (PagedAttention page size; typical: 16 or 32).
- $N_{\text{blocks}}(S)$ — Blocks allocated for a sequence of length $S$: $\lceil S / \text{BLK}_{KV} \rceil$.
- $\varphi(S)$ — Fragmentation factor: ratio of allocated KV memory to ideally occupied KV memory.
  $$\varphi(S) = \frac{\lceil S / \text{BLK}_{KV} \rceil \times \text{BLK}_{KV}}{S}, \qquad \varphi_{\text{avg}} \approx 1 + \frac{\text{BLK}_{KV}}{2S}$$
- $M_{\text{HBM,KV,avail}}$ — HBM capacity available for KV storage after weights and activations.
- $S_{\max}$ — Maximum supportable context length given $M_{\text{HBM,KV,avail}}$ and $\varphi$:
  $$S_{\max} \approx \frac{M_{\text{HBM,KV,avail}} \times TP \times SP}{\varphi_{\text{avg}} \times 2 H_{kv} b \times L/PP}$$

---

## 13. Framework Overhead
_(→ framework.md)_

Scope: CPU / software-stack overhead only. Network-fabric overheads (e.g., disaggregated KV transfer) are handled in §11 / prefill.md §6.

Per-request (once):
- $t_{\text{tok}}$ — Tokenization latency (CPU BPE/SP processing).
- $t_{\text{sched}}$ — Request scheduling / batch assembly latency.

Per-step (each decode iteration):
- $t_{\text{launch}}$ — CUDA kernel launch overhead per decode step (no CUDA graph).
- $t_{\text{graph}}$ — CUDA graph replay latency per decode step.
- $t_{\text{sample}}$ — Token sampling latency (logits → token ID).
- $t_{\text{detok}}$ — Response streaming / detokenization latency per output token.

Request scope:
- $T_{\text{out}}$ — Number of output tokens per request.

_(Total framework overhead $t_{\text{framework}}$ is assembled in framework.md §3.)_

---

## 14. End-to-End Metrics
_(→ e2e.md)_

Core metrics:
- $TTFT$ — Time To First Token (defined in §11 above; assembled in e2e.md §2).
- $\text{TPOT}(B)$ — Time Per Output Token (user-observed): $t_{\text{step,user}}(B)$ (defined in §10 above).
- $\text{Tput/GPU}$ — Output tokens per second per GPU: $TTPS / N_{\text{GPUs}}$.
- $\text{Interactivity}$ — Per-user output rate: $1 / \text{TPOT} = 1 / t_{\text{step,user}}(B)$ (tokens/s/request).
- $\text{Goodput}$ — Maximum request arrival rate $\lambda$ the cluster can sustain while keeping both TTFT and TPOT below operator-set SLOs at percentile $p$ (e2e.md §1.5):
  $$\text{Goodput} = \max\,\lambda \;\;\text{s.t.}\;\; P_{p}[TTFT(\lambda)] \le TTFT_{\text{SLO}} \;\text{and}\; P_{p}[\text{TPOT}(\lambda)] \le \text{TPOT}_{\text{SLO}}$$
  **Scope note:** speculative decoding, preemption-driven recompute, and cancellation effects are real goodput drains but not modeled in this suite.
- $TTFT_{\text{SLO}}$ — Operator-set upper bound on TTFT (seconds) used in the goodput definition.
- $\text{TPOT}_{\text{SLO}}$ — Operator-set upper bound on TPOT (seconds) used in the goodput definition.
- $p$ — SLO compliance percentile (typically 90 or 99).

Continuous batching:
- $\lambda$ — Request arrival rate (requests/second).
- $N_{\text{out}}$ — Number of output tokens in a single response.
- $N_{\text{out}}^{\star}$ — Crossover output length where TTFT equals the decode contribution: $N_{\text{out}}^{\star} \approx TTFT / \text{TPOT} + 1$ (e2e.md §4).
- $\overline{B_{\text{eff}}}$ — Mean effective batch size in steady-state continuous batching.
- $\overline{\text{TPOT}}$ — Average TPOT over a request's decode lifetime.
- $N_{\text{GPUs,per-replica}}$ — GPUs per DP replica: $PP \cdot TP \cdot EP \cdot SP$.

Pareto relationship (per-replica, parameterized by $B$):
$$\text{Tput/GPU} \times \text{TPOT} = \frac{B}{N_{\text{GPUs,per-replica}}}$$
The ceiling $1/N_{\text{GPUs,per-replica}}$ applies at $B=1$; for $B>1$, a single replica can sit on a higher hyperbola.

---

## 15. 3D DRAM
_(→ dram3d.md)_

Physical parameters:
- $A_{\text{die}}$ — DRAM die area (mm²).
- $p_{HB}$ — Hybrid bonding pitch: center-to-center pad spacing (µm).
- $\eta_{\text{data}}$ — Data pin fraction: proportion of pads carrying data signals.
- $f_{\text{data}}$ — Data rate per pin (Gbps).
- $N_{\text{dies}}$ — Number of DRAM dies stacked on the logic die.

Derived:
- $N_{\text{pins,total}}$ — Total pad count (area-limited): $\lfloor A_{\text{die}} / p_{HB}^2 \rfloor$.
- $N_{\text{pins,data}}$ — Data pad count: $N_{\text{pins,total}} \times \eta_{\text{data}}$.
- $BW_{\text{die}}$ — Raw bandwidth per die interface: $N_{\text{pins,data}} \times f_{\text{data}} / 8$ (GB/s).
- $BW_{\text{conservative}}$ — Lower bound: single logic-facing interface ($= BW_{\text{die}}$).
- $BW_{\text{optimistic}}$ — Upper bound: independent per-die interfaces ($= N_{\text{dies}} \times BW_{\text{die}}$).

Latency:
- $k_{\text{interconnect}}$ — Latency reduction factor vs. standard HBM bump interconnect.
- $\ell_{3D}$ — Estimated 3D DRAM read latency: $\ell_{\text{HBM}} / k_{\text{interconnect}}$ (ns).
