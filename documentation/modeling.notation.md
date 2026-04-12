# Notation Reference

Shared symbol definitions and architectural conventions for all `modeling.*.md` documents.  
Aligned with Megatron-LM (Shoeybi et al., 2019), DeepSpeed-MoE (Rajbhandari et al., 2022),
FlashAttention (Dao et al., 2022, 2023), and NVIDIA CUTLASS / cuBLAS GEMM dataflow.

Each section notes which document first uses or extends the symbols.

---

## 0. Parallelism Architecture
_(→ modeling.tpot.md §0.2)_

All documents in this suite assume a fixed nesting order for parallelism dimensions:

$$
\boxed{\text{DP} \;\rightarrow\; \text{PP} \;\rightarrow\; \text{EP} \;\rightarrow\; \text{TP} \;\rightarrow\; \text{SP}}
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

## 1. Parallelism Dimensions
_(→ modeling.tpot.md)_

- $DP$ — Data Parallelism. Number of full model replicas; each handles disjoint input batches.
  $$DP = \left\lfloor \frac{N_{\text{GPUs}}}{PP \cdot EP \cdot TP \cdot SP} \right\rfloor$$
- $PP$ — Pipeline Parallelism. Layers split into stages; each stage holds $L_{\text{stage}} = L / PP$ layers.
- $TP$ — Tensor Parallelism. Splits matrix multiplies across devices (Megatron-LM column/row parallel).
- $EP$ — Expert Parallelism. For MoE: experts partitioned across devices via all-to-all routing.
- $SP$ — Sequence Parallelism. Ring-attention-style KV sharding for inference; sequence dimension
  partitioned across devices for KV storage.

---

## 2. Model Dimensions
_(→ modeling.tpot.md)_

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

## 3. Sequence, Batch, and Precision
_(→ modeling.tpot.md §6.4 for batch scaling; → modeling.prefill.md for prefill batch)_

- $S$ — Decode context length (tokens in KV cache during decoding).
- $S_{\text{input}}$ — Input sequence length for prefill.
- $B$ — Static batch size (number of sequences processed together). $B=1$ for single-request decode.
- $B_{\text{eff}}$ — Effective batch size under continuous batching (varies per iteration).
- $B_{\text{prefill}}$ — Number of requests batched together during a prefill pass.
- $b$ — Bytes per parameter/activation element (e.g., bf16 → $b=2$, fp8 → $b=1$).

---

## 4. Memory
_(→ modeling.tpot.md; → modeling.kv.md for paging extensions)_

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
- $c_{\text{act}}$ — Empirical activation I/O constant per layer (unavoidable even with FlashAttention).

---

## 5. Device Compute and Bandwidth
_(→ modeling.tpot.md; → modeling.dram3d.md for 3D DRAM extensions)_

- $N_{\text{GPUs}}$ — Total devices in the cluster.
- $R_{\text{GPU}}$ — Peak compute throughput (FLOPs/s).
- $B_{\text{eff,mem}}$ — Effective HBM bandwidth (bytes/s).

---

## 6. Networking
_(→ modeling.tpot.md)_

- $\alpha_{TP}, \alpha_{EP}, \alpha_{SP}, \alpha_{PP}$ — Per-collective startup latency (α–β model).
- $B_{\text{eff,TP}}, B_{\text{eff,EP}}, B_{\text{eff,SP}}, B_{\text{eff,PP}}$ — Effective interconnect bandwidths.
- $n_{TP}$ — Number of TP collective iterations per layer per token.
- $n_{EP}$ — Number of EP collective iterations per layer per token.
- $n_{SP}$ — Number of SP collective iterations per layer per token.

---

## 7. FLOPs
_(→ modeling.tpot.md)_

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
- $F_{\text{norm}}$ — Normalization FLOPs per layer ($\approx c_{\text{norm}} H$).
- $c_{\text{norm}}$ — Norm FLOP coefficient (typically 5–10).
- $F_{\text{layer}}$ — Total FLOPs per layer (attention + FFN + norm).
- $F_{\text{layer,device}}$ — FLOPs per layer per device after sharding.
- $F_{\text{token,device}}$ — Total FLOPs per generated token on this device (decode).

---

## 8. Decode Timing and Throughput
_(→ modeling.tpot.md)_

- $t_{\text{compute}}$ — Per-token compute time: $F_{\text{token,device}} / R_{\text{GPU}}$.
- $t_{\text{mem}}$ — Per-token memory time: $T_{\text{token,device}}^{\text{eff}} / B_{\text{eff,mem}}$.
- $t_{\text{local}}$ — Roofline local time: $\max(t_{\text{compute}}, t_{\text{mem}})$.
- $t_{TP}, t_{EP}, t_{SP}, t_{PP}$ — Communication time per token per parallelism type.
- $t_{\text{comm}}$ — Combined communication time per token per PP stage.
- $t_{\text{token}}$ — Effective per-token decode time (overlap-aware):
  $$t_{\text{token}} = t_{\text{local}} + \max(0,\; t_{\text{comm}} - \rho \cdot t_{\text{local}})$$
- $\rho$ — Overlap factor $\in [0,1]$: fraction of $t_{\text{local}}$ that hides $t_{\text{comm}}$.
- $TPS_{\text{single}}$ — Per-DP-replica decode throughput (tokens/s).
- $TTPS$ — Global decode throughput across all DP replicas (tokens/s).

---

## 9. Batch Scaling
_(→ modeling.tpot.md §6.4 — to be populated)_

_Symbols for arithmetic intensity as a function of B, batched TPOT, and the
throughput–latency Pareto curve. Defined in modeling.tpot.md §6.4._

---

## 10. Prefill and TTFT
_(→ modeling.prefill.md — to be populated)_

- $F_{\text{prefill}}$ — Total FLOPs for the prefill pass (processes full input sequence).
- $t_{\text{prefill,local}}$ — Prefill compute + memory time on device.
- $t_{\text{prefill,comm}}$ — Prefill communication time.
- $TTFT$ — Time To First Token: total latency from request receipt to first generated token.

_Additional prefill batch and chunked-prefill symbols defined in modeling.prefill.md._

---

## 11. KV Cache Management
_(→ modeling.kv.md — to be populated)_

_Symbols for PagedAttention block size, fragmentation factor, and effective
HBM capacity after paging overhead. Defined in modeling.kv.md._

---

## 12. Framework Overhead
_(→ modeling.framework.md — to be populated)_

- $t_{\text{startup}}$ — Kernel warmup and CUDA graph capture overhead.

_Additional per-phase framework latency constants defined in modeling.framework.md._

---

## 13. End-to-End Metrics
_(→ modeling.e2e.md — to be populated)_

_Symbols for TPOT, interactivity, throughput/GPU, and Pareto frontier
under continuous batching. Defined in modeling.e2e.md._

---

## 14. 3D DRAM
_(→ modeling.dram3d.md — to be populated)_

_Symbols for hybrid bonding pitch, die area, pin density, data rate per pin,
and derived equivalent bandwidth. Defined in modeling.dram3d.md._
