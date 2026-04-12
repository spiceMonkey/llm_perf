# Notation Reference

Shared symbol definitions and architectural conventions for all `modeling.*.md` documents.  
Aligned with Megatron-LM (Shoeybi et al., 2019), DeepSpeed-MoE (Rajbhandari et al., 2022),
FlashAttention (Dao et al., 2022, 2023), and NVIDIA CUTLASS / cuBLAS GEMM dataflow.

Each section notes which document first uses or extends the symbols.

---

## 1. Parallelism Architecture
_(→ modeling.tpot.md)_

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

## 2. Parallelism Dimensions
_(→ modeling.tpot.md)_

- $DP$ — Data Parallelism. Number of full model replicas; each handles disjoint input batches.
  $$DP = \left\lfloor \frac{N_{\text{GPUs}}}{PP \cdot EP \cdot TP \cdot SP} \right\rfloor$$
- $PP$ — Pipeline Parallelism. Layers split into stages; each stage holds $L_{\text{stage}} = L / PP$ layers.
- $TP$ — Tensor Parallelism. Splits matrix multiplies across devices (Megatron-LM column/row parallel).
- $EP$ — Expert Parallelism. For MoE: experts partitioned across devices via all-to-all routing.
- $SP$ — Sequence Parallelism. Ring-attention-style KV sharding for inference; sequence dimension
  partitioned across devices for KV storage.

---

## 3. Model Dimensions
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

## 4. Sequence, Batch, and Precision
_(→ modeling.tpot.md §6.4 for batch scaling; → modeling.prefill.md for prefill batch)_

- $S$ — Decode context length (tokens in KV cache during decoding).
- $S_{\text{input}}$ — Input sequence length for prefill.
- $B$ — Static batch size (number of sequences processed together). $B=1$ for single-request decode.
- $B_{\text{eff}}$ — Effective batch size under continuous batching (varies per iteration).
- $B_{\text{prefill}}$ — Number of requests batched together during a prefill pass.
- $b$ — Bytes per parameter/activation element (e.g., bf16 → $b=2$, fp8 → $b=1$).

---

## 5. Memory
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

---

## 6. Device Compute and Bandwidth
_(→ modeling.tpot.md; → modeling.dram3d.md for 3D DRAM extensions)_

- $N_{\text{GPUs}}$ — Total devices in the cluster.
- $R_{\text{GPU}}$ — Peak compute throughput (FLOPs/s).
- $B_{\text{eff,mem}}$ — Effective HBM bandwidth (bytes/s).

---

## 7. Networking
_(→ modeling.tpot.md)_

- $\alpha_{TP}, \alpha_{EP}, \alpha_{SP}, \alpha_{PP}$ — Per-collective startup latency (α–β model).
- $B_{\text{eff,TP}}, B_{\text{eff,EP}}, B_{\text{eff,SP}}, B_{\text{eff,PP}}$ — Effective interconnect bandwidths.
- $n_{TP}$ — Number of TP collective iterations per layer per token.
- $n_{EP}$ — Number of EP collective iterations per layer per token.
- $n_{SP}$ — Number of SP collective iterations per layer per token.

---

## 8. FLOPs
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
- $F_{\text{layer}}$ — Total FLOPs per layer (attention + FFN; norm dropped as negligible).
- $F_{\text{layer,device}}$ — FLOPs per layer per device after sharding.
- $F_{\text{token,device}}$ — Total FLOPs per generated token on this device (decode).

---

## 9. Decode Timing and Throughput
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

## 10. Batch Scaling
_(→ modeling.tpot.md §6.4)_

- $OI(B)$ — Operational intensity as a function of batch size $B$:
  $$OI(B) = \frac{B \times F_{\text{token,device}}}{T_{\theta,\text{device}} + B \times T_{\text{KV,device}}}$$
- $B^*$ — Crossover batch size where the roofline transitions from memory-bound to compute-bound:
  $$B^* = \frac{T_{\theta,\text{device}} \times R_{\text{GPU}}}{F_{\text{token,device}} \times B_{\text{eff,mem}} - T_{\text{KV,device}} \times R_{\text{GPU}}}$$
- $\text{TPOT}(B)$ — Batched Time Per Output Token (per sequence): $t_{\text{token}}(B) / B$.
  Memory-bound: $\approx T_{\theta,\text{device}} / (B \times B_{\text{eff,mem}})$; compute-bound: $\approx F_{\text{token,device}} / R_{\text{GPU}}$.

---

## 11. Prefill and TTFT
_(→ modeling.prefill.md)_

FLOPs:
- $F_{\text{proj,prefill}}$ — Q/K/V/O projection FLOPs for prefill: $(2H^2 + 6HH_{kv}) S_{\text{input}}$.
- $F_{\text{score,prefill}}$ — Attention score FLOPs: $2 S_{\text{input}}^2 H_{kv}$.
- $F_{\text{value,prefill}}$ — Value application FLOPs: $2 S_{\text{input}}^2 H_{kv}$.
- $F_{\text{ffn,prefill}}$ — FFN FLOPs for prefill: $4 H I_{\text{eff}} S_{\text{input}}$.
- $F_{\text{layer,prefill}}$ — Per-layer prefill FLOPs (projections + $S^2$ attention).
- $F_{\text{prefill,device}}$ — Total prefill FLOPs per device across all layers on this PP stage.

Timing:
- $t_{\text{prefill,compute}}$ — Prefill compute time: $F_{\text{prefill,device}} / R_{\text{GPU}}$.
- $t_{\text{prefill,mem}}$ — Prefill memory time: $T_{\text{prefill,device}} / B_{\text{eff,mem}}$.
- $t_{\text{prefill,local}}$ — Prefill roofline local time: $\max(t_{\text{prefill,compute}}, t_{\text{prefill,mem}})$.
- $t_{\text{prefill,comm}}$ — Total prefill communication time (TP/EP/SP/PP collectives).
- $t_{\text{chunk}}$ — Latency of one chunked-prefill iteration.
- $t_{\text{pipeline,warmup}}$ — Pipeline fill time: $(PP-1) \times t_{\text{stage}}$.
- $TTFT$ — Time To First Token: $t_{\text{sched}} + t_{\text{tok}} + t_{\text{prefill}} + t_{\text{KV\_transfer}} + t_{\text{token}}$.
- $TTFT_{\text{disagg}}$ — TTFT for disaggregated prefill architecture (includes KV transfer).

Chunked prefill:
- $C$ — Chunk size in tokens.
- $N_{\text{chunks}}$ — Number of chunks: $\lceil S_{\text{input}} / C \rceil$.

Crossover:
- $S_{\text{input}}^{\star}$ — Prefill compute-bound crossover: $S_{\text{input}}^{\star} = (b/2) \times R_{\text{ridge}}$.
- $R_{\text{ridge}}$ — Device ridge point: $R_{\text{GPU}} / B_{\text{eff,mem}}$ (FLOPs/byte).

KV transfer:
- $M_{\text{KV,total}}$ — Total KV bytes from one prefill pass: $2 L S_{\text{input}} H_{kv} b$.
- $t_{\text{KV,transfer}}$ — KV transfer latency: $\alpha_{\text{inter}} + M_{\text{KV,total}} / B_{\text{eff,inter}}$.
- $B_{\text{eff,inter}}$ — Effective inter-cluster bandwidth.
- $\alpha_{\text{inter}}$ — Inter-cluster link startup latency.

---

## 12. KV Cache Management
_(→ modeling.kv.md)_

- $B_{\text{block}}$ — KV block size in tokens (PagedAttention page size; typical: 16 or 32).
- $N_{\text{blocks}}(S)$ — Blocks allocated for a sequence of length $S$: $\lceil S / B_{\text{block}} \rceil$.
- $\varphi(S)$ — Fragmentation factor: ratio of allocated KV memory to ideally occupied KV memory.
  $$\varphi(S) = \frac{\lceil S / B_{\text{block}} \rceil \times B_{\text{block}}}{S}, \qquad \varphi_{\text{avg}} \approx 1 + \frac{B_{\text{block}}}{2S}$$
- $M_{\text{HBM,KV,avail}}$ — HBM capacity available for KV storage after weights and activations.
- $S_{\max}$ — Maximum supportable context length given $M_{\text{HBM,KV,avail}}$ and $\varphi$:
  $$S_{\max} \approx \frac{M_{\text{HBM,KV,avail}} \times TP \times SP}{\varphi_{\text{avg}} \times 2 H_{kv} b \times L/PP}$$

---

## 13. Framework Overhead
_(→ modeling.framework.md)_

Per-request (once):
- $t_{\text{tok}}$ — Tokenization latency (CPU BPE/SP processing).
- $t_{\text{sched}}$ — Request scheduling / batch assembly latency.

Per-step (each decode iteration):
- $t_{\text{launch}}$ — CUDA kernel launch overhead per decode step (no CUDA graph).
- $t_{\text{graph}}$ — CUDA graph replay latency per decode step.
- $t_{\text{sample}}$ — Token sampling latency (logits → token ID).
- $t_{\text{detok}}$ — Response streaming / detokenization latency per output token.

Disaggregated serving:
- $t_{\text{KV\_transfer}}$ — KV cache transfer latency from prefill to decode cluster (α–β model).
- $M_{\text{KV\_transfer}}$ — KV bytes transferred per device: $2 S_{\text{input}} H_{kv} b \times L / (PP \times TP \times SP)$.

Empirical calibration constants (defined in modeling.framework.md; negligible in roofline):
- $c_{\text{act}}$ — Activation I/O constant: unavoidable hidden-state HBM transfers per layer (~8–12).
- $c_{\text{norm}}$ — Norm FLOP constant: per-element norm operation count (~5–20).

Total framework overhead:
- $T_{\text{out}}$ — Number of output tokens per request.
- $t_{\text{framework}} = t_{\text{tok}} + t_{\text{sched}} + T_{\text{out}} \times (t_{\text{graph}} + t_{\text{sample}} + t_{\text{detok}}) + t_{\text{KV\_transfer}}$.

---

## 14. End-to-End Metrics
_(→ modeling.e2e.md)_

Core metrics:
- $TTFT$ — Time To First Token (defined in §11 above; assembled in modeling.e2e.md §2).
- $\text{TPOT}(B)$ — Time Per Output Token (per sequence): $t_{\text{token}}(B) / B$ (defined in §10 above).
- $\text{E2E}(N_{\text{out}})$ — End-to-end latency: $TTFT + (N_{\text{out}} - 1) \times \text{TPOT}$.
- $\text{Tput/GPU}$ — Output tokens per second per GPU: $TTPS / N_{\text{GPUs}}$.
- $\text{Interactivity}$ — Per-user output rate: $1 / \text{TPOT} = B / t_{\text{token}}(B)$ (tokens/s/request).
- $\text{Goodput}$ — Fraction of GPU time spent on useful token generation.

Continuous batching:
- $\lambda$ — Request arrival rate (requests/second).
- $N_{\text{out}}$ — Number of output tokens in a single response.
- $N_{\text{out}}^{\star}$ — Crossover output length where TTFT = decode contribution: $N_{\text{out}}^{\star} \approx TTFT / \text{TPOT}$.
- $\overline{B_{\text{eff}}}$ — Mean effective batch size in steady-state continuous batching.
- $\overline{\text{TPOT}}$ — Average TPOT over a request's decode lifetime.
- $N_{\text{GPUs,per-replica}}$ — GPUs per DP replica: $PP \cdot TP \cdot EP \cdot SP$.

Pareto ceiling:
$$\text{Tput/GPU} \times \text{TPOT} = \frac{1}{N_{\text{GPUs,per-replica}}}$$

---

## 15. 3D DRAM
_(→ modeling.dram3d.md)_

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
