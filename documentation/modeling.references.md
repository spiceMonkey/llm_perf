# References

Shared bibliography for all `modeling.*.md` documents.  
Each entry carries a short tag (e.g. `[MEGATRON]`) used for inline citations across the suite.

---

## Parallelism and Distributed Training

**[MEGATRON]**  
Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019).  
*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*  
arXiv:1909.08053.  
→ Column/row-parallel linear layers (TP); pipeline stage scheduling.

**[MEGATRON3]**  
Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., & Zaharia, M. (2021).  
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*  
SC '21. arXiv:2104.04473.  
→ Interleaved pipeline schedules; 1F1B; sequence parallelism (SP).

**[DEEPSPEED-MOE]**  
Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R.Y., Awan, A.A., Rasley, J., & He, Y. (2022).  
*DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.*  
ICML 2022. arXiv:2201.05596.  
→ Expert parallelism (EP); all-to-all routing; top-k gating FLOPs.

---

## Attention and Memory Efficiency

**[FA1]**  
Dao, T., Fu, D.Y., Ermon, S., Rudra, A., & Ré, C. (2022).  
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*  
NeurIPS 2022. arXiv:2205.14135.  
→ Tiled attention; HBM traffic reduction; activation I/O lower bound.

**[FA2]**  
Dao, T. (2023).  
*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*  
ICLR 2024. arXiv:2307.08691.  
→ Improved parallelism across heads and sequence; GPU utilization analysis.

**[GQA]**  
Ainslie, J., Lee-Thorp, J., de Jong, M., Zelaski, T., Sanghai, S., & Xu, Y. (2023).  
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*  
EMNLP 2023. arXiv:2305.13245.  
→ Grouped-query attention; $n_{kv} < n_q$; KV cache size reduction.

**[MQA]**  
Shazeer, N. (2019).  
*Fast Transformer Decoding: One Write-Head is All You Need.*  
arXiv:1911.02150.  
→ Multi-query attention origin; $n_{kv}=1$.

---

## Mixture of Experts

**[SWITCH]**  
Fedus, W., Zoph, B., & Shazeer, N. (2022).  
*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*  
JMLR 2022. arXiv:2101.03961.  
→ Top-1 routing; capacity factor; load balancing loss.

**[MIXTRAL]**  
Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., de las Casas, D., Hanna, E.B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L.R., Saulnier, L., Lachaux, M-A., Stock, P., Subramanian, P., Yang, J., Antoniak, S., Le Scao, T., Gervet, T., Lavril, T., Wang, T., Lacroix, T., & El Sayed, W. (2024).  
*Mixtral of Experts.*  
arXiv:2401.04088.  
→ Sliding-window attention + MoE; top-2 routing in practice.

---

## Roofline and Performance Modeling

**[ROOFLINE]**  
Williams, S., Waterman, A., & Patterson, D. (2009).  
*Roofline: An Insightful Visual Performance Model for Multicore Architectures.*  
CACM 52(4). doi:10.1145/1498765.1498785.  
→ Operational intensity; memory-bound vs compute-bound regimes; ridge point.

**[ALPHA-BETA]**  
Hockney, R. (1994).  
*The Communication Challenge for MPP: Intel Paragon and Meiko CS-2.*  
Parallel Computing 20(3).  
→ α–β latency model for collective communication: $t = \alpha + n/\beta$.

---

## Serving Systems and KV Cache

**[VLLM]**  
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023).  
*Efficient Memory Management for Large Language Model Serving with PagedAttention.*  
SOSP 2023. arXiv:2309.06180.  
→ PagedAttention; block-based KV paging; fragmentation analysis.

**[SARATHI]**  
Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B.S., & Ramjee, R. (2023).  
*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.*  
arXiv:2308.16369.  
→ Chunked prefill scheduling; head-of-line blocking reduction.

**[DISAGG-PREFILL]**  
Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024).  
*DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving.*  
OSDI 2024. arXiv:2401.09670.  
→ Prefill–decode disaggregation; KV transfer latency between clusters.

---

## Hardware Specifications

**[H100-SPEC]**  
NVIDIA Corporation. (2022).  
*NVIDIA H100 Tensor Core GPU Architecture.*  
NVIDIA Whitepaper WP-10792-001.  
→ H100 SXM5: 989 TF/s (bf16 TC), 3.35 TB/s HBM3E, 80 GB HBM per GPU; NVLink 4.0 900 GB/s bisection.

**[HBM-SPEC]**  
JEDEC Solid State Technology Association. (2023).  
*High Bandwidth Memory (HBM) DRAM Standard JESD235D.*  
→ HBM2E / HBM3 / HBM3E pin bandwidth and capacity specs.

---

## 3D Stacking and Advanced Packaging

**[ACCELSTACK]**  
Bai, Y., et al. (2025).  
*AccelStack: A Co-Design Framework for 3D-Stacked Accelerators.*  
HKUST FACT Lab. arXiv (preprint).  
→ §III-C2: 3D DRAM bandwidth from hybrid bonding pitch and pin count; Eq. 2: GEMM compute latency (two-level tiling).  
_Note: Eq. 2 is GEMM latency, not BW. The 3D DRAM BW derivation in §III-C2 has no dedicated equation — `modeling.dram3d.md` extends it from first principles._

---

## LLM Scaling and Architecture Surveys

**[KAPLAN-SCALING]**  
Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020).  
*Scaling Laws for Neural Language Models.*  
arXiv:2001.08361.  
→ FLOPs ≈ 6N per training token; parameter count estimates.

**[CHINCHILLA]**  
Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D.D.L., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O., & Sifre, L. (2022).  
*Training Compute-Optimal Large Language Models.*  
NeurIPS 2022. arXiv:2203.15556.  
→ Compute-optimal token/parameter ratio; inference FLOPs context.

---

## Benchmarks and Evaluation

**[INFERENCEX]**  
SemiAnalysis. (2024–2025).  
*InferenceX: LLM Inference Benchmark.*  
https://inferencex.semianalysis.com/inference  
→ Industry benchmark axes: Throughput/GPU vs Interactivity (output tokens/s per request); TPOT; E2E latency definitions.

---

## Sequence Parallelism

**[RING-ATTN]**  
Liu, H., Zaharia, M., & Abbeel, P. (2023).  
*Ring Attention with Blockwise Transformers for Near-Infinite Context.*  
ICLR 2024. arXiv:2310.01889.  
→ Ring-based KV sharding; pass-KV variant; (SP-1) rotation steps; single-pass streaming (no scatter phase). Basis for SP collective latency equation in §5.4.

**[HUANG-CP-2024]**  
Huang, Y., et al. (2024).  
*Context Parallelism for Scalable Million-Token Inference.*  
arXiv:2411.01783.  
→ Extends Ring Attention with pass-KV and pass-Q variants; latency analysis for decode vs. prefill; used to confirm TP × SP interaction and per-step message size in §5.4.

**[DEEPSPEED-ULYSSES]**  
Rajbhandari, S., et al. (2023).  
*DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.*  
arXiv:2309.14509.  
→ Alternative SP approach using all-to-all (not ring). Bounded by number of attention heads (cannot exceed $n_{kv}$). Contrasted with ring SP in §5.4 to justify ring assumption for large-context inference.

---

## Empirical Constants and Profiling

**[TENSORRT-LLM]**  
NVIDIA Corporation. (2023–2025).  
*TensorRT-LLM: Open-Source Library for Optimized LLM Inference.*  
https://github.com/NVIDIA/TensorRT-LLM  
→ Fused kernel implementations for attention, LayerNorm/RMSNorm, FFN. Source for empirical activation I/O constant $c_{\text{act}}$: kernel traces on H100 show ~8–12 unavoidable hidden-state reads/writes per layer even with FlashAttention fusion.

**[XFORMERS]**  
Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Castrejon, F., Chen, S., Du, S., Eubank, N., Han, S., Hu, J., Huang, P., Khabsa, M., Tan, A., Wang, H., Wehrsteiner, S., Xu, S., Zhang, Z., Girshick, R., Rabin, S., Stoica, I., & Li, Y. (2022).  
*xFormers: A Modular and Hackable Transformer Modelling Library.*  
arXiv:2209.14970.  
→ Modular attention and FFN kernel library; kernel profiling data for activation I/O and norm costs.

---

## Original Modeling Contributions

The following constants and models are **original to this document suite** and are not derived from a published paper. They should be marked as such in inline citations:

- **$\rho$ (overlap factor)** — The parameterization $t_{\text{token}} = t_{\text{local}} + \max(0, t_{\text{comm}} - \rho \cdot t_{\text{local}})$ is an original formulation introduced in `modeling.tpot.md`. The concept of compute–communication overlap is discussed in [MEGATRON3] and [DEEPSPEED-MOE] but without this exact model. Use "this work" as the citation.

- **$c_{\text{act}}$ and $c_{\text{norm}}$** — Removed from `modeling.tpot.md` (negligible at large model scale: ~3–4 orders of magnitude below dominant weight and FFN terms). Defined as empirical calibration constants in `modeling.framework.md`. Sources: [TENSORRT-LLM] and [XFORMERS] for $c_{\text{act}}$; first-principles for $c_{\text{norm}}$ (RMSNorm ~5H ops, LayerNorm ~10H ops).

---

_To cite in a doc: use the tag in brackets, e.g. "per [ROOFLINE], the ridge point is $R_{\text{GPU}} / B_{\text{eff,mem}}$". Original-to-this-work items use "this work". New references should be added here and tagged consistently._
