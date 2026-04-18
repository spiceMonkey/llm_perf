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

**[MOONCAKE]**  
Qin, R., Li, Z., He, W., Zhang, M., Wu, Y., Zheng, W., & Xu, X. / Moonshot AI. (2024).  
*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.*  
arXiv:2407.00079.  
→ KV-centric disaggregated serving; layer-wise KV streaming over RDMA; production deployment.

**[DYNAMO]**  
NVIDIA Corporation. (2024–2025).  
*NVIDIA Dynamo: Distributed Inference Serving Framework.*  
NVIDIA technical documentation, https://github.com/ai-dynamo/dynamo.  
→ Production disaggregated inference; layer-wise KV transfer; multi-cluster orchestration.

---

## Hardware Specifications

**[H100-SPEC]**  
NVIDIA Corporation. (2022).  
*NVIDIA H100 Tensor Core GPU Architecture.*  
NVIDIA Whitepaper WP-10792-001.  
→ H100 SXM5: 989 TF/s (bf16 TC), 3.35 TB/s HBM3, 80 GB HBM per GPU; NVLink 4.0 900 GB/s bisection. (HBM3E appears in H200/B200; H100 SXM5 ships with HBM3.)

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
→ §III-C2: 3D DRAM bandwidth from hybrid bonding pitch and pin count. Eq. 2: two-level tiled-GEMM compute latency — tile size $(t_m, t_n, t_k)$ sets operand reuse and therefore BW demand per compute cycle. §III-C (Fig. 6, p. 4) places "distributed controllers and PHYs, aligned with PEs" on the logic die, and reports 3D hybrid-bonded DRAM access latency "over 2×" better than HBM's ~300 ns. Introduction (p. 2) gives the qualitative ratio: hybrid bonding pitch is "5–30× smaller than that of microbumps" (citing TSMC SoIC [Chen 2019 ECTC]).  
_Note: §III-C2 contains no dedicated BW equation — `dram3d.md` §2 extends it from first principles. `dram3d.md` §2.5 uses Eq. 2 and the PE-aligned-PHY structure of §III-C to argue that 3D stacking brings effective BW closer to the peak ceiling (another 3D DRAM advantage beyond the raw pin × data-rate lift), without introducing a separate formal symbol._

**[LAU-PKG]**  
Lau, J.H. (2021).  
*State-of-the-Art and Outlooks of Chiplets Heterogeneous Integration and Hybrid Bonding.*  
Journal of Microelectronics and Electronic Packaging, 18(4):145–160.  
→ Comprehensive industry review of advanced packaging pitch regimes. Primary datapoint for Intel Foveros Direct (2020): "bumpless pad pitch reduces from **50 µm (for microbumps) to 10 µm**, density increases from 400 bumps/mm² to 10,000 pads/mm²" (p. 149). Covers AMD 3D V-Cache (2021), Xilinx CoWoS, Ponte Vecchio.

**[SOIC-UHD]**  
Chen, M.F., Yang, C.L., et al. (2020).  
*Ultra High Density SoIC with Sub-micron Bond Pitch.*  
IEEE ECTC 2020, pp. 576–581.  
→ Sub-micron pitch hybrid bonding demonstrated at ≥1.2 M bonds/mm². Follows the 2019 foundational SoIC paper (Chen et al., ECTC 2019, pp. 594–599). Used as the near-term-ceiling datapoint for §3 Scenario 3 (aggressive hybrid bonding).

**[SEMIANALYSIS-HB]**  
Patel, D., Nishball, D. (2024).  
*Hybrid Bonding Process Flow — Advanced Packaging Part 5.*  
SemiAnalysis. https://newsletter.semianalysis.com/p/hybrid-bonding-process-flow-advanced  
→ Cross-section analysis of Nvidia A100: **~130 µm C4 flip-chip**, **~50 µm copper pillars (microbumps)**. Characterizes the Sony CIS production regime (~1 µm hybrid bonding). Industry-press reference for the commercial pitch landscape.

---

## Network Topologies and Collective Algorithms

**[PY09]**  
Patarasuk, P., & Yuan, X. (2009).  
*Bandwidth Optimal All-Reduce Algorithms for Clusters of Workstations.*  
Journal of Parallel and Distributed Computing, 69(2):117–124.  
→ Ring all-reduce achieves $2(N-1)/N \cdot M/BW$ bandwidth-optimal bound on any tree-connected fabric. Used for torus dim-by-dim AR bandwidth analysis in switching.md §8.3.

**[CHPV07]**  
Chan, E., Heimlich, M., Purkayastha, A., & van de Geijn, R. (2007).  
*Collective Communication: Theory, Practice, and Experience.*  
Concurrency and Computation: Practice and Experience, 19(13):1749–1783.  
→ Dimension-decomposed all-reduce framework; telescoping derivation of multi-dim ring costs. Used in switching.md §8.3 for the dim-by-dim ring AR derivation.

**[KDSA08]**  
Kim, J., Dally, W.J., Scott, S., & Abts, D. (2008).  
*Technology-Driven, Highly-Scalable Dragonfly Topology.*  
ISCA 2008, pp. 77–88.  
→ Dragonfly $(p, a, h, g)$ parameterization; minimal adaptive routing (diameter 3) vs. Valiant routing (diameter 5); canonical balanced construction $g = a h + 1$. Foundational reference for switching.md §9.

**[JAIN22]**  
Jain, P., et al. (2022).  
*Optimized MPI Collective Algorithms for Dragonfly Topology.*  
ICS 2022.  
→ Hierarchical three-tier AR decomposition for dragonfly; confirms full global-link utilization under uniform admissible routing. Basis for switching.md §9.4 formulas.

**[SLINGSHOT]**  
De Sensi, D., Di Girolamo, S., McMahon, K., Roweth, D., & Hoefler, T. (2020).  
*An In-Depth Analysis of the Slingshot Interconnect.*  
SC 2020.  
→ Rosetta 64-port dragonfly; measured α-calibration across router/group/global tiers; adaptive-routing effectiveness under real workloads. Calibration source for the `slingshot11.dragonfly.json` system.

**[SWING]**  
Cascagrande, M., De Sensi, D., et al. (2024).  
*Swing: Short-cutting Rings for Higher-Bandwidth Allreduce.*  
arXiv:2401.09356.  
→ Alternative torus AR algorithm that short-cuts non-adjacent rank pairs; flagged as future work in switching.md §8.7 (not implemented — `torus_algorithm="swing"` raises `NotImplementedError`).

**[HAMMESH]**  
Hoefler, T., Bonato, S., De Sensi, D., Di Girolamo, S., Li, S., Heddes, M., Belk, J., Goel, D., Castro, M., & Scott, S. (2022).  
*HammingMesh: A Network Topology for Large-Scale Deep Learning.*  
SC 2022.  
→ Irregular torus-family topology with improved bisection at fixed link count; alternative tier type referenced in switching.md §8.7 (not modeled).

**[TPU-V4]**  
Jouppi, N.P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., Patil, N., Subramanian, S., Swing, A., Towles, B., Young, C., Zhou, X., Zhou, Z., & Patterson, D. (2023).  
*TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings.*  
ISCA 2023.  
→ 3D torus with optical circuit switching for slice reconfiguration; twisted-torus 1.63× A2A gain on asymmetric layouts (§V). Motivates switching.md §8.5's $D_\mathrm{max}$ layout-sensitivity analysis.

**[PAARD]**  
*Proximity-Aware All-Reduce on Dragonfly.*  
ISPA 2021.  
→ Topology-aware AR scheduling on dragonfly under multi-job interference. Referenced in switching.md §9.6 open questions.

**[VALIANT81]**  
Valiant, L.G. (1981).  
*A Scheme for Fast Parallel Communication.*  
SIAM Journal on Computing, 11(2):350–361.  
→ Randomized two-hop routing via intermediate node for adversarial-traffic balancing; cited in switching.md §9.3 as the worst-case dragonfly routing fallback.

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
→ Fused kernel implementations for attention, LayerNorm/RMSNorm, FFN; optimized sampling kernels; CUDA graph integration.

---

## Original Modeling Contributions

The following constants and models are **original to this document suite** and are not derived from a published paper. They should be marked as such in inline citations:

- **$\rho$ (overlap factor)** — The parameterization $t_{\text{step,user}} = t_{\text{local}} + \max(0, t_{\text{comm}} - \rho \cdot t_{\text{local}})$ is an original formulation introduced in `decode.md`. The concept of compute–communication overlap is discussed in [MEGATRON3] and [DEEPSPEED-MOE] but without this exact model. Use "this work" as the citation.

---

_To cite in a doc: use the tag in brackets, e.g. "per [ROOFLINE], the ridge point is $R_{\text{GPU}} / BW_{\text{mem}}$". Original-to-this-work items use "this work". New references should be added here and tagged consistently._
