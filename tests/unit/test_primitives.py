"""Unit tests for core/primitives/* — hand-computed closed-form checks.

Each test fixtures a tiny model/partition, hand-computes the expected byte
or FLOP count from the closed-form expression in the primitive's docstring,
and asserts bit-equality. These pin down the upstream-synced
``collective_cost.py`` API surface — see ``scratch/collectives_cost_CHANGES.md``
for the post-sync conventions (M = full per-rank payload for AG/RS;
``tree_all_reduce`` defaults to non-pipelined; MoE Dispatch+Combine 2×
wrap lives in the dispatcher, not the primitive).

Usage:  PYTHONPATH=. python tests/unit/test_primitives.py
"""
import math
import sys
from dataclasses import replace

from llm_perf.core.primitives import (
    p2p_hop,
    ring_all_reduce,
    tree_all_reduce,
    pairwise_a2a,
    ring_all_gather,
    ring_reduce_scatter,
    torus_all_reduce,
    torus_all_gather,
    torus_reduce_scatter,
    torus_a2a,
    inc_all_reduce,
    inc_all_gather,
    inc_reduce_scatter,
    inc_a2a,
    aggregate_per_stage,
    dense_weight_bytes,
    moe_weight_bytes,
    embedding_bytes,
    kv_bytes_per_seq,
    linear_flops_per_token,
)
from llm_perf.specs.model_spec import LlmModelSpec, MoESpec
from llm_perf.specs.partition_spec import PartitionSpec


# ────────────────────────────────────────────────────────────
# Tiny fixtures — small enough to do arithmetic in head/margin
# ────────────────────────────────────────────────────────────

DENSE = LlmModelSpec(
    name="tiny-dense",
    L=2, H=4, n_q=2, n_kv=2, I_dense=8,
    vocab_size=100, max_seq_len=1024, bytes_per_param=2.0,
)

MOE = LlmModelSpec(
    name="tiny-moe",
    L=2, H=4, n_q=2, n_kv=2, I_dense=8,
    vocab_size=100, max_seq_len=1024, bytes_per_param=2.0,
    moe=MoESpec(n_experts=4, k_active=2, I_moe=8, n_moe_layers=1),
)

PART_UNSHARDED = PartitionSpec(PP=1, TP=1, EP=1, SP=1)


# ────────────────────────────────────────────────────────────
# Assertion helper
# ────────────────────────────────────────────────────────────

_failures: list[str] = []


def _check(label: str, got: float, want: float) -> None:
    if math.isclose(got, want, rel_tol=0.0, abs_tol=0.0):
        print(f"  PASS  {label}: {got!r}")
    else:
        _failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def _check_near(label: str, got: float, want: float, rel_tol: float) -> None:
    if math.isclose(got, want, rel_tol=rel_tol, abs_tol=0.0):
        print(f"  PASS  {label}: {got!r} ≈ {want!r} (rel_tol={rel_tol})")
    else:
        _failures.append(f"{label}: got {got!r}, want ~{want!r} (rel_tol={rel_tol})")
        print(f"  FAIL  {label}: got {got!r}, want ~{want!r}")


# ────────────────────────────────────────────────────────────
# collective_cost — α-β primitives (upstream-synced)
# ────────────────────────────────────────────────────────────

def test_collective_cost():
    print("collective_cost:")
    # p2p_hop at G=2, α=1e-6, BW=1 GB/s = 1e9, M=1000 → 1e-6 + 1000/1e9 = 2e-6
    _check("p2p_hop", p2p_hop(1000, 1e-6, 1e9), 1e-6 + 1000 / 1e9)
    # ring_all_reduce no-op at G=1
    _check("ring_all_reduce G=1", ring_all_reduce(1000, 1, 1e-6, 1e9), 0.0)
    # ring_all_reduce at G=2: 2·1·α + 2·1/2·M/BW = 2α + M/BW
    _check("ring_all_reduce G=2", ring_all_reduce(1000, 2, 1e-6, 1e9), 2e-6 + 1000 / 1e9)
    # tree_all_reduce default (pipelined=False, P=1 schedule):
    #   2·⌈log₂G⌉·α + ⌈log₂G⌉·M/BW
    # G=4: 2·2·α + 2·M/BW = 4α + 2·M/BW
    _check("tree_all_reduce G=4 (P=1)",
           tree_all_reduce(1000, 4, 1e-6, 1e9),
           4e-6 + 2 * 1000 / 1e9)
    # tree_all_reduce pipelined=True (asymptotic P→P*): 2·⌈log₂G⌉·α + M/BW
    _check("tree_all_reduce G=4 pipelined",
           tree_all_reduce(1000, 4, 1e-6, 1e9, pipelined=True),
           4e-6 + 1000 / 1e9)
    # pairwise_a2a — single direction (no Dispatch+Combine 2× — that lives
    # in the dispatcher's MoE wrap). G=2: 1·α + (1/2)·M/BW = α + M/(2·BW)
    _check("pairwise_a2a G=2",
           pairwise_a2a(1000, 2, 1e-6, 1e9),
           1e-6 + 1000 / (2 * 1e9))
    # ring_all_gather under new M=full convention:
    #   (G-1)·α + (G-1)/G · M/BW
    # G=2, M=1000: 1·α + (1/2)·1000/BW = α + 500/BW
    _check("ring_all_gather G=2 (M=full)",
           ring_all_gather(1000, 2, 1e-6, 1e9),
           1e-6 + 500 / 1e9)
    # Zero BW → 0.0 guard
    _check("p2p_hop BW=0", p2p_hop(1000, 1e-6, 0), 0.0)
    _check("ring_all_reduce BW=0", ring_all_reduce(1000, 4, 1e-6, 0), 0.0)
    # aggregate_per_stage: L=4, L_moe=2, PP=2, n_TP=2, t_TP=1.0, n_SP=1, t_SP=2.0,
    # n_EP=1, t_EP=3.0, t_PP=4.0
    #   (4/2)·(2·1 + 1·2) + (2/2)·(1·3) + 4 = 2·4 + 1·3 + 4 = 15
    _check(
        "aggregate_per_stage",
        aggregate_per_stage(L=4, L_moe=2, PP=2, n_TP=2, t_TP=1.0,
                            n_SP=1, t_SP=2.0, n_EP=1, t_EP=3.0, t_PP=4.0),
        15.0,
    )


# ────────────────────────────────────────────────────────────
# torus collective_cost
# ────────────────────────────────────────────────────────────

def test_torus_collective_cost():
    print("torus_collective_cost:")
    M, alpha, bw = 1000.0, 1e-6, 1e9

    # T5 continuity: torus_all_reduce(dims=(N,)) == ring_all_reduce(N).
    # G=4: 2·3·α + 2·(3/4)·M/BW = 6e-6 + 1.5e-6 = 7.5e-6
    _check(
        "torus_AR k=1 continuity",
        torus_all_reduce(M, (4,), alpha, bw),
        ring_all_reduce(M, 4, alpha, bw),
    )
    # 2D 4×4: hops = 3+3 = 6, N=16 → 2·6·α + 2·(15/16)·M/BW
    _check(
        "torus_AR 4x4",
        torus_all_reduce(M, (4, 4), alpha, bw),
        12 * alpha + 2 * (15 / 16) * (M / bw),
    )
    # BW-term invariance across layouts at fixed N=16 (α=0 isolates):
    # all of (16,), (2,8), (4,4), (2,2,4) yield 2·(15/16)·M/BW.
    bw_term = 2 * (15 / 16) * (M / bw)
    _check("torus_AR BW-term (16,)",   torus_all_reduce(M, (16,),     0.0, bw), bw_term)
    _check("torus_AR BW-term (2,8)",   torus_all_reduce(M, (2, 8),    0.0, bw), bw_term)
    _check("torus_AR BW-term (4,4)",   torus_all_reduce(M, (4, 4),    0.0, bw), bw_term)
    _check("torus_AR BW-term (2,2,4)", torus_all_reduce(M, (2, 2, 4), 0.0, bw), bw_term)

    # torus_all_gather under new M=full convention: Σ(D-1)·α + (N-1)/N · M/BW
    # G=4: 3·α + 3/4·M/BW
    _check(
        "torus_AG k=1 continuity",
        torus_all_gather(M, (4,), alpha, bw),
        ring_all_gather(M, 4, alpha, bw),
    )
    # 2D 4×4: hops=6, N=16 → 6·α + (15/16)·M/BW
    _check(
        "torus_AG 4x4 (M=full)",
        torus_all_gather(M, (4, 4), alpha, bw),
        6 * alpha + (15 / 16) * (M / bw),
    )

    # torus_a2a: bisection-limited A2A (no MoE 2× wrap — that's dispatcher-side).
    # Balanced 8×8×8 wraparound: diam=4+4+4=12, D_max=8 → 12·α + 8·M/(8·BW) = 12α + M/BW
    _check(
        "torus_A2A 8x8x8 balanced (wraparound)",
        torus_a2a(M, (8, 8, 8), alpha, bw),
        12 * alpha + 8 * M / (8 * bw),
    )
    # Extreme 2×2×128 wraparound: diam=1+1+64=66, D_max=128 → BW-term 16× the balanced case
    _check(
        "torus_A2A 2x2x128 extreme",
        torus_a2a(M, (2, 2, 128), alpha, bw),
        66 * alpha + 128 * M / (8 * bw),
    )
    # Layout sensitivity: BW-term scales exactly as D_max ratio (128/8 = 16×)
    _check(
        "torus_A2A D_max ratio (128/8)",
        torus_a2a(M, (2, 2, 128), 0.0, bw),
        16 * torus_a2a(M, (8, 8, 8), 0.0, bw),
    )
    # k-D mesh (wraparound=False): open-line diameter Σ(D-1), bisection /4.
    # 4×4 mesh: diam=3+3=6, D_max=4 → 6α + 4·M/(4·BW) = 6α + M/BW.
    # vs same-shape torus 4×4: diam=2+2=4, D_max=4 → 4α + 4·M/(8·BW) = 4α + M/(2·BW).
    # Mesh BW term is exactly 2× torus BW term.
    _check(
        "torus_A2A 4x4 mesh BW = 2× wraparound",
        torus_a2a(M, (4, 4), 0.0, bw, wraparound=False),
        2 * torus_a2a(M, (4, 4), 0.0, bw, wraparound=True),
    )

    # No-op / zero-guard parity with crossbar primitives.
    _check("torus_AR N=1",     torus_all_reduce(M, (1,),  alpha, bw), 0.0)
    _check("torus_AR empty",   torus_all_reduce(M, (),    alpha, bw), 0.0)
    _check("torus_AR BW=0",    torus_all_reduce(M, (4, 4), alpha, 0.0), 0.0)
    _check("torus_AG BW=0",    torus_all_gather(M, (4, 4), alpha, 0.0), 0.0)
    _check("torus_A2A BW=0",   torus_a2a(M, (4, 4), alpha, 0.0), 0.0)


# ────────────────────────────────────────────────────────────
# INC collective_cost — 04_in_network_collectives.md
# ────────────────────────────────────────────────────────────

def test_inc_collective_cost():
    print("inc_collective_cost:")
    M, alpha, bw = 1_000.0, 1e-6, 1e9

    # AR single-tier (scale-up): n_α = 2 regardless of N.
    # inc_all_reduce(M, α_total, BW) = 2α + M/BW
    _check(
        "inc_AR single-tier",
        inc_all_reduce(M, alpha, bw),
        2 * alpha + M / bw,
    )
    # AR multi-tier (scale-out k=2): pass summed α; primitive applies the 2× factor.
    # For α_total = 2α: result = 2·2α + M/BW = 4α + M/BW.
    _check(
        "inc_AR two-tier scale-out",
        inc_all_reduce(M, 2 * alpha, bw),
        4 * alpha + M / bw,
    )
    # AG single-tier under new M=full convention: 2α + (G-1)/G · M/BW
    # G=8: 2α + (7/8)·M/BW
    _check(
        "inc_AG G=8 (M=full)",
        inc_all_gather(M, 8, alpha, bw),
        2 * alpha + (7 / 8) * (M / bw),
    )
    # BW-eff doubling vs ring at fixed α, large N: inc_AR ≈ half the BW term of ring_AR.
    # N=1024, α=0 isolates BW term: ring = 2·(1023/1024)·M/BW; inc = M/BW.
    ring_bw = 2 * (1023 / 1024) * (M / bw)
    inc_bw = M / bw
    _check("inc_AR BW-term halved (N=1024)", inc_bw, ring_bw / (2 * 1023 / 1024))

    # Algorithmic-ceiling sanity check vs doc §3.1 (N=512, M=16MB, α=0.5us, BW=900GB/s).
    # INC AR should total ~18.8 μs.
    M_doc = 16 * 2**20  # 16 MiB
    alpha_doc = 0.5e-6
    bw_doc = 900 * 1e9
    inc_ar_doc = inc_all_reduce(M_doc, alpha_doc, bw_doc)
    # 2·0.5us + 16MiB/900GB/s = 1 μs + ~18.64 μs ≈ 19.6 μs (doc quotes 18.8 for decimal MB).
    _check_near("inc_AR N=512 doc anchor", inc_ar_doc, 19.6e-6, rel_tol=0.02)

    # No-op guards
    _check("inc_AR BW=0",     inc_all_reduce(M, alpha, 0.0), 0.0)
    _check("inc_AG G=1",      inc_all_gather(M, 1, alpha, bw), 0.0)
    _check("inc_AG BW=0",     inc_all_gather(M, 4, alpha, 0.0), 0.0)

    # inc_reduce_scatter under new M=full convention: 2α + (G-1)/G · M/BW
    _check(
        "inc_RS G=8 (M=full)",
        inc_reduce_scatter(M, 8, alpha, bw),
        2 * alpha + (7 / 8) * (M / bw),
    )
    _check("inc_RS G=1",       inc_reduce_scatter(M, 1, alpha, bw), 0.0)
    _check("inc_RS BW=0",      inc_reduce_scatter(M, 4, alpha, 0.0), 0.0)

    # inc_a2a — α_switch (single transaction, NOT 2×) + (G-1)/G · M/BW.
    # The MoE Dispatch+Combine 2× wrap lives in dispatch.py, not the primitive.
    _check(
        "inc_A2A G=8",
        inc_a2a(M, 8, alpha, bw),
        alpha + (7 / 8) * (M / bw),
    )
    # α-collapse vs software pairwise: at G=64, α=1us, both are single-direction
    # primitives now. pairwise_a2a α-term = (G-1)·α = 63·α. inc_a2a α-term = α.
    # Ratio = 63×.
    pairwise_alpha = pairwise_a2a(0.0, 64, 1e-6, 1e12)
    inc_a2a_alpha = inc_a2a(0.0, 64, 1e-6, 1e12)  # M=0 isolates α-term
    _check("inc_A2A α-collapse vs pairwise (G=64)", inc_a2a_alpha, 1e-6)
    assert pairwise_alpha / inc_a2a_alpha == 63.0, (
        f"α ratio mismatch: pairwise/inc = {pairwise_alpha / inc_a2a_alpha}"
    )
    print(f"  PASS  α-collapse ratio (pairwise/inc, G=64): 63.0×")
    _check("inc_A2A G=1",      inc_a2a(M, 1, alpha, bw), 0.0)
    _check("inc_A2A BW=0",     inc_a2a(M, 4, alpha, 0.0), 0.0)

    # ring_reduce_scatter / torus_reduce_scatter under new M=full convention.
    # ring_RS G=4: (G-1)·α + (G-1)/G · M/BW = 3α + (3/4)·M/BW
    _check(
        "ring_RS G=4 (M=full)",
        ring_reduce_scatter(M, 4, alpha, bw),
        3 * alpha + (3 / 4) * (M / bw),
    )
    # torus_RS 4×4: hops=6, N=16 → 6α + (15/16)·M/BW
    _check(
        "torus_RS 4x4 (M=full)",
        torus_reduce_scatter(M, (4, 4), alpha, bw),
        6 * alpha + (15 / 16) * (M / bw),
    )
    _check("ring_RS G=1",      ring_reduce_scatter(M, 1, alpha, bw), 0.0)
    _check("torus_RS empty",   torus_reduce_scatter(M, (), alpha, bw), 0.0)


# ────────────────────────────────────────────────────────────
# weight_footprint
# ────────────────────────────────────────────────────────────

def test_weight_footprint():
    print("weight_footprint:")
    # Dense model: L=2 dense, PP=1, TP=1, H=4, H_kv=4, I_dense=8, b=2
    #   M_theta_dense = 2 · ((2·16 + 2·16) + 3·4·8) · 2
    #                 = 2 · (32 + 32 + 96) · 2 = 2 · 160 · 2 = 640
    _check("dense (dense model)", dense_weight_bytes(DENSE, PART_UNSHARDED), 640.0)
    # Dense model has no MoE contribution
    _check("moe (dense model)", moe_weight_bytes(DENSE, PART_UNSHARDED), 0.0)
    # Embedding: V·H/TP·b = 100·4·2 = 800
    _check("embedding (dense)", embedding_bytes(DENSE, PART_UNSHARDED), 800.0)

    # MoE model: L=2 total, L_moe=1, so L_dense=1, PP=1, TP=1, EP=1, N_exp=4, I_moe=8
    #   M_theta_dense = 1 · ((32+32) + 3·4·8) · 2 = (64 + 96) · 2 = 320
    _check("dense (moe model)", dense_weight_bytes(MOE, PART_UNSHARDED), 320.0)
    #   M_theta_moe = 1 · ((32+32) + 3·4·8·4/(1·1)) · 2 = (64 + 384) · 2 = 896
    _check("moe (moe model)", moe_weight_bytes(MOE, PART_UNSHARDED), 896.0)
    # With EP=2: dispatches I_moe·N_exp across 2 ranks
    #   M_theta_moe = 1 · ((32+32) + 3·4·8·4/(1·2)) · 2 = (64 + 192) · 2 = 512
    moe_ep2 = replace(PART_UNSHARDED, EP=2)
    _check("moe EP=2", moe_weight_bytes(MOE, moe_ep2), 512.0)
    # EP > N_exp clamp: EP=8 on N_exp=4 → behaves like EP=4
    moe_ep8 = replace(PART_UNSHARDED, EP=8)
    moe_ep4 = replace(PART_UNSHARDED, EP=4)
    _check("EP>N_exp clamp", moe_weight_bytes(MOE, moe_ep8), moe_weight_bytes(MOE, moe_ep4))


# ────────────────────────────────────────────────────────────
# kv_footprint
# ────────────────────────────────────────────────────────────

def test_kv_footprint():
    print("kv_footprint:")
    # L=2, H_kv=4, b=2, n_tokens=16, PP=1, TP=1, SP=1
    #   = 2 · 2 · 16 · 4 · 2 / (1·1) = 512
    _check("unsharded", kv_bytes_per_seq(DENSE, PART_UNSHARDED, 16), 512.0)
    # With TP=2, SP=2: 512 / (2·2) = 128
    p = replace(PART_UNSHARDED, TP=2, SP=2)
    _check("TP=2 SP=2", kv_bytes_per_seq(DENSE, p, 16), 128.0)
    # With PP=2: 256
    p = replace(PART_UNSHARDED, PP=2)
    _check("PP=2", kv_bytes_per_seq(DENSE, p, 16), 256.0)
    # n_tokens=0 → 0
    _check("n_tokens=0", kv_bytes_per_seq(DENSE, PART_UNSHARDED, 0), 0.0)


# ────────────────────────────────────────────────────────────
# linear_flops_per_token
# ────────────────────────────────────────────────────────────

def test_linear_flops_per_token():
    print("linear_flops_per_token:")
    # Dense-only: L=2, H=4, H_kv=4, I_dense=8, PP=1, TP=1
    #   F_layer = (4·16 + 4·16)/1 + 6·4·8/1 = 64+64 + 192 = 320
    #   F_token = 2 · 320 = 640
    _check("dense", linear_flops_per_token(DENSE, PART_UNSHARDED), 640.0)
    # MoE: L=2 (1 dense + 1 moe), N_exp=4, k=2, I_moe=8, EP=1, TP=1
    #   F_layer_dense = 128 + 192 = 320
    #   F_layer_moe = 128 + 6·4·2·8 + 2·4·4 = 128 + 384 + 32 = 544
    #   F_token = 1·320 + 1·544 = 864
    _check("moe", linear_flops_per_token(MOE, PART_UNSHARDED), 864.0)
    # TP=2 halves proj + dense_ffn; router stays unsharded
    #   F_layer_dense = 160/1 + 96 = 160 wait: (4·16+4·16)/2 + (6·4·8)/2 = 64 + 96 = 160
    #   F_layer_moe = 64 + (6·4·2·8)/(2·1) + 2·4·4 = 64 + 192 + 32 = 288
    #   F_token = 160 + 288 = 448
    p = replace(PART_UNSHARDED, TP=2)
    _check("moe TP=2", linear_flops_per_token(MOE, p), 448.0)
    # EP=2 halves expert FFN only
    #   F_layer_moe = 128 + (6·4·2·8)/(1·2) + 2·4·4 = 128 + 192 + 32 = 352
    #   F_token = 320 + 352 = 672
    p = replace(PART_UNSHARDED, EP=2)
    _check("moe EP=2", linear_flops_per_token(MOE, p), 672.0)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    test_collective_cost()
    test_torus_collective_cost()
    test_inc_collective_cost()
    test_weight_footprint()
    test_kv_footprint()
    test_linear_flops_per_token()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll primitive unit tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
