"""Unit tests for core/primitives/* — hand-computed closed-form checks.

Each test fixtures a tiny model/partition, hand-computes the expected byte
or FLOP count from the closed-form expression in the primitive's docstring,
and asserts bit-equality. These catch accidental formula changes during the
refactor (Phase 2 onward must reproduce these numbers exactly).

Usage:  PYTHONPATH=. python tests/unit/test_primitives.py
"""
import math
import sys
from dataclasses import replace

from llm_perf.core.primitives import (
    p2p_hop,
    ring_all_reduce,
    tree_all_reduce,
    ring_moe_all_to_all,
    tree_moe_all_to_all,
    ring_all_gather,
    torus_all_reduce,
    torus_all_gather,
    torus_moe_all_to_all,
    dragonfly_all_reduce,
    dragonfly_all_gather,
    dragonfly_moe_all_to_all,
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


# ────────────────────────────────────────────────────────────
# collective_cost
# ────────────────────────────────────────────────────────────

def test_collective_cost():
    print("collective_cost:")
    # p2p_hop at G=2, α=1e-6, BW=1 GB/s = 1e9, M=1000 → 1e-6 + 1000/1e9 = 2e-6
    _check("p2p_hop", p2p_hop(1000, 1e-6, 1e9), 1e-6 + 1000 / 1e9)
    # ring_all_reduce no-op at G=1
    _check("ring_all_reduce G=1", ring_all_reduce(1000, 1, 1e-6, 1e9), 0.0)
    # ring_all_reduce at G=2: 2·1·α + 2·1/2·M/BW = 2α + M/BW
    _check("ring_all_reduce G=2", ring_all_reduce(1000, 2, 1e-6, 1e9), 2e-6 + 1000 / 1e9)
    # tree_all_reduce at G=4: 2·2·α + 2·M/BW
    _check("tree_all_reduce G=4", tree_all_reduce(1000, 4, 1e-6, 1e9), 4e-6 + 2 * 1000 / 1e9)
    # ring_moe_all_to_all at G=2: 2·1·α + 2·1·M/(2·BW) = 2α + M/BW
    _check("ring_moe_all_to_all G=2", ring_moe_all_to_all(1000, 2, 1e-6, 1e9), 2e-6 + 1000 / 1e9)
    # tree_moe_all_to_all at G=4: 2·2·α + 2·M/BW
    _check("tree_moe_all_to_all G=4", tree_moe_all_to_all(1000, 4, 1e-6, 1e9), 4e-6 + 2 * 1000 / 1e9)
    # ring_all_gather at G=2: 1·α + 1·M/BW
    _check("ring_all_gather G=2", ring_all_gather(1000, 2, 1e-6, 1e9), 1e-6 + 1000 / 1e9)
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

    # torus_all_gather: T5 continuity + 2D
    # G=4: 3·α + 3·M/BW
    _check(
        "torus_AG k=1 continuity",
        torus_all_gather(M, (4,), alpha, bw),
        ring_all_gather(M, 4, alpha, bw),
    )
    # 2D 4×4: hops=6, N=16 → 6·α + 15·M/BW
    _check(
        "torus_AG 4x4",
        torus_all_gather(M, (4, 4), alpha, bw),
        6 * alpha + 15 * (M / bw),
    )

    # torus_moe_all_to_all: bisection-limited A2A.
    # Balanced 8×8×8: diam=4+4+4=12, D_max=8 → 12·α + 8·M/(4·BW)
    _check(
        "torus_A2A 8x8x8 balanced",
        torus_moe_all_to_all(M, (8, 8, 8), alpha, bw),
        12 * alpha + 8 * M / (4 * bw),
    )
    # Extreme 2×2×128: diam=1+1+64=66, D_max=128 → BW-term 16× the balanced case.
    _check(
        "torus_A2A 2x2x128 extreme",
        torus_moe_all_to_all(M, (2, 2, 128), alpha, bw),
        66 * alpha + 128 * M / (4 * bw),
    )
    # Layout sensitivity: BW-term scales exactly as D_max ratio (128/8 = 16×).
    _check(
        "torus_A2A D_max ratio (128/8)",
        torus_moe_all_to_all(M, (2, 2, 128), 0.0, bw),
        16 * torus_moe_all_to_all(M, (8, 8, 8), 0.0, bw),
    )

    # No-op / zero-guard parity with crossbar primitives.
    _check("torus_AR N=1",     torus_all_reduce(M, (1,),  alpha, bw), 0.0)
    _check("torus_AR empty",   torus_all_reduce(M, (),    alpha, bw), 0.0)
    _check("torus_AR BW=0",    torus_all_reduce(M, (4, 4), alpha, 0.0), 0.0)
    _check("torus_AG BW=0",    torus_all_gather(M, (4, 4), alpha, 0.0), 0.0)
    _check("torus_A2A BW=0",   torus_moe_all_to_all(M, (4, 4), alpha, 0.0), 0.0)


# ────────────────────────────────────────────────────────────
# dragonfly collective_cost
# ────────────────────────────────────────────────────────────

def test_dragonfly_collective_cost():
    print("dragonfly_collective_cost:")
    M, bw = 1000.0, 1e9
    alpha_r, alpha_l, alpha_g = 1e-6, 2e-6, 5e-6
    bw_r, bw_l, bw_g = 2e9, 1e9, 0.5e9

    # Trivial-tier continuity: a=g=1 reduces to ring_all_reduce(M, p, α_r, BW_r).
    # p=4, a=1, g=1 → 2·3·α_r + 2·(3/4)·M/BW_r.
    _check(
        "df_AR a=1 g=1 continuity",
        dragonfly_all_reduce(
            M, 4, 1, 1,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        ring_all_reduce(M, 4, alpha_r, bw_r),
    )
    # p=4, a=2, g=1: adds L1 contribution, no L2. Hand-compute.
    # L0 = 2·3·α_r + 2·(3/4)·M/BW_r = 6e-6 + 1.5·M/BW_r
    # L1 = 2·1·α_l + 2·(1/2)·(M/4)/BW_l = 2·α_l + (M/4)/BW_l
    want = (6 * alpha_r + 1.5 * M / bw_r) + (2 * alpha_l + (M / 4) / bw_l)
    _check(
        "df_AR p=4 a=2 g=1",
        dragonfly_all_reduce(
            M, 4, 2, 1,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        want,
    )
    # Full three-tier p=4, a=2, g=3.
    # L0 = 6·α_r + 1.5·M/BW_r
    # L1 = 2·α_l + (M/4)/BW_l
    # L2 = 2·2·α_g + 2·(2/3)·(M/8)/BW_g
    l0 = 6 * alpha_r + 1.5 * M / bw_r
    l1 = 2 * alpha_l + (M / 4) / bw_l
    l2 = 4 * alpha_g + (4 / 3) * (M / 8) / bw_g
    _check(
        "df_AR p=4 a=2 g=3 best",
        dragonfly_all_reduce(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        l0 + l1 + l2,
    )
    # worst_case=True doubles L2 only.
    _check(
        "df_AR p=4 a=2 g=3 worst_case",
        dragonfly_all_reduce(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
            worst_case=True,
        ),
        l0 + l1 + 2 * l2,
    )

    # AG trivial-tier continuity: a=g=1 reduces to ring_all_gather(M, p, α_r, BW_r).
    _check(
        "df_AG a=1 g=1 continuity",
        dragonfly_all_gather(
            M, 4, 1, 1,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        ring_all_gather(M, 4, alpha_r, bw_r),
    )
    # AG p=2, a=2, g=2 hand-compute:
    # L0 = 1·α_r + 1·M/BW_r
    # L1 = 1·α_l + 1·(2·M)/BW_l
    # L2 = 1·α_g + 1·(2·2·M)/BW_g = α_g + 4·M/BW_g
    want_ag = (
        (alpha_r + M / bw_r)
        + (alpha_l + 2 * M / bw_l)
        + (alpha_g + 4 * M / bw_g)
    )
    _check(
        "df_AG p=2 a=2 g=2 best",
        dragonfly_all_gather(
            M, 2, 2, 2,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        want_ag,
    )
    # AG worst_case doubles L2 only.
    _check(
        "df_AG p=2 a=2 g=2 worst_case",
        dragonfly_all_gather(
            M, 2, 2, 2,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
            worst_case=True,
        ),
        (alpha_r + M / bw_r)
        + (alpha_l + 2 * M / bw_l)
        + 2 * (alpha_g + 4 * M / bw_g),
    )

    # MoE A2A == AR formula identity (mirrors ring_moe_all_to_all == ring_all_reduce).
    _check(
        "df_A2A == df_AR identity",
        dragonfly_moe_all_to_all(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
        dragonfly_all_reduce(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
        ),
    )
    # Worst-case propagates through the alias.
    _check(
        "df_A2A worst_case",
        dragonfly_moe_all_to_all(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
            worst_case=True,
        ),
        dragonfly_all_reduce(
            M, 4, 2, 3,
            alpha_r, bw_r,
            alpha_l, bw_l,
            alpha_g, bw_g,
            worst_case=True,
        ),
    )

    # No-op / zero-guards.
    _check(
        "df_AR p=a=g=1",
        dragonfly_all_reduce(M, 1, 1, 1, alpha_r, bw_r, alpha_l, bw_l, alpha_g, bw_g),
        0.0,
    )
    # Zero BW on any tier suppresses that tier's contribution only.
    # p=2, a=2, g=1 with BW_l=0 → just L0.
    _check(
        "df_AR BW_l=0 suppresses L1",
        dragonfly_all_reduce(M, 2, 2, 1, alpha_r, bw_r, alpha_l, 0.0, alpha_g, bw_g),
        2 * 1 * alpha_r + 2 * (1 / 2) * (M / bw_r),
    )


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
    test_dragonfly_collective_cost()
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
