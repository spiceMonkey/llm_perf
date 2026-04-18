"""Dispatcher equivalence suite (Phase C) — scratch/switching_upgrade.md §10.4.

Proves that `cost_collective` on any pure-crossbar tier chain returns exactly
the same float as the pre-refactor flatten-then-apply pattern:

    α_sum, BW_min, _ = span_tiers(tiers, G)
    t = ring_or_tree(M, G, α_sum·1e-6, BW_min·1e9)

Anything non-bit-identical here is a regression. Covers T1 single-tier, T2
multi-tier crossbar chain, T3 randomized property test, T4 crossbar-silence
of the torus dim-alignment warning, T5 k=1 torus continuity.

Usage:  PYTHONPATH=. python tests/unit/test_topology_equivalence.py
"""
import math
import random
import sys
import warnings

from llm_perf.core.primitives import (
    cost_collective,
    p2p_hop,
    ring_all_gather,
    ring_all_reduce,
    ring_moe_all_to_all,
    torus_all_reduce,
    tree_all_reduce,
    tree_moe_all_to_all,
)
from llm_perf.specs.system_spec import CrossbarTier, TorusTier, span_tiers
from llm_perf.utils import GB_TO_BYTES, US_TO_SECONDS


_failures: list[str] = []


def _check(label: str, got: float, want: float) -> None:
    if math.isclose(got, want, rel_tol=0.0, abs_tol=0.0):
        print(f"  PASS  {label}")
    else:
        _failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def _old_path(op: str, tiers, M: float, G: int, algorithm: str) -> float:
    """Pre-refactor flatten-then-apply reference path."""
    if G <= 1 or not tiers:
        return 0.0
    alpha_us, bw_GBps, _ = span_tiers(tiers, G)
    alpha_s = alpha_us * US_TO_SECONDS
    bw_Bps = bw_GBps * GB_TO_BYTES
    if op == "p2p":
        return p2p_hop(M, alpha_s, bw_Bps)
    if op == "all_gather":
        return ring_all_gather(M, G, alpha_s, bw_Bps)
    if op == "all_reduce":
        if algorithm == "ring":
            return ring_all_reduce(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_all_reduce(M, G, alpha_s, bw_Bps)
    if op == "moe_a2a":
        if algorithm == "ring":
            return ring_moe_all_to_all(M, G, alpha_s, bw_Bps)
        if algorithm == "tree":
            return tree_moe_all_to_all(M, G, alpha_s, bw_Bps)
    raise ValueError(f"bad op/algorithm: {op!r}/{algorithm!r}")


# ────────────────────────────────────────────────────────────
# T1 — Single-tier crossbar equivalence (every op × algorithm)
# ────────────────────────────────────────────────────────────

def test_T1_single_tier():
    print("T1 single-tier crossbar equivalence:")
    tier = CrossbarTier(name="t0", ports=32, bw_per_port_GBps=900.0, alpha_us=0.5)
    cases = [
        ("p2p",        "ring", 1_024.0,        2),
        ("p2p",        "ring", 10_000_000.0,   2),
        ("all_gather", "ring", 4_096.0,        8),
        ("all_gather", "ring", 100_000_000.0,  16),
        ("all_reduce", "ring", 1_000_000.0,    8),
        ("all_reduce", "ring", 1_000_000.0,    32),
        ("all_reduce", "tree", 1_000_000.0,    8),
        ("all_reduce", "tree", 1_000_000.0,    16),
        ("moe_a2a",    "ring", 500_000.0,      8),
        ("moe_a2a",    "ring", 500_000.0,      32),
        ("moe_a2a",    "tree", 500_000.0,      8),
    ]
    for op, alg, M, G in cases:
        old = _old_path(op, [tier], M, G, alg)
        new = cost_collective([tier], op, M, G, algorithm=alg)
        _check(f"{op}/{alg} M={M:.0e} G={G}", new, old)


# ────────────────────────────────────────────────────────────
# T2 — Multi-tier crossbar chain equivalence
#   Modeled after gb200.nvl576.hierarchical (NVLink scale-up + IB scale-out).
# ────────────────────────────────────────────────────────────

def test_T2_multi_tier():
    print("T2 multi-tier crossbar chain equivalence:")
    tier0 = CrossbarTier(name="nvlink", ports=72, bw_per_port_GBps=900.0, alpha_us=0.5)
    tier1 = CrossbarTier(name="ib",     ports=8,  bw_per_port_GBps=50.0,  alpha_us=2.0)
    chain = [tier0, tier1]
    cases = [
        # Stays in tier 0
        ("all_reduce", "ring", 1_000_000.0, 8),
        ("all_reduce", "ring", 1_000_000.0, 64),
        # Boundary — exactly tier0.ports
        ("all_reduce", "ring", 1_000_000.0, 72),
        ("all_reduce", "tree", 1_000_000.0, 72),
        # Spills into tier 1
        ("all_reduce", "ring", 1_000_000.0, 144),
        ("all_reduce", "ring", 1_000_000.0, 576),
        ("all_reduce", "tree", 1_000_000.0, 576),
        # All-gather + MoE-A2A across both tiers
        ("all_gather", "ring", 10_000.0,   288),
        ("moe_a2a",    "ring", 50_000.0,   288),
        ("moe_a2a",    "tree", 50_000.0,   288),
        # p2p always stays in tier 0 (G=2)
        ("p2p",        "ring", 500_000.0,  2),
    ]
    for op, alg, M, G in cases:
        old = _old_path(op, chain, M, G, alg)
        new = cost_collective(chain, op, M, G, algorithm=alg)
        _check(f"{op}/{alg} G={G}", new, old)


# ────────────────────────────────────────────────────────────
# T3 — Randomized property test: 1000 crossbar-only chains
# ────────────────────────────────────────────────────────────

def test_T3_randomized():
    print("T3 randomized property test (1000 draws):")
    rng = random.Random(0xC0FFEE)
    ops = ["all_reduce", "all_gather", "moe_a2a", "p2p"]
    algs = ["ring", "tree"]
    mismatches = 0
    for i in range(1000):
        n_tiers = rng.randint(1, 4)
        tiers = [
            CrossbarTier(
                name=f"t{j}",
                ports=rng.choice([2, 4, 8, 16, 32, 64, 128]),
                bw_per_port_GBps=rng.uniform(10.0, 2000.0),
                alpha_us=rng.uniform(0.1, 5.0),
            )
            for j in range(n_tiers)
        ]
        op = rng.choice(ops)
        alg = rng.choice(algs) if op in ("all_reduce", "moe_a2a") else "ring"
        M = rng.uniform(1.0, 1e8)
        # Group size spans inside tier 0 through full-chain reach.
        max_reach = 1
        for t in tiers:
            max_reach *= t.ports
        G = rng.randint(2, max_reach) if op != "p2p" else 2
        old = _old_path(op, tiers, M, G, alg)
        new = cost_collective(tiers, op, M, G, algorithm=alg)
        if old != new:
            mismatches += 1
            if mismatches <= 5:
                print(f"  FAIL draw#{i} {op}/{alg}: old={old!r} new={new!r}")
    if mismatches == 0:
        print("  PASS  1000/1000 bit-identical")
    else:
        _failures.append(f"T3 randomized: {mismatches}/1000 mismatches")


# ────────────────────────────────────────────────────────────
# T4 — Crossbar path must NOT emit torus dim-alignment warning
# ────────────────────────────────────────────────────────────

def test_T4_crossbar_silent():
    print("T4 crossbar path emits no UserWarning:")
    tier = CrossbarTier(name="t0", ports=16, bw_per_port_GBps=900.0, alpha_us=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost_collective([tier], "all_reduce", 1e6, 12, algorithm="ring")
        cost_collective([tier], "moe_a2a",    1e6, 13, algorithm="ring")
        cost_collective([tier], "all_gather", 1e6, 7)
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    if not user_warns:
        print("  PASS  no UserWarning on crossbar inputs")
    else:
        _failures.append(f"T4: unexpected warnings: {[str(w.message) for w in user_warns]}")
        for w in user_warns:
            print(f"  FAIL  unexpected warning: {w.message}")


# ────────────────────────────────────────────────────────────
# T5 — torus_all_reduce(dims=(N,)) == ring_all_reduce(N)
#   Continuity check guarding against silent divergence in the torus path.
# ────────────────────────────────────────────────────────────

def test_T5_torus_k1_continuity():
    print("T5 torus k=1 continuity:")
    for N in (2, 4, 8, 16, 64, 128):
        for M in (1_024.0, 1e6, 1e8):
            want = ring_all_reduce(M, N, 1e-6, 1e9)
            got = torus_all_reduce(M, (N,), 1e-6, 1e9)
            _check(f"N={N} M={M:.0e}", got, want)
            # Also via the dispatcher on a single TorusTier:
            ttier = TorusTier(name="t", dims=(N,), bw_per_port_GBps=1.0, alpha_us=1.0)
            got_disp = cost_collective([ttier], "all_reduce", M, N, torus_algorithm="ring")
            _check(f"dispatch N={N} M={M:.0e}", got_disp, want)


# ────────────────────────────────────────────────────────────

def main() -> int:
    test_T1_single_tier()
    test_T2_multi_tier()
    test_T3_randomized()
    test_T4_crossbar_silent()
    test_T5_torus_k1_continuity()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll topology-equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
