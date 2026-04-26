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
    inc_a2a,
    inc_all_gather,
    inc_all_reduce,
    p2p_hop,
    pairwise_a2a,
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
# T6 — INC dispatch: inc_enabled routing, opt-out parity, scope limits
# ────────────────────────────────────────────────────────────

def test_T6_inc_dispatch():
    print("T6 INC dispatch:")
    M = 1_000_000.0
    alpha_endpoint = 0.5   # software α, μs
    alpha_switch = 0.2     # cut-through α, μs
    bw = 900.0             # GB/s

    inc_tier = CrossbarTier(
        name="nvls", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls", inc_alpha_us=alpha_switch,
    )
    plain_tier = CrossbarTier(
        name="plain", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
    )

    alpha_s_switch = alpha_switch * US_TO_SECONDS
    alpha_s_endpoint = alpha_endpoint * US_TO_SECONDS
    bw_Bps = bw * GB_TO_BYTES

    # (a) AR single-tier INC — should match inc_all_reduce(M, α_switch, BW) exactly.
    got = cost_collective([inc_tier], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_switch, bw_Bps)
    _check("AR G=64 inc on", got, want)

    # (b) AR single-tier with inc_enabled=False — falls back to ring_all_reduce
    # using the tier's endpoint α (not the switch cut-through).
    got = cost_collective([inc_tier], "all_reduce", M, 64, inc_enabled=False)
    want = ring_all_reduce(M, 64, alpha_s_endpoint, bw_Bps)
    _check("AR G=64 inc off → ring", got, want)

    # (c) AG routes through inc_all_gather when enabled.
    got = cost_collective([inc_tier], "all_gather", M, 64)
    want = inc_all_gather(M, 64, alpha_s_switch, bw_Bps)
    _check("AG G=64 inc on", got, want)

    # (d) MoE A2A on an INC tier must stay on ring (no structural INC win).
    got = cost_collective([inc_tier], "moe_a2a", M, 64, algorithm="ring")
    want = ring_moe_all_to_all(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A G=64 inc tier → ring", got, want)

    # (e) p2p on an INC tier stays on single-hop.
    got = cost_collective([inc_tier], "p2p", M, 2)
    want = p2p_hop(M, alpha_s_endpoint, bw_Bps)
    _check("p2p G=2 inc tier → hop", got, want)

    # (f) Partial-INC chain (plain + inc) must fall back to software — INC requires
    # every crossed tier to support in-fabric reduction.
    # G=64 ≤ plain_tier.ports=72 so the walker stops at tier 0 and only plain α counts.
    got = cost_collective([plain_tier, inc_tier], "all_reduce", M, 64)
    want = ring_all_reduce(M, 64, alpha_s_endpoint, bw_Bps)
    _check("AR partial-INC chain → software ring", got, want)

    # (g) Multi-tier full-INC chain: scale-out aggregation tree.
    # G > tier0.ports forces walk into tier1; both tiers contribute α.
    inc_tier2 = CrossbarTier(
        name="sharp-ib", ports=8, bw_per_port_GBps=400.0, alpha_us=2.5,
        inc="sharp", inc_alpha_us=0.4,
    )
    got = cost_collective([inc_tier, inc_tier2], "all_reduce", M, 576)
    # α_total = 0.2 + 0.4 = 0.6 μs; BW_min = min(900, 400) = 400 GB/s.
    alpha_total_s = (0.2 + 0.4) * US_TO_SECONDS
    bw_min_Bps = 400.0 * GB_TO_BYTES
    want = inc_all_reduce(M, alpha_total_s, bw_min_Bps)
    _check("AR multi-tier INC scale-out", got, want)

    # (h) η coefficients flow through the INC path identically to crossbar.
    inc_tier_eta = CrossbarTier(
        name="nvls-eta", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls", inc_alpha_us=alpha_switch,
        eta_alpha=1.5, eta_beta=0.8,
    )
    got = cost_collective([inc_tier_eta], "all_reduce", M, 64)
    want = inc_all_reduce(
        M,
        alpha_switch * 1.5 * US_TO_SECONDS,
        bw * 0.8 * GB_TO_BYTES,
    )
    _check("AR INC with η≠1", got, want)

    # (i) inc_alpha_us=0 sentinel — reuses alpha_us on the INC path.
    inc_tier_reuse = CrossbarTier(
        name="inc-reuse", ports=72, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="nvls",  # no inc_alpha_us override → sentinel 0.0
    )
    got = cost_collective([inc_tier_reuse], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_endpoint, bw_Bps)
    _check("AR INC with inc_alpha_us=0 sentinel", got, want)

    # (j) HW A2A tier — A2A routes through inc_a2a (collectives.md §5.4).
    hw_a2a_tier = CrossbarTier(
        name="tomahawk-ultra", ports=64, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="hw_a2a", inc_alpha_us=alpha_switch,
    )
    got = cost_collective([hw_a2a_tier], "moe_a2a", M, 64, algorithm="ring")
    want = inc_a2a(M, 64, alpha_s_switch, bw_Bps)
    _check("A2A G=64 hw_a2a tier → inc_a2a", got, want)

    # (k) sharp_class tier with A2A — must NOT route through inc_a2a; falls through
    # to software pairwise (sharp_class only covers AR / AG / RS).
    sharp_tier = CrossbarTier(
        name="sharp-class", ports=64, bw_per_port_GBps=bw, alpha_us=alpha_endpoint,
        inc="sharp_class", inc_alpha_us=alpha_switch,
    )
    got = cost_collective([sharp_tier], "moe_a2a", M, 64, algorithm="ring")
    want = pairwise_a2a(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A G=64 sharp_class tier → pairwise (no INC for A2A)", got, want)

    # (l) HW A2A tier with AR/AG — also routes through SHARP-class INC paths
    # (hw_a2a is a superset of sharp_class capability).
    got = cost_collective([hw_a2a_tier], "all_reduce", M, 64)
    want = inc_all_reduce(M, alpha_s_switch, bw_Bps)
    _check("AR G=64 hw_a2a tier → inc_all_reduce", got, want)

    # (m) Mixed sharp_class + hw_a2a chain for A2A: gating requires *every* tier
    # to be hw_a2a, so this falls back to software.
    got = cost_collective([sharp_tier, hw_a2a_tier], "moe_a2a", M, 4096, algorithm="ring")
    # G=4096 forces walk into both tiers. _is_hw_a2a fails on sharp_tier → fallback.
    alpha_total_s = (alpha_endpoint + alpha_endpoint) * US_TO_SECONDS
    bw_min_Bps = bw * GB_TO_BYTES
    want = pairwise_a2a(M, 4096, alpha_total_s, bw_min_Bps)
    _check("A2A mixed sharp+hw_a2a → software", got, want)

    # (n) inc_enabled=False forces software fallback even on hw_a2a tier.
    got = cost_collective([hw_a2a_tier], "moe_a2a", M, 64, inc_enabled=False, algorithm="ring")
    want = pairwise_a2a(M, 64, alpha_s_endpoint, bw_Bps)
    _check("A2A hw_a2a tier inc_enabled=False → pairwise", got, want)


# ────────────────────────────────────────────────────────────
# T7 — system_loaders: inc enum back-compat + oversubscription clip
# ────────────────────────────────────────────────────────────

def test_T7_loader_inc_and_oversub():
    print("T7 loader (inc enum + oversubscription):")
    from llm_perf.io.system_loaders import system_spec_from_json_dict

    base_cfg = {
        "schema": "llm_perf.system",
        "name": "test",
        "num_devices": 8,
        "device": {"name": "x", "hbm_capacity_GB": 80.0,
                   "hbm_bandwidth_GBps": 3000.0, "peak_flops_TF": 1000.0},
        "fabrics": {
            "f": {"tiers": [{"name": "t0", "ports": 8,
                             "bw_per_port_GBps": 900.0, "alpha_us": 0.5}]}
        },
        "collective_fabrics": {"TP": "f", "EP": "f", "SP": "f", "PP": "f"},
    }

    # (a) Legacy `inc: nvls` parses to canonical `sharp_class`.
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "nvls"
    sys_a = system_spec_from_json_dict(cfg)
    tier = sys_a.fabrics["f"].tiers[0]
    if tier.inc != "sharp_class":
        _failures.append(f"T7a: inc='nvls' → expected 'sharp_class', got {tier.inc!r}")
    else:
        print("  PASS  inc='nvls' → 'sharp_class'")

    # (b) Legacy `inc: sharp` parses to canonical `sharp_class`.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "sharp"
    sys_b = system_spec_from_json_dict(cfg)
    if sys_b.fabrics["f"].tiers[0].inc != "sharp_class":
        _failures.append(f"T7b: inc='sharp' → expected 'sharp_class', got {sys_b.fabrics['f'].tiers[0].inc!r}")
    else:
        print("  PASS  inc='sharp' → 'sharp_class'")

    # (c) New `inc: hw_a2a` parses through.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "hw_a2a"
    sys_c = system_spec_from_json_dict(cfg)
    if sys_c.fabrics["f"].tiers[0].inc != "hw_a2a":
        _failures.append(f"T7c: inc='hw_a2a' lost: {sys_c.fabrics['f'].tiers[0].inc!r}")
    else:
        print("  PASS  inc='hw_a2a' parses through")

    # (d) Unknown `inc` value rejected.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["inc"] = "bogus"
    try:
        system_spec_from_json_dict(cfg)
        _failures.append("T7d: expected ValueError for inc='bogus'")
    except ValueError:
        print("  PASS  inc='bogus' rejected")

    # (e) oversubscription default = 1.0; eta_beta passes through unchanged.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.8
    sys_e = system_spec_from_json_dict(cfg)
    t = sys_e.fabrics["f"].tiers[0]
    if t.eta_beta != 0.8 or t.oversubscription != 1.0:
        _failures.append(f"T7e: defaults broken (eta_beta={t.eta_beta}, s={t.oversubscription})")
    else:
        print("  PASS  oversubscription=1.0 default; eta_beta passes through")

    # (f) oversubscription > 1 with eta_beta > 1/s — silent min(eta_beta, 1/s).
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.9
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys_f = system_spec_from_json_dict(cfg)
    t = sys_f.fabrics["f"].tiers[0]
    if t.eta_beta != 0.5 or t.oversubscription != 2.0:
        _failures.append(f"T7f: clip wrong (eta_beta={t.eta_beta}, expected 0.5)")
    elif caught:
        _failures.append(f"T7f: unexpected warnings: {[str(w.message) for w in caught]}")
    else:
        print("  PASS  eta_beta=0.9, s=2 → silent min → 0.5")

    # (g) oversubscription > 1 with eta_beta ≤ 1/s passes through unchanged.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["eta_beta"] = 0.3
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 2.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys_g = system_spec_from_json_dict(cfg)
    if sys_g.fabrics["f"].tiers[0].eta_beta != 0.3:
        _failures.append(f"T7g: passthrough wrong (eta_beta={sys_g.fabrics['f'].tiers[0].eta_beta})")
    elif caught:
        _failures.append(f"T7g: unexpected warnings: {[str(w.message) for w in caught]}")
    else:
        print("  PASS  eta_beta=0.3, s=2 → unchanged (min picks 0.3)")

    # (h) oversubscription < 1 rejected.
    cfg = copy.deepcopy(base_cfg)
    cfg["fabrics"]["f"]["tiers"][0]["oversubscription"] = 0.5
    try:
        system_spec_from_json_dict(cfg)
        _failures.append("T7h: expected ValueError for oversubscription < 1")
    except ValueError:
        print("  PASS  oversubscription < 1 rejected")


# ────────────────────────────────────────────────────────────

def main() -> int:
    test_T1_single_tier()
    test_T2_multi_tier()
    test_T3_randomized()
    test_T4_crossbar_silent()
    test_T5_torus_k1_continuity()
    test_T6_inc_dispatch()
    test_T7_loader_inc_and_oversub()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll topology-equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
