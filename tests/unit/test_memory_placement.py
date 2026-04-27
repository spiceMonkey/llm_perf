"""Unit tests for the multi-tier placement and roofline (PR2, sram.md §1.3, §2).

Pins down:
  - Single-tier reduction (legacy shim path) → t_mem matches the legacy
    (T_θ + B·T_KV) / BW_mem expression to floating-point equality.
  - Two-tier "auto" greedy: weights fill fast tier, KV spills to slow tier
    when fast tier is full.
  - Two-tier operator-pinned: weights pinned to slow tier (Capacity Mode);
    KV pinned to a chosen tier.
  - CapacityError on pinned overflow; auto mode is permissive (overflow
    accumulates on the last tier so latency math remains computable).
  - placement_fits predicate matches the same boundary.
  - Two-tier crossover formula (sram.md §2.2) — numerical sweep finds B*
    matching the closed form.

Usage:  PYTHONPATH=. python tests/unit/test_memory_placement.py
"""
import json
import sys
import tempfile
from pathlib import Path

from llm_perf import MemoryTierSpec
from llm_perf.core.memory_placement import (
    CapacityError,
    placement_fits,
    resolve_placement,
    t_mem_from_placement,
)
from llm_perf.io.tuner_loaders import load_tuning_spec
from llm_perf.specs.tuner_spec import MemoryPlacementSpec
from llm_perf.utils import GB_TO_BYTES


_failures = []


def _check(label, cond, detail=""):
    print(f"{'OK' if cond else 'FAIL'}: {label}{(' — ' + detail) if detail and not cond else ''}")
    if not cond:
        _failures.append(label)


def _check_close(label, got, expected, rel=1e-9):
    ok = abs(got - expected) <= rel * max(abs(expected), 1.0)
    print(f"{'OK' if ok else 'FAIL'}: {label}: got={got!r} expected≈{expected!r}")
    if not ok:
        _failures.append(label)


# Reusable tier list factories. Bandwidth in GB/s, capacity in GB.
def _single_hbm():
    return [MemoryTierSpec(name="hbm", capacity_GB=80.0, bandwidth_GBps=3350.0,
                           eta_beta=1.0)]


def _two_tier_dmatrix():
    """SRAM (fast, small) + LPDDR5 (slow, large), sram.md §3.2 numbers."""
    return [
        MemoryTierSpec(name="sram", capacity_GB=2.0, bandwidth_GBps=150_000.0,
                       eta_beta=1.0),
        MemoryTierSpec(name="lpddr5", capacity_GB=256.0, bandwidth_GBps=400.0,
                       eta_beta=0.85),
    ]


# ────────────────────────────────────────────────────────────
# Single-tier reduction (sram.md §2.1 reduces to legacy form)
# ────────────────────────────────────────────────────────────

def test_single_tier_t_mem_equals_legacy_form():
    """Legacy: t_mem = (T_θ + B·T_KV) / BW_mem (no η_β deflator). PR2 with
    a single-tier shim and η_β=1.0 must produce the same number bit-for-bit."""
    tiers = _single_hbm()
    T_theta, T_kv, B = 8.75e9, 1.05e9, 16  # GB-scale, sram.md §3.1 example
    placement = resolve_placement(T_theta, T_kv, B, tiers, MemoryPlacementSpec())
    t_mem = t_mem_from_placement(placement, B, tiers)
    BW_bytes = tiers[0].bandwidth_GBps * GB_TO_BYTES  # eta=1.0
    expected = (T_theta + B * T_kv) / BW_bytes
    _check_close("single-tier t_mem matches legacy expression", t_mem, expected)


def test_single_tier_overflow_is_permissive():
    """Auto mode with overflow accumulates onto the last tier; t_mem still
    computable. fits() returns False. Matches legacy behavior where t_mem
    was computed regardless of fits_in_HBM."""
    tiers = _single_hbm()  # 80 GB
    T_theta, T_kv, B = 70e9, 1e9, 64  # 70 + 64 = 134 GB > 80 GB
    placement = resolve_placement(T_theta, T_kv, B, tiers, MemoryPlacementSpec())
    _check("auto overflow does not raise", True)
    _check("placement_fits=False on overflow",
           not placement_fits(placement, B, tiers))
    # Conservation still holds — overflow accumulates on the last (only) tier.
    _check_close("conservation: weights sum",
                 sum(placement.weights_per_tier), T_theta)
    _check_close("conservation: KV sum",
                 sum(placement.kv_per_request_per_tier), T_kv)


# ────────────────────────────────────────────────────────────
# Two-tier auto (sram.md §1.3 greedy fastest-first)
# ────────────────────────────────────────────────────────────

def test_two_tier_auto_fits_in_fast_tier():
    """Workload fits entirely in SRAM — nothing should land on LPDDR5."""
    tiers = _two_tier_dmatrix()
    T_theta, T_kv, B = 1.0e9, 0.05e9, 16  # 1 + 16·0.05 = 1.8 GB ≤ 2 GB SRAM
    placement = resolve_placement(T_theta, T_kv, B, tiers, MemoryPlacementSpec())
    _check_close("auto: weights all on SRAM", placement.weights_per_tier[0], T_theta)
    _check_close("auto: nothing on LPDDR5 (weights)", placement.weights_per_tier[1], 0.0)
    _check_close("auto: KV all on SRAM", placement.kv_per_request_per_tier[0], T_kv)
    _check_close("auto: nothing on LPDDR5 (KV)", placement.kv_per_request_per_tier[1], 0.0)
    _check("auto: fits=True", placement_fits(placement, B, tiers))


def test_two_tier_auto_kv_spills_to_lpddr5():
    """Weights + KV exceed SRAM capacity; KV gets the partial spill since
    weights were placed first per the greedy policy."""
    tiers = _two_tier_dmatrix()
    T_theta, T_kv, B = 1.5e9, 0.1e9, 16  # 1.5 weights + 1.6 KV = 3.1 > 2 GB
    placement = resolve_placement(T_theta, T_kv, B, tiers, MemoryPlacementSpec())
    _check_close("greedy: all weights stay on SRAM",
                 placement.weights_per_tier[0], 1.5e9)
    # SRAM remaining = 2 - 1.5 = 0.5 GB; KV total = 1.6 GB → 0.5 on SRAM, 1.1 on LPDDR5
    kv_sram_total = placement.kv_per_request_per_tier[0] * B
    _check_close("greedy: SRAM KV uses remaining 0.5 GB", kv_sram_total, 0.5e9)
    kv_lpddr_total = placement.kv_per_request_per_tier[1] * B
    _check_close("greedy: LPDDR5 KV gets the 1.1 GB spill", kv_lpddr_total, 1.1e9)


def test_two_tier_t_mem_dominated_by_slow_tier():
    """When weights spill to LPDDR5 (Capacity Mode), t_mem should be ≈
    weights / BW_LPDDR5 — the slow-tier load swamps everything."""
    tiers = _two_tier_dmatrix()
    T_theta, T_kv, B = 4.4e9, 0.084e9, 16  # sram.md §3.3 numbers
    # Pin weights to LPDDR5 (Capacity Mode); KV stays on SRAM by default.
    plc = resolve_placement(
        T_theta, T_kv, B, tiers,
        MemoryPlacementSpec(weights_tier="lpddr5", kv_tier="sram"),
    )
    t_mem = t_mem_from_placement(plc, B, tiers)
    # Expected: 4.4 GB / (400 GB/s · 0.85) + 1.34 GB / 150,000 GB/s
    BW_lpddr_eff = 400.0 * 0.85 * GB_TO_BYTES
    BW_sram_eff = 150_000.0 * 1.0 * GB_TO_BYTES
    expected = 4.4e9 / BW_lpddr_eff + (B * 0.084e9) / BW_sram_eff
    _check_close("Capacity Mode t_mem matches §2.2 form", t_mem, expected, rel=1e-6)


# ────────────────────────────────────────────────────────────
# Operator-pinned mode (sram.md §1.3 second policy)
# ────────────────────────────────────────────────────────────

def test_pin_weights_to_slow_tier():
    tiers = _two_tier_dmatrix()
    plc = resolve_placement(
        1.0e9, 0.05e9, 16, tiers,
        MemoryPlacementSpec(weights_tier="lpddr5"),
    )
    _check_close("pin: nothing on SRAM (weights)", plc.weights_per_tier[0], 0.0)
    _check_close("pin: all weights on LPDDR5", plc.weights_per_tier[1], 1.0e9)


def test_pinned_overflow_raises():
    tiers = _two_tier_dmatrix()
    try:
        resolve_placement(
            5.0e9, 0.0, 1, tiers,  # 5 GB weights pinned to 2 GB SRAM
            MemoryPlacementSpec(weights_tier="sram"),
        )
        _check("pinned overflow raises CapacityError", False, "no exception")
    except CapacityError as e:
        _check("pinned overflow raises CapacityError", True)
        _check("error names overflowing class", e.data_class == "weights")


def test_auto_priority_weights_default_fills_weights_first():
    """auto_priority='weights' (the default) puts weights on the fast tier
    first; KV gets remainder. This is the existing behavior — pinned by the
    test to guard against regression."""
    tiers = _two_tier_dmatrix()
    # Both classes individually fit but together exceed SRAM.
    T_theta, T_kv, B = 1.5e9, 0.04e9, 16  # 1.5 weights + 0.64 KV = 2.14 > 2 GB SRAM
    plc = resolve_placement(T_theta, T_kv, B, tiers, MemoryPlacementSpec())
    _check_close("weights-first: weights all on SRAM",
                 plc.weights_per_tier[0], T_theta)
    _check_close("weights-first: weights nothing on LPDDR5",
                 plc.weights_per_tier[1], 0.0)


def test_auto_priority_kv_fills_kv_first():
    """auto_priority='kv' flips the order — KV claims SRAM first; weights
    spill to LPDDR5 if there's not enough room."""
    tiers = _two_tier_dmatrix()
    T_theta, T_kv, B = 1.5e9, 0.04e9, 16
    plc = resolve_placement(
        T_theta, T_kv, B, tiers,
        MemoryPlacementSpec(auto_priority="kv"),
    )
    # KV total = 16 * 0.04 = 0.64 GB; fits on SRAM with margin
    _check_close("kv-first: KV all on SRAM",
                 plc.kv_per_request_per_tier[0], T_kv)
    _check_close("kv-first: KV nothing on LPDDR5",
                 plc.kv_per_request_per_tier[1], 0.0)
    # Weights then fill remaining SRAM (2 - 0.64 = 1.36 GB) and spill 0.14 GB
    _check_close("kv-first: weights on SRAM = remaining capacity",
                 plc.weights_per_tier[0], 1.36e9)
    _check_close("kv-first: weights spill 0.14 GB to LPDDR5",
                 plc.weights_per_tier[1], 0.14e9)


def test_auto_priority_inert_when_pinned():
    """When weights or KV is explicitly pinned, auto_priority has no effect
    on the pinned class — pin always wins."""
    tiers = _two_tier_dmatrix()
    plc = resolve_placement(
        1.0e9, 0.05e9, 16, tiers,
        MemoryPlacementSpec(weights_tier="lpddr5", auto_priority="kv"),
    )
    # weights pinned → all on LPDDR5 regardless of priority
    _check_close("pinned weights override priority",
                 plc.weights_per_tier[1], 1.0e9)
    _check_close("pinned weights nothing on SRAM",
                 plc.weights_per_tier[0], 0.0)


def test_auto_priority_invalid_raises():
    tiers = _two_tier_dmatrix()
    try:
        resolve_placement(1e9, 0.0, 1, tiers,
                          MemoryPlacementSpec(auto_priority="banana"))
        _check("bad auto_priority raises", False, "no exception")
    except ValueError as e:
        _check("bad auto_priority raises ValueError", "auto_priority" in str(e))


def test_pin_unknown_tier_raises():
    tiers = _two_tier_dmatrix()
    try:
        resolve_placement(
            1.0e9, 0.05e9, 16, tiers,
            MemoryPlacementSpec(weights_tier="exotic_mram"),
        )
        _check("unknown tier name raises ValueError", False, "no exception")
    except ValueError:
        _check("unknown tier name raises ValueError", True)


# ────────────────────────────────────────────────────────────
# Two-tier crossover (sram.md §2.2)
# ────────────────────────────────────────────────────────────

def test_two_tier_crossover_matches_closed_form():
    """Numerical sweep B → find B* (where t_compute = t_mem) and compare to
    the §2.2 closed form B*_W,K = (R · T_θ / BW_W) / (F - R · T_KV / BW_K).
    Setup: weights on SRAM (fast), KV on LPDDR5 (slow). For pure single-token
    matmul, F_token would be linear in H; we use synthetic numbers tuned so
    the crossover lives in a sensible B range."""
    tiers = _two_tier_dmatrix()
    T_theta, T_kv = 1.0e9, 0.001e9
    F_token = 1e10  # FLOPs/token for one device
    R = 2.4e15      # peak FLOPs/s (sram.md §3.2 d-Matrix)
    BW_W = 150_000.0 * 1.0 * GB_TO_BYTES
    BW_K = 400.0 * 0.85 * GB_TO_BYTES

    # §2.2 closed form
    denom = F_token - R * T_kv / BW_K
    if denom <= 0:
        _check("crossover formula has positive denominator", False,
               f"denom={denom}")
        return
    B_star_closed = R * T_theta / BW_W / denom

    # Numerical: find smallest B where B·F/R ≥ T_θ/BW_W + B·T_KV/BW_K
    def residual(B):
        return (B * F_token / R) - (T_theta / BW_W + B * T_kv / BW_K)

    # Bisection in [B_lo, B_hi] where residual flips sign
    B_lo, B_hi = 1.0, 1e9
    while B_hi - B_lo > 1e-6 * B_star_closed:
        B_mid = 0.5 * (B_lo + B_hi)
        if residual(B_mid) < 0:
            B_lo = B_mid
        else:
            B_hi = B_mid
    B_star_num = 0.5 * (B_lo + B_hi)
    _check_close("numerical B* matches closed form", B_star_num, B_star_closed, rel=1e-4)


# ────────────────────────────────────────────────────────────
# Tuner JSON loader (PR3): parses optional `placement` block
# ────────────────────────────────────────────────────────────

def _load_temp_tuner(cfg: dict):
    """Inject the required collective-count fields if absent so each test
    can stay focused on the placement block."""
    cfg.setdefault("n_TP_collectives", 2)
    cfg.setdefault("n_EP_collectives", 1)
    cfg.setdefault("n_SP_collectives", 1)
    cfg.setdefault("overlap_factor", 0.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    try:
        return load_tuning_spec(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_tuner_loader_default_placement_is_auto_auto():
    """A tuner JSON with no placement block defaults to MemoryPlacementSpec()."""
    spec = _load_temp_tuner({
        "schema": "llm_perf.tuner",
        "name": "no-placement",
        "S_decode": 2048,
    })
    _check("default weights_tier=auto", spec.placement.weights_tier == "auto")
    _check("default kv_tier=auto", spec.placement.kv_tier == "auto")


def test_tuner_loader_placement_block_round_trips():
    """Capacity Mode shape: weights pinned to 'hbm', KV on 'sram'."""
    spec = _load_temp_tuner({
        "schema": "llm_perf.tuner",
        "name": "capacity-mode",
        "S_decode": 8192,
        "placement": {"weights_tier": "hbm", "kv_tier": "sram"},
    })
    _check("loader weights_tier='hbm'", spec.placement.weights_tier == "hbm")
    _check("loader kv_tier='sram'", spec.placement.kv_tier == "sram")


def test_tuner_loader_rejects_non_dict_placement():
    try:
        _load_temp_tuner({
            "schema": "llm_perf.tuner",
            "name": "bad-placement",
            "S_decode": 2048,
            "placement": "auto",  # must be an object, not a string
        })
        _check("non-dict placement raises", False, "no exception")
    except ValueError:
        _check("non-dict placement raises ValueError", True)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    test_single_tier_t_mem_equals_legacy_form()
    test_single_tier_overflow_is_permissive()
    test_two_tier_auto_fits_in_fast_tier()
    test_two_tier_auto_kv_spills_to_lpddr5()
    test_two_tier_t_mem_dominated_by_slow_tier()
    test_pin_weights_to_slow_tier()
    test_pinned_overflow_raises()
    test_pin_unknown_tier_raises()
    test_auto_priority_weights_default_fills_weights_first()
    test_auto_priority_kv_fills_kv_first()
    test_auto_priority_inert_when_pinned()
    test_auto_priority_invalid_raises()
    test_two_tier_crossover_matches_closed_form()
    test_tuner_loader_default_placement_is_auto_auto()
    test_tuner_loader_placement_block_round_trips()
    test_tuner_loader_rejects_non_dict_placement()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll memory-placement unit tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
