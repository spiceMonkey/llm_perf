"""Unit tests for the SW-overhead extension to the decode roofline
(documentation/explaining/kernel_launch_overhead.md §5).

Covers four corner cases:
  (a) kernel_launch_us = 0 → behavior bit-identical to legacy roofline.
  (b) Partition-shape dependence via the k_collective term.
  (c) η_TC ramp at small mb derates t_compute.
  (d) ρ_SW = 1 (default, full overlap) makes t_SW a floor;
      ρ_SW = 0 makes it strictly additive.

Usage:  PYTHONPATH=. python tests/unit/test_sw_overhead.py
"""
import math
import sys
from dataclasses import replace

from llm_perf.calculators.inference_calculator import InferenceCalculator
from llm_perf.io import load_model_spec, load_system_spec, load_tuning_spec
from llm_perf.specs.partition_spec import PartitionSpec


_failures: list[str] = []


def _check(label: str, got: float, want: float) -> None:
    if math.isclose(got, want, rel_tol=0.0, abs_tol=0.0):
        print(f"  PASS  {label}: {got!r}")
    else:
        _failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")


def _check_near(label: str, got: float, want: float, rel_tol: float) -> None:
    if math.isclose(got, want, rel_tol=rel_tol, abs_tol=0.0):
        print(f"  PASS  {label}: {got!r} ≈ {want!r}")
    else:
        _failures.append(f"{label}: got {got!r}, want ~{want!r}")
        print(f"  FAIL  {label}: got {got!r}, want ~{want!r}")


def _check_truthy(label: str, cond: bool, why: str) -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        _failures.append(f"{label}: {why}")
        print(f"  FAIL  {label}: {why}")


# ────────────────────────────────────────────────────────────
# Fixture: small dense model on the example single-GPU system.
# ────────────────────────────────────────────────────────────

def _fixture():
    m = load_model_spec("llm_perf/database/model/example.model.dense.json")
    s = load_system_spec("llm_perf/database/system/example.sys.json")
    t = load_tuning_spec("llm_perf/database/tuner/example.tuner.json")
    return m, s, t


# ────────────────────────────────────────────────────────────
# (a) kernel_launch_us = 0 → legacy bit-identity
# ────────────────────────────────────────────────────────────

def test_sw_disabled_matches_legacy_roofline() -> None:
    print("\ntest_sw_disabled_matches_legacy_roofline")
    m, s, t_prod = _fixture()
    p = PartitionSpec(PP=2, TP=2, EP=1, SP=1)

    # Disable SW: t_SW must be 0 and t_step_user must equal t_stage * γ_pp.
    t_off = replace(
        t_prod,
        kernel_launch_us=0.0,
        kernels_per_layer_compute=0,
        kernels_per_collective_call=0,
        tensor_core_efficiency=None,
    )
    r = InferenceCalculator(m, s, p, t_off).run().latency
    _check("t_SW", r.t_SW, 0.0)
    _check("eta_TC (None curve)", r.eta_TC, 1.0)
    _check("t_compute_eff = t_compute when η_TC=1", r.t_compute_eff, r.t_compute)
    expected_step = r.t_stage * r.pp_bubble_factor
    _check_near("t_step_user = t_stage · γ_pp", r.t_step_user, expected_step, rel_tol=1e-12)


# ────────────────────────────────────────────────────────────
# (b) k_collective makes t_SW partition-shape-dependent
# ────────────────────────────────────────────────────────────

def test_t_sw_scales_with_collective_count() -> None:
    print("\ntest_t_sw_scales_with_collective_count")
    m, s, t = _fixture()
    # Use known production knobs.
    t = replace(t, kernel_launch_us=1.5, kernels_per_layer_compute=10, kernels_per_collective_call=2)

    # PP=4, TP=1, EP=1, SP=1 → no collectives fire → k = 10
    r_nocoll = InferenceCalculator(m, s, PartitionSpec(PP=4, TP=1, EP=1, SP=1), t).run().latency
    # PP=4, TP=2, EP=1, SP=1 → only TP fires → k = 10 + 2 · n_TP_collectives = 10 + 4 = 14
    r_TP = InferenceCalculator(m, s, PartitionSpec(PP=4, TP=2, EP=1, SP=1), t).run().latency

    L = m.L
    expected_nocoll = L * 10 * 1.5e-6
    expected_TP = L * (10 + 2 * t.n_TP_collectives) * 1.5e-6

    _check_near("t_SW (TP=1, no coll)", r_nocoll.t_SW, expected_nocoll, rel_tol=1e-12)
    _check_near("t_SW (TP=2, n_TP=2 collectives)", r_TP.t_SW, expected_TP, rel_tol=1e-12)
    _check_truthy(
        "TP-on t_SW > TP-off t_SW",
        r_TP.t_SW > r_nocoll.t_SW,
        f"got {r_TP.t_SW} vs {r_nocoll.t_SW}",
    )


# ────────────────────────────────────────────────────────────
# (c) η_TC ramp derates t_compute at small mb
# ────────────────────────────────────────────────────────────

def test_eta_tc_derates_compute_at_small_mb() -> None:
    print("\ntest_eta_tc_derates_compute_at_small_mb")
    m, s, t = _fixture()
    curve = {1: 0.05, 16: 0.4, 64: 0.8, 256: 1.0}
    t = replace(t, tensor_core_efficiency=curve, B_decode=1)
    p = PartitionSpec(PP=1, TP=1, EP=1, SP=1)  # mb = B/PP = 1

    r = InferenceCalculator(m, s, p, t).run().latency
    _check_near("eta_TC at mb=1", r.eta_TC, 0.05, rel_tol=1e-12)
    _check_near(
        "t_compute_eff = t_compute / 0.05 (= 20× at mb=1)",
        r.t_compute_eff, r.t_compute / 0.05, rel_tol=1e-12,
    )

    # Linear interpolation between the curve's keys.
    # mb = 4 between (1, 0.05) and (16, 0.4): expected η_TC = 0.05 + (4-1)/(16-1) · (0.4 - 0.05) = 0.12
    t8 = replace(t, B_decode=4)  # PP=1 → mb=4
    r8 = InferenceCalculator(m, s, p, t8).run().latency
    _check_near("eta_TC at mb=4 (interpolated)", r8.eta_TC, 0.05 + (3/15) * 0.35, rel_tol=1e-12)

    # Above the max key clamps to the max value.
    t1024 = replace(t, B_decode=1024)
    r1024 = InferenceCalculator(m, s, p, t1024).run().latency
    _check("eta_TC at mb=1024 clamps to 1.0", r1024.eta_TC, 1.0)


# ────────────────────────────────────────────────────────────
# (d) ρ_SW boundary cases: full overlap (max) vs no overlap (additive)
# ────────────────────────────────────────────────────────────

def test_sw_overlap_factor_boundaries() -> None:
    print("\ntest_sw_overlap_factor_boundaries")
    m, s, t_base = _fixture()
    # Crank SW high so it's clearly the bottleneck on a small-model fixture.
    t_base = replace(
        t_base,
        kernel_launch_us=20.0,            # exaggerated to make SW dominate
        kernels_per_layer_compute=20,
        kernels_per_collective_call=4,
        tensor_core_efficiency=None,
    )
    p = PartitionSpec(PP=1, TP=1, EP=1, SP=1)

    # ρ_SW = 1: full overlap; SW is a floor.
    t1 = replace(t_base, sw_overlap_factor=1.0)
    r1 = InferenceCalculator(m, s, p, t1).run().latency
    expected_ceiling = max(r1.t_stage, r1.t_SW) * r1.pp_bubble_factor
    _check_near(
        "ρ_SW=1 → t_step_user = max(t_stage, t_SW) · γ_pp",
        r1.t_step_user, expected_ceiling, rel_tol=1e-12,
    )

    # ρ_SW = 0: no overlap; SW adds to t_stage.
    t0 = replace(t_base, sw_overlap_factor=0.0)
    r0 = InferenceCalculator(m, s, p, t0).run().latency
    expected_sum = (r0.t_stage + r0.t_SW) * r0.pp_bubble_factor
    _check_near(
        "ρ_SW=0 → t_step_user = (t_stage + t_SW) · γ_pp",
        r0.t_step_user, expected_sum, rel_tol=1e-12,
    )

    _check_truthy(
        "ρ_SW=0 t_step_user > ρ_SW=1 t_step_user when t_SW > 0",
        r0.t_step_user > r1.t_step_user,
        f"ρ_SW=0 gave {r0.t_step_user}, ρ_SW=1 gave {r1.t_step_user}",
    )


# ────────────────────────────────────────────────────────────


def main() -> int:
    test_sw_disabled_matches_legacy_roofline()
    test_t_sw_scales_with_collective_count()
    test_eta_tc_derates_compute_at_small_mb()
    test_sw_overlap_factor_boundaries()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll SW-overhead unit tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
