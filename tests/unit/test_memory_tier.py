"""Unit tests for the multi-tier memory spec layer (PR1, sram.md §1.1).

PR1 adds `MemoryTierSpec` and `DeviceSpec.tiers` plus a `get_tiers()` shim.
PR1 is purely additive — these tests pin down the spec/loader behavior so
later PRs (placement, multi-tier roofline) cannot regress the back-compat
shim that keeps existing systems numerically identical to legacy.

Usage:  PYTHONPATH=. python tests/unit/test_memory_tier.py
"""
import json
import sys
import tempfile
from pathlib import Path

from llm_perf import DeviceSpec, MemoryTierSpec
from llm_perf.io.system_loaders import load_system_spec


_failures = []


def _check(label, got, expected):
    ok = got == expected
    print(f"{'OK' if ok else 'FAIL'}: {label}: got={got!r} expected={expected!r}")
    if not ok:
        _failures.append(label)


def _check_close(label, got, expected, rel=1e-9):
    ok = abs(got - expected) <= rel * max(abs(expected), 1.0)
    print(f"{'OK' if ok else 'FAIL'}: {label}: got={got!r} expected≈{expected!r}")
    if not ok:
        _failures.append(label)


# ────────────────────────────────────────────────────────────
# DeviceSpec shim — legacy fields → single-tier list (sram.md §1.1)
# ────────────────────────────────────────────────────────────

def test_legacy_shim_materializes_single_tier():
    d = DeviceSpec(
        name="legacy", hbm_capacity_GB=80.0, hbm_bandwidth_GBps=3350.0,
        peak_flops_TF=1000.0,
    )
    tiers = d.get_tiers()
    _check("legacy shim produces 1 tier", len(tiers), 1)
    t0 = tiers[0]
    _check("legacy shim tier name", t0.name, "hbm")
    _check_close("legacy shim capacity", t0.capacity_GB, 80.0)
    _check_close("legacy shim bandwidth", t0.bandwidth_GBps, 3350.0)
    _check("legacy shim alpha_us", t0.alpha_us, 0.0)
    # Critical: shim eta_beta = 1.0 to preserve regression numbers exactly
    # (sram.md §1.1 second-paragraph footnote).
    _check("legacy shim eta_beta = 1.0 (preserves regression)", t0.eta_beta, 1.0)


def test_explicit_tiers_pass_through():
    sram = MemoryTierSpec(
        name="sram", capacity_GB=2.0, bandwidth_GBps=150_000.0, eta_beta=1.0,
    )
    lpddr = MemoryTierSpec(
        name="lpddr5", capacity_GB=256.0, bandwidth_GBps=400.0, eta_beta=0.85,
    )
    d = DeviceSpec(
        name="d-matrix", hbm_capacity_GB=2.0, hbm_bandwidth_GBps=150_000.0,
        peak_flops_TF=2400.0, tiers=[sram, lpddr],
    )
    tiers = d.get_tiers()
    _check("explicit tiers count", len(tiers), 2)
    _check("explicit tier 0 is SRAM", tiers[0].name, "sram")
    _check("explicit tier 1 is LPDDR5", tiers[1].name, "lpddr5")
    _check_close("explicit tier 1 eta_beta passes through", tiers[1].eta_beta, 0.85)


# ────────────────────────────────────────────────────────────
# JSON loader — optional tiers[] block (sram.md §1.1)
# ────────────────────────────────────────────────────────────

# A minimal valid system JSON, parameterized by an optional `tiers` block on
# `device`. Mirrors the shape of llm_perf/database/system/h100.8gpu.json so
# the test exercises the real loader path end-to-end.
def _minimal_system_json(device_extra: dict | None = None) -> dict:
    device = {
        "name": "Test-Device",
        "hbm_capacity_GB": 80.0,
        "hbm_bandwidth_GBps": 3350.0,
        "peak_flops_TF": 1000.0,
    }
    if device_extra:
        device.update(device_extra)
    return {
        "schema": "llm_perf.system",
        "name": "test_system",
        "num_devices": 8,
        "device": device,
        "fabrics": {
            "single": {
                "tiers": [
                    {"name": "tier0", "ports": 8, "bw_per_port_GBps": 900.0,
                     "alpha_us": 0.5}
                ]
            }
        },
        "collective_fabrics": {
            "TP": "single", "EP": "single", "SP": "single", "PP": "single",
        },
    }


def _load_temp_system(cfg: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    try:
        return load_system_spec(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_loader_no_tiers_block_keeps_legacy_path():
    """When device JSON has no `tiers` field, DeviceSpec.tiers is empty and
    get_tiers() materializes the legacy shim. This is the path every
    existing system JSON in the repo takes."""
    sys_obj = _load_temp_system(_minimal_system_json())
    _check("loader: no tiers → empty list", len(sys_obj.device.tiers), 0)
    tiers = sys_obj.device.get_tiers()
    _check("loader: shim still works", len(tiers), 1)
    _check("loader: shim tier name", tiers[0].name, "hbm")
    _check("loader: shim eta_beta = 1.0", tiers[0].eta_beta, 1.0)


def test_loader_tiers_block_parses_two_tiers():
    """Explicit `tiers` block parses with type-name eta_beta defaults
    (sram.md §1.2: SRAM=1.0, HBM=0.92, LPDDR5=0.85)."""
    cfg = _minimal_system_json({
        "tiers": [
            {"name": "sram", "capacity_GB": 2.0, "bandwidth_GBps": 150000.0},
            {"name": "lpddr5", "capacity_GB": 256.0, "bandwidth_GBps": 400.0},
        ]
    })
    sys_obj = _load_temp_system(cfg)
    tiers = sys_obj.device.get_tiers()
    _check("loader: 2-tier count", len(tiers), 2)
    _check("loader: tier 0 SRAM eta default", tiers[0].eta_beta, 1.0)
    _check("loader: tier 1 LPDDR5 eta default", tiers[1].eta_beta, 0.85)


def test_loader_explicit_eta_beta_overrides_default():
    cfg = _minimal_system_json({
        "tiers": [
            {"name": "hbm", "capacity_GB": 192.0, "bandwidth_GBps": 8000.0,
             "eta_beta": 0.95},
        ]
    })
    sys_obj = _load_temp_system(cfg)
    _check("loader: explicit eta_beta wins over name default",
           sys_obj.device.get_tiers()[0].eta_beta, 0.95)


def test_loader_unknown_tier_name_defaults_to_one():
    cfg = _minimal_system_json({
        "tiers": [
            {"name": "exotic_mram", "capacity_GB": 4.0, "bandwidth_GBps": 50000.0},
        ]
    })
    sys_obj = _load_temp_system(cfg)
    _check("loader: unknown tier name → eta_beta default 1.0",
           sys_obj.device.get_tiers()[0].eta_beta, 1.0)


def test_loader_rejects_empty_tiers_list():
    cfg = _minimal_system_json({"tiers": []})
    try:
        _load_temp_system(cfg)
        _failures.append("empty tiers list should raise")
        print("FAIL: empty tiers list should raise ValueError")
    except ValueError as e:
        print(f"OK: empty tiers list raises: {e!s}")


def test_loader_rejects_bad_eta_beta():
    cfg = _minimal_system_json({
        "tiers": [
            {"name": "hbm", "capacity_GB": 80.0, "bandwidth_GBps": 3350.0,
             "eta_beta": 1.5},
        ]
    })
    try:
        _load_temp_system(cfg)
        _failures.append("eta_beta > 1 should raise")
        print("FAIL: eta_beta > 1 should raise ValueError")
    except ValueError as e:
        print(f"OK: eta_beta > 1 raises: {e!s}")


def test_loader_rejects_negative_capacity():
    cfg = _minimal_system_json({
        "tiers": [
            {"name": "hbm", "capacity_GB": -1.0, "bandwidth_GBps": 3350.0},
        ]
    })
    try:
        _load_temp_system(cfg)
        _failures.append("negative capacity should raise")
        print("FAIL: negative capacity should raise ValueError")
    except ValueError as e:
        print(f"OK: negative capacity raises: {e!s}")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main() -> int:
    test_legacy_shim_materializes_single_tier()
    test_explicit_tiers_pass_through()
    test_loader_no_tiers_block_keeps_legacy_path()
    test_loader_tiers_block_parses_two_tiers()
    test_loader_explicit_eta_beta_overrides_default()
    test_loader_unknown_tier_name_defaults_to_one()
    test_loader_rejects_empty_tiers_list()
    test_loader_rejects_bad_eta_beta()
    test_loader_rejects_negative_capacity()
    if _failures:
        print(f"\n{len(_failures)} FAILURES:")
        for f in _failures:
            print(f"  {f}")
        return 1
    print("\nAll memory-tier unit tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
