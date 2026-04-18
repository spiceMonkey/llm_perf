"""Regression gate: replay the golden grid and diff against golden.json.

Run this after each refactor phase. It re-instantiates every (model,
system, partition, tuner) tuple recorded in `golden.json`, executes the
decode and prefill calculators, and compares every numeric field. Any
deviation outside a single-ULP tolerance is reported as a failure with
full context (case id, field path, golden vs current value).

Exit code 0 if all cases match, 1 otherwise.

Usage:  PYTHONPATH=. python tests/regression/verify.py
"""
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

# Silence expected dispatcher UserWarnings on torus misalignment and
# mixed-topology fallback — same rationale as capture_golden.py.
warnings.filterwarnings("ignore", category=UserWarning)

from llm_perf.calculators.inference_calculator import InferenceCalculator
from llm_perf.calculators.prefill_calculator import PrefillCalculator
from llm_perf.io import (
    load_model_from_db,
    load_system_from_db,
)
from llm_perf.specs.partition_spec import PartitionSpec
from llm_perf.specs.tuner_spec import TuningSpec


# Tolerance: default is bit-identical. If the refactor reorders FP
# operations inside primitives, single-ULP rel tolerance is acceptable
# (set via env var or CLI — left at 0 to keep the gate strict).
REL_TOL = 0.0
ABS_TOL = 0.0


def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return repr(obj)


def _diff(path: str, a, b, failures: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            failures.append((path, f"key mismatch: golden={sorted(a)} current={sorted(b)}"))
            return
        for k in a:
            _diff(f"{path}.{k}", a[k], b[k], failures)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            failures.append((path, f"length mismatch: golden={len(a)} current={len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(f"{path}[{i}]", x, y, failures)
        return
    if isinstance(a, float) and isinstance(b, float):
        if math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return
        failures.append((path, f"golden={a!r} current={b!r}"))
        return
    if a != b:
        failures.append((path, f"golden={a!r} current={b!r}"))


def _run_case(inputs):
    model = load_model_from_db(inputs["model"])
    system = load_system_from_db(inputs["system"])
    p = inputs["partition"]
    partition = PartitionSpec(PP=p["PP"], TP=p["TP"], EP=p["EP"], SP=p["SP"])
    t = inputs["tuner"]
    from llm_perf.io import load_tuner_from_db
    base_tuner = load_tuner_from_db("example.tuner")
    tuner = dataclasses.replace(
        base_tuner,
        S_decode=t["S_decode"],
        B_decode=t["B_decode"],
        S_input=t["S_input"],
        B_prefill=t["B_prefill"],
        chunk_size=t["chunk_size"],
    )
    ir = InferenceCalculator(model, system, partition, tuner).run()
    pr = PrefillCalculator(model, system, partition, tuner).run()
    return {
        "decode": {
            "memory": _to_jsonable(ir.memory),
            "flops": _to_jsonable(ir.flops),
            "traffic": _to_jsonable(ir.traffic),
            "comm": _to_jsonable(ir.comm),
            "latency": _to_jsonable(ir.latency),
        },
        "prefill": {
            "flops": _to_jsonable(pr.flops),
            "traffic": _to_jsonable(pr.traffic),
            "comm": _to_jsonable(pr.comm),
            "latency": _to_jsonable(pr.latency),
        },
    }


def main() -> int:
    golden_path = Path(__file__).parent / "golden.json"
    if not golden_path.exists():
        print(f"FAIL: golden fixture missing at {golden_path}", file=sys.stderr)
        print("  Run: PYTHONPATH=. python tests/regression/capture_golden.py", file=sys.stderr)
        return 2

    with golden_path.open() as f:
        fixture = json.load(f)

    n_pass = 0
    n_fail = 0
    first_failure_reported = False
    fail_details = []

    for case in fixture["cases"]:
        if case.get("status") != "ok":
            continue
        current = _run_case(case["inputs"])
        golden = case["results"]
        failures: list = []
        _diff("", golden, current, failures)
        if failures:
            n_fail += 1
            if not first_failure_reported:
                print(f"FAIL: first deviation in case:\n  {case['id']}")
                for path, msg in failures[:5]:
                    print(f"  {path.lstrip('.')}: {msg}")
                if len(failures) > 5:
                    print(f"  ... and {len(failures) - 5} more fields in this case")
                first_failure_reported = True
            fail_details.append((case["id"], len(failures)))
        else:
            n_pass += 1

    print(f"\nRegression summary: {n_pass} passed, {n_fail} failed, total {n_pass + n_fail}")
    if n_fail:
        print(f"\nCases with deviations ({n_fail}):")
        for cid, nfld in fail_details[:20]:
            print(f"  {cid}  ({nfld} fields)")
        if len(fail_details) > 20:
            print(f"  ... and {len(fail_details) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
