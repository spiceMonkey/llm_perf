"""Sweep PP/TP/EP/SP configurations for a fixed cluster budget and report TPS."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_perf.calculators.inference_calculator import InferenceCalculator
from llm_perf.io import load_model_spec, load_system_spec, load_tuning_spec
from llm_perf.specs.partition_spec import PartitionSpec
from llm_perf.utils import GB_TO_BYTES, MB_TO_BYTES, US_TO_SECONDS, save_config_tps_scatter


@dataclass
class SweepRecord:
    partition: PartitionSpec
    total_devices: int
    fits_memory: bool
    tps_single: float
    tps_total: float
    t_pp: float
    t_tp: float
    t_ep: float
    t_sp: float
    t_comm: float
    t_compute: float
    t_mem: float
    t_local: float
    t_token: float
    memory_param_gb: float
    memory_act_gb: float
    memory_kv_gb: float
    memory_total_gb: float
    msg_pp_mb: float
    msg_tp_mb: float
    msg_ep_mb: float
    msg_sp_mb: float
    traffic_theta_gb: float
    traffic_act_gb: float
    traffic_kv_gb: float
    traffic_total_gb: float


def divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def enumerate_partitions(cluster_size: int) -> Iterable[Tuple[int, int, int, int]]:
    for PP in divisors(cluster_size):
        rem_after_pp = cluster_size // PP
        for TP in divisors(rem_after_pp):
            rem_after_tp = rem_after_pp // TP
            for EP in divisors(rem_after_tp):
                SP = rem_after_tp // EP
                yield PP, TP, EP, SP


def sweep_partitions(
    model_path: Path,
    system_path: Path,
    tuner_path: Path,
    cluster_size: int,
    dp: int = 1,
) -> Tuple[List[SweepRecord], str]:
    model = load_model_spec(model_path)
    system = load_system_spec(system_path)
    tuner = load_tuning_spec(tuner_path)

    if cluster_size > system.num_devices:
        raise ValueError(
            f"Cluster size {cluster_size} exceeds system capacity {system.num_devices}"
        )

    max_ep = model.moe.n_experts if model.moe is not None else 1

    records: List[SweepRecord] = []
    for PP, TP, EP, SP in enumerate_partitions(cluster_size):
        if EP > max_ep:
            continue
        if model.moe is None and EP != 1:
            continue

        partition = PartitionSpec(DP=dp, PP=PP, TP=TP, EP=EP, SP=SP)
        calc = InferenceCalculator(model, system, partition, tuner)
        result = calc.run()
        total_devices = dp * PP * TP * EP * SP
        memory_param_gb = result.memory.M_theta_device / GB_TO_BYTES
        memory_act_gb = result.memory.M_act_device / GB_TO_BYTES
        memory_kv_gb = result.memory.M_kv_device / GB_TO_BYTES
        memory_total_gb = result.memory.M_total_device / GB_TO_BYTES
        msg_pp_mb = result.comm.msg_PP_bytes / MB_TO_BYTES
        msg_tp_mb = result.comm.msg_TP_bytes / MB_TO_BYTES
        msg_ep_mb = result.comm.msg_EP_bytes / MB_TO_BYTES
        msg_sp_mb = result.comm.msg_SP_bytes / MB_TO_BYTES
        traffic_theta_gb = result.traffic.T_theta / GB_TO_BYTES
        traffic_act_gb = result.traffic.T_act / GB_TO_BYTES
        traffic_kv_gb = result.traffic.T_kv / GB_TO_BYTES
        traffic_total_gb = result.traffic.T_token_eff / GB_TO_BYTES
        records.append(
            SweepRecord(
                partition=partition,
                total_devices=total_devices,
                fits_memory=result.memory.fits_in_HBM,
                tps_single=result.latency.TPS_single,
                tps_total=result.latency.TTPS,
                t_pp=result.comm.t_PP,
                t_tp=result.comm.t_TP,
                t_ep=result.comm.t_EP,
                t_sp=result.comm.t_SP,
                t_comm=result.comm.t_comm_stage,
                t_compute=result.latency.t_compute,
                t_mem=result.latency.t_mem,
                t_local=result.latency.t_local,
                t_token=result.latency.t_token,
                memory_param_gb=memory_param_gb,
                memory_act_gb=memory_act_gb,
                memory_kv_gb=memory_kv_gb,
                memory_total_gb=memory_total_gb,
                msg_pp_mb=msg_pp_mb,
                msg_tp_mb=msg_tp_mb,
                msg_ep_mb=msg_ep_mb,
                msg_sp_mb=msg_sp_mb,
                traffic_theta_gb=traffic_theta_gb,
                traffic_act_gb=traffic_act_gb,
                traffic_kv_gb=traffic_kv_gb,
                traffic_total_gb=traffic_total_gb,
            )
        )

    records.sort(key=lambda r: r.tps_total, reverse=True)
    return records, system.name


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    col_widths = [max(len(row[i]) for row in [headers] + list(rows)) for i in range(len(headers))]

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(entry.ljust(col_widths[i]) for i, entry in enumerate(row))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in col_widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def format_latency_memory_table(records: Sequence[SweepRecord]) -> str:
    headers = [
        "Config",
        "Devices",
        "HBM ok",
        "TPS",
        "TTPS",
        "t_PP",
        "t_TP",
        "t_EP",
        "t_SP",
        "t_comm",
        "t_compute",
        "t_mem",
        "t_local",
        "t_token",
        "Param_GB",
        "Act_GB",
        "KV_GB",
        "Total_GB",
    ]
    rows = []
    for rec in records:
        cfg = rec.partition
        label = f"PP{cfg.PP}-TP{cfg.TP}-EP{cfg.EP}-SP{cfg.SP}"
        rows.append(
            [
                label,
                str(rec.total_devices),
                "yes" if rec.fits_memory else "no",
                f"{rec.tps_single:,.2f}",
                f"{rec.tps_total:,.2f}",
                f"{rec.t_pp / US_TO_SECONDS:.2f}",
                f"{rec.t_tp / US_TO_SECONDS:.2f}",
                f"{rec.t_ep / US_TO_SECONDS:.2f}",
                f"{rec.t_sp / US_TO_SECONDS:.2f}",
                f"{rec.t_comm / US_TO_SECONDS:.2f}",
                f"{rec.t_compute / US_TO_SECONDS:.2f}",
                f"{rec.t_mem / US_TO_SECONDS:.2f}",
                f"{rec.t_local / US_TO_SECONDS:.2f}",
                f"{rec.t_token / US_TO_SECONDS:.2f}",
                f"{rec.memory_param_gb:.2f}",
                f"{rec.memory_act_gb:.2f}",
                f"{rec.memory_kv_gb:.2f}",
                f"{rec.memory_total_gb:.2f}",
            ]
        )
    return _render_table(headers, rows)


def format_comm_traffic_table(records: Sequence[SweepRecord]) -> str:
    headers = [
        "Config",
        "Devices",
        "TPS",
        "TTPS",
        "msgPP_MB",
        "msgTP_MB",
        "msgEP_MB",
        "msgSP_MB",
        "Tparam_GB",
        "Tact_GB",
        "Tkv_GB",
        "Ttotal_GB",
    ]
    rows = []
    for rec in records:
        cfg = rec.partition
        label = f"PP{cfg.PP}-TP{cfg.TP}-EP{cfg.EP}-SP{cfg.SP}"
        rows.append(
            [
                label,
                str(rec.total_devices),
                f"{rec.tps_single:,.2f}",
                f"{rec.tps_total:,.2f}",
                f"{rec.msg_pp_mb:.2f}",
                f"{rec.msg_tp_mb:.2f}",
                f"{rec.msg_ep_mb:.2f}",
                f"{rec.msg_sp_mb:.2f}",
                f"{rec.traffic_theta_gb:.2f}",
                f"{rec.traffic_act_gb:.2f}",
                f"{rec.traffic_kv_gb:.2f}",
                f"{rec.traffic_total_gb:.2f}",
            ]
        )
    return _render_table(headers, rows)


def build_plot(records: Sequence[SweepRecord], plot_path: Path) -> Path:
    labels = [f"PP{r.partition.PP}-TP{r.partition.TP}-EP{r.partition.EP}-SP{r.partition.SP}" for r in records]
    tps = [r.tps_total for r in records]
    title = f"Partition sweep ({len(records)} configs)"
    return save_config_tps_scatter(labels, tps, plot_path, title=title, ylabel="TTPS (tokens/s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="llm_perf/database/model/qwen3_vl_235b_fp8.json",
        type=Path,
        help="Path to model JSON",
    )
    parser.add_argument(
        "--system",
        default="llm_perf/database/system/h100.32gpu.json",
        type=Path,
        help="Path to system JSON",
    )
    parser.add_argument(
        "--tuner",
        default="llm_perf/database/tuner/qwen3.tuner.json",
        type=Path,
        help="Path to tuner JSON",
    )
    parser.add_argument(
        "--cluster-size",
        default=32,
        type=int,
        help="Total number of devices available (DP*PP*TP*EP*SP)",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        type=Path,
        help="Optional custom path for the scatter plot (default auto-includes system name)",
    )
    parser.add_argument(
        "--dp",
        default=1,
        type=int,
        help="Data parallel replicas (fixed for this sweep)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, system_name = sweep_partitions(
        args.model,
        args.system,
        args.tuner,
        args.cluster_size,
        dp=args.dp,
    )
    if not records:
        raise SystemExit("No valid configurations satisfied the constraints.")

    print(format_latency_memory_table(records))
    print()
    print(format_comm_traffic_table(records))
    if args.plot_path is not None:
        plot_path = args.plot_path
    else:
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in system_name)
        plot_path = Path("artifacts") / f"partition_sweep_{safe_name}.png"
    plot_path = build_plot(records, plot_path)
    print(f"\nSaved scatter plot to {plot_path}")


if __name__ == "__main__":
    main()
