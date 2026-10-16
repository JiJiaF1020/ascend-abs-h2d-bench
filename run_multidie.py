#!/usr/bin/env python3
"""Launch one H2D probe process per die concurrently and merge results.

Each die runs the selected worker in its own process with
ASCEND_RT_VISIBLE_DEVICES pinned to that die, so all dies measure at the
same time and share host DRAM / PCIe exactly like a real multi-die run.
Per-die JSON parts plus a merged JSON (with per-config bandwidth sums) are
written under the results directory.

Workers:
  abs     test_abs_h2d_bandwidth.py       (grouped by op/dtype/size_mb)
  scatter test_scatter_contig_vs_random.py (grouped by pattern/copy_count;
          requires bash build_c8.sh first and uses torch_extension via PYTHONPATH)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

WORKERS = {
    "abs": {
        "script": HERE / "test_abs_h2d_bandwidth.py",
        "device_flag": "--devices",
        "group_keys": ("op", "dtype", "size_mb"),
    },
    "scatter": {
        "script": HERE / "test_scatter_contig_vs_random.py",
        "device_flag": "--device",
        "group_keys": ("pattern", "copy_count"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=tuple(WORKERS), default="abs")
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)),
                        help="Physical die indexes; each gets ASCEND_RT_VISIBLE_DEVICES=<idx>.")
    parser.add_argument("--json-output", type=Path, default=Path("results/abs_h2d_multidie.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/multidie_parts"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Forwarded to the worker after '--', e.g. -- --dtypes fp16 --sizes-mb 1024 2048")
    args = parser.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if not args.devices or any(dev < 0 for dev in args.devices):
        parser.error("--devices must be non-negative die indexes")
    return args


def launch_workers(args: argparse.Namespace) -> list[tuple[int, Path, Path]]:
    worker = WORKERS[args.worker]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dev in args.devices:
        part_json = args.results_dir / f"die{dev}.json"
        part_log = args.results_dir / f"die{dev}.log"
        pythonpath = f"{HERE / 'torch_extension'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(dev), PYTHONPATH=pythonpath)
        cmd = [args.python, str(worker["script"]), worker["device_flag"], "npu:0",
               "--json-output", str(part_json), *args.extra]
        log_file = part_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        jobs.append((dev, proc, part_json, part_log, log_file))
    failures = []
    for dev, proc, part_json, part_log, log_file in jobs:
        proc.wait()
        log_file.close()
        if proc.returncode != 0:
            tail = part_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            print(f"A5_H2D_MULTIDIE_WORKER_FAILED die={dev} returncode={proc.returncode}", flush=True)
            print("\n".join(tail), flush=True)
            failures.append(dev)
    if failures:
        raise SystemExit(f"workers failed on dies {failures}; see {args.results_dir}")
    return [(dev, part_json, part_log) for dev, _, part_json, part_log, _ in jobs]


def merge(args: argparse.Namespace, parts: list[tuple[int, Path, Path]]) -> dict:
    group_keys = WORKERS[args.worker]["group_keys"]
    rows = []
    environment = None
    worker_config = None
    for dev, part_json, part_log in parts:
        part = json.loads(part_json.read_text(encoding="utf-8"))
        if part.get("status") != "passed":
            raise SystemExit(f"die {dev} did not pass; see {part_log}")
        environment = environment or part.get("environment")
        worker_config = worker_config or part.get("config")
        for row in part["results"]:
            row = dict(row)
            row["device"] = f"npu:{dev}"
            rows.append(row)
    aggregate: dict[tuple, float] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        aggregate[key] = aggregate.get(key, 0.0) + row["bandwidth_gbps"]
    agg_rows = [
        {**dict(zip(group_keys, key)), "sum_bandwidth_gbps": total}
        for key, total in sorted(aggregate.items(), key=lambda item: item[0])
    ]
    best = max(agg_rows, key=lambda row: row["sum_bandwidth_gbps"])
    header = "".join(f"{k:>12}" for k in group_keys) + f"{'sum_bw_gbps':>14}"
    print(f"\n{header}", flush=True)
    for row in agg_rows:
        line = "".join(f"{str(row[k]):>12}" for k in group_keys)
        print(f"{line}{row['sum_bandwidth_gbps']:>14.3f}", flush=True)
    best_desc = " ".join(f"{k}={best[k]}" for k in group_keys)
    print(
        f"A5_H2D_MULTIDIE_MAX_BANDWIDTH worker={args.worker} dies={len(parts)} "
        f"{best_desc} sum_bandwidth_gbps={best['sum_bandwidth_gbps']:.3f} ok=1",
        flush=True,
    )
    merged = {
        "schema_version": 2, "test": f"a5_h2d_multidie_{args.worker}",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_count": len(parts), "physical_devices": [dev for dev, _, _ in parts],
        "worker": args.worker, "group_keys": list(group_keys),
        "environment": environment, "worker_config": worker_config,
        "aggregate": agg_rows,
        "aggregate_max_bandwidth": best,
        "results": rows,
        "parts": [str(part_json) for _, part_json, _ in parts],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("A5_H2D_MULTIDIE_OK", flush=True)
    return merged


def main() -> None:
    args = parse_args()
    parts = launch_workers(args)
    merge(args, parts)


if __name__ == "__main__":
    main()
