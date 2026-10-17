#!/usr/bin/env python3
"""One-command H2D comparison: random vs contiguous, 1:8 vs 1:4 topology.

Runs four 8-die concurrent passes via run_multidie.py and prints one matrix:

  worker=scatter (kvcache_scatter_copy_c8, 656-byte rows)
    - pattern=random:      discrete H2D, one row per token, scattered addresses
    - pattern=contiguous:  same kernel, sequential addresses (A/B control)
  worker=contig (VllmA5H2dContigCopy, DataCopyPad 16 KB chunks)
    - fully contiguous H2D block move, kernel-path ceiling

Topologies (physical die indexes, ASCEND_RT_VISIBLE_DEVICES per process):
  1to8  dies 0-7                 (8 dies behind one CPU socket)
  1to4  dies 0 2 4 6 8 10 12 14  (4 dies per socket, two sockets)

Sizes are nominal KiB per iteration (default 128 512 1024 2048 4096 8192).
For scatter they are converted to per-request copy counts via
round(kb*1024/(656*batch_size)); the table reports the nominal size.

Prerequisites: bash build_c8.sh once, and source set_env.sh before running.
Output: <results-dir>/compare.md, compare.json, plus per-pass merged JSONs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOPOLOGIES = {
    "1to8": [0, 1, 2, 3, 4, 5, 6, 7],
    "1to4": [0, 2, 4, 6, 8, 10, 12, 14],
}

# matrix columns: (worker, pattern or None, topology, header)
COLUMNS = [
    ("scatter", "random", "1to8", "rand_1:8"),
    ("scatter", "random", "1to4", "rand_1:4"),
    ("scatter", "contiguous", "1to8", "sc_contig_1:8"),
    ("scatter", "contiguous", "1to4", "sc_contig_1:4"),
    ("contig", None, "1to8", "contig_1:8"),
    ("contig", None, "1to4", "contig_1:4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-kb", type=int, nargs="+",
                        default=[128, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Scatter worker batch size (rows = tokens per request).")
    parser.add_argument("--hbm-slots", type=int, default=8192,
                        help="Scatter worker HBM slots per row; must cover the largest size.")
    parser.add_argument("--devices-1to8", type=int, nargs="+", default=TOPOLOGIES["1to8"])
    parser.add_argument("--devices-1to4", type=int, nargs="+", default=TOPOLOGIES["1to4"])
    parser.add_argument("--results-dir", type=Path, default=Path("results/compare"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if not args.sizes_kb or any(kb <= 0 for kb in args.sizes_kb):
        parser.error("--sizes-kb must be a non-empty list of positive integers")
    return args


def run_pass(args: argparse.Namespace, worker: str, topo: str, devices: list[int]) -> Path:
    json_output = args.results_dir / f"{worker}_{topo}.json"
    parts_dir = args.results_dir / f"parts_{worker}_{topo}"
    cmd = [
        args.python, str(HERE / "run_multidie.py"),
        "--worker", worker,
        "--devices", *[str(dev) for dev in devices],
        "--json-output", str(json_output),
        "--results-dir", str(parts_dir),
        "--",
        "--sizes-kb", *[str(kb) for kb in args.sizes_kb],
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]
    if worker == "scatter":
        cmd += ["--batch-size", str(args.batch_size), "--hbm-slots", str(args.hbm_slots)]
    print(f"A5_COMPARE_LAUNCH worker={worker} topo={topo} devices={devices}", flush=True)
    proc = subprocess.run(cmd, cwd=HERE)
    if proc.returncode != 0:
        raise SystemExit(f"pass failed: worker={worker} topo={topo}; see {parts_dir}")
    return json_output


def sum_by_size(rows: list[dict], worker: str, pattern: str | None) -> dict[int, float]:
    totals: dict[int, float] = {}
    for row in rows:
        if worker == "scatter" and row["pattern"] != pattern:
            continue
        nominal = row.get("nominal_size_kb")
        if nominal is None:
            raise SystemExit("worker rows lack nominal_size_kb; rerun with --sizes-kb")
        key = int(nominal)
        totals[key] = totals.get(key, 0.0) + row["bandwidth_gbps"]
    return totals


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    passes: dict[tuple[str, str], Path] = {}
    for worker in ("scatter", "contig"):
        for topo, devices in (("1to8", args.devices_1to8), ("1to4", args.devices_1to4)):
            passes[(worker, topo)] = run_pass(args, worker, topo, devices)

    table: dict[str, dict[int, float]] = {}
    for worker, pattern, topo, header in COLUMNS:
        merged = json.loads(passes[(worker, topo)].read_text(encoding="utf-8"))
        table[header] = sum_by_size(merged["results"], worker, pattern)

    sizes = sorted(args.sizes_kb)
    headers = [header for _, _, _, header in COLUMNS]
    lines = [f"{'size_kib':>8} " + " ".join(f"{h:>14}" for h in headers)]
    for kb in sizes:
        cells = []
        for h in headers:
            value = table[h].get(kb)
            cells.append(f"{value:>14.3f}" if value is not None else f"{'-':>14}")
        lines.append(f"{kb:>8} " + " ".join(cells))
    print("\n" + "\n".join(lines), flush=True)

    md = [
        "# H2D compare: random vs contiguous, 1:8 vs 1:4 (8-die sums, GB/s)",
        "",
        "| size (KiB) | " + " | ".join(headers) + " |",
        "|---|" + "---|" * len(headers),
    ]
    for kb in sizes:
        cells = [f"{table[h][kb]:.3f}" if kb in table[h] else "-" for h in headers]
        md.append(f"| {kb} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("rand_* = scatter kernel, random addresses; sc_contig_* = same scatter "
              "kernel, contiguous addresses (A/B control); contig_* = contiguous-copy "
              "kernel (DataCopyPad). 1:8 = dies 0-7; 1:4 = dies 0 2 4 6 8 10 12 14.")
    (args.results_dir / "compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    compare = {
        "schema_version": 1, "test": "a5_h2d_compare_random_vs_contig_topology",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "sizes_kb": sizes, "warmup": args.warmup, "iters": args.iters,
            "batch_size": args.batch_size, "hbm_slots": args.hbm_slots,
            "devices_1to8": args.devices_1to8, "devices_1to4": args.devices_1to4,
        },
        "columns": headers,
        "matrix_gbps": {str(kb): {h: table[h].get(kb) for h in headers} for kb in sizes},
        "passes": {f"{worker}_{topo}": str(path) for (worker, topo), path in passes.items()},
    }
    (args.results_dir / "compare.json").write_text(
        json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"A5_COMPARE_OK results_dir={args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
