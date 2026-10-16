#!/usr/bin/env python3
"""Ascend 950 C8 scatter-copy A/B: random vs contiguous source access.

Runs the SAME kvcache_scatter_copy_c8 kernel (depth-8 MTE2 queue, 656-byte
rows) twice per config: once with fully randomized metadata (block tables,
source ids, destination slots), once with identity/sequential metadata so the
source read is fully contiguous. Identical kernel, row granularity, queue
depth and payload — only the address pattern differs, isolating the
random-vs-contiguous effect on effective H2D bandwidth.

Bandwidth convention (same as the scatter and abs probes):
copied_tokens * 656 / avg_us / 1000 (GB/s), payload counted once.

Requires the vendored operator build: bash build_c8.sh, and
PYTHONPATH=$PWD/torch_extension when running this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BLOCK_SIZE = 128
ROW_BYTES = 656
COPY_CAP = 16384
MAX_SOURCE_LEN = 1 << 18
POISON = 65
DEFAULT_COPY_COUNTS = [300, 2048, 8192, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--copy-counts", type=int, nargs="+", default=DEFAULT_COPY_COUNTS,
                        help="Fixed tokens copied per request for each sweep point.")
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--hbm-slots", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.source_len <= MAX_SOURCE_LEN:
        raise ValueError("--source-len must be in [1,262144].")
    if not args.copy_counts:
        raise ValueError("--copy-counts must be non-empty.")
    cap = min(COPY_CAP, args.source_len, args.hbm_slots)
    if any(count <= 0 or count > cap for count in args.copy_counts):
        raise ValueError(f"each copy count must be in [1,{cap}] "
                         "(min of copy-cap 16384, source-len, hbm-slots).")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def require_a5(device: "torch.device", allow_non_a5: bool) -> str:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(f"expected Ascend 950, got {name!r}; use --allow-non-a5 only for debugging")
    return name


def swapped_from_cpu(cpu: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if allocator is None:
        raise RuntimeError("torch_npu.empty_with_swapped_memory is unavailable; "
                           "the test refuses to replace DRAM with an HBM tensor")
    tensor = allocator(cpu.shape, dtype=cpu.dtype, device=device)
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def npu_smi_info() -> str | None:
    try:
        proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()[:8000] or None


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": min(values), "max": max(values), "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p50": statistics.median(values), "p95": ordered[p95_index],
    }


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class Case:
    pattern: str
    copy_count: int
    dram_cpu: "torch.Tensor"
    dram: "torch.Tensor"
    hbm: "torch.Tensor"
    dram_table_cpu: "torch.Tensor"
    hbm_table_cpu: "torch.Tensor"
    src_ids_cpu: "torch.Tensor"
    dst_slots_cpu: "torch.Tensor"
    copy_counts_cpu: "torch.Tensor"
    dram_table: "torch.Tensor"
    hbm_table: "torch.Tensor"
    src_ids: "torch.Tensor"
    dst_slots: "torch.Tensor"
    copy_counts: "torch.Tensor"


def make_case(args: argparse.Namespace, device: "torch.device", pattern: str, copy_count: int) -> Case:
    generator = torch.Generator().manual_seed(args.seed)
    source_blocks = (args.source_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    hbm_blocks_per_row = (args.hbm_slots + BLOCK_SIZE - 1) // BLOCK_SIZE
    mapped_blocks = args.batch_size * hbm_blocks_per_row
    dram_cpu = torch.randint(-128, 128, (source_blocks, BLOCK_SIZE, 1, ROW_BYTES),
                             dtype=torch.int8, generator=generator)
    if pattern == "random":
        dram_table_cpu = torch.stack([
            torch.randperm(source_blocks, generator=generator).to(torch.int32)
            for _ in range(args.batch_size)
        ])
        hbm_table_cpu = torch.randperm(mapped_blocks + 1, generator=generator)[:mapped_blocks] \
            .to(torch.int32).reshape(args.batch_size, hbm_blocks_per_row)
        src = torch.randperm(args.source_len, generator=generator)[:copy_count].to(torch.int32)
        dst = torch.randperm(args.hbm_slots, generator=generator)[:copy_count].to(torch.int32)
    else:
        dram_table_cpu = torch.arange(source_blocks, dtype=torch.int32) \
            .expand(args.batch_size, source_blocks).contiguous()
        hbm_table_cpu = torch.arange(mapped_blocks, dtype=torch.int32) \
            .reshape(args.batch_size, hbm_blocks_per_row).contiguous()
        src = torch.arange(copy_count, dtype=torch.int32)
        dst = torch.arange(copy_count, dtype=torch.int32)
    src_ids_cpu = torch.full((args.batch_size, 1, COPY_CAP), -1, dtype=torch.int32)
    dst_slots_cpu = torch.full_like(src_ids_cpu, -1)
    src_ids_cpu[:, 0, :copy_count] = src
    dst_slots_cpu[:, 0, :copy_count] = dst
    copy_counts_cpu = torch.full((args.batch_size,), copy_count, dtype=torch.int32)
    return Case(
        pattern=pattern, copy_count=copy_count,
        dram_cpu=dram_cpu, dram=swapped_from_cpu(dram_cpu, device),
        hbm=torch.empty((mapped_blocks + 1, BLOCK_SIZE, 1, ROW_BYTES), dtype=torch.int8, device=device),
        dram_table_cpu=dram_table_cpu, hbm_table_cpu=hbm_table_cpu,
        src_ids_cpu=src_ids_cpu, dst_slots_cpu=dst_slots_cpu, copy_counts_cpu=copy_counts_cpu,
        dram_table=dram_table_cpu.to(device), hbm_table=hbm_table_cpu.to(device),
        src_ids=src_ids_cpu.to(device), dst_slots=dst_slots_cpu.to(device),
        copy_counts=copy_counts_cpu.to(device),
    )


def call_scatter(case: Case, copy_counts: "torch.Tensor | None" = None) -> None:
    return vllm_dsa_a5.kvcache_scatter_copy_c8(
        case.hbm, case.dram, case.hbm_table, case.dram_table,
        case.src_ids, case.dst_slots,
        case.copy_counts if copy_counts is None else copy_counts,
    )


def assert_copied(case: Case) -> None:
    sources, destinations = [], []
    for row in range(case.copy_counts_cpu.numel()):
        count = int(case.copy_counts_cpu[row])
        source = case.src_ids_cpu[row, 0, :count].long()
        destination = case.dst_slots_cpu[row, 0, :count].long()
        sources.append(case.dram_table_cpu[row].long()[source // BLOCK_SIZE] * BLOCK_SIZE + source % BLOCK_SIZE)
        destinations.append(case.hbm_table_cpu[row].long()[destination // BLOCK_SIZE] * BLOCK_SIZE + destination % BLOCK_SIZE)
    source = torch.cat(sources)
    destination = torch.cat(destinations)
    expected = torch.full((case.hbm.shape[0] * BLOCK_SIZE, ROW_BYTES), POISON, dtype=torch.int8)
    expected[destination] = case.dram_cpu.reshape(-1, ROW_BYTES)[source]
    if not torch.equal(case.hbm.cpu().reshape(-1, ROW_BYTES), expected):
        raise AssertionError(f"{case.pattern}: DRAM->HBM byte mismatch or inactive bytes modified.")


def check_copy(case: Case) -> None:
    case.hbm.fill_(POISON)
    torch.npu.synchronize()
    if call_scatter(case) is not None:
        raise AssertionError("C8 scatter must return None (caller-owned HBM).")
    torch.npu.synchronize()
    assert_copied(case)
    if call_scatter(case, torch.zeros_like(case.copy_counts)) is not None:
        raise AssertionError("Zero-copy scatter must return None.")
    torch.npu.synchronize()
    if not bool((case.hbm == POISON).all().item()):
        raise AssertionError("copy_count=0 modified the HBM cache.")
    print(f"A5_SCATTER_AB_CHECK pattern={case.pattern} copy_count={case.copy_count} "
          "byte_exact=1 zero_count_unchanged=1 ok=1", flush=True)


def measure_config(args: argparse.Namespace, device: "torch.device", pattern: str, copy_count: int) -> dict:
    case = make_case(args, device, pattern, copy_count)
    check_copy(case)
    for _ in range(args.warmup):
        call_scatter(case)
    torch.npu.synchronize()
    events = [torch.npu.Event(enable_timing=True) for _ in range(args.iters + 1)]
    events[0].record()
    wall_start = time.perf_counter()
    for index in range(args.iters):
        call_scatter(case)
        events[index + 1].record()
    events[-1].synchronize()
    wall_s = time.perf_counter() - wall_start
    iter_us = [events[i].elapsed_time(events[i + 1]) * 1000 for i in range(args.iters)]
    assert_copied(case)

    copied_tokens = args.batch_size * copy_count
    payload_bytes = copied_tokens * ROW_BYTES
    iter_bw = [payload_bytes / (t * 1000) for t in iter_us]
    avg_us = statistics.fmean(iter_us)
    del case
    torch.npu.empty_cache()
    return {
        "pattern": pattern, "batch_size": args.batch_size, "copy_count": copy_count,
        "copied_tokens": copied_tokens, "payload_bytes": payload_bytes,
        "warmup": args.warmup, "iters": args.iters,
        "avg_us": avg_us, "bandwidth_gbps": payload_bytes / (avg_us * 1000),
        "wall_us": wall_s * 1e6 / args.iters,
        "wall_bandwidth_gbps": payload_bytes * args.iters / wall_s / 1e9,
        "per_iter_us": summarize(iter_us),
        "bandwidth_per_iter_gbps": summarize(iter_bw),
        "raw_iter_us": [round(t, 3) for t in iter_us],
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    print(
        f"A5_SCATTER_AB_CONFIG device={device} device_name={device_name!r} "
        f"batch={args.batch_size} source_len={args.source_len} hbm_slots={args.hbm_slots} "
        f"copy_counts={args.copy_counts} warmup={args.warmup} iters={args.iters} "
        f"row_bytes={ROW_BYTES} queue_depth=8 opapi={vllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    rows = []
    for copy_count in args.copy_counts:
        for pattern in ("random", "contiguous"):
            row = measure_config(args, device, pattern, copy_count)
            rows.append(row)
            bw = row["bandwidth_per_iter_gbps"]
            print(
                f"A5_SCATTER_AB_RESULT pattern={pattern} copy_count={copy_count} "
                f"payload_bytes={row['payload_bytes']} avg_us={row['avg_us']:.3f} "
                f"wall_us={row['wall_us']:.3f} bw_mean={bw['mean']:.3f} "
                f"bw_min={bw['min']:.3f} bw_max={bw['max']:.3f} ok=1",
                flush=True,
            )
    print(f"\n{'copy_count':>10} {'random_gbps':>12} {'contig_gbps':>12} {'contig/random':>14}", flush=True)
    ratios = []
    for copy_count in args.copy_counts:
        random_row = next(row for row in rows if row["pattern"] == "random" and row["copy_count"] == copy_count)
        contig_row = next(row for row in rows if row["pattern"] == "contiguous" and row["copy_count"] == copy_count)
        ratio = contig_row["bandwidth_gbps"] / random_row["bandwidth_gbps"]
        ratios.append(ratio)
        print(f"{copy_count:>10} {random_row['bandwidth_gbps']:>12.3f} "
              f"{contig_row['bandwidth_gbps']:>12.3f} {ratio:>14.3f}", flush=True)
    result = {
        "schema_version": 1, "test": "a5_scatter_copy_random_vs_contiguous",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
            "device_name": device_name, "npu_smi_info": npu_smi_info(),
        },
        "config": {
            "device": str(device), "batch_size": args.batch_size,
            "source_len": args.source_len, "hbm_slots": args.hbm_slots,
            "copy_counts": args.copy_counts, "warmup": args.warmup, "iters": args.iters,
            "seed": args.seed, "row_bytes": ROW_BYTES, "queue_depth": 8,
            "bandwidth_convention": "copied payload counted once",
        },
        "results": rows,
        "contiguous_over_random_ratio": ratios,
        "performance": {"timer": "npu_event_per_iteration", "cross_check": "wall_clock"},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
    print("A5_SCATTER_AB_UT_OK", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    global torch, torch_npu, vllm_dsa_a5
    import vllm_dsa_a5
    import torch
    import torch_npu
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
