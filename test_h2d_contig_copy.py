#!/usr/bin/env python3
"""Ascend 950 contiguous-copy kernel probe: host DRAM -> HBM via DataCopyPad.

Runs the custom VllmA5H2dContigCopy operator: one flat contiguous range read
from a swapped-memory tensor (real host DRAM) into caller-owned HBM, in 16 KB
DataCopyPad chunks with a depth-4 MTE queue per AIV. This is the kernel-path
ceiling for fully contiguous H2D movement (no DMA copy engine involved).

Bandwidth convention: payload_bytes / avg_us / 1000 (GB/s), counted once.

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
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SIZES_MB = [1, 1024, 2048]
SAMPLE_SLICES = 8
SAMPLE_LEN = 1 << 20
FULL_COMPARE_MAX = 64 * (1 << 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=DEFAULT_SIZES_MB,
                        help="Payload sizes to sweep, in MiB per iteration.")
    parser.add_argument("--sizes-kb", type=int, nargs="+", default=None,
                        help="Payload sizes in KiB per iteration; overrides --sizes-mb when set.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--timing-ready-prefix", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timing-start-prefix", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    sizes_kb = args.sizes_kb if args.sizes_kb else [mb * 1024 for mb in args.sizes_mb]
    if not sizes_kb or any(size <= 0 for size in sizes_kb):
        raise ValueError("sizes must be a non-empty list of positive integers.")
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


def timing_barrier(args: argparse.Namespace, index: int) -> None:
    """Rendezvous so all multidie workers time config `index` concurrently."""
    if args.timing_ready_prefix is None and args.timing_start_prefix is None:
        return
    if args.timing_ready_prefix is None or args.timing_start_prefix is None:
        raise ValueError("timing ready/start prefixes must be specified together")
    Path(f"{args.timing_ready_prefix}.{index}").write_text("ready\n", encoding="utf-8")
    start = Path(f"{args.timing_start_prefix}.{index}")
    deadline = time.monotonic() + 1800
    print(f"A5_H2D_BARRIER_WAIT index={index}", flush=True)
    while not start.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for multidie timing barrier {index}")
        time.sleep(0.001)


def check_data(cpu: "torch.Tensor", dst: "torch.Tensor", seed: int) -> None:
    numel = cpu.numel()
    if numel <= FULL_COMPARE_MAX:
        if not torch.equal(dst.cpu(), cpu):
            raise AssertionError("contiguous copy result mismatch (full compare).")
        return
    generator = torch.Generator().manual_seed(seed + 1)
    offsets = {0, numel - SAMPLE_LEN}
    offsets.update(torch.randint(0, numel - SAMPLE_LEN + 1, (SAMPLE_SLICES - 2,), generator=generator).tolist())
    for offset in sorted(offsets):
        if not torch.equal(dst[offset:offset + SAMPLE_LEN].cpu(), cpu[offset:offset + SAMPLE_LEN]):
            raise AssertionError(f"contiguous copy result mismatch at byte offset {offset}.")


def measure_config(args: argparse.Namespace, device: "torch.device", device_name: str,
                   size_bytes: int, barrier_index: int) -> dict:
    numel = size_bytes
    size_kb = size_bytes / 1024
    generator = torch.Generator().manual_seed(args.seed)
    cpu = torch.randint(-128, 128, (numel,), dtype=torch.int8, generator=generator)
    src = swapped_from_cpu(cpu, device)
    dst = torch.empty(numel, dtype=torch.int8, device=device)
    torch.npu.synchronize()

    def call_op() -> None:
        vllm_dsa_a5.h2d_contig_copy(dst, src)

    call_op()
    torch.npu.synchronize()
    check_data(cpu, dst, args.seed)
    print(f"A5_CONTIG_COPY_CHECK size_kb={size_kb:g} byte_exact=1 ok=1", flush=True)

    for _ in range(args.warmup):
        call_op()
    torch.npu.synchronize()
    timing_barrier(args, barrier_index)
    events = [torch.npu.Event(enable_timing=True) for _ in range(args.iters + 1)]
    events[0].record()
    wall_start = time.perf_counter()
    for index in range(args.iters):
        call_op()
        events[index + 1].record()
    events[-1].synchronize()
    wall_s = time.perf_counter() - wall_start
    iter_us = [events[i].elapsed_time(events[i + 1]) * 1000 for i in range(args.iters)]
    check_data(cpu, dst, args.seed + 1)

    payload_bytes = numel
    iter_bw = [payload_bytes / (t * 1000) for t in iter_us]
    avg_us = statistics.fmean(iter_us)
    del cpu, src, dst, events
    torch.npu.empty_cache()
    return {
        "device": str(device), "device_name": device_name,
        "size_kb": size_kb, "nominal_size_kb": size_kb, "payload_bytes": payload_bytes,
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
    sizes_bytes = ([kb * 1024 for kb in args.sizes_kb] if args.sizes_kb
                   else [mb * (1 << 20) for mb in args.sizes_mb])
    print(
        f"A5_CONTIG_COPY_CONFIG device={device} device_name={device_name!r} "
        f"sizes_kb={[s // 1024 for s in sizes_bytes]} warmup={args.warmup} iters={args.iters} "
        "chunk_bytes=16384 queue_depth=4 "
        f"opapi={vllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    rows = []
    barrier_index = 0
    for size_bytes in sizes_bytes:
        row = measure_config(args, device, device_name, size_bytes, barrier_index)
        barrier_index += 1
        rows.append(row)
        bw = row["bandwidth_per_iter_gbps"]
        print(
            f"A5_CONTIG_COPY_RESULT device={device} size_kb={row['size_kb']:g} "
            f"payload_bytes={row['payload_bytes']} avg_us={row['avg_us']:.3f} "
            f"wall_us={row['wall_us']:.3f} bw_mean={bw['mean']:.3f} "
            f"bw_min={bw['min']:.3f} bw_max={bw['max']:.3f} ok=1",
            flush=True,
        )
    best = max(rows, key=lambda row: row["bandwidth_gbps"])
    print(f"\n{'size_kb':>8} {'avg_us':>12} {'bw_mean':>9} {'bw_min':>9} {'bw_max':>9}", flush=True)
    for row in rows:
        bw = row["bandwidth_per_iter_gbps"]
        print(f"{row['size_kb']:>8g} {row['avg_us']:>12.3f} {bw['mean']:>9.3f} "
              f"{bw['min']:>9.3f} {bw['max']:>9.3f}", flush=True)
    print(
        f"A5_CONTIG_COPY_MAX_BANDWIDTH device={device} size_kb={best['size_kb']:g} "
        f"bandwidth_gbps={best['bandwidth_gbps']:.3f} ok=1",
        flush=True,
    )
    result = {
        "schema_version": 1, "test": "a5_h2d_contig_copy_kernel",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
            "device_name": device_name, "npu_smi_info": npu_smi_info(),
        },
        "config": {
            "device": str(device), "sizes_kb": [s // 1024 for s in sizes_bytes],
            "warmup": args.warmup,
            "iters": args.iters, "seed": args.seed, "chunk_bytes": 16384,
            "queue_depth": 4, "allocator": "empty_with_swapped_memory",
            "access": "contiguous", "bandwidth_convention": "payload counted once",
        },
        "results": rows,
        "max_bandwidth": {"size_kb": best["size_kb"], "bandwidth_gbps": best["bandwidth_gbps"]},
        "performance": {"timer": "npu_event_per_iteration", "cross_check": "wall_clock"},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
    print("A5_CONTIG_COPY_UT_OK", flush=True)
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
