#!/usr/bin/env python3
"""Ascend 950 abs H2D probe: fully-contiguous host-DRAM input -> HBM output.

Uses torch.abs on a swapped-memory tensor (real host DRAM allocated via
torch_npu.empty_with_swapped_memory) with the output on HBM, so the device
kernel reads the whole input over PCIe once per call. Bandwidth counts the
H2D payload once: input_bytes / avg_us / 1000 (GB/s); the HBM-side output
write is not counted. Optional baselines (default on) measure the same sizes
through the plain H2D copy paths for comparison: `out.copy_(host)` from
swapped DRAM, and `pinned_copy` (pinned host memory -> HBM, non_blocking),
which is the DMA-engine ceiling for contiguous H2D.

Pure torch_npu: no custom operator build is required. Per-iteration raw
timings (NPU events) and wall-clock cross-checks are kept in the JSON.
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

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
DEFAULT_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SAMPLE_LEN = 1 << 20
OPS = ("abs", "copy", "pinned_copy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["npu:0"],
                        help="NPUs to sweep sequentially, e.g. --devices npu:0 npu:1.")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=DEFAULT_SIZES_MB,
                        help="Payload sizes to sweep, in MiB per iteration.")
    parser.add_argument("--dtypes", nargs="+", choices=tuple(DTYPES), default=list(DTYPES),
                        help="Data types to sweep; default runs all three.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copy-baseline", dest="copy_baseline", action="store_true", default=True,
                        help="Also measure out.copy_(host) for the same sizes (default).")
    parser.add_argument("--no-copy-baseline", dest="copy_baseline", action="store_false")
    parser.add_argument("--pinned-baseline", dest="pinned_baseline", action="store_true", default=True,
                        help="Also measure pinned-memory -> HBM non_blocking copy, the DMA-engine "
                             "ceiling for contiguous H2D (default).")
    parser.add_argument("--no-pinned-baseline", dest="pinned_baseline", action="store_false")
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.devices:
        raise ValueError("--devices must be non-empty.")
    if not args.sizes_mb or any(size <= 0 for size in args.sizes_mb):
        raise ValueError("--sizes-mb must be a non-empty list of positive integers.")
    if not args.dtypes:
        raise ValueError("--dtypes must be non-empty.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def require_a5(device: "torch.device", allow_non_a5: bool) -> str:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(
            f"expected Ascend 950, got {name!r}; "
            "use --allow-non-a5 only for debugging"
        )
    return name


def swapped_from_cpu(cpu: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    """Create real swapped-memory (host DRAM) storage initialized from a CPU tensor."""

    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if allocator is None:
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable; "
            "the test refuses to replace DRAM with an HBM tensor"
        )
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["device,op,dtype,size_mb,payload_bytes,avg_us,wall_us,"
             "bw_mean_gbps,bw_min_gbps,bw_max_gbps,iter_us_std,iter_us_p95"]
    for row in rows:
        stats = row["per_iter_us"]
        bw = row["bandwidth_per_iter_gbps"]
        lines.append(
            f"{row['device']},{row['op']},{row['dtype']},{row['size_mb']},{row['payload_bytes']},"
            f"{row['avg_us']:.3f},{row['wall_us']:.3f},{bw['mean']:.3f},{bw['min']:.3f},"
            f"{bw['max']:.3f},{stats['std']:.3f},{stats['p95']:.3f}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_samples(cpu: "torch.Tensor", out: "torch.Tensor", op: str, seed: int) -> None:
    numel = cpu.numel()
    generator = torch.Generator().manual_seed(seed + 1)
    length = min(SAMPLE_LEN, numel)
    offsets = {0, numel - length}
    offsets.update(torch.randint(0, numel - length + 1, (4,), generator=generator).tolist())
    flat_ref = (cpu.abs() if op == "abs" else cpu).view(-1)
    flat_out = out.view(-1)
    for offset in sorted(offsets):
        if not torch.equal(flat_out[offset:offset + length].cpu(), flat_ref[offset:offset + length]):
            raise AssertionError(f"{op} result mismatch at element offset {offset}.")


def measure_config(args: argparse.Namespace, device: "torch.device", device_name: str,
                   op: str, dtype_name: str, size_mb: int) -> dict:
    dtype = getattr(torch, DTYPES[dtype_name])
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = size_mb * (1 << 20) // element_size
    generator = torch.Generator().manual_seed(args.seed)
    cpu = torch.randn(numel, dtype=dtype, generator=generator)
    if op == "pinned_copy":
        try:
            host = torch.empty(numel, dtype=dtype, pin_memory=True)
        except Exception as exc:
            raise RuntimeError(
                "torch.empty(pin_memory=True) failed; this torch/torch_npu build lacks "
                "pinned host memory support. Report the error back unchanged, or rerun "
                "with --no-pinned-baseline."
            ) from exc
        host.copy_(cpu)
    else:
        host = swapped_from_cpu(cpu, device)
    out = torch.empty(numel, dtype=dtype, device=device)
    torch.npu.synchronize()

    def call_op() -> None:
        if op == "abs":
            torch.abs(host, out=out)
        elif op == "copy":
            out.copy_(host)
        else:
            out.copy_(host, non_blocking=True)

    try:
        call_op()
    except Exception as exc:
        raise RuntimeError(
            f"{op} with host-resident input and HBM output failed on device {device}; "
            "this torch_npu build may not accept this host-memory operand. "
            "Report the original error back unchanged."
        ) from exc
    torch.npu.synchronize()
    check_samples(cpu, out, op, args.seed)

    for _ in range(args.warmup):
        call_op()
    torch.npu.synchronize()
    events = [torch.npu.Event(enable_timing=True) for _ in range(args.iters + 1)]
    events[0].record()
    wall_start = time.perf_counter()
    for index in range(args.iters):
        call_op()
        events[index + 1].record()
    events[-1].synchronize()
    wall_s = time.perf_counter() - wall_start
    iter_us = [events[i].elapsed_time(events[i + 1]) * 1000 for i in range(args.iters)]

    payload_bytes = numel * element_size
    iter_bw = [payload_bytes / (t * 1000) for t in iter_us]
    avg_us = statistics.fmean(iter_us)
    del cpu, host, out, events
    torch.npu.empty_cache()
    return {
        "device": str(device), "device_name": device_name, "op": op, "dtype": dtype_name,
        "size_mb": size_mb, "numel": numel, "payload_bytes": payload_bytes,
        "warmup": args.warmup, "iters": args.iters,
        "avg_us": avg_us, "bandwidth_gbps": payload_bytes / (avg_us * 1000),
        "wall_us": wall_s * 1e6 / args.iters,
        "wall_bandwidth_gbps": payload_bytes * args.iters / wall_s / 1e9,
        "per_iter_us": summarize(iter_us),
        "bandwidth_per_iter_gbps": summarize(iter_bw),
        "raw_iter_us": [round(t, 3) for t in iter_us],
    }


def run(args: argparse.Namespace) -> dict:
    ops = ["abs"] + (["copy"] if args.copy_baseline else []) + (["pinned_copy"] if args.pinned_baseline else [])
    rows: list[dict] = []
    environment = {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
        "launch_blocking": os.getenv("ASCEND_LAUNCH_BLOCKING", "unset"),
        "npu_smi_info": npu_smi_info(),
    }
    for device_str in args.devices:
        device = torch.device(device_str)
        torch.npu.set_device(device)
        torch.npu.config.allow_internal_format = False
        device_name = require_a5(device, args.allow_non_a5)
        print(
            f"A5_ABS_H2D_CONFIG device={device} device_name={device_name!r} "
            f"ops={ops} dtypes={args.dtypes} sizes_mb={args.sizes_mb} "
            f"warmup={args.warmup} iters={args.iters} "
            "allocator=empty_with_swapped_memory access=contiguous",
            flush=True,
        )
        for dtype_name in args.dtypes:
            for size_mb in args.sizes_mb:
                for op in ops:
                    row = measure_config(args, device, device_name, op, dtype_name, size_mb)
                    rows.append(row)
                    bw = row["bandwidth_per_iter_gbps"]
                    print(
                        f"A5_ABS_H2D_RESULT device={device} op={op} dtype={dtype_name} "
                        f"size_mb={size_mb} payload_bytes={row['payload_bytes']} "
                        f"avg_us={row['avg_us']:.3f} wall_us={row['wall_us']:.3f} "
                        f"bw_mean={bw['mean']:.3f} bw_min={bw['min']:.3f} bw_max={bw['max']:.3f} ok=1",
                        flush=True,
                    )
    abs_rows = [row for row in rows if row["op"] == "abs"]
    print(f"\n{'device':>8} {'op':>5} {'dtype':>5} {'size_mb':>8} {'avg_us':>12} "
          f"{'bw_mean':>9} {'bw_min':>9} {'bw_max':>9}", flush=True)
    for row in rows:
        bw = row["bandwidth_per_iter_gbps"]
        print(f"{row['device']:>8} {row['op']:>5} {row['dtype']:>5} {row['size_mb']:>8} "
              f"{row['avg_us']:>12.3f} {bw['mean']:>9.3f} {bw['min']:>9.3f} {bw['max']:>9.3f}",
              flush=True)
    best = max(abs_rows, key=lambda row: row["bandwidth_gbps"])
    print(
        f"A5_ABS_H2D_MAX_BANDWIDTH device={best['device']} dtype={best['dtype']} "
        f"size_mb={best['size_mb']} bandwidth_gbps={best['bandwidth_gbps']:.3f} ok=1",
        flush=True,
    )
    result = {
        "schema_version": 2, "test": "a5_abs_h2d_max_bandwidth",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "config": {
            "devices": args.devices, "ops": ops, "dtypes": args.dtypes,
            "sizes_mb": args.sizes_mb, "warmup": args.warmup, "iters": args.iters,
            "seed": args.seed, "allocator": "empty_with_swapped_memory",
            "access": "contiguous", "bandwidth_convention": "input payload counted once",
        },
        "results": rows,
        "max_bandwidth": {
            "device": best["device"], "dtype": best["dtype"],
            "size_mb": best["size_mb"], "bandwidth_gbps": best["bandwidth_gbps"],
        },
        "performance": {"timer": "npu_event_per_iteration", "cross_check": "wall_clock"},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
        write_csv(args.json_output.with_suffix(".csv"), rows)
    print("A5_ABS_H2D_UT_OK", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    global torch, torch_npu
    import torch
    import torch_npu
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
