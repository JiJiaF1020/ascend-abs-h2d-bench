#!/usr/bin/env python3
"""Ascend 950 abs H2D probe: fully-contiguous host-DRAM input -> HBM output.

Uses torch.abs on a swapped-memory tensor (real host DRAM allocated via
torch_npu.empty_with_swapped_memory) with the output on HBM, so the device
kernel reads the whole input over PCIe once per call. Bandwidth counts the
H2D payload once: input_bytes / avg_us / 1000 (GB/s); the HBM-side output
write is not counted. Pure torch_npu: no custom operator build is required.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
DEFAULT_SIZES_MB = [64, 256, 1024, 2048]
SAMPLE_LEN = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=DEFAULT_SIZES_MB,
                        help="Payload sizes to sweep, in MiB per iteration.")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.sizes_mb or any(size <= 0 for size in args.sizes_mb):
        raise ValueError("--sizes-mb must be a non-empty list of positive integers.")
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


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_samples(cpu: "torch.Tensor", out: "torch.Tensor", seed: int) -> None:
    numel = cpu.numel()
    generator = torch.Generator().manual_seed(seed + 1)
    length = min(SAMPLE_LEN, numel)
    offsets = {0, numel - length}
    offsets.update(torch.randint(0, numel - length + 1, (4,), generator=generator).tolist())
    flat_ref = cpu.abs().view(-1)
    flat_out = out.view(-1)
    for offset in sorted(offsets):
        if not torch.equal(flat_out[offset:offset + length].cpu(), flat_ref[offset:offset + length]):
            raise AssertionError(f"abs result mismatch at element offset {offset}.")


def measure_size(args: argparse.Namespace, size_mb: int) -> dict:
    dtype = getattr(torch, DTYPES[args.dtype])
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = size_mb * (1 << 20) // element_size
    generator = torch.Generator().manual_seed(args.seed)
    cpu = torch.randn(numel, dtype=dtype, generator=generator)
    host = swapped_from_cpu(cpu, args.device_obj)
    out = torch.empty(numel, dtype=dtype, device=args.device_obj)
    torch.npu.synchronize()

    def call_abs() -> None:
        torch.abs(host, out=out)

    try:
        call_abs()
    except Exception as exc:
        raise RuntimeError(
            "torch.abs(swapped_memory_input, out=hbm_output) failed; this torch_npu "
            "build may not accept swapped-memory operands with an HBM out tensor. "
            "Report the original error back unchanged."
        ) from exc
    torch.npu.synchronize()
    check_samples(cpu, out, args.seed)

    for _ in range(args.warmup):
        call_abs()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        call_abs()
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters

    payload_bytes = numel * element_size
    bandwidth_gbps = payload_bytes / (avg_us * 1000)
    del cpu, host, out
    torch.npu.empty_cache()
    return {
        "size_mb": size_mb, "numel": numel, "payload_bytes": payload_bytes,
        "avg_us": avg_us, "bandwidth_gbps": bandwidth_gbps,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.device_obj = device
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    print(
        f"A5_ABS_H2D_CONFIG device={device} device_name={device_name!r} "
        f"dtype={args.dtype} sizes_mb={args.sizes_mb} warmup={args.warmup} iters={args.iters} "
        "allocator=empty_with_swapped_memory access=contiguous "
        f"launch_blocking={os.getenv('ASCEND_LAUNCH_BLOCKING', 'unset')}",
        flush=True,
    )
    rows = []
    for size_mb in args.sizes_mb:
        row = measure_size(args, size_mb)
        rows.append(row)
        print(
            f"A5_ABS_H2D_RESULT size_mb={row['size_mb']} payload_bytes={row['payload_bytes']} "
            f"avg_us={row['avg_us']:.3f} bandwidth_gbps={row['bandwidth_gbps']:.3f}",
            flush=True,
        )
    best = max(rows, key=lambda row: row["bandwidth_gbps"])
    print(f"\n{'size_mb':>8} {'avg_us':>12} {'bandwidth_gbps':>15}", flush=True)
    for row in rows:
        print(f"{row['size_mb']:>8} {row['avg_us']:>12.3f} {row['bandwidth_gbps']:>15.3f}", flush=True)
    print(
        f"A5_ABS_H2D_MAX_BANDWIDTH size_mb={best['size_mb']} "
        f"bandwidth_gbps={best['bandwidth_gbps']:.3f} ok=1",
        flush=True,
    )
    result = {
        "schema_version": 1, "test": "a5_abs_h2d_max_bandwidth",
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_count": 1,
        "config": {
            "device": str(device), "device_name": device_name, "dtype": args.dtype,
            "sizes_mb": args.sizes_mb, "warmup": args.warmup, "iters": args.iters,
            "seed": args.seed, "allocator": "empty_with_swapped_memory",
            "access": "contiguous", "bandwidth_convention": "input payload counted once",
        },
        "results": rows,
        "max_bandwidth": {"size_mb": best["size_mb"], "bandwidth_gbps": best["bandwidth_gbps"]},
        "performance": {"timer": "npu_event"},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
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
