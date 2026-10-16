"""Repository-local Ascend 950 packed-C8 scatter operator."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPP = _ROOT / "_custom_opp_c8"
_LIBS = tuple((_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so"))
if len(_LIBS) != 1:
    raise RuntimeError("Build this repository with bash build_c8.sh first.")
_OPAPI = _LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]

# Always bind this checkout's C8 build, never a leftover BF16/external OPP.
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR)
os.environ["NANOVLLM_A5_INSTALL_OPP_PATH"] = str(_LOCAL_OPP)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401


def local_opapi_path() -> str:
    return str(_OPAPI)


def kvcache_scatter_copy_c8(
    hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
    source_token_ids, destination_slots, copy_counts,
) -> None:
    return torch.ops.vllm_dsa_a5.kvcache_scatter_copy_c8(
        hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
        source_token_ids, destination_slots, copy_counts,
    )


__all__ = ["kvcache_scatter_copy_c8", "local_opapi_path"]
