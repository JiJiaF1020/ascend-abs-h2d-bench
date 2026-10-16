#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"
#include "ops_common.h"

namespace {
using namespace vllm_dsa_a5;

void CheckKvcacheScatterCopyC8Inputs(
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  TORCH_CHECK(
      hbm_kv_bytes.dim() == 4 && hbm_kv_bytes.size(0) > 0 &&
          hbm_kv_bytes.size(1) == kBlockSize &&
          hbm_kv_bytes.size(2) == 1 &&
          hbm_kv_bytes.size(3) == kPackedKvDim &&
          dram_kv_bytes.dim() == 4 && dram_kv_bytes.size(0) > 0 &&
          dram_kv_bytes.size(1) == kBlockSize &&
          dram_kv_bytes.size(2) == 1 &&
          dram_kv_bytes.size(3) == kPackedKvDim,
      "kvcache_scatter_copy_c8 KV byte views must be "
      "[blocks,128,1,656].");
  TORCH_CHECK(
      hbm_kv_bytes.scalar_type() == at::kChar &&
          dram_kv_bytes.scalar_type() == at::kChar,
      "kvcache_scatter_copy_c8 internal KV views must be int8.");
  TORCH_CHECK(
      copy_counts.dim() == 1 && copy_counts.size(0) > 0,
      "kvcache_scatter_copy_c8 copy_counts must be [B].");
  const int64_t batch = copy_counts.size(0);
  TORCH_CHECK(
      hbm_block_table.dim() == 2 && hbm_block_table.size(0) == batch &&
          hbm_block_table.size(1) > 0 &&
          dram_block_table.dim() == 2 &&
          dram_block_table.size(0) == batch &&
          dram_block_table.size(1) > 0 &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity &&
          source_token_ids.dim() == 3 &&
          source_token_ids.size(0) == batch &&
          source_token_ids.size(1) == 1 &&
          source_token_ids.size(2) == kLiManageOutputCapacity &&
          destination_slots.sizes() == source_token_ids.sizes(),
      "kvcache_scatter_copy_c8 metadata must be block tables [B,*], "
      "source_token_ids/destination_slots [B,1,16384], and "
      "copy_counts [B].");
  for (const at::Tensor* tensor :
       {&hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "kvcache_scatter_copy_c8 metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kv_bytes,
      {&hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
       &dram_block_table, &source_token_ids, &destination_slots,
       &copy_counts},
      "kvcache_scatter_copy_c8");
}

void KvcacheScatterCopyC8Npu(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  CheckKvcacheScatterCopyC8Inputs(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts);
  auto keepalive = std::make_tuple(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnVllmA5KvcacheScatterCopyC8,
      keepalive,
      hbm_kv_bytes,
      dram_kv_bytes,
      hbm_block_table,
      dram_block_table,
      source_token_ids,
      destination_slots,
      copy_counts);
}

void KvcacheScatterCopyC8Meta(
    at::Tensor,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {}

}  // namespace

TORCH_LIBRARY_IMPL(vllm_dsa_a5, PrivateUse1, m) {
  m.impl(
      "kvcache_scatter_copy_c8",
      &KvcacheScatterCopyC8Npu);
}

TORCH_LIBRARY_IMPL(vllm_dsa_a5, Meta, m) {
  m.impl(
      "kvcache_scatter_copy_c8",
      &KvcacheScatterCopyC8Meta);
}
