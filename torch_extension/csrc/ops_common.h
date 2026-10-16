#pragma once

#include <initializer_list>

#include <torch/extension.h>

namespace vllm_dsa_a5 {

inline constexpr int64_t kBlockSize = 128;
inline constexpr int64_t kPackedKvDim = 656;
inline constexpr int64_t kLiManageOutputCapacity = 16384;
inline constexpr int64_t kMaxSourceCapacity = 1 << 18;

inline void CheckOneDeviceAndContiguous(
    const at::Tensor& reference,
    std::initializer_list<const at::Tensor*> tensors,
    const char* op_name) {
  TORCH_CHECK(
      reference.device().is_privateuseone(),
      op_name,
      " inputs must be on NPU.");
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device() && tensor->is_contiguous(),
        op_name,
        " inputs must be contiguous tensors on one NPU.");
  }
}

}  // namespace vllm_dsa_a5
