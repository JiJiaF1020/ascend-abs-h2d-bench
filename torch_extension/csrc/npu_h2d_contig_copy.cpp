#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"
#include "ops_common.h"

namespace {
using namespace vllm_dsa_a5;

void H2dContigCopyNpu(at::Tensor dst, const at::Tensor& src) {
  TORCH_CHECK(
      dst.scalar_type() == at::kChar && src.scalar_type() == at::kChar,
      "h2d_contig_copy dst/src must be int8.");
  TORCH_CHECK(
      dst.numel() == src.numel() && src.numel() > 0,
      "h2d_contig_copy dst/src must have the same non-zero element count.");
  CheckOneDeviceAndContiguous(dst, {&dst, &src}, "h2d_contig_copy");
  auto keepalive = std::make_tuple(dst, src);
  EXEC_NPU_CMD_ORDERED(
      aclnnVllmA5H2dContigCopy,
      keepalive,
      dst,
      src);
}

void H2dContigCopyMeta(at::Tensor, const at::Tensor&) {}

}  // namespace

TORCH_LIBRARY_IMPL(vllm_dsa_a5, PrivateUse1, m) {
  m.impl(
      "h2d_contig_copy",
      &H2dContigCopyNpu);
}

TORCH_LIBRARY_IMPL(vllm_dsa_a5, Meta, m) {
  m.impl(
      "h2d_contig_copy",
      &H2dContigCopyMeta);
}
