#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(vllm_dsa_a5, m) {
  m.def(
      "kvcache_scatter_copy_c8(Tensor(a!) hbm_kv_bytes, "
      "Tensor dram_kv_bytes, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts) -> ()");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
