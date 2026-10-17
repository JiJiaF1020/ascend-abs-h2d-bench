#ifndef VLLM_A5_H2D_CONTIG_COPY_TILING_H
#define VLLM_A5_H2D_CONTIG_COPY_TILING_H

#include <cstdint>

// Per-DataCopy chunk and in-flight queue depth. UB budget is
// QUEUE_DEPTH * CHUNK_BYTES = 64 KB, plus a little queue metadata, which
// fits any AIV. In-flight bytes per die = 64 KB * usedCoreNum, far above
// what is needed to cover host-DRAM-over-PCIe latency at Gen5 x16 rates.
constexpr uint32_t H2D_CONTIG_COPY_CHUNK_BYTES = 16 * 1024;
constexpr uint32_t H2D_CONTIG_COPY_QUEUE_DEPTH = 4;

struct VllmA5H2dContigCopyTilingData {
    uint64_t totalBytes;
    uint64_t spanBytes;
    uint32_t usedCoreNum;
    uint32_t chunkBytes;
};

#endif
