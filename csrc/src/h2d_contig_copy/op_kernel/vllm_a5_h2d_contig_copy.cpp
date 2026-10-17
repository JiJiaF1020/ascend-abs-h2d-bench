/**
 * Ascend 950 contiguous host-DRAM -> HBM copy probe.
 *
 * Unlike the scatter-copy operator this kernel needs no metadata: the whole
 * transfer is one flat contiguous range. Each AIV owns a contiguous span
 * (coreIdx * spanBytes) and streams it through a depth-N MTE2/MTE3 queue of
 * 16 KB DataCopyPad chunks, so QUEUE_DEPTH large reads are always in flight
 * per core. This measures the kernel-path (not DMA-engine) ceiling for fully
 * contiguous host reads: the burst length and in-flight depth are the only
 * knobs that matter at PCIe latencies.
 *
 * The source may be backed by torch_npu.empty_with_swapped_memory (real host
 * DRAM); the destination is caller-owned HBM, written in place.
 */

#include "kernel_operator.h"
#include "vllm_a5_h2d_contig_copy_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t CHUNK_BYTES = H2D_CONTIG_COPY_CHUNK_BYTES;
constexpr uint32_t QUEUE_DEPTH = H2D_CONTIG_COPY_QUEUE_DEPTH;

class VllmA5H2dContigCopyKernel {
public:
    __aicore__ inline VllmA5H2dContigCopyKernel(
        TPipe *pipe, const VllmA5H2dContigCopyTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src)
    {
        const uint32_t coreIdx = GetBlockIdx();
        pipe_->InitBuffer(copyQueue_, QUEUE_DEPTH, CHUNK_BYTES);
        dstGm_.SetGlobalBuffer((__gm__ uint8_t *)dst);
        srcGm_.SetGlobalBuffer((__gm__ uint8_t *)src);
        begin_ = static_cast<uint64_t>(coreIdx) * tiling_->spanBytes;
        length_ = 0;
        if (begin_ < tiling_->totalBytes) {
            length_ = tiling_->totalBytes - begin_;
            if (length_ > tiling_->spanBytes) {
                length_ = tiling_->spanBytes;
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t issued = 0;
        uint64_t drained = 0;
        uint32_t inflight = 0;
        while (drained < length_) {
            if (inflight == QUEUE_DEPTH || issued >= length_) {
                CopyOut(drained);
                drained += ChunkLen(drained);
                --inflight;
                continue;
            }
            CopyIn(issued);
            issued += ChunkLen(issued);
            ++inflight;
        }
    }

private:
    __aicore__ inline uint32_t ChunkLen(uint64_t offset) const
    {
        const uint64_t remaining = length_ - offset;
        return remaining < CHUNK_BYTES ? static_cast<uint32_t>(remaining)
                                       : CHUNK_BYTES;
    }

    __aicore__ inline void CopyIn(uint64_t offset)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyExtParams params{1, ChunkLen(offset), 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, srcGm_[begin_ + offset], params, pad);
        copyQueue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(uint64_t offset)
    {
        LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{1, ChunkLen(offset), 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            dstGm_[begin_ + offset], local, params);
        copyQueue_.FreeTensor(local);
    }

    TPipe *pipe_;
    const VllmA5H2dContigCopyTilingData *tiling_;
    uint64_t begin_ = 0;
    uint64_t length_ = 0;
    GlobalTensor<uint8_t> dstGm_;
    GlobalTensor<uint8_t> srcGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, QUEUE_DEPTH> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void vllm_a5_h2d_contig_copy(
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR dstOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)dstOut;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(VllmA5H2dContigCopyTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    VllmA5H2dContigCopyKernel op(&pipe, &tilingData);
    op.Init(dst, src);
    op.Process();
}
