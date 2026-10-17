/** Tiling registration for the A5 contiguous host->device copy probe. */

#include <cstddef>
#include <cstdint>

#include "../op_kernel/vllm_a5_h2d_contig_copy_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t DST = 0;
constexpr size_t SRC = 1;
} // namespace

namespace optiling {
static ge::graphStatus TilingVllmA5H2dContigCopy(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(DST) == nullptr ||
        context->GetInputDesc(DST) == nullptr ||
        context->GetInputShape(SRC) == nullptr ||
        context->GetInputDesc(SRC) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(DST)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(SRC)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape dst = context->GetInputShape(DST)->GetStorageShape();
    const gert::Shape src = context->GetInputShape(SRC)->GetStorageShape();
    const int64_t total = src.GetShapeSize();
    if (total <= 0 || dst.GetShapeSize() != total) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalBytes = static_cast<uint64_t>(total);
    const uint64_t chunkBytes = H2D_CONTIG_COPY_CHUNK_BYTES;
    const uint64_t totalChunks = (totalBytes + chunkBytes - 1) / chunkBytes;
    const uint64_t usedCores = totalChunks < static_cast<uint64_t>(aivCount)
        ? totalChunks : static_cast<uint64_t>(aivCount);
    const uint32_t usedCoreNum = static_cast<uint32_t>(usedCores);
    const uint64_t spanChunks =
        (totalChunks + usedCores - 1) / usedCores;
    auto *tiling = context->GetTilingData<VllmA5H2dContigCopyTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalBytes = totalBytes;
    tiling->spanBytes = spanChunks * chunkBytes;
    tiling->usedCoreNum = usedCoreNum;
    tiling->chunkBytes = static_cast<uint32_t>(chunkBytes);
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

struct VllmA5H2dContigCopyCompileInfo {};

static ge::graphStatus TilingParseVllmA5H2dContigCopy(
    gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmA5H2dContigCopy)
    .Tiling(TilingVllmA5H2dContigCopy)
    .TilingParse<VllmA5H2dContigCopyCompileInfo>(
        TilingParseVllmA5H2dContigCopy);
} // namespace optiling
