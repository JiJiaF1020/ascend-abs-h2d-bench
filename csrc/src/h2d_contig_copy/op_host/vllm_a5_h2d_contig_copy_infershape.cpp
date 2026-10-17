/** Shape and dtype inference for the A5 contiguous host->device copy probe. */

#include <cstddef>

#include "register/op_impl_registry.h"

namespace {
constexpr size_t DST = 0;
} // namespace

namespace ops {
static ge::graphStatus InferVllmA5H2dContigCopyShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(DST) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(DST);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferVllmA5H2dContigCopyDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(VllmA5H2dContigCopy)
    .InferShape(InferVllmA5H2dContigCopyShape)
    .InferDataType(InferVllmA5H2dContigCopyDataType);
} // namespace ops
