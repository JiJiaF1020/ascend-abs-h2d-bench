/** Host definition for the Ascend 950 contiguous host->device copy probe. */

#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class VllmA5H2dContigCopy : public OpDef {
public:
    explicit VllmA5H2dContigCopy(const char *name)
        : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("dst")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("src")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);
        // The matching input/output name declares an in-place reference, so
        // the generated ACLNN host API receives dst only once (caller-owned);
        // the device kernel still gets the framework-generated output alias.
        this->Output("dst")
            .ParamType(REQUIRED).DataType(bytes).Format(formats);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(VllmA5H2dContigCopy);
} // namespace ops
