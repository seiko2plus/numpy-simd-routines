// Body of a per-op foreach_target TU; no include guard, re-included once per
// target. The op .cpp defines NPSR_OP + HWY_TARGET_INCLUDE, then includes this.
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "npsr/npsr.h"
#include "python/_numpy_sr/target-inl.h"

HWY_BEFORE_NAMESPACE();
namespace npsr::py::HWY_NAMESPACE {
namespace sr = ::npsr::HWY_NAMESPACE;

template <>
void TargetHighway::Bind<Operation::NPSR_OP::kID>(Binder& b) {
  BindKernel(b, [](auto& prec, auto v) { return sr::NPSR_OP(prec, v); });
}

}  // namespace npsr::py::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
