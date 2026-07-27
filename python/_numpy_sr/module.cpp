#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "python/_numpy_sr/module.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include <pybind11/pybind11.h>

#include "hwy/highway.h"
#include "npsr/precise.h"

#include "python/_numpy_sr/target-inl.h"

#if HWY_ONCE
#include <pybind11/native_enum.h>

#include <cfenv>
#include <tuple>

namespace npsr::py {
namespace {

// A native enum.Enum, not pb::enum_/IntEnum: no int can stand in. Must be
// finalize()d before any def() defaulting to it.
template <typename... ACCS>
void BindAccuracy(Module& m, std::tuple<ACCS...>) {
  pb::native_enum<AccuracyID> e(m, "Accuracy", "enum.Enum",
                                "Accuracy profile selecting a kernel's error "
                                "bound.");
  (e.value(ACCS::kPyName, ACCS::kID), ...);
  e.finalize();
}

// Flag primitives; scoping lives in numpy_sr.fpenv. Covers INEXACT, which the
// kernels leave unmanaged, so ERRORS masks it back out.
struct FPEnv {
  static void Clear() noexcept { std::feclearexcept(FPExceptions::kAll); }
  static int Test() noexcept { return std::fetestexcept(FPExceptions::kAll); }
};

// Bound without a constructor: a namespace of flag bits, not an object.
void BindFPEnv(Module& m) {
  pb::class_<FPEnv> c(m, "FPEnv",
                      "Thread FP-exception flags: the FE_* bits plus the two "
                      "primitives acting on them.");
  c.def_static("clear", &FPEnv::Clear,
               "Clear all floating-point exception flags.");
  c.def_static("test", &FPEnv::Test,
               "Currently raised FP exception flags (FE_* bits).");
  c.attr("INVALID") = FPExceptions::kInvalid;
  c.attr("DIVBYZERO") = FPExceptions::kDivByZero;
  c.attr("OVERFLOW") = FPExceptions::kOverflow;
  c.attr("UNDERFLOW") = FPExceptions::kUnderflow;
  c.attr("INEXACT") = FPExceptions::kInexact;
  c.attr("ERRORS") = FPExceptions::kAll;
}

template <typename T, typename OP>
void BindOperation(Module& m) {
  Binder binder(m, OP::kPyName);
  T::template Bind<OP::kID>(binder);
}

template <typename T, typename... OPS>
void BindOperations(Module& m, std::tuple<OPS...>) {
  (BindOperation<T, OPS>(m), ...);
}

template <typename T>
void BindTarget(Module& m, pb::dict& d) {
  Module sub = m.def_submodule(T::kPyName, "");
  bool supported = T::kIsEnabled;
  if constexpr (T::kIsEnabled) {
    if constexpr (T::kHighwayID != 0) {
      supported = (::hwy::SupportedTargets() & T::kHighwayID) != 0;
    }
    if (supported) {
      BindOperations<T>(sub, typename T::Operations{});
    }
  }
  sub.attr("__name__") = T::kPyName;
  sub.attr("HAVE_FMA") = T::kHaveFMA;
  sub.attr("HAVE_FLOAT64") = T::kHaveF64;
  sub.attr("IS_REFERENCE") = T::kIsReference;
  sub.attr("IS_ORACLE") = T::kIsOracle;
  sub.attr("IS_SUPPORTED") = supported;
  // Highway's target bit: lower = better ISA, 0 for non-Highway. The dict is
  // emit-order (alphabetic), so Python ranks targets with this.
  sub.attr("HWY_TARGET") = T::kHighwayID;
  d[T::kPyName] = sub;
}

}  // namespace
}  // namespace npsr::py

PYBIND11_MODULE(_numpy_sr, m) {
  using namespace npsr::py;
  m.doc() = "NumPy SIMD routines: per-target vectorized math kernels.";

  BindAccuracy(m, Accuracy::All{});
  BindFPEnv(m);
  BindULP(m);

  pb::dict targets_dict;
  BindTarget<TargetSVML>(m, targets_dict);
  BindTarget<TargetMPFR>(m, targets_dict);

#define NPSR_BIND_HWY_TARGET(TARGET, NAMESPACE) \
  BindTarget<NAMESPACE::TargetHighway>(m, targets_dict);
  HWY_VISIT_TARGETS(NPSR_BIND_HWY_TARGET)
#undef NPSR_BIND_HWY_TARGET

  m.attr("targets") = targets_dict;
}
#endif  // HWY_ONCE
