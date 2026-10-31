#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "python/_numpy_sr/module.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep


#include <pybind11/pybind11.h>

#include "hwy/highway.h"
#include "npsr/precise.h"
#include "python/_numpy_sr/target-inl.h"

#if HWY_ONCE
#include <pybind11/native_enum.h>

#include <tuple>
#include <utility>

namespace npsr::py {
namespace {

template <typename... ACCS>
void BindAccuracy(Module& m, std::tuple<ACCS...>) {
  pb::native_enum<AccuracyID> e(
      m, "Accuracy", "enum.Enum",
      "Accuracy profile of an NPSR operation.\n\n"
      "The NPSR C++ operations take a ``Precise`` configuration: compile-time "
      "flags that change how an operation behaves (accuracy, large-argument "
      "reduction, special cases, FP exceptions, subnormals). Only two "
      "configurations are exposed here:\n\n"
      "- ``High``: ``Precise{}``, all defaults. High accuracy (~1 ULP), with "
      "large arguments, special cases (NaN, Inf) and FP exceptions handled, "
      "and IEEE 754 subnormals.\n"
      "- ``Low``: ``Precise{kLowAccuracy}``. Faster algorithms with lower "
      "accuracy (~1-4 ULP). All other flags keep their defaults.");
  (e.value(ACCS::kPyName, kAccuracyID<ACCS>), ...);
  e.finalize();
}

struct FPEnv {};

void BindFPEnv(Module& m) {
  pb::class_<FPEnv> c(m, "FPEnv",
                      "Thread FP-exception flags: the FE_* bits plus the two "
                      "primitives acting on them.");
  c.def_static(
      "clear", []() noexcept { std::feclearexcept(FPExceptions::kAll); },
      "Clear the FP exception flags in ERRORS.");
  c.def_static(
      "test", []() noexcept { return std::fetestexcept(FPExceptions::kAll); },
      "Currently raised FP exception flags in ERRORS (FE_* bits).");
  c.attr("INVALID") = FPExceptions::kInvalid;
  c.attr("DIVBYZERO") = FPExceptions::kDivByZero;
  c.attr("OVERFLOW") = FPExceptions::kOverflow;
  c.attr("UNDERFLOW") = FPExceptions::kUnderflow;
  c.attr("ERRORS") = FPExceptions::kAll;
}

template <typename T, typename OP>
void BindOperation(Module& m) {
  Binder binder(m, OP::kName);
  T::template Bind<OP>(binder);
}

template <typename T, typename... OPS>
void BindOperations(Module& m, std::tuple<OPS...>) {
  (BindOperation<T, OPS>(m), ...);
}

template <typename T>
void BindTarget(Module& m, pb::dict& d) {
  Module sub = m.def_submodule(T::kPyName, T::kPyDoc);
  // Reachable only through `targets`.
  pb::delattr(m, T::kPyName);
  bool supported = T::kIsEnabled;
  if constexpr (T::kIsEnabled) {
    if constexpr (T::kHighwayID != 0) {
      supported = (::hwy::SupportedTargets() & T::kHighwayID) != 0;
    }
    if (supported) {
      BindOperations<T>(sub, Operation::All{});
    }
  }
  sub.attr("__name__") = T::kPyName;
  sub.attr("HAVE_FMA") = T::kHaveFMA;
  sub.attr("HAVE_FLOAT64") = T::kHaveF64;
  sub.attr("IS_REFERENCE") = T::kIsReference;
  sub.attr("IS_ORACLE") = T::kIsOracle;
  sub.attr("IS_SUPPORTED") = supported;
  // The Highway target the backend needs (svml: HWY_AVX3), 0 for none (mpfr).
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
