#ifndef NPSR_PY_TARGET_H_
#define NPSR_PY_TARGET_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <tuple>
#include <utility>

#include "hwy/detect_targets.h"
#include "npsr/precise.h"

// Set by meson; 0 keeps a TU that misses the flag from silently failing to
// resolve TargetSVML::kIsEnabled.
#ifndef NPSR_HAVE_SVML
#define NPSR_HAVE_SVML 0
#endif

namespace npsr::py {

namespace pb = ::pybind11;

struct Operation {
  struct Sin {
    static constexpr int kID = 0;
    static constexpr const char* kPyName = "sin";
  };
  struct Cos {
    static constexpr int kID = 1;
    static constexpr const char* kPyName = "cos";
  };
  using All = std::tuple<Sin, Cos>;
};

enum class AccuracyID { kHigh = 0, kLow = 1};

struct Accuracy {
  struct High {
    static constexpr AccuracyID kID = AccuracyID::kHigh;
    static constexpr const char* kPyName = "High";
    using Prec = decltype(Precise{});
  };
  struct Low {
    static constexpr AccuracyID kID = AccuracyID::kLow;
    static constexpr const char* kPyName = "Low";
    using Prec = decltype(Precise{kLowAccuracy});
  };
  using All = std::tuple<High, Low>;
};

template <typename T>
using Array = pb::array_t<T>;

template <typename T>
using CArray = pb::array_t<T, pb::array::c_style>;

// Same dtype, C-contiguous: copies strided or broadcast input, never casts.
template <typename T>
CArray<T> Contiguous(const Array<T>& x) {
  CArray<T> c = CArray<T>::ensure(x);
  if (!c) throw pb::value_error("could not make the input C-contiguous");
  return c;
}

// Bind every array argument through this: dropping forcecast blocks only
// *unsafe* casts, pybind's convert pass still promotes int/f32/list/scalar into
// the wider overload. Layout is left to Contiguous() instead.
inline pb::arg SafeArg(const char* name) { return pb::arg(name).noconvert(); }

using Module = pb::module_;

// The one def() site, so every op shares `(x, accuracy=Accuracy.High)`.
// Accuracy must be finalize()d before the default value is read.
class Binder {
 public:
  Binder(Module& m, const char* op_name) : mod_(m), op_name_(op_name) {}

  template <typename F>
  void operator()(F&& f) {
    mod_.def(op_name_, std::forward<F>(f), SafeArg("x"),
             pb::arg("accuracy") = Accuracy::High::kID);
  }

 private:
  Module& mod_;
  const char* op_name_;
};

struct Target {
  static constexpr const char* kPyName = "?";
  static constexpr bool kHaveFMA = true;
  static constexpr bool kHaveF64 = true;
  static constexpr bool kIsReference = false;
  static constexpr bool kIsOracle = false;
  static constexpr bool kIsEnabled = true;
  static constexpr bool kIsHighway = false;
  // Non-zero = the HWY_* ISA bit this target needs; checked against
  // hwy::SupportedTargets() before binding.
  static constexpr int64_t kHighwayID = 0;
};

struct TargetSVML : public Target {
  using Operations = Operation::All;
  using Accuracies = std::tuple<Accuracy::High, Accuracy::Low>;

  static constexpr const char* kPyName = "svml_avx512";
  static constexpr bool kIsReference = true;
  // x86-64 Linux only (see meson.build); AVX-512 kernels, hence the ISA gate.
  static constexpr bool kIsEnabled = NPSR_HAVE_SVML != 0;
  static constexpr int64_t kHighwayID = HWY_AVX3;

  template <int OPID>
  static void Bind(Binder& b);
};

struct TargetMPFR : public Target {
  using Operations = Operation::All;
  using Accuracies = std::tuple<Accuracy::High>;

  static constexpr const char* kPyName = "mpfr";
  static constexpr bool kIsOracle = true;

  template <int OPID>
  static void Bind(Binder& b);
};

// Defined in ulp.cpp, dispatched to the best supported target at call time.
void BindULP(Module& m);

}  // namespace npsr::py

#endif  // NPSR_PY_TARGET_H_
