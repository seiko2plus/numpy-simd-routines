#if defined(NPSR_PY_TARGET_INL_H_) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef NPSR_PY_TARGET_INL_H_
#undef NPSR_PY_TARGET_INL_H_
#else
#define NPSR_PY_TARGET_INL_H_
#endif

#include <cstddef>

#include "hwy/highway.h"
#include "python/_numpy_sr/target.h"

HWY_BEFORE_NAMESPACE();
namespace npsr::py::HWY_NAMESPACE {
namespace hn = ::hwy::HWY_NAMESPACE;

struct TargetHighway : public Target {
  using Operations = Operation::All;
  using Accuracies = Accuracy::All;
  // +2 skips the "N_" of HWY_NAMESPACE: TargetName(), but at compile time.
  static constexpr const char* kPyName = HWY_STR(HWY_NAMESPACE) + 2;
  static constexpr bool kHaveFMA = HWY_NATIVE_FMA != 0;
  static constexpr bool kHaveF64 = HWY_HAVE_FLOAT64 != 0;
  static constexpr bool kIsHighway = true;
  static constexpr int64_t kHighwayID = HWY_TARGET;

  template <int OPID>
  static void Bind(Binder& b);
};

// Masked LoadN/StoreN so full vectors and the tail share one op call site.
template <typename Prec, typename Op, typename T>
void Forward(Op op, const T* HWY_RESTRICT src, T* HWY_RESTRICT dst,
             size_t len) {
  Prec prec;
  const hn::ScalableTag<T> d;
  const size_t N = hn::Lanes(d);
  for (size_t i = 0; i < len; i += N) {
    const size_t rem = len - i;
    hn::StoreN(op(prec, hn::LoadN(d, src + i, rem)), d, dst + i, rem);
  }
}

template <typename T, typename Op, typename... ACCS>
Array<T> Apply(std::tuple<ACCS...>, Op op, Array<T> x, AccuracyID acc_id) {
  const CArray<T> xc = Contiguous(x);
  Array<T> out(xc.request().shape);
  const T* src = xc.data();
  T* dst = out.mutable_data();
  const size_t n = static_cast<size_t>(xc.size());
  bool found = false;
  {
    pb::gil_scoped_release nogil;  // released before the array_t return
    const auto run = [&](auto acc) {
      using Acc = decltype(acc);
      if (Acc::kID == acc_id) {
        found = true;
        Forward<typename Acc::Prec>(op, src, dst, n);
      }
    };
    (run(ACCS{}), ...);
  }
  if (!found) throw pb::value_error("unsupported accuracy");
  return out;
}

// The f32/f64 overloads of one op; dtype is exact, so their order is free.
template <typename Op>
void BindKernel(Binder& b, Op op) {
  const auto def = [&](auto zero) {
    using T = decltype(zero);
    b([op](Array<T> x, AccuracyID acc_id) {
      return Apply<T>(TargetHighway::Accuracies{}, op, x, acc_id);
    });
  };
  def(float{});
#if HWY_HAVE_FLOAT64
  def(double{});
#endif
}

}  // namespace npsr::py::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#endif  // NPSR_PY_TARGET_INL_H_
