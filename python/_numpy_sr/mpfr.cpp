// Correctly-rounded scalar oracle: <op>(x) -> (ref, residual), where residual
// is the exact value's signed offset from ref in ULP of ref.
#include <mpfr.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "python/_numpy_sr/target.h"

namespace npsr::py {
namespace {

using MpfrFn = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

template <MpfrFn Fn, typename T>
void MpfrRefResidual(const T* src, T* ref_out, double* res_out, size_t n) {
  constexpr bool kIsF32 = std::is_same_v<T, float>;
  constexpr mpfr_prec_t kPrec = kIsF32 ? 24 : 53;
  const mpfr_exp_t emin = mpfr_get_emin(), emax = mpfr_get_emax();
  mpfr_set_emin(kIsF32 ? -148 : -1073);
  mpfr_set_emax(kIsF32 ? 128 : 1024);
  mpfr_t x, hi, ref, diff;
  mpfr_init2(x, kPrec);
  mpfr_init2(hi, kPrec + 64);  // guard bits: hi is the exact value
  mpfr_init2(ref, kPrec);
  mpfr_init2(diff, kPrec + 64);
  for (size_t i = 0; i < n; ++i) {
    if constexpr (kIsF32) {
      mpfr_set_flt(x, src[i], MPFR_RNDN);
    } else {
      mpfr_set_d(x, src[i], MPFR_RNDN);
    }
    Fn(hi, x, MPFR_RNDN);
    const int t = mpfr_set(ref, hi, MPFR_RNDN);
    mpfr_subnormalize(ref, t, MPFR_RNDN);
    const T r = kIsF32 ? mpfr_get_flt(ref, MPFR_RNDN)
                       : static_cast<T>(mpfr_get_d(ref, MPFR_RNDN));
    ref_out[i] = r;
    if (!std::isfinite(r)) {
      res_out[i] = 0.0;
      continue;
    }
    const T a = std::fabs(r);
    const double ulp =
        a == T(0)
            ? std::ldexp(1.0, kIsF32 ? -149 : -1074)
            : static_cast<double>(
                  std::nextafter(a, std::numeric_limits<T>::infinity()) - a);
    mpfr_sub(diff, ref, hi, MPFR_RNDN);
    res_out[i] = mpfr_get_d(diff, MPFR_RNDN) / ulp;
  }
  mpfr_clear(x);
  mpfr_clear(hi);
  mpfr_clear(ref);
  mpfr_clear(diff);
  mpfr_set_emin(emin);
  mpfr_set_emax(emax);
}

template <MpfrFn Fn, typename T>
pb::tuple MpfrRefUlp(Array<T> x) {
  const CArray<T> xc = Contiguous(x);
  const auto shape = xc.request().shape;
  Array<T> ref(shape);
  Array<double> res(shape);
  const size_t n = static_cast<size_t>(xc.size());
  {
    pb::gil_scoped_release nogil;
    MpfrRefResidual<Fn, T>(xc.data(), ref.mutable_data(), res.mutable_data(),
                           n);
  }
  return pb::make_tuple(std::move(ref), std::move(res));
}

// Accuracy is taken for signature parity and ignored: always correctly rounded.
template <MpfrFn Fn>
void BindOp(Binder& b) {
  b([](Array<float> x, AccuracyID) { return MpfrRefUlp<Fn, float>(x); });
  b([](Array<double> x, AccuracyID) { return MpfrRefUlp<Fn, double>(x); });
}

}  // namespace

template <>
void TargetMPFR::Bind<Operation::Sin::kID>(Binder& b) {
  BindOp<mpfr_sin>(b);
}

template <>
void TargetMPFR::Bind<Operation::Cos::kID>(Binder& b) {
  BindOp<mpfr_cos>(b);
}

}  // namespace npsr::py
