// Correctly-rounded scalar oracle: <op>(x) -> (ref, residual), where residual
// is ref's signed offset from the exact value (ref - exact), in ULP of the
// binade the exact value lies in. That sign is what makes a consumer's
// `key_diff(computed, ref) + residual` telescope to `computed - exact`.
#include <mpfr.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "python/_numpy_sr/target.h"

namespace npsr::py {
namespace {

using MpfrFn = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

template <typename T>
struct MpfrFormat {
  static_assert(std::numeric_limits<T>::is_iec559);

  static constexpr mpfr_prec_t kPrec = std::numeric_limits<T>::digits;
  static constexpr mpfr_exp_t kExpMax = std::numeric_limits<T>::max_exponent;
  static constexpr mpfr_exp_t kExpMin =
      std::numeric_limits<T>::min_exponent - kPrec + 1;

  // MPFR's significand is in [1/2, 1), so kExpMin admits 2^(kExpMin - 1).
  static constexpr int kMinSubExp = static_cast<int>(kExpMin) - 1;
  static_assert(kMinSubExp == std::numeric_limits<T>::min_exponent -
                                  std::numeric_limits<T>::digits);
};

// Scoped override of MPFR's global exponent range. It constrains every
// operation in scope, not just the rounding to T.
class ExpRange {
 public:
  [[nodiscard]] static ExpRange Widest() {
    return ExpRange(mpfr_get_emin_min(), mpfr_get_emax_max());
  }

  template <typename T>
  [[nodiscard]] static ExpRange OfFormat() {
    return ExpRange(MpfrFormat<T>::kExpMin, MpfrFormat<T>::kExpMax);
  }

  ExpRange(mpfr_exp_t emin, mpfr_exp_t emax) {
    mpfr_set_emin(emin);
    mpfr_set_emax(emax);
  }
  ExpRange(const ExpRange&) = delete;
  ExpRange& operator=(const ExpRange&) = delete;
  ~ExpRange() {
    mpfr_set_emin(saved_emin_);
    mpfr_set_emax(saved_emax_);
  }

 private:
  const mpfr_exp_t saved_emin_ = mpfr_get_emin();
  const mpfr_exp_t saved_emax_ = mpfr_get_emax();
};

template <typename T, mpfr_prec_t kExtraPrec = 0>
class Var {
 public:
  Var() { mpfr_init2(var_, MpfrFormat<T>::kPrec + kExtraPrec); }
  Var(const Var&) = delete;
  Var& operator=(const Var&) = delete;
  ~Var() { mpfr_clear(var_); }

  void Set(T value) {
    if constexpr (std::is_same_v<T, float>) {
      mpfr_set_flt(var_, value, MPFR_RNDN);
    } else {
      mpfr_set_d(var_, value, MPFR_RNDN);
    }
  }

  T Get() const {
    if constexpr (std::is_same_v<T, float>) {
      return mpfr_get_flt(var_, MPFR_RNDN);
    } else {
      return mpfr_get_d(var_, MPFR_RNDN);
    }
  }

  template <mpfr_prec_t kSrcExtraPrec>
  void RoundFrom(const Var<T, kSrcExtraPrec>& src) {
    static_assert(kExtraPrec == 0);
    const ExpRange narrow = ExpRange::OfFormat<T>();
    const int ternary = mpfr_set(var_, src, MPFR_RNDN);
    mpfr_subnormalize(var_, ternary, MPFR_RNDN);
  }

  operator mpfr_ptr() { return var_; }           // NOLINT(runtime/explicit)
  operator mpfr_srcptr() const { return var_; }  // NOLINT(runtime/explicit)

 private:
  mpfr_t var_;
};

// Distance from |value| to its neighbor on the given side.
template <typename T>
double UlpGap(T value, bool toward_zero) {
  if (value == T(0)) {
    return std::ldexp(1.0, MpfrFormat<T>::kMinSubExp);
  }
  // The top binade's gap is uniform; stepping up would give infinity.
  if (toward_zero || value == std::numeric_limits<T>::max()) {
    return static_cast<double>(value - std::nextafter(value, T(0)));
  }
  return static_cast<double>(
      std::nextafter(value, std::numeric_limits<T>::infinity()) - value);
}

// Writes Fn's correctly rounded value to ref_out and (rounded - exact), in
// ulps of the side the exact value falls on, to res_out. Add res_out to a
// candidate's integer key difference from ref_out for its true error. NaN
// where the reference is not finite.
template <MpfrFn Fn, typename T>
void MpfrRefResidual(const T* src, T* ref_out, double* res_out, size_t n) {
  const ExpRange wide = ExpRange::Widest();
  Var<T> x;
  Var<T> ref;
  Var<T, 64> hi;
  Var<T, 128> diff;

  for (size_t i = 0; i < n; ++i) {
    x.Set(src[i]);
    Fn(hi, x, MPFR_RNDN);
    ref.RoundFrom(hi);

    const T rounded = ref.Get();
    ref_out[i] = rounded;
    if (!std::isfinite(rounded)) {
      res_out[i] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }

    const double ulp = UlpGap(std::fabs(rounded), mpfr_cmpabs(hi, ref) < 0);
    mpfr_sub(diff, ref, hi, MPFR_RNDN);
    // Scale in MPFR: unscaled, a subnormal's diff underflows a double.
    mpfr_div_d(diff, diff, ulp, MPFR_RNDN);
    res_out[i] = mpfr_get_d(diff, MPFR_RNDN);
  }
}

template <MpfrFn Fn, typename T>
std::tuple<Array<T>, Array<double>> MpfrRefUlp(Array<T> x) {
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
  return {std::move(ref), std::move(res)};
}

// No accuracy argument: always correctly rounded.
template <typename OP, auto Fn>
void BindOp(Binder& b) {
  if constexpr (OP::kSignature == Signature::kUnary) {
    b.Oracle([](Array<float> x) { return MpfrRefUlp<Fn, float>(x); });
    b.Oracle([](Array<double> x) { return MpfrRefUlp<Fn, double>(x); });
  }
}

}  // namespace

#define NPSR_OPERATION(OP, op, Sig)                  \
  template <>                                        \
  void TargetMPFR::Bind<Operation::OP>(Binder & b) { \
    BindOp<Operation::OP, mpfr_##op>(b);             \
  }

NPSR_OPERATIONS

#undef NPSR_OPERATION

}  // namespace npsr::py
