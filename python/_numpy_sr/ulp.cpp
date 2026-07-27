// Fractional-ULP comparators: utilities, so they bind at module level and
// dispatch to the best supported target at call time.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "python/_numpy_sr/ulp.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include <pybind11/stl.h>  // std::optional<CArray<double>> residual

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "hwy/highway.h"
#include "python/_numpy_sr/target.h"

HWY_BEFORE_NAMESPACE();
namespace npsr::py::HWY_NAMESPACE {
namespace hn = ::hwy::HWY_NAMESPACE;

template <typename V>
V UlpError32(V c, V o) {
  using namespace hn;
  using D = DFromV<V>;
  using DU = RebindToUnsigned<D>;
  using VU = VFromD<DU>;
  const D d;
  const DU du;
  const VU bias = Set(du, uint64_t{2069});
  const VU expmask = Set(du, uint64_t{0x7FF});
  const VU e = And(ShiftRight<52>(BitCast(du, o)), expmask);
  const V inv = BitCast(d, ShiftLeft<52>(Sub(bias, e)));
  return Mul(Abs(Sub(c, o)), inv);
}

// f32 lanes widen to the double lane width; f64 already match.
template <class D, class V>
hn::VFromD<D> WidenTo(D d, V v) {
  if constexpr (hwy::IsSame<hn::TFromV<V>, hn::TFromD<D>>()) {
    return v;
  } else {
    return hn::PromoteTo(d, v);
  }
}

template <typename V, typename VD>
VD UlpError(V ct, V rt, VD res) {
  using namespace hn;
  using T = TFromV<V>;
  using TI = hwy::MakeSigned<T>;
  using DD = DFromV<VD>;
  using DI = RebindToSigned<DD>;
  using DTI = Rebind<TI, DD>;
  using VTI = VFromD<DTI>;
  using MD = MFromD<DD>;
  const DD dd;
  const DI di;
  const DTI dti;
  const VTI kmin = Set(dti, hwy::LimitsMin<TI>());
  const VD inf = Set(dd, std::numeric_limits<double>::infinity());
  const VTI cbits = BitCast(dti, ct);
  const VTI rbits = BitCast(dti, rt);
  const VTI ckey = IfNegativeThenElse(cbits, Sub(kmin, cbits), cbits);
  const VTI rkey = IfNegativeThenElse(rbits, Sub(kmin, rbits), rbits);
  // int64 wraps like the numpy key subtraction; non-finites are fixed below.
  const VFromD<DI> diff = Sub(WidenTo(di, ckey), WidenTo(di, rkey));
  VD err = Add(ConvertTo(dd, diff), res);
  // Classify in f64: promotion preserves NaN-ness, Inf-ness and sign.
  const VD c = WidenTo(dd, ct);
  const VD r = WidenTo(dd, rt);
  const MD c_nan = IsNaN(c), r_nan = IsNaN(r);
  const MD c_inf = IsInf(c), r_inf = IsInf(r);
  const MD sign_ne = RebindMask(
      dd, Xor(Lt(BitCast(di, c), Zero(di)), Lt(BitCast(di, r), Zero(di))));
  const MD both_inf = And(c_inf, r_inf);
  const MD match = Or(And(c_nan, r_nan), AndNot(sign_ne, both_inf));
  const MD mismatch = Or(Or(Xor(c_nan, r_nan), Xor(c_inf, r_inf)),
                         And(both_inf, sign_ne));
  err = IfThenZeroElse(match, err);
  return IfThenElse(mismatch, inf, err);
}

#if HWY_HAVE_FLOAT64
// |computed - oracle| in f32-ULP units; 1/ulp32(oracle) = 2^(2069-E) built
// straight from the biased f64 exponent E, no frexp/divide.
void KernelUlpError32(const float* HWY_RESTRICT comp,
                      const double* HWY_RESTRICT oracle,
                      double* HWY_RESTRICT out, size_t n) {
  const hn::ScalableTag<double> dd;
  const hn::Rebind<float, decltype(dd)> df;
  const size_t N = hn::Lanes(dd);
  for (size_t i = 0; i < n; i += N) {
    const size_t rem = n - i;
    const auto c = hn::PromoteTo(dd, hn::LoadN(df, comp + i, rem));
    const auto o = hn::LoadN(dd, oracle + i, rem);
    hn::StoreN(UlpError32(c, o), dd, out + i, rem);
  }
}
// Signed fractional-ULP error: IEEE total-order key difference plus the
// nullable MPFR residual. Matching non-finites score 0, mismatches +inf.
template <typename T>
void LoopUlpError(const T* HWY_RESTRICT comp, const T* HWY_RESTRICT ref,
                  const double* HWY_RESTRICT res, double* HWY_RESTRICT out,
                  size_t n) {
  const hn::ScalableTag<double> dd;
  const hn::Rebind<T, decltype(dd)> dt;
  const size_t N = hn::Lanes(dd);
  for (size_t i = 0; i < n; i += N) {
    const size_t rem = n - i;
    const auto ct = hn::LoadN(dt, comp + i, rem);
    const auto rt = hn::LoadN(dt, ref + i, rem);
    const auto rs = res ? hn::LoadN(dd, res + i, rem) : hn::Zero(dd);
    hn::StoreN(UlpError(ct, rt, rs), dd, out + i, rem);
  }
}
#else
// Bit-identical scalar fallbacks for targets without native float64 vectors.
void KernelUlpError32(const float* HWY_RESTRICT comp,
                      const double* HWY_RESTRICT oracle,
                      double* HWY_RESTRICT out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const double o = oracle[i];
    const uint64_t e = (hwy::BitCastScalar<uint64_t>(o) >> 52) & 0x7FF;
    const double inv = hwy::BitCastScalar<double>((uint64_t{2069} - e) << 52);
    out[i] = std::fabs(static_cast<double>(comp[i]) - o) * inv;
  }
}

template <typename T>
void LoopUlpError(const T* HWY_RESTRICT comp, const T* HWY_RESTRICT ref,
                  const double* HWY_RESTRICT res, double* HWY_RESTRICT out,
                  size_t n) {
  using TI = hwy::MakeSigned<T>;
  const auto key = [](T v) {
    const TI bits = hwy::BitCastScalar<TI>(v);
    return bits < 0 ? static_cast<TI>(hwy::LimitsMin<TI>() - bits) : bits;
  };
  for (size_t i = 0; i < n; ++i) {
    const T c = comp[i], r = ref[i];
    // unsigned subtraction wraps like the int64 vector/numpy key math
    const int64_t diff = static_cast<int64_t>(
        static_cast<uint64_t>(key(c)) - static_cast<uint64_t>(key(r)));
    double err = static_cast<double>(diff);
    if (res) err += res[i];
    const bool c_nan = std::isnan(c), r_nan = std::isnan(r);
    const bool c_inf = std::isinf(c), r_inf = std::isinf(r);
    const bool sign_ne =
        (hwy::BitCastScalar<TI>(c) < 0) != (hwy::BitCastScalar<TI>(r) < 0);
    if ((c_nan && r_nan) || (c_inf && r_inf && !sign_ne)) err = 0.0;
    if ((c_nan != r_nan) || (c_inf != r_inf) || (c_inf && r_inf && sign_ne))
      err = std::numeric_limits<double>::infinity();
    out[i] = err;
  }
}
#endif  // HWY_HAVE_FLOAT64

// HWY_EXPORT needs one non-template symbol per dispatch table.
void KernelUlpErrorF32(const float* HWY_RESTRICT comp,
                       const float* HWY_RESTRICT ref,
                       const double* HWY_RESTRICT res, double* HWY_RESTRICT out,
                       size_t n) {
  LoopUlpError(comp, ref, res, out, n);
}

void KernelUlpErrorF64(const double* HWY_RESTRICT comp,
                       const double* HWY_RESTRICT ref,
                       const double* HWY_RESTRICT res, double* HWY_RESTRICT out,
                       size_t n) {
  LoopUlpError(comp, ref, res, out, n);
}

}  // namespace npsr::py::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace npsr::py {
HWY_EXPORT(KernelUlpError32);
HWY_EXPORT(KernelUlpErrorF32);
HWY_EXPORT(KernelUlpErrorF64);

namespace {
template <typename T>
using UlpFn = void (*)(const T*, const T*, const double*, double*, size_t);

template <typename T>
Array<double> UlpError(UlpFn<T> fn, Array<T> comp, Array<T> ref,
                       std::optional<Array<double>> residual) {
  if (comp.size() != ref.size())
    throw pb::value_error("computed/ref size mismatch");
  if (residual && residual->size() != ref.size())
    throw pb::value_error("residual size mismatch");
  const CArray<T> c = Contiguous(comp), r = Contiguous(ref);
  const std::optional<CArray<double>> rs =
      residual ? std::optional(Contiguous(*residual)) : std::nullopt;
  Array<double> out(c.request().shape);
  const size_t n = static_cast<size_t>(c.size());
  {
    pb::gil_scoped_release nogil;
    fn(c.data(), r.data(), rs ? rs->data() : nullptr, out.mutable_data(), n);
  }
  return out;
}
}  // namespace

void BindULP(Module& m) {
  m.def(
      "ulp_error32",
      [](Array<float> comp, Array<double> oracle) {
        if (comp.size() != oracle.size())
          throw pb::value_error("computed/oracle size mismatch");
        const CArray<float> c = Contiguous(comp);
        const CArray<double> o = Contiguous(oracle);
        Array<double> out(o.request().shape);
        const size_t n = static_cast<size_t>(o.size());
        {
          pb::gil_scoped_release nogil;
          HWY_DYNAMIC_DISPATCH(KernelUlpError32)
          (c.data(), o.data(), out.mutable_data(), n);
        }
        return out;
      },
      SafeArg("computed"), SafeArg("oracle"));

  m.def(
      "ulp_error",
      [](Array<float> comp, Array<float> ref,
         std::optional<Array<double>> residual) {
        return UlpError<float>(HWY_DYNAMIC_POINTER(KernelUlpErrorF32), comp,
                               ref, residual);
      },
      SafeArg("computed"), SafeArg("ref"), SafeArg("residual") = pb::none());
  m.def(
      "ulp_error",
      [](Array<double> comp, Array<double> ref,
         std::optional<Array<double>> residual) {
        return UlpError<double>(HWY_DYNAMIC_POINTER(KernelUlpErrorF64), comp,
                                ref, residual);
      },
      SafeArg("computed"), SafeArg("ref"), SafeArg("residual") = pb::none());
}
}  // namespace npsr::py
#endif  // HWY_ONCE
