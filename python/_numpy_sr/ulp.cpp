// Fractional-ULP comparators: utilities, so they bind at module level and
// dispatch to the best supported target at call time.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "python/_numpy_sr/ulp.cpp"
#include <pybind11/stl.h>  // std::optional<CArray<double>> residual

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "python/_numpy_sr/target.h"

HWY_BEFORE_NAMESPACE();
namespace npsr::py::HWY_NAMESPACE {
namespace hn = ::hwy::HWY_NAMESPACE;

template <typename T>
constexpr hwy::MakeSigned<T> InfKey() {
  using TU = hwy::MakeUnsigned<T>;
  constexpr int kMantissaBits = std::numeric_limits<T>::digits - 1;  // 23 / 52
  constexpr int kExponentBits =
      static_cast<int>(sizeof(T) * 8) - kMantissaBits - 1;  // 8 / 11
  return static_cast<hwy::MakeSigned<T>>(((TU{1} << kExponentBits) - 1)
                                         << kMantissaBits);
}

template <typename T>
double ScalarUlpError(T c, T r, double res) {
  using TI = hwy::MakeSigned<T>;
  constexpr TI kInf = InfKey<T>();
  const auto key = [](T v) -> TI {
    const TI bits = hwy::BitCastScalar<TI>(v);
    // bits < 0 puts LimitsMin - bits in [LimitsMin + 1, 0]: the subtraction
    // cannot overflow and the key is never LimitsMin, so |key| is well defined.
    return bits < 0 ? static_cast<TI>(hwy::LimitsMin<TI>() - bits) : bits;
  };
  const TI ckey = key(c), rkey = key(r);
  // Unsigned subtraction wraps like the i64 vector (and numpy) key math.
  const int64_t diff = static_cast<int64_t>(static_cast<uint64_t>(ckey) -
                                            static_cast<uint64_t>(rkey));
  double err = static_cast<double>(diff) + res;
  const TI cabs = ckey < 0 ? static_cast<TI>(-ckey) : ckey;
  const TI rabs = rkey < 0 ? static_cast<TI>(-rkey) : rkey;
  const bool c_nan = cabs > kInf, r_nan = rabs > kInf;
  const bool c_inf = cabs == kInf, r_inf = rabs == kInf;
  const bool sign_ne = (ckey < 0) != (rkey < 0);
  if ((c_nan && r_nan) || (c_inf && r_inf && !sign_ne)) err = 0.0;
  if ((c_nan != r_nan) || (c_inf != r_inf) || (c_inf && r_inf && sign_ne))
    err = std::numeric_limits<double>::infinity();
  return err;
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

// err = key(computed) - key(ref) + residual, in ULPs of T. See the module
// docstring for the conventions.
template <typename V, typename VD>
VD UlpError(V ct, V rt, VD res) {
  using namespace hn;
  using T = TFromV<V>;
  using TI = hwy::MakeSigned<T>;
  using DD = DFromV<VD>;
  using DI = RebindToSigned<DD>;
  using VI = VFromD<DI>;
  using MI = MFromD<DI>;
  using DTI = Rebind<TI, DD>;
  using VTI = VFromD<DTI>;
  const DD dd;
  const DI di;
  const DTI dti;
 
  const VTI kmin = Set(dti, hwy::LimitsMin<TI>());
  const VTI cbits = BitCast(dti, ct);
  const VTI rbits = BitCast(dti, rt);
  const VI ckey =
      WidenTo(di, IfNegativeThenElse(cbits, Sub(kmin, cbits), cbits));
  const VI rkey =
      WidenTo(di, IfNegativeThenElse(rbits, Sub(kmin, rbits), rbits));
 
  const VD err0 = Add(ConvertTo(dd, Sub(ckey, rkey)), res);
  // Classify from the keys: no second promotion, no FP compares (so signalling
  // NaN inputs never raise invalid), and Abs is safe because key != LimitsMin.
  const VI kinf = Set(di, static_cast<int64_t>(InfKey<T>()));
  const VI cabs = Abs(ckey), rabs = Abs(rkey);
  const MI c_nan = Gt(cabs, kinf), r_nan = Gt(rabs, kinf);
  const MI c_inf = Eq(cabs, kinf), r_inf = Eq(rabs, kinf);
  const MI sign_ne = Xor(Lt(ckey, Zero(di)), Lt(rkey, Zero(di)));
  const MI both_inf = And(c_inf, r_inf);
  const MI match = Or(And(c_nan, r_nan), AndNot(sign_ne, both_inf));
  const MI mismatch =
      Or(Or(Xor(c_nan, r_nan), Xor(c_inf, r_inf)), And(both_inf, sign_ne));
 
  const VD err = IfThenZeroElse(RebindMask(dd, match), err0);
  return IfThenElse(RebindMask(dd, mismatch),
                    Set(dd, std::numeric_limits<double>::infinity()), err);
}

template <typename T>
void LoopUlpError(const T* HWY_RESTRICT comp, const T* HWY_RESTRICT ref,
                  const double* HWY_RESTRICT res, double* HWY_RESTRICT out,
                  size_t n) {
#if HWY_HAVE_FLOAT64
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
#else
  for (size_t i = 0; i < n; ++i) {
    out[i] = ScalarUlpError(comp[i], ref[i], res ? res[i] : 0.0);
  }
#endif
}
HWY_NOINLINE void UlpErrorF32(const float* HWY_RESTRICT comp,
                              const float* HWY_RESTRICT ref,
                              const double* HWY_RESTRICT res,
                              double* HWY_RESTRICT out, size_t n) {
  LoopUlpError(comp, ref, res, out, n);
}
HWY_NOINLINE void UlpErrorF64(const double* HWY_RESTRICT comp,
                              const double* HWY_RESTRICT ref,
                              const double* HWY_RESTRICT res,
                              double* HWY_RESTRICT out, size_t n) {
  LoopUlpError(comp, ref, res, out, n);
}
}  // namespace npsr::py::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace npsr::py {
HWY_EXPORT(UlpErrorF32);
HWY_EXPORT(UlpErrorF64);

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

constexpr const char* kUlpErrorDoc =
    R"(ulp_error(computed, ref, residual=None) -> numpy.ndarray

Fractional-ULP error of ``computed`` against the exact value, as float64.

All three must have the same size. Runs on the best target the CPU supports.
See :mod:`numpy_sr.testing` for the conventions.

:param computed: float32 or float64 array, not converted.
:param ref: the correctly rounded result, of ``computed``'s dtype.
:param residual: the oracle's float64 residual, or None for 0.)";

void BindULP(Module& m) {
  // pybind's per-overload signatures would replace the docstring with
  // "Overloaded function."; the first line of kUlpErrorDoc is the signature.
  pb::options opts;
  opts.disable_function_signatures();
  m.def(
      "ulp_error",
      [](Array<float> comp, Array<float> ref,
         std::optional<Array<double>> residual) {
        return UlpError<float>(HWY_DYNAMIC_POINTER(UlpErrorF32), comp, ref,
                               residual);
      },
      kUlpErrorDoc, SafeArg("computed"), SafeArg("ref"),
      SafeArg("residual") = pb::none());
  m.def(
      "ulp_error",
      [](Array<double> comp, Array<double> ref,
         std::optional<Array<double>> residual) {
        return UlpError<double>(HWY_DYNAMIC_POINTER(UlpErrorF64), comp, ref,
                                residual);
      },
      SafeArg("computed"), SafeArg("ref"), SafeArg("residual") = pb::none());
}
}  // namespace npsr::py
#endif  // HWY_ONCE
