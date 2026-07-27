#if defined(NPSR_TRIG_LOW_INL_H_) == defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef NPSR_TRIG_LOW_INL_H_
#undef NPSR_TRIG_LOW_INL_H_
#else
#define NPSR_TRIG_LOW_INL_H_
#endif

#include "npsr/hwy.h"
#include "npsr/trig/data/data.h"

HWY_BEFORE_NAMESPACE();

namespace npsr::HWY_NAMESPACE::trig {

enum class Operation { kSin = 0, kCos = 1 };

template <Operation OP, typename V, HWY_IF_F32(TFromV<V>)>
NPSR_INTRIN V PolyLow(V r, V r2) {
  using namespace hn;
  namespace data = ::npsr::trig::data;

  const DFromV<V> d;
  constexpr bool kCos = OP == Operation::kCos;
  // cos reduces around (N + 0.5)·π, so both ops evaluate sin(r) here; the
  // per-op/per-mode sets differ only in pinning (see data/polyf32.h.sol).
  constexpr auto kSet = (kCos ? data::kCosPolyLowF32<kNativeFMA>
                              : data::kSinPolyLowF32<kNativeFMA>);
  const V c9 = Set(d, kSet[3]);
  const V c7 = Set(d, kSet[2]);
  const V c5 = Set(d, kSet[1]);
  const V c3 = Set(d, kSet[0]);
  V poly = MulAdd(c9, r2, c7);
  poly = MulAdd(r2, poly, c5);
  poly = MulAdd(r2, poly, c3);
  // cos was mapped to sin via cos(x) = sin(x + π/2) during reduction, so all
  // branches evaluate the same sine polynomial sin(r) ≈ r + r³ · P(r²), not a
  // cosine series. All three build r + r³·poly in three ops; only the grouping
  // differs. Without FMA every intermediate is rounded and r·poly lands in the
  // lowest binade of the three, worth ~0.1 ULP. With FMA the choice is nearly
  // free and cos measures ~0.06 ULP better in SVML's order (r³ first, matching
  // SVML's R = X + X·X²·(...)).
  if constexpr (!kNativeFMA) {
    poly = Mul(r, poly);
    poly = MulAdd(poly, r2, r);
  } else if constexpr (kCos) {
    V r3 = Mul(r2, r);
    poly = MulAdd(r3, poly, r);
  } else {
    poly = Mul(poly, r2);
    poly = MulAdd(r, poly, r);
  }
  return poly;
}

template <Operation OP, typename V, HWY_IF_F64(TFromV<V>)>
NPSR_INTRIN V PolyLow(V r, V r2) {
  using namespace hn;

  const DFromV<V> d;
  // Both ops fit sin(r) on the same interval, so the two coefficient sets
  // differ only in trailing bits; each matches its SVML counterpart
  // (__svml_sin_d_la / __svml_cos_d_la) bit-for-bit.
  constexpr bool kCos = OP == Operation::kCos;
  const V c15 = Set(d, kCos ? -0x1.9f0d60811aac8p-41 : -0x1.9f1517e9f65fp-41);
  const V c13 = Set(d, kCos ? 0x1.60e6857a2f220p-33 : 0x1.60e6bee01d83ep-33);
  const V c11 = Set(d, kCos ? -0x1.ae63546002231p-26 : -0x1.ae6355aaa4a53p-26);
  const V c9 = Set(d, kCos ? 0x1.71de38030fea0p-19 : 0x1.71de3806add1ap-19);
  const V c7 = Set(d, kCos ? -0x1.a01a019a5b87bp-13 : -0x1.a01a019a659ddp-13);
  const V c5 = Set(d, kCos ? 0x1.111111110a4a8p-7 : 0x1.111111110a573p-7);
  const V c3 = Set(d, kCos ? -0x1.55555555554a7p-3 : -0x1.55555555554a8p-3);
  V poly = MulAdd(c15, r2, c13);
  poly = MulAdd(r2, poly, c11);
  poly = MulAdd(r2, poly, c9);
  poly = MulAdd(r2, poly, c7);
  poly = MulAdd(r2, poly, c5);
  poly = MulAdd(r2, poly, c3);
  return poly;
}

template <Operation OP, typename V>
NPSR_INTRIN V Low(V x) {
  using namespace hn;
  using hwy::SignMask;
  namespace data = ::npsr::trig::data;

  const DFromV<V> d;
  const RebindToUnsigned<decltype(d)> du;
  using T = TFromV<V>;
  // Load frequently used constants as vector registers
  const V abs_mask = BitCast(d, Set(du, SignMask<T>() - 1));
  const V x_abs = And(abs_mask, x);
  const V x_sign = AndNot(x_abs, x);

  constexpr bool kIsSingle = std::is_same_v<T, float>;
  // Transform cosine to sine using identity: cos(x) = sin(x + π/2)
  V x_trans = x_abs;
  if constexpr (OP == Operation::kCos) {
    x_trans = Add(x_abs, Set(d, data::kHalfPi<T>));
  }
  // cos(x) rounds to exactly 1 up to these (largest such value per type).
  // Answering 1 directly instead of catching only x_trans == half_pi matters
  // for correctness, not just speed: tiny x reduces to r hugging pi/2, where
  // the poly's per-op rounding cannot land on 1.0 for every such r.
  constexpr T kCosOneMax =
      static_cast<T>(kIsSingle ? 0x1p-12 : 0x1.6a09e667f3bccp-27);
  const auto is_cos_near_zero = Le(x_abs, Set(d, kCosOneMax));

  // Compute N = round(x/π) using "magic number" technique
  // and stores integer part in mantissa
  const V magic_round = Set(d, kIsSingle ? 0x1.8p23f : 0x1.8p52);
  V n_biased = MulAdd(x_trans, Set(d, data::kInvPi<T>), magic_round);
  V n = Sub(n_biased, magic_round);

  // Adjust quotient for cosine (accounts for π/2 phase shift)
  if constexpr (OP == Operation::kCos) {
    // For cosine, we computed N = round((x + π/2)/π) but need N' for x:
    //   N = round((x + π/2)/π) = round(x/π + 0.5)
    // This is often 1 more than round(x/π), so we subtract 0.5:
    //   N' = N - 0.5
    n = Sub(n, Set(d, static_cast<T>(0.5)));
  }
  // Cody-Waite reduction with multi-word π (3 words with FMA, 4 without)
  constexpr auto kPi = data::kPi<T, kNativeFMA>;
  V r = NegMulAdd(n, Set(d, kPi[0]), x_abs);
  r = NegMulAdd(n, Set(d, kPi[1]), r);
  V r_lo;
  if constexpr (!kNativeFMA) {
    // n·π[2] is exact, so Fast2Sum recovers the π[2] rounding error into the
    // π[3] step. 
    const V p2 = Mul(n, Set(d, kPi[2]));
    const V ra = Sub(r, p2);
    const V p2_err = Sub(Sub(r, ra), p2);
    r_lo = Sub(ra, Sub(Mul(n, Set(d, kPi[3])), p2_err));
  } else {
    r_lo = NegMulAdd(n, Set(d, kPi[2]), r);
    if constexpr (!kNativeFMA) {
      r_lo = NegMulAdd(n, Set(d, kPi[3]), r_lo);
    }
  }
  if constexpr (kIsSingle || !kNativeFMA) {
    // The polynomial needs the fully reduced value; r still owes the last
    // π word(s).
    r = r_lo;
  }

  V r2 = Mul(r, r);
  V poly = PolyLow<OP>(r, r2);

  if constexpr (!kIsSingle) {
    // Non-FMA has r == r_lo, giving the plain r + r³·poly. With FMA this is
    // r_lo·(1 + r²·poly) ≈ sin(r_lo): r and r_lo differ by n·π[2] (~2^-83)
    // and sin(r)/r varies slowly, so the error stays sub-ULP.
    V r2_corr = Mul(r2, r_lo);
    poly = MulAdd(r2_corr, poly, r_lo);
  }

  // Extract octant sign information from quotient and flip the sign bit
  poly = Xor(poly,
             BitCast(d, ShiftLeft<sizeof(T) * 8 - 1>(BitCast(du, n_biased))));
  if constexpr (OP == Operation::kCos) {
    poly = IfThenElse(is_cos_near_zero, Set(d, static_cast<T>(1.0)), poly);
  } else {
    // Restore original sign for sine (odd function)
    poly = Xor(poly, x_sign);
  }
  return poly;
}
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace npsr::HWY_NAMESPACE::trig

HWY_AFTER_NAMESPACE();

#endif  // NPSR_TRIG_LOW_INL_H_
