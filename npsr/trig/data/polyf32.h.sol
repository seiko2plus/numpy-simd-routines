// f32 Low trig coefficients: sin(x) ~= x + x^3*P(x^2), degree 9, odd.
// 333/346/368 are recoverable Remez warnings the constrained refits below
// always trip; the driver would treat them as errors.
suppressmessage(346, 333, 368);

I = [0x1p-20; pi/2];
vFree = fpminimax(sin(x), [|3, 5, 7, 9|], [|single...|], I, relative, floating, x);

// Relative error in ULP at 1 (= 2^-24); guards regeneration against a worse fit.
procedure ErrULP(pP) { return dirtyinfnorm(pP(x)/sin(x) - 1, I) * 0x1p24; };
Assert(ErrULP(vFree) < 0.12, "f32 Low free fit error regressed: " @ ErrULP(vFree));

// One f32 ULP at a coefficient's scale. No coefficient sits within a step of
// a power of two, so the constant step is exact.
procedure Ulp(pC) { return 2^(floor(log2(abs(pC))) - 23); };

// Pin c3 (then c5, then optionally c7) an integer number of ULPs off the free
// fit and refit the rest under the pin. Pin indices come from exhaustive
// measurement of the compiled kernel over [0, 1e4); no analytic criterion
// sees the eval DAG's per-operation rounding.
procedure Stage2(pK3, pJ5) {
  var vC3, vP5, vC5, vP7;
  vC3 = round(coeff(vFree, 3) + pK3 * Ulp(coeff(vFree, 3)), single, RN);
  vP5 = fpminimax(sin(x), [|5, 7, 9|], [|single...|], I, relative, floating,
                  x + vC3 * x^3);
  vC5 = round(coeff(vP5, 5) + pJ5 * Ulp(coeff(vP5, 5)), single, RN);
  vP7 = fpminimax(sin(x), [|7, 9|], [|single...|], I, relative, floating,
                  x + vC3 * x^3 + vC5 * x^5);
  return [| vC3, vC5, vP7 |];
};

procedure CheckPinned(pSet, pTag) {
  Assert(ErrULP(pSet[0]*x^3 + pSet[1]*x^5 + pSet[2]*x^7 + pSet[3]*x^9 + x) < 1.0,
         "pinned set " @ pTag @ " analytic error drifted");
  return pSet;
};

// c7/c9 come jointly from the stage-two fit.
procedure Pinned(pK3, pJ5) {
  var vS;
  vS = Stage2(pK3, pJ5);
  return CheckPinned([| vS[0], vS[1], coeff(vS[2], 7), coeff(vS[2], 9) |],
                     "(" @ pK3 @ "," @ pJ5 @ ")");
};

// Additionally pin c7 and refit c9 alone. Only FMA cos gains from it.
procedure Pinned3(pK3, pJ5, pL7) {
  var vS, vC7, vP9;
  vS = Stage2(pK3, pJ5);
  vC7 = round(coeff(vS[2], 7) + pL7 * Ulp(coeff(vS[2], 7)), single, RN);
  vP9 = fpminimax(sin(x), [|9|], [|single...|], I, relative, floating,
                  x + vS[0] * x^3 + vS[1] * x^5 + vC7 * x^7);
  return CheckPinned([| vS[0], vS[1], vC7, coeff(vP9, 9) |],
                     "(" @ pK3 @ "," @ pJ5 @ "," @ pL7 @ ")");
};

// Exhaustive [0, 1e4) worst case, pinned vs free fit, in ULP:
//   FMA:     sin 1.886 (free 2.191)   cos 1.815 (free 2.110)
//   non-FMA: sin 1.912 (free 1.939)   cos 1.948 (free 1.966)
// Non-FMA is eval-rounding bound (~1.9 of the ~1.95 total), so pins only
// shift which argument realizes the worst alignment; the whole (k3, j5, l7)
// grid stays within 0.02 of the sets below.
vFmaSin = Pinned(4, 1);
vFmaCos = Pinned3(2, 1, 8);
vNoFmaSin = Pinned(1, 1);
vNoFmaCos = Pinned(1, 5);

Append(
  "// Degree-9 odd minimax sin(x) = x + x^3*P(x^2) on [2^-20, pi/2].",
  "// Coefficients {c3, c5, c7, c9} for the f32 PolyLow in npsr/trig/low-inl.h.",
  "// See poly.h.sol for the recipe, pins and measured worst cases.",
  "template <bool FMA> inline constexpr float kSinPolyLowF32[] = " @ CArrayT(Float32, vFmaSin, 4) @ ";",
  "template <bool FMA> inline constexpr float kCosPolyLowF32[] = " @ CArrayT(Float32, vFmaCos, 4) @ ";",
  "template <> inline constexpr float kSinPolyLowF32<false>[] = " @ CArrayT(Float32, vNoFmaSin, 4) @ ";",
  "template <> inline constexpr float kCosPolyLowF32<false>[] = " @ CArrayT(Float32, vNoFmaCos, 4) @ ";",
  ""
);

// f32 High coefficients, held and evaluated in double by npsr/trig/high-inl.h.
//
// c1 is fitted, not pinned to 1: MulAdd(r2, poly, c1) + Mul(poly, r) closes in
// the same two ops as the pinned x + x^3*P(x^2), so the free parameter is free
// -- 0.1099 -> 0.0965 ulp. No pinning stage either: eval is in double (~2^-52),
// so unlike the single Low sets above there is no rounding for pins to steer.
//
// The interval tracks the cutover in npsr/trig/inl.h and must move with it: n
// comes from a f32 magic-round, so |r| overshoots pi/2 by up to |x|*1.7e-7.
// 0.012 is the exhaustive ceiling below both cutovers, and widening costs
// accuracy fast: 0.0892 ulp at pi/2, 0.0965 here, 0.1073 at 0.028.
IHigh = [0x1p-30; pi/2 + 0.012];
vHigh = fpminimax(sin(x), [|1, 3, 5, 7, 9|], [|double...|], IHigh, relative, floating);

// Leaves High<> at 0.5 (final f32 rounding) + 0.097, the f32 ceiling: the
// Extended kernel above the cutover measures 0.5031, so this fit -- not the
// Payne-Hanek reduction -- is what keeps sin/cos off correct rounding.
errHigh = dirtyinfnorm(vHigh(x)/sin(x) - 1, IHigh) * 0x1p24;
Assert(errHigh < 0.0970, "f32 High fit error regressed: " @ errHigh);
// The driver leaves display in hexadecimal, which is unreadable for an error.
display = decimal!;
errHighStr = "" @ round(errHigh, 24, RN);
display = hexadecimal!;

Append(
  "// Degree-9 odd minimax sin(r) on [2^-30, pi/2 + 0.012] for the f32 High path",
  "// in trig/high-inl.h; c1 is fitted, not pinned to 1. See poly.h.sol.",
  "// Relative error " @ errHighStr @ " ulp(f32). Coefficients {c1, c3, c5, c7, c9}.",
  "inline constexpr double kSinPolyHighF32[] = " @
  CArrayT(Float64, [| coeff(vHigh, 1), coeff(vHigh, 3), coeff(vHigh, 5),
                      coeff(vHigh, 7), coeff(vHigh, 9) |], 3) @ ";",
  ""
);

WriteCPPHeader("npsr::trig::data");
