// Intel SVML as the `svml_avx512` reference target: a baseline, not an oracle.
// Built -mavx512f in its own library, behind TargetSVML's ISA gate.
#include <immintrin.h>

#include <cstddef>

#include "python/_numpy_sr/target.h"

namespace npsr::py {
namespace {

// Bare name = low accuracy, _ha = high. The vector ABI passes and returns in
// zmm0, so a plain C declaration and call is correct (as numpy does it).
extern "C" {
__m512d __svml_sin8(__m512d);
__m512d __svml_sin8_ha(__m512d);
__m512d __svml_cos8(__m512d);
__m512d __svml_cos8_ha(__m512d);
__m512 __svml_sinf16(__m512);
__m512 __svml_sinf16_ha(__m512);
__m512 __svml_cosf16(__m512);
__m512 __svml_cosf16_ha(__m512);
}

using F64Fn = __m512d (*)(__m512d);
using F32Fn = __m512 (*)(__m512);

// Full vectors, then a masked tail pass: padded lanes compute but never store.
void Run(F64Fn fn, const double* src, double* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm512_storeu_pd(dst + i, fn(_mm512_loadu_pd(src + i)));
  }
  if (const size_t rem = n - i) {
    const __mmask8 m = static_cast<__mmask8>((1u << rem) - 1u);
    _mm512_mask_storeu_pd(dst + i, m, fn(_mm512_maskz_loadu_pd(m, src + i)));
  }
}

void Run(F32Fn fn, const float* src, float* dst, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i, fn(_mm512_loadu_ps(src + i)));
  }
  if (const size_t rem = n - i) {
    const __mmask16 m = static_cast<__mmask16>((1u << rem) - 1u);
    _mm512_mask_storeu_ps(dst + i, m, fn(_mm512_maskz_loadu_ps(m, src + i)));
  }
}

// TargetSVML::Accuracies is {High, Low}; anything else is a Python-level error.
template <typename T, typename Fn>
Array<T> Apply(Fn ha, Fn la, Array<T> x, AccuracyID acc_id) {
  if (acc_id != Accuracy::High::kID && acc_id != Accuracy::Low::kID) {
    throw pb::value_error("unsupported accuracy");
  }
  const Fn fn = acc_id == Accuracy::High::kID ? ha : la;
  const CArray<T> xc = Contiguous(x);
  Array<T> out(xc.request().shape);
  const T* src = xc.data();
  T* dst = out.mutable_data();
  const size_t n = static_cast<size_t>(xc.size());
  {
    pb::gil_scoped_release nogil;
    Run(fn, src, dst, n);
  }
  return out;
}

void BindOp(Binder& b, F32Fn s_ha, F32Fn s_la, F64Fn d_ha, F64Fn d_la) {
  b([s_ha, s_la](Array<float> x, AccuracyID acc_id) {
    return Apply<float>(s_ha, s_la, x, acc_id);
  });
  b([d_ha, d_la](Array<double> x, AccuracyID acc_id) {
    return Apply<double>(d_ha, d_la, x, acc_id);
  });
}

}  // namespace

template <>
void TargetSVML::Bind<Operation::Sin::kID>(Binder& b) {
  BindOp(b, __svml_sinf16_ha, __svml_sinf16, __svml_sin8_ha, __svml_sin8);
}

template <>
void TargetSVML::Bind<Operation::Cos::kID>(Binder& b) {
  BindOp(b, __svml_cosf16_ha, __svml_cosf16, __svml_cos8_ha, __svml_cos8);
}

}  // namespace npsr::py
