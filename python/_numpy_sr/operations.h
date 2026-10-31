#ifndef NPSR_PY_OPERATIONS_H_
#define NPSR_PY_OPERATIONS_H_

#include <tuple>
#include <type_traits>

#include "npsr/precise.h"

namespace npsr::py {
enum class Signature {
  kUnary,
};

// NPSR_OPERATION(Op, op, Sig) per op, from
// `operations` in meson.build.
#ifndef NPSR_OPERATIONS
#error "NPSR_OPERATIONS must be defined by meson.build"
#endif

// Drops the leading void, so each op can add `, Op` with no trailing comma.
template <typename, typename... TS>
using TupleTail = std::tuple<TS...>;

struct Operation {
#define NPSR_OPERATION(Op, op, Sig)                            \
  struct Op {                                                  \
    static constexpr const char* kName = #op;                  \
    static constexpr Signature kSignature = Signature::k##Sig; \
  };
  NPSR_OPERATIONS
#undef NPSR_OPERATION

#define NPSR_OPERATION(Op, ...) , Op
  using All = TupleTail<void NPSR_OPERATIONS>;
#undef NPSR_OPERATION
};

struct Accuracy {
  struct High {
    static constexpr const char* kPyName = "High";
    using Prec = decltype(Precise{});
  };
  struct Low {
    static constexpr const char* kPyName = "Low";
    using Prec = decltype(Precise{kLowAccuracy});
  };
  using All = std::tuple<High, Low>;
};

// The Python value of an accuracy: its index in Accuracy::All.
enum class AccuracyID : int {};

template <typename ACC, typename... ACCS>
constexpr AccuracyID AccuracyIndex(std::tuple<ACCS...>) {
  static_assert((std::is_same_v<ACC, ACCS> || ...), "not in Accuracy::All");
  int i = 0, id = -1;
  ((std::is_same_v<ACC, ACCS> ? id = i : 0, ++i), ...);
  return AccuracyID{id};
}

template <typename ACC>
inline constexpr AccuracyID kAccuracyID = AccuracyIndex<ACC>(Accuracy::All{});

}  // namespace npsr::py

#endif  // NPSR_PY_OPERATIONS_H_
