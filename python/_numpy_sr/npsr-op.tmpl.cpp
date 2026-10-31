// Template for one op's foreach_target TU; meson.build fills it per op.
#define NPSR_OP @OP@
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "@SELF@"
#include "python/_numpy_sr/operation-inl.h"
