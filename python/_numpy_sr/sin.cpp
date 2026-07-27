#define NPSR_OP Sin
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "python/_numpy_sr/sin.cpp"
#include "python/_numpy_sr/op-inl.h"
