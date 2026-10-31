"""Type stub for the compiled ``_numpy_sr`` extension."""

from __future__ import annotations

import enum
from typing import overload

import numpy as np
import numpy.typing as npt

from .target import Target

# Arguments are bound via SafeArg (no convert): dtype must match exactly, so ArrayLike
# would overpromise. Any layout is accepted; strided input is copied.
_F32 = npt.NDArray[np.float32]
_F64 = npt.NDArray[np.float64]
_Float = _F32 | _F64

class Accuracy(enum.Enum):
    """Accuracy profile selecting a kernel's error bound."""

    High = 0
    Low = 1

targets: dict[str, Target]

# Dispatched to the best supported target at call time.
@overload
def ulp_error(computed: _F32, ref: _F32, residual: _F64 | None = ...) -> _F64: ...
@overload
def ulp_error(computed: _F64, ref: _F64, residual: _F64 | None = ...) -> _F64: ...
