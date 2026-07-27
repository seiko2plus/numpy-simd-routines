"""Type stub for the compiled ``_numpy_sr`` extension."""

from __future__ import annotations

import enum
from typing import Any, Protocol, overload

import numpy as np
import numpy.typing as npt

# Arguments are bound via SafeArg (no convert): dtype must match exactly, so ArrayLike
# would overpromise. Any layout is accepted; strided input is copied.
_F32 = npt.NDArray[np.float32]
_F64 = npt.NDArray[np.float64]
_Float = _F32 | _F64
_RefResidual = tuple[npt.NDArray[Any], npt.NDArray[Any]]

class Accuracy(enum.Enum):
    """Accuracy profile selecting a kernel's error bound."""

    High = 0
    Low = 1

class _Backend(Protocol):
    """Flags every target submodule carries, kernel or reference alike.

    ``IS_SUPPORTED`` is False when the build included the target but this CPU
    cannot run it -- such a submodule is bound *without* its ops.
    ``HWY_TARGET`` is Highway's target bit (lower = better ISA, 0 when the
    backend is not a Highway target)."""

    HAVE_FMA: bool
    HAVE_FLOAT64: bool
    IS_REFERENCE: bool
    IS_ORACLE: bool
    IS_SUPPORTED: bool
    HWY_TARGET: int

class _Target(_Backend, Protocol):
    """One dispatched ISA target (e.g. ``numpy_sr.AVX2``)."""
    def sin(self, x: _Float, accuracy: Accuracy = ...) -> npt.NDArray[Any]: ...
    def cos(self, x: _Float, accuracy: Accuracy = ...) -> npt.NDArray[Any]: ...

targets: dict[str, _Target]

# Dispatched to the best supported target at call time.
@overload
def ulp_error(computed: _F32, ref: _F32, residual: _F64 | None = ...) -> _F64: ...
@overload
def ulp_error(computed: _F64, ref: _F64, residual: _F64 | None = ...) -> _F64: ...
def ulp_error32(computed: _F32, oracle: _F64) -> _F64: ...
