"""Scoped capture of the thread's floating-point exception flags."""

from __future__ import annotations

from typing import Final

# The stub omits the internal flags-only class; this one shadows it.
from ._numpy_sr import FPEnv as _flags  # pyright: ignore[reportAttributeAccessIssue]


class FPEnv:
    """Clear the FP exception flags on entry, capture on exit; none restored."""

    INVALID: Final = _flags.INVALID
    DIVBYZERO: Final = _flags.DIVBYZERO
    OVERFLOW: Final = _flags.OVERFLOW
    UNDERFLOW: Final = _flags.UNDERFLOW
    INEXACT: Final = _flags.INEXACT
    ERRORS: Final = _flags.ERRORS

    clear = staticmethod(_flags.clear)
    test = staticmethod(_flags.test)

    def __init__(self) -> None:
        self.raised = 0

    @property
    def errors(self) -> int:
        return self.raised & self.ERRORS

    def __enter__(self) -> FPEnv:
        self.raised = 0
        _flags.clear()
        return self

    def __exit__(self, *exc: object) -> None:
        self.raised = _flags.test()
