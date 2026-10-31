"""Scoped capture of the thread's floating-point exception flags."""

from __future__ import annotations

from typing import Final

# The stub omits the internal flags-only class; this one shadows it.
from ._numpy_sr import FPEnv as _flags  # pyright: ignore[reportAttributeAccessIssue]


class FPEnv:
    """Clear the thread's FP exception flags on entry, read them on exit.

    The flags are not restored.

    .. code-block:: python

       with sr.FPEnv() as f:
           target.sin(x, accuracy=acc)
       assert f.errors == 0

    ``errors`` holds the raised ``INVALID``, ``DIVBYZERO``, ``OVERFLOW`` and
    ``UNDERFLOW`` flags. ``INEXACT`` is left out: nearly every rounded
    operation raises it.
    """

    INVALID: Final = _flags.INVALID
    DIVBYZERO: Final = _flags.DIVBYZERO
    OVERFLOW: Final = _flags.OVERFLOW
    UNDERFLOW: Final = _flags.UNDERFLOW
    ERRORS: Final = _flags.ERRORS

    clear = staticmethod(_flags.clear)
    test = staticmethod(_flags.test)

    def __init__(self) -> None:
        self.errors = 0

    def __enter__(self) -> FPEnv:
        self.errors = 0
        _flags.clear()
        return self

    def __exit__(self, *exc: object) -> None:
        self.errors = _flags.test()
