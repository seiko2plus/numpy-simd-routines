from __future__ import annotations

from . import argred, ieee754, testing
from ._numpy_sr import Accuracy, targets, ulp_error, ulp_error32

# Context manager over the flags-only `_numpy_sr.FPEnv`, which it shadows here.
from .fpenv import FPEnv

mpfr = targets["mpfr"]

__all__ = [
    "Accuracy",
    "FPEnv",
    "argred",
    "ieee754",
    "mpfr",
    "targets",
    "testing",
    "ulp_error",
    "ulp_error32",
]
