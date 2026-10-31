"""``numpy_sr`` and its C++ extension are not a second NumPy. They exist only
for deep numeric testing of the kernels, and they go further than NumPy's own
tests:

- NumPy tests the one target its dispatcher picks for the CPU. ``numpy_sr``
  tests every target.
- NumPy checks against libm and a basic set of precomputed data. ``numpy_sr``
  scores against MPFR, a slow but exact oracle.
- It searches for the hard inputs, and for float32 it tests every input.
- It tests every accuracy profile.
- It reports accuracy in more detail, and compares against reference targets:
  the current NumPy, Intel SVML, and maybe others later. NPSR is meant to run
  as a submodule inside NumPy, so the NumPy comparison shows what a change
  would do there.

.. code-block:: python

   import numpy_sr as sr

   sr.targets            # dict: name -> submodule
   sr.targets["AVX2"].sin(x, accuracy=sr.Accuracy.Low)
   sr.mpfr.sin(x)        # (ref, residual)
   sr.ulp_error(computed, ref, residual)
   sr.Accuracy           # High, Low
   sr.FPEnv              # capture of the FP exception flags
"""

from __future__ import annotations

from typing import cast

from . import _numpy_sr, target, utils
from ._numpy_sr import Accuracy, ulp_error

# Context manager over the flags-only `_numpy_sr.FPEnv`, which it shadows here.
from .fpenv import FPEnv

#: Every target submodule by name: the NPSR targets, the ``mpfr`` oracle and
#: the ``svml_avx512`` reference (see :mod:`numpy_sr.target`).
targets: dict[str, target.Target] = _numpy_sr.targets
for _t in targets.values():
    _t.DTYPES = ("float32", "float64") if _t.HAVE_FLOAT64 else ("float32",)
#: The MPFR oracle: correctly rounded results to score against.
# `targets` is typed for the NPSR targets; this entry is the oracle.
mpfr = cast("target.Oracle", _numpy_sr.targets["mpfr"])

__all__ = [
    "Accuracy",
    "FPEnv",
    "mpfr",
    "target",
    "targets",
    "ulp_error",
    "utils",
]
