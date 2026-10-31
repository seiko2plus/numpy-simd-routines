"""The interface of every ``numpy_sr.targets`` submodule.

``numpy_sr._numpy_sr`` is a pybind11 module built by
``python/_numpy_sr/meson.build``. ``python/_numpy_sr/target.h`` describes each
backend as a struct, and its flags become the attributes of the submodule
(:class:`Backend`).

There are three kinds of targets:

- **NPSR targets**: the NPSR operations, built for each target that the
  architecture and the compiler support within Highway.
- **Oracle targets**: only ``mpfr`` for now. The NPSR targets are tested
  against it, and their error counts the oracle's residual.
- **Reference targets**: ``svml_avx512`` and ``numpy``. They are only for
  comparison in the ULP report: their error is measured against the oracle,
  but never asserted.

All targets have exactly the same operations, but not all of them support the
accuracy profiles:

- ``mpfr`` takes no ``accuracy``: its result is always correctly rounded, so
  there is no profile to choose.
- ``numpy`` accepts ``accuracy`` but ignores it: a NumPy ufunc has a single
  implementation.

NPSR relies on the `Highway dispatcher
<https://github.com/google/highway#quick-start>`_ to generate the targets
and detect them at runtime. This isn't NumPy's dispatcher, and the Highway
targets (sets of CPU features) differ from the NumPy ones. Testing on them is
still valid for checking the accuracy and sanity of the math functions.

The dispatching differs too, because this is a test unit. It doesn't pick
the best target for the CPU. It tests every target that the build enables
and the host supports.

Each NPSR operation has its own TU, generated from
``python/_numpy_sr/npsr-op.tmpl.cpp``. Highway's ``hwy/foreach_target.h``
re-includes the TU once per target, compiled with that target's ``#pragma``
attributes, so a single TU holds every target. ``python/_numpy_sr/module.cpp``
binds each one with ``HWY_VISIT_TARGETS``. Because of this limitation, one TU
per operation keeps compilation time acceptable: with every operation in one
source, that single TU would hold all operations for every target, which is
slow, uses a lot of memory and limits parallel builds.

.. note::

   To work on one operation, build only it with
   ``spin build -- -Doperation=<name>``, which saves recompiling every
   operation after a shared header changes. ``spin test --operation`` only
   filters at runtime.

Every op is bound through ``Binder`` (``python/_numpy_sr/target.h``), so all
ops of the same signature share their arguments. For example, a ``Unary`` op
takes ``(x, accuracy=Accuracy.High)`` on a target and ``(x)`` on the oracle.
The ``x`` argument is bound with ``noconvert()``, so the dtype must be exactly
``float32`` or ``float64``. Any layout works: strided input is copied to
contiguous first, since performance doesn't matter here.

References
----------

- ``numpy``: added on the Python side (``python/conftest.py``). It
  calls the NumPy ufunc of the same name, ignores ``accuracy``, and is counted
  as FMA and f64 capable.
- ``svml_avx512``: Intel SVML from numpy/SVML (``python/_numpy_sr/svml.cpp``).
  It's built only on Linux and FreeBSD x86_64, in its own ``-mavx512f`` static
  library so the module still loads on CPUs without AVX-512. ``High`` maps to
  the ``_ha`` variants, ``Low`` to the default ones.
"""

from __future__ import annotations

from typing import Final, Protocol

# Absolute: _ops.py is generated into the build, not the source tree.
from numpy_sr import _ops
from numpy_sr._ops import OracleOps, TargetOps

__all__ = ["OPERATIONS", "SIGNATURES", "Backend", "Oracle", "Target"]

#: Every op the extension binds; the oracle has them all.
OPERATIONS: Final = _ops.OPERATIONS
#: Each op's signature (``operations.h``), e.g. ``"Unary"``.
SIGNATURES: Final = _ops.SIGNATURES


class Backend(Protocol):
    """Flags every target submodule carries, kernel or reference alike."""

    #: Native FMA (``HWY_NATIVE_FMA``). Selects the books.
    HAVE_FMA: bool
    #: f64 lanes (``HWY_HAVE_FLOAT64``). Without it only the f32 overload is
    #: bound.
    HAVE_FLOAT64: bool
    #: dtype names the ops take, from ``HAVE_FLOAT64``.
    DTYPES: tuple[str, ...]
    #: Compared, not asserted (``svml_avx512``).
    IS_REFERENCE: bool
    #: The scoring reference (``mpfr``).
    IS_ORACLE: bool
    #: This CPU can run it. If not, the submodule exists but has no ops.
    IS_SUPPORTED: bool
    #: The Highway target bit the backend needs (``svml_avx512``: ``HWY_AVX3``),
    #: 0 for none (``mpfr``).
    HWY_TARGET: int


class Target(Backend, TargetOps, Protocol):
    """One dispatched NPSR target (e.g. ``numpy_sr.targets["AVX2"]``).

    Every op takes ``(x, accuracy=Accuracy.High)`` and returns an array of
    ``x``'s dtype.

    :param x: float32 or float64 array.
    :param accuracy: the accuracy profile.
    """


class Oracle(Backend, OracleOps, Protocol):
    """The MPFR oracle (``numpy_sr.mpfr``).

    ``python/_numpy_sr/mpfr.cpp`` computes each element at the format's
    precision plus 64 bits, rounds it to the format with ``mpfr_subnormalize``
    inside the format's exponent range (so subnormals round like IEEE), and
    returns two arrays:

    - ``ref``: the correctly rounded result.
    - ``residual``: ``ref - exact``, in ULP of the binade the exact value lies
      in, with the 64-extra-bit value standing in for the exact one. Its
      magnitude is at most 0.5, and it's NaN where ``ref`` isn't finite.

    The oracle takes no ``accuracy``: its result is always correctly rounded.

    :param x: float32 or float64 array.
    """
