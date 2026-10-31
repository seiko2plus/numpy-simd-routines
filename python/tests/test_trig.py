"""Trigonometric accuracy, IEEE specials, math identities and fenv discipline.

Bounds
------

========  =======  =======
Accuracy  float32  float64
========  =======  =======
High      1 ULP    1 ULP
Low       2 ULP    4 ULP
========  =======  =======

The hard inputs
---------------

There are too many inputs to sweep, so the tests aim at the ones where the
kernel is most likely to be wrong:

- ``test_simple`` (``rand``): random inputs, log-uniform so every binade
  gets the same share, split into 8 binade groups. A new seed on every run.
- ``test_simple`` (``argred``): per binade, the float closest to a grid in
  each of 4 equal slices, computed exactly by ``python/argred.py`` with
  a 1300-bit π, with its ±2 ULP neighbors and both signs. The grids
  are kπ and (k+½)π, the zeros of sin and cos, where argument reduction
  loses the most bits, and the seams where the kernel picks a different
  branch or table entry: the same two grids for the other op, since Low and
  float32 High round x/π, (k+½)π/16 for the float64 High π/16 reduction,
  and the centres and seams of the Extended table, kπ/128 and (k+½)π/128
  for float32, kπ/256 and (k+½)π/256 for float64. ``test_argred.py``
  checks them against the published worst case (Muller) and brute force at
  low precisions.
- ``test_reduction_exhaustive`` (``exhaustive``, float64): every zero of the
  op up to 2**24, the High and Low paths' range, rounded, with its ±1 ULP
  neighbors and both signs.
- ``test_worst_adaptive`` (``slow``): starts from the ``argred`` cases plus
  random ones and hill-climbs: score against the oracle, keep the worst 512, probe
  ±3 ULP around them and add fresh random inputs, 128 rounds. The winners are
  then checked with their ±2 ULP neighbors and both signs.
- ``test_simple``: named angles (π/6, π/4, ..., 2π), the smallest normal and
  subnormal, 10000 and 2**24 (where float32 and float64 switch to Extended),
  their ±2 ULP neighbors and both signs.

Both dtypes: bit-exact
----------------------

- ``test_parity``: ``f(-x) == -f(x)`` for odd ops (sin) and ``f(-x) == f(x)``
  for even ones (cos).
- ``test_tiny``: ±0, subnormals and anything below half an ULP of π/2 give
  ``f(x) == x`` for odd ops and ``f(x) == 1`` for even ones.
- ``test_special``: inf, qNaN and sNaN all give a quiet NaN, and ±0 and
  subnormals in the same vectors still pass the ``test_tiny`` identity.
- ``test_fp_errors``: across normals, zeros, subnormals, NaNs, inf and a mix
  of them, the only flag raised is ``INVALID``, and only when there is an inf
  or an sNaN in the input.
"""

import functools
import math
from itertools import product
from typing import Any

import numpy as np
import numpy_sr as sr
from numpy_sr import utils
from numpy_sr.testing import assert_bits, ulp_error
from pytest import mark, param

_BOUNDS = {
    sr.Accuracy.High: {np.float32: 1.0, np.float64: 1.0},
    sr.Accuracy.Low: {np.float32: 2.0, np.float64: 4.0},
}

# op -> (odd, argred grid). An odd op has f(-x) == -f(x) and f(tiny) == x,
# an even one f(-x) == f(x) and f(tiny) == 1. The grid holds its zeros.
_OPS = {
    "sin": (True, sr.argred.Grid.PI),
    "cos": (False, sr.argred.Grid.PI_HALF),
}
# Hard for every op: Low and f32 High round x/pi, so each op's zeros are the
# other's seams; f64 High reduces by pi/16; then the Extended tables.
_SEAMS = {
    np.float32: (
        sr.argred.Grid.PI,
        sr.argred.Grid.PI_HALF,
        sr.argred.Grid.PI_128,
        sr.argred.Grid.PI_128_HALF,
    ),
    np.float64: (
        sr.argred.Grid.PI,
        sr.argred.Grid.PI_HALF,
        sr.argred.Grid.PI_16_HALF,
        sr.argred.Grid.PI_256,
        sr.argred.Grid.PI_256_HALF,
    ),
}
_DTYPES = [np.float32, np.float64]
_MIN_BLOCK = 4096
_MAX_BLOCK = 2**16
# Binades are dealt round-robin into this many cases, which xdist runs in parallel.
_GROUPS = 8
# Champions per binade, one per slice.
_SLICES = 4
# Every grid point is tested up to here, the High and Low paths' range:
# above 2**24 the kernels switch to Extended.
_REDUCTION_EMAX = 24


def _group(dtype, group):
    return utils.binades(dtype, subnormals=True)[group::_GROUPS]


def _report_combos(test, operation, dtype, x, ref, combos):
    """Score one block against every combo; `ref` is one shared oracle pass."""
    for target, accuracy in combos:
        test(
            getattr(target, operation),
            x,
            ref,
            _BOUNDS[accuracy][dtype],
            accuracy=accuracy,
        )


@functools.cache
def _basic(rand, oracle, operation, dtype) -> tuple[np.ndarray, Any]:
    """Named args over the quadrant seams, both signs, +-2 ULP neighbors."""
    fi, pi = np.finfo(dtype), math.pi
    named = [0.0, pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3]
    named += [pi, 3 * pi / 2, 2 * pi, fi.tiny, fi.smallest_subnormal]
    named += [10000.0, 2.0**24]  # the f32 and f64 switches to Extended
    x = utils.both_signs(utils.ulp_neighbors(np.array(named, dtype=dtype)))
    return x, getattr(oracle, operation)(x)


@functools.cache
def _rand(group, rand, oracle, operation, dtype) -> tuple[np.ndarray, Any]:
    """Log-uniform over binade group `group`."""
    x = rand(dtype, _group(dtype, group), _MAX_BLOCK)
    return x, getattr(oracle, operation)(x)


@functools.cache
def _argred(rand, oracle, operation, dtype) -> tuple[np.ndarray, Any]:
    """Per binade, the champion of each slice of the dtype's grids, +-2 ULP."""
    champions = [
        ar.worst_cases(e, _SLICES)
        for ar in (sr.argred.Argred(dtype, g) for g in _SEAMS[dtype])
        for e in ar.valid_binades()
    ]
    x = utils.both_signs(utils.ulp_neighbors(np.concatenate(champions)))
    return x, getattr(oracle, operation)(x)


_DATA = [
    param(_basic, id="basic"),
    param(_argred, id="argred"),
    *(param(functools.partial(_rand, i), id=f"rand{i}") for i in range(_GROUPS)),
]


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
@mark.parametrize("data", _DATA)
def test_simple(test, rand, oracle, target, accuracy, operation, dtype, data):
    """Cached on `rand`, so every target shares one block and one oracle pass."""
    x, ref = data(rand, oracle, operation, dtype)
    op_fn = getattr(target, operation)
    test(op_fn, x, ref, _BOUNDS[accuracy][dtype], accuracy=accuracy)


# Adaptive worst-ULP hunt
_HUNT_ROUNDS = 128
_HUNT_KEEP = 512
_HUNT_RADIUS = 3


def _hunt_worst(combos, operation, oracle, rand, dtype, seeds):
    """Hill-climb the ULP landscape: score vs the oracle, keep top-KEEP, re-probe."""
    binades = utils.binades(dtype)
    ref_op = getattr(oracle, operation)
    centers = pool = np.asarray(seeds, dtype=dtype)
    best: list[tuple[np.ndarray, np.ndarray]] = [
        (np.empty(0, dtype), np.empty(0)) for _ in combos
    ]
    for _ in range(_HUNT_ROUNDS):
        fresh = rand(dtype, binades, _MAX_BLOCK, sign=rand.POSITIVE)
        pool = np.concatenate([pool, fresh])
        pool = np.unique(pool[np.isfinite(pool) & (pool > 0)])
        ref = ref_op(pool)  # one oracle pass, shared by every combo below
        for i, (target, accuracy) in enumerate(combos):
            op_fn = getattr(target, operation)
            err = np.abs(ulp_error(op_fn(pool, accuracy=accuracy), ref))
            keep = np.argsort(-err, kind="stable")[:_HUNT_KEEP]
            best[i] = pool[keep], err[keep]
        centers = np.unique(np.concatenate([bx for bx, _ in best]))
        pool = np.concatenate([centers, utils.ulp_neighbors(centers, _HUNT_RADIUS)])
    return centers


@mark.slow
@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", [np.float64])
def test_worst_adaptive(test, rand, oracle, targets, accuracies, operation, dtype):
    """`_hunt_worst` seeded from `argred`'s champions, +-2 ULP window at the end."""
    combos = [
        (t, a)
        for t, a in product(targets, accuracies)
        if np.dtype(dtype).name in t.DTYPES
    ]
    seeds = [
        ar.worst_case(e)
        for ar in (sr.argred.Argred(dtype, g) for g in _SEAMS[dtype])
        for e in ar.valid_binades()
    ]

    champions = _hunt_worst(combos, operation, oracle, rand, dtype, seeds)

    x = utils.both_signs(utils.ulp_neighbors(champions, radius=2))
    ref = getattr(oracle, operation)(x)
    _report_combos(test, operation, dtype, x, ref, combos)


@mark.exhaustive
@mark.parametrize("operation", _OPS)
@mark.parametrize("group", range(_GROUPS))
@mark.parametrize("dtype", [np.float32])
def test_exhaustive(test, oracle, targets, accuracies, operation, dtype, group):
    """f32: every float of binade group `group`, scored for x >= 0; on an NPSR
    target, op(-x) must equal -op(x) (odd) or op(x) (even) bit for bit."""
    combos = list(product(targets, accuracies))  # reused across binades
    odd, _ = _OPS[operation]
    for lo, hi in _group(dtype, group):
        x = utils.float_range(dtype, lo, hi)
        ref = getattr(oracle, operation)(x)
        for target, accuracy in combos:
            op_fn = getattr(target, operation)
            got = test(op_fn, x, ref, _BOUNDS[accuracy][dtype], accuracy=accuracy)
            if not target.IS_REFERENCE:
                neg = op_fn(-x, accuracy=accuracy)
                msg = f"{operation}(-x) parity on {target.__name__} {accuracy.name}"
                assert_bits(neg, -got if odd else got, -x, msg)
        del x, ref


@mark.exhaustive
@mark.parametrize("operation", _OPS)
@mark.parametrize("group", range(_GROUPS))
@mark.parametrize("dtype", [np.float64])
def test_reduction_exhaustive(
    test, oracle, targets, accuracies, operation, dtype, group
):
    """f64: every zero of the op up to 2**24, rounded, +-1 ULP."""
    combos = [
        (t, a)
        for t, a in product(targets, accuracies)
        if np.dtype(dtype).name in t.DTYPES
    ]
    _, zero = _OPS[operation]
    ar = sr.argred.Argred(dtype, zero)
    binades = [e for e in ar.valid_binades() if e <= _REDUCTION_EMAX]
    # deal zeros, not binades: each binade has twice the zeros of the one below
    zeros = np.concatenate([ar.nearest(e) for e in binades])
    x = utils.both_signs(utils.ulp_neighbors(zeros[group::_GROUPS], radius=1))
    ref = getattr(oracle, operation)(x)
    _report_combos(test, operation, dtype, x, ref, combos)


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_parity(rand, npsr, accuracy, operation, dtype):
    """Odd or even: op(-x) must equal +-op(x) bit for bit."""
    op_fn = getattr(npsr, operation)
    binades = utils.binades(dtype)
    x = rand(dtype, binades, _MIN_BLOCK, sign=rand.POSITIVE, subnormal=0.25)
    pos = op_fn(x, accuracy=accuracy)
    neg = op_fn(-x, accuracy=accuracy)
    odd, _ = _OPS[operation]
    want = -pos if odd else pos
    assert_bits(neg, want, -x, f"{operation}(-x) parity")


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_tiny(rand, npsr, accuracy, operation, dtype):
    """Below the round-to-1 branch, op(x) == x (odd) or 1.0 (even) bit-exactly."""
    # The stricter High round-to-1 threshold, |x| <= half ulp(pi/2).
    cap = np.finfo(dtype).eps / 2
    tiny = np.finfo(dtype).smallest_subnormal
    x = rand(dtype, [(tiny, cap)], _MIN_BLOCK, subnormal=0.5, zero=0.1)
    # rand never draws the cap itself
    x = np.concatenate([x, np.array([cap, -cap], dtype)])
    got = getattr(npsr, operation)(x, accuracy=accuracy)
    odd, _ = _OPS[operation]
    want = x if odd else np.ones_like(x)
    assert_bits(got, want, x, f"{operation} tiny identity")


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_special(rand, npsr, accuracy, operation, dtype):
    """Specials scattered through a normal block: inf, qNaN and sNaN give a quiet
    NaN, and ±0 and subnormals still give x (odd) or 1.0 (even) bit-exactly."""
    specials = dict.fromkeys(("zero", "subnormal", "qnan", "snan", "inf"), 0.15)
    x = rand(dtype, utils.binades(dtype), _MIN_BLOCK, **specials)
    got = getattr(npsr, operation)(x, accuracy=accuracy)
    nonfinite = ~np.isfinite(x)
    assert np.isnan(got[nonfinite]).all(), f"{operation}: non-finite input -> non-NaN"
    quiet = 1 << (np.finfo(dtype).nmant - 1)
    signaling = utils.bits(got[nonfinite]) & quiet == 0
    assert not signaling.any(), f"{operation}: non-finite input -> sNaN"
    small = np.isfinite(x) & (np.abs(x) < np.finfo(dtype).tiny)
    odd, _ = _OPS[operation]
    want = x[small] if odd else np.ones(int(small.sum()), dtype)
    assert_bits(got[small], want, x[small], f"{operation} sub-tiny")


# `rand` configs; INVALID is raised solely for inf and sNaN, so the expected
# outcome is read off the block rather than tabulated per case.
_FLAG_CASES = {
    "normals": {},
    "zeros": {"zero": 1.0},
    "subnormals": {"subnormal": 1.0},
    "qnan": {"qnan": 1.0},
    "snan": {"snan": 1.0},
    "inf": {"inf": 1.0},
    "mixed": {"subnormal": 0.1, "qnan": 0.1, "snan": 0.1, "inf": 0.1},
}


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
@mark.parametrize("case", _FLAG_CASES)
def test_fp_errors(rand, npsr, accuracy, operation, dtype, case):
    """fenv discipline: spurious flags suppressed, INVALID set only for inf and sNaN."""
    x = rand(dtype, utils.binades(dtype), _MIN_BLOCK, **_FLAG_CASES[case])
    with sr.FPEnv() as f:
        getattr(npsr, operation)(x, accuracy=accuracy)
    quiet = 1 << (np.finfo(dtype).nmant - 1)
    snan = np.isnan(x) & (utils.bits(x) & quiet == 0)
    assert f.errors == (sr.FPEnv.INVALID if (np.isinf(x) | snan).any() else 0)
