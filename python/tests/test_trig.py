"""sin/cos accuracy, IEEE specials, math identities and fenv discipline."""

import functools
import math
from collections.abc import Sequence
from itertools import product
from typing import Any, cast

import numpy as np
import numpy_sr as sr
from numpy_sr import ieee754 as ie
from numpy_sr.tests.test_worst import register as _register_worst
from pytest import mark, param, skip


def _bound_low(target, dtype):
    if dtype is np.float32:
        return 2.0
    return 3.5 if target.HAVE_FMA else 4.0


def _bound_high(target, dtype):
    return 0.6 if dtype is np.float32 else 1.0


_BOUNDS = {
    sr.Accuracy.High: _bound_high,
    sr.Accuracy.Low: _bound_low,
}

_OPS = ["sin", "cos"]
_DTYPES = [np.float32, np.float64]
_MIN_BLOCK = 4096


# Enroll sin/cos in the shared frozen worst-case gate (test_worst.py).
for _op in _OPS:
    _register_worst(_op, lambda target, dtype, acc: _BOUNDS[acc](target, dtype))


def _sweeps_binades(dtypes_jobs, subnormals=True):
    """Binades dealt round-robin into `jobs` cases per dtype.

    Splitting serves two ends: the cases run in parallel under `pytest -n`, and
    each books its own worst cases, widening the range the book covers.
    """
    for dtype, jobs in dtypes_jobs:
        bins = ie.binades(dtype, subnormals=subnormals)
        for i in range(jobs):
            yield param(dtype, bins[i::jobs], id=f"{dtype.__name__}-{i}")


def _report_combos(report_ulp, operation, dtype, x, ref, combos, store_worst=None):
    """Score one block against every combo; `ref` is one shared MPFR pass."""
    for target, accuracy in combos:
        computed = getattr(target, operation)(x, accuracy=accuracy)
        bound = _BOUNDS[accuracy](target, dtype)
        report_ulp(
            x,
            computed,
            ref,
            bound,
            accuracy=accuracy,
            target=target,
            store_worst=store_worst,
        )


@functools.cache
def _basic(operation, dtype) -> tuple[np.ndarray, Any]:
    """Named args over the quadrant seams, both signs, +-2 ULP neighbors."""
    fi, pi = np.finfo(dtype), math.pi
    named = [0.0, pi / 6, pi / 4, pi / 3, pi / 2, 2 * pi / 3]
    named += [pi, 3 * pi / 2, 2 * pi, fi.tiny, fi.smallest_subnormal]
    x = ie.both_signs(ie.ulp_neighbors(np.array(named, dtype=dtype)))
    return x, getattr(sr.mpfr, operation)(x)


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
@mark.parametrize("data", [_basic])
def test_simple(report_ulp, target, accuracy, operation, dtype, data):
    op_fn = getattr(target, operation)
    x, ref = data(operation, dtype)
    computed = op_fn(x, accuracy=accuracy)
    report_ulp(x, computed, ref, _BOUNDS[accuracy](target, dtype))


def _seed_key(rng: np.random.Generator) -> tuple:
    """A Generator hashes by identity; its seed is what the data depends on."""
    seed_seq = cast("np.random.SeedSequence", rng.bit_generator.seed_seq)
    entropy = seed_seq.entropy
    return tuple(entropy) if isinstance(entropy, Sequence) else (entropy,)


@functools.cache
def _rand(seed, operation, dtype, binades) -> tuple[np.ndarray, Any]:
    """Cached on `_seed_key`, so every target shares one block and one MPFR pass."""
    x = ie.rand(np.random.default_rng(list(seed)), dtype, binades, 2**16)
    return x, getattr(sr.mpfr, operation)(x)


@mark.parametrize("operation", _OPS)
@mark.parametrize(
    "dtype,binades", list(_sweeps_binades([(np.float32, 2), (np.float64, 8)]))
)
def test_rand(report_ulp, rng, target, accuracy, operation, dtype, binades):
    op_fn = getattr(target, operation)
    x, ref = _rand(_seed_key(rng), operation, dtype, binades)
    if not x.size:
        skip(f"binades lie above the {accuracy.name} cap")
    computed = op_fn(x, accuracy=accuracy)
    report_ulp(x, computed, ref, _BOUNDS[accuracy](target, dtype))


# Adaptive worst-ULP hunt
_HUNT_ROUNDS = 128
_HUNT_POP = 2**16
_HUNT_KEEP = 512
_HUNT_RADIUS = 3


def _hunt_worst(combos, operation, ref_op, rng, dtype, seeds):
    """Hill-climb the ULP landscape: score vs MPFR, keep top-KEEP, re-probe."""
    binades = ie.binades(dtype)
    fresh = ie.rand(rng, dtype, binades, _HUNT_POP, sign="positive")
    pool = np.concatenate([np.asarray(seeds, dtype=dtype), fresh])
    best: list[tuple[np.ndarray, np.ndarray]] = [
        (np.empty(0, dtype), np.empty(0)) for _ in combos
    ]
    for _ in range(_HUNT_ROUNDS):
        pool = np.unique(pool[np.isfinite(pool) & (pool > 0)])
        ref = ref_op(pool)  # one MPFR pass, shared by every combo below
        for i, (target, accuracy) in enumerate(combos):
            op_fn = getattr(target, operation)
            err = np.abs(sr.testing.ulp_error(op_fn(pool, accuracy=accuracy), ref))
            cand_x = np.concatenate([best[i][0], pool])
            cand_e = np.concatenate([best[i][1], err])
            keep = np.argsort(-cand_e, kind="stable")[:_HUNT_KEEP]
            best[i] = cand_x[keep], cand_e[keep]
        centers = np.unique(np.concatenate([bx for bx, _ in best]))
        fresh = ie.rand(rng, dtype, binades, _HUNT_POP, sign="positive")
        pool = np.concatenate([centers, ie.ulp_neighbors(centers, _HUNT_RADIUS), fresh])
    return np.unique(np.concatenate([bx for bx, _ in best]))


@mark.slow
@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", [np.float64])
def test_worst_adaptive(report_ulp, rng, targets, accuracies, operation, dtype):
    """`_hunt_worst` seeded from `argred`'s champions, +-2 ULP window at the end."""
    combos = list(product(targets, accuracies))
    ar = sr.argred.Argred(
        dtype, sr.argred.Grid.PI if operation == "sin" else sr.argred.Grid.PI_HALF
    )
    seeds = [ar.worst_case(e) for e in ar.valid_binades()]

    ref_op = getattr(sr.mpfr, operation)
    champions = _hunt_worst(combos, operation, ref_op, rng, dtype, seeds)

    x = ie.both_signs(ie.ulp_neighbors(champions, radius=2))
    _report_combos(report_ulp, operation, dtype, x, ref_op(x), combos)


@mark.exhaustive
@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype,binades", list(_sweeps_binades([(np.float32, 8)])))
def test_exhaustive(report_ulp, targets, accuracies, operation, dtype, binades):
    combos = list(product(targets, accuracies))  # reused across binades
    for lo, hi in binades:
        x = ie.float_range(dtype, lo, hi)
        ref = getattr(sr.mpfr, operation)(x)
        # the only f32 book worth keeping: this sweep sees every input there is
        _report_combos(report_ulp, operation, dtype, x, ref, combos, store_worst=True)
        del x, ref


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_parity(rng, target, accuracy, operation, dtype):
    """sin odd / cos even: op(-x) must equal +-op(x) bit for bit."""
    op_fn = getattr(target, operation)
    binades = ie.binades(dtype)
    x = ie.rand(rng, dtype, binades, _MIN_BLOCK, sign="positive", subnormal=0.25)
    pos = op_fn(x, accuracy=accuracy)
    neg = op_fn(-x, accuracy=accuracy)
    want = -pos if operation == "sin" else pos
    sr.testing.assert_bits(neg, want, -x, f"{operation}(-x) parity")


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_tiny(rng, target, accuracy, operation, dtype):
    """Below the round-to-1 branch, sin(x) == x and cos(x) == 1.0 bit-exactly."""
    # The stricter High round-to-1 threshold, |x| <= half ulp(pi/2).
    cap = {np.float32: 2.0**-24, np.float64: 2.0**-53}[dtype]
    tiny = np.finfo(dtype).smallest_subnormal
    x = ie.rand(rng, dtype, [(tiny, cap)], _MIN_BLOCK, subnormal=0.5)
    got = getattr(target, operation)(x, accuracy=accuracy)
    want = x if operation == "sin" else np.ones_like(x)
    sr.testing.assert_bits(got, want, x, f"{operation} tiny identity")


@mark.parametrize("operation", _OPS)
@mark.parametrize("dtype", _DTYPES)
def test_special(rng, target, accuracy, operation, dtype):
    """Bit-exact IEEE specials scattered through a normal block."""
    specials = dict.fromkeys(("zero", "subnormal", "qnan", "snan", "inf"), 0.15)
    x = ie.rand(rng, dtype, ie.binades(dtype), _MIN_BLOCK, **specials)
    got = getattr(target, operation)(x, accuracy=accuracy)

    # sub-tiny lanes (+-0, subnormals): sin preserves, cos snaps to 1.0.
    small = np.isfinite(x) & (np.abs(x) < np.finfo(dtype).tiny)
    want = x[small] if operation == "sin" else np.ones(int(small.sum()), dtype)
    sr.testing.assert_bits(got[small], want, x[small], f"{operation} sub-tiny")

    # inf / qNaN / sNaN all collapse to a quiet NaN.
    nonfinite = ~np.isfinite(x)
    assert np.isnan(got[nonfinite]).all(), f"{operation}: non-finite input -> non-NaN"


# `rand` configs; INVALID is raised solely for inf, so the expected outcome is
# read off the block rather than tabulated per case.
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
def test_fp_errors(rng, target, accuracy, operation, dtype, case):
    """fenv discipline: spurious flags suppressed, INVALID set only for inf."""
    if target.IS_REFERENCE:
        skip("kernel fenv not applicable for reference targets")
    x = ie.rand(rng, dtype, ie.binades(dtype), _MIN_BLOCK, **_FLAG_CASES[case])
    with sr.FPEnv() as f:
        getattr(target, operation)(x, accuracy=accuracy)
    assert f.errors == (sr.FPEnv.INVALID if np.isinf(x).any() else 0)
