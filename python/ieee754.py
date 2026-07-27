"""Bit-level helpers for exhaustive IEEE-754 testing."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

Float = np.floating[Any]
FloatArray = npt.NDArray[Float]
# A magnitude bound: numpy float scalar or plain float.
Bound = float | Float
Binades = Sequence[tuple[Bound, Bound]]

_SIGNED = {np.dtype(np.float32): np.int32, np.dtype(np.float64): np.int64}


def _float_dtype(dtype: npt.DTypeLike) -> np.dtype[np.floating[Any]]:
    """Narrow ``DTypeLike`` to a concrete float dtype for type-checkers."""
    return cast("np.dtype[np.floating[Any]]", np.dtype(dtype))


def bits(a: np.ndarray) -> np.ndarray:
    """Reinterpret floats as same-width unsigned ints for bit-exact compares."""
    a = np.asarray(a)
    return a.view(np.dtype(f"uint{a.dtype.itemsize * 8}"))


def float_range(
    dtype: npt.DTypeLike,
    lo: float | Float,
    hi: float | Float,
) -> FloatArray:
    """Every representable value in [0 <= lo, hi], ascending and inclusive."""
    dt = _float_dtype(dtype)
    uint = np.dtype(f"uint{dt.itemsize * 8}")
    a = int(np.asarray(lo, dt).view(uint))
    b = int(np.asarray(hi, dt).view(uint))
    if a > b:
        raise ValueError("need lo <= hi, and both non-negative")
    return np.arange(a, b + 1, dtype=uint).view(dt)


@functools.cache
def binades(
    dtype: npt.DTypeLike,
    step: int = 1,
    subnormals: bool = False,
) -> tuple[tuple[Float, Float], ...]:
    """(lo, hi) pairs covering the finite positives, `step` binades per pair."""
    # Tuple, not list: the result is cached and handed to every caller.
    dt = _float_dtype(dtype)
    fi = np.finfo(dt)  # raises on non-float dtypes
    uint = np.dtype(f"uint{dt.itemsize * 8}")
    bias = 1 - fi.minexp

    lows = [(e + bias) << fi.nmant for e in range(fi.minexp, fi.maxexp, step)]
    if subnormals:
        lows.insert(0, 0)  # +0.0
    highs = [*lows[1:], (fi.maxexp + bias) << fi.nmant]  # sentinel is +inf
    highs = [b - 1 for b in highs]  # one ULP below

    def as_float(bits: list[int]) -> FloatArray:
        return np.array(bits, uint).view(dt)

    return tuple(zip(as_float(lows), as_float(highs), strict=True))


def both_signs(x: np.ndarray) -> np.ndarray:
    """Mirror to both signs and drop the non-finites bit-stepping can spawn."""
    x = np.concatenate([x, -x])
    return x[np.isfinite(x)]


def ulp_neighbors(x: np.ndarray, radius: int = 2) -> np.ndarray:
    """Each finite positive x plus its +-radius ULP neighbors, by bit stepping."""
    i = np.ascontiguousarray(x).view(_SIGNED[x.dtype])
    off = np.arange(-radius, radius + 1, dtype=i.dtype)
    return (i[:, None] + off).ravel().view(x.dtype)


def _uint(dtype: npt.DTypeLike) -> np.dtype:
    return np.dtype(f"uint{np.dtype(dtype).itemsize * 8}")


def _neg(rng: np.random.Generator, n: int, sign: str) -> np.ndarray:
    """`n`-long 0/1 is-negative mask per `sign` (positive/negative/mixed)."""
    if sign == "mixed":
        return rng.integers(0, 2, size=n)
    if sign == "positive":
        return np.zeros(n, dtype=np.int64)
    if sign == "negative":
        return np.ones(n, dtype=np.int64)
    raise ValueError(f"sign must be 'positive', 'negative', or 'mixed', got {sign!r}")


def _signs(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """`n` sign bits already shifted into the MSB, as uints, per `sign`."""
    u = _uint(dtype)
    return _neg(rng, n, sign).astype(u) << (np.finfo(dtype).bits - 1)


def _normals(
    rng: np.random.Generator,
    dtype: npt.DTypeLike,
    n: int,
    binades: Binades,
    sign: str,
) -> np.ndarray:
    """`n` floats log-uniform over `binades`; `lo == 0` clamps to subnormal."""
    tiny = float(np.finfo(dtype).smallest_subnormal)
    lows = np.maximum([float(lo) for lo, _ in binades], tiny)
    highs = np.array([float(hi) for _, hi in binades])
    idx = rng.integers(0, len(binades), size=n)
    e = rng.uniform(np.log2(lows[idx]), np.log2(highs[idx]))
    s = np.where(_neg(rng, n, sign), -1.0, 1.0)
    return (s * 2.0**e).astype(dtype)


def _subnormals(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """Nonzero subnormals: random mantissa in [1, 2^nmant)."""
    u = _uint(dtype)
    m = rng.integers(1, 1 << np.finfo(dtype).nmant, size=n, dtype=u)
    return (m | _signs(rng, dtype, n, sign)).view(dtype)


def _qnans(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """Quiet NaNs: all-ones exponent, quiet bit set, random payload."""
    fi = np.finfo(dtype)
    u = _uint(dtype)
    quiet = u.type(1) << (fi.nmant - 1)
    exp = ((u.type(1) << (fi.bits - 1 - fi.nmant)) - 1) << fi.nmant
    payload = rng.integers(0, quiet, size=n, dtype=u)
    return (exp | quiet | payload | _signs(rng, dtype, n, sign)).view(dtype)


def _snans(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """Signaling NaNs: quiet bit clear, nonzero payload."""
    fi = np.finfo(dtype)
    u = _uint(dtype)
    quiet = u.type(1) << (fi.nmant - 1)
    exp = ((u.type(1) << (fi.bits - 1 - fi.nmant)) - 1) << fi.nmant
    payload = rng.integers(1, quiet, size=n, dtype=u)
    return (exp | payload | _signs(rng, dtype, n, sign)).view(dtype)


def _infs(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """+-inf, by bit assembly."""
    fi = np.finfo(dtype)
    u = _uint(dtype)
    exp = ((u.type(1) << (fi.bits - 1 - fi.nmant)) - 1) << fi.nmant
    return (exp | _signs(rng, dtype, n, sign)).view(dtype)


def _zeros(
    rng: np.random.Generator, dtype: npt.DTypeLike, n: int, sign: str
) -> np.ndarray:
    """+-0.0, by bit assembly."""
    return _signs(rng, dtype, n, sign).view(dtype)


_SPECIALS = {
    "subnormal": _subnormals,
    "zero": _zeros,
    "qnan": _qnans,
    "snan": _snans,
    "inf": _infs,
}


def rand(
    rng: np.random.Generator,
    dtype: npt.DTypeLike,
    binades: Binades,
    n: int,
    *,
    sign: str = "mixed",
    subnormal: float = 0.0,
    zero: float = 0.0,
    qnan: float = 0.0,
    snan: float = 0.0,
    inf: float = 0.0,
) -> FloatArray:
    """`n` shuffled values: log-uniform normals, each keyword its block share."""
    dt = _float_dtype(dtype)
    fracs = {
        "subnormal": subnormal,
        "zero": zero,
        "qnan": qnan,
        "snan": snan,
        "inf": inf,
    }
    parts = []
    used = 0
    for name, frac in fracs.items():
        if frac <= 0.0:
            continue
        k = min(round(frac * n), n - used)
        if k <= 0:
            continue
        parts.append(_SPECIALS[name](rng, dt, k, sign))
        used += k
    if used < n:
        parts.append(_normals(rng, dt, n - used, binades, sign))
    out = np.concatenate(parts).astype(dt, copy=False)
    rng.shuffle(out)
    return out
