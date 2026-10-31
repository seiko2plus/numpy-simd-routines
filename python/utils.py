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


def _float_dtype(dtype: npt.DTypeLike) -> np.dtype[np.floating[Any]]:
    """Narrow ``DTypeLike`` to a concrete float dtype for type-checkers."""
    return cast("np.dtype[np.floating[Any]]", np.dtype(dtype))


def bits_dtype(dtype: npt.DTypeLike, signed: bool = False) -> np.dtype:
    """The int dtype as wide as `dtype`.

    :param dtype: float32 or float64.
    :param signed: signed instead of unsigned.
    """
    return np.dtype(f"{'i' if signed else 'u'}{np.dtype(dtype).itemsize}")


def bits(a: npt.ArrayLike, signed: bool = False) -> np.ndarray:
    """Reinterpret floats as same-width ints, for bit-exact compares.

    :param a: float32 or float64 array.
    :param signed: signed instead of unsigned ints.
    """
    a = np.asarray(a)
    return a.view(bits_dtype(a.dtype, signed))


def from_bits(b: npt.ArrayLike) -> FloatArray:
    """Reinterpret ints as same-width floats; :func:`bits` inverted.

    :param b: 32- or 64-bit int array.
    """
    b = np.asarray(b)
    return b.view(_float_dtype(f"f{b.dtype.itemsize}"))


def float_range(
    dtype: npt.DTypeLike,
    lo: float | Float,
    hi: float | Float,
) -> FloatArray:
    """Every representable value in [0 <= lo, hi], ascending and inclusive.

    :param dtype: float32 or float64.
    :param lo: smallest value, >= 0.
    :param hi: largest value, >= ``lo``.
    """
    dt = _float_dtype(dtype)
    a = int(bits(np.asarray(lo, dt)))
    b = int(bits(np.asarray(hi, dt)))
    if a > b:
        raise ValueError("need lo <= hi, and both non-negative")
    return from_bits(np.arange(a, b + 1, dtype=bits_dtype(dt)))


@functools.cache
def binades(
    dtype: npt.DTypeLike,
    step: int = 1,
    subnormals: bool = False,
) -> tuple[tuple[Float, Float], ...]:
    """(lo, hi) pairs covering the finite positives, `step` binades per pair.

    :param dtype: float32 or float64.
    :param step: binades per pair.
    :param subnormals: start with a pair for +0 and the subnormals.
    """
    # Tuple, not list: the result is cached and handed to every caller.
    dt = _float_dtype(dtype)
    fi = np.finfo(dt)  # raises on non-float dtypes
    bias = 1 - fi.minexp

    lows = [(e + bias) << fi.nmant for e in range(fi.minexp, fi.maxexp, step)]
    if subnormals:
        lows.insert(0, 0)  # +0.0
    highs = [*lows[1:], (fi.maxexp + bias) << fi.nmant]  # sentinel is +inf
    highs = [b - 1 for b in highs]  # one ULP below

    def as_float(b: list[int]) -> FloatArray:
        return from_bits(np.array(b, bits_dtype(dt)))

    return tuple(zip(as_float(lows), as_float(highs), strict=True))


def both_signs(x: np.ndarray) -> np.ndarray:
    """Mirror to both signs and drop the non-finites bit-stepping can spawn.

    :param x: float array.
    """
    x = np.concatenate([x, -x])
    return x[np.isfinite(x)]


def ulp_neighbors(x: np.ndarray, radius: int = 2) -> np.ndarray:
    """Each finite positive x plus its +-radius ULP neighbors, by bit stepping.

    :param x: finite positive float32 or float64 array.
    :param radius: neighbors on each side, in ULP.
    """
    i = bits(x, signed=True)
    off = np.arange(-radius, radius + 1, dtype=i.dtype)
    return from_bits((i[:, None] + off).ravel())


def _neg(rng: np.random.Generator, n: int, sign: int) -> np.ndarray:
    """`n`-long 0/1 is-negative mask per `sign`."""
    if sign == Rand.MIXED:
        return rng.integers(0, 2, size=n)
    return np.full(n, sign == Rand.NEGATIVE, dtype=np.int64)


def _normals(
    rng: np.random.Generator,
    dtype: np.dtype,
    n: int,
    binades: Binades,
    sign: int,
) -> np.ndarray:
    """`n` floats log-uniform over `binades`; `lo == 0` clamps to subnormal."""
    tiny = float(np.finfo(dtype).smallest_subnormal)
    lows = np.maximum([float(lo) for lo, _ in binades], tiny)
    highs = np.array([float(hi) for _, hi in binades])
    idx = rng.integers(0, len(binades), size=n)
    e = rng.uniform(np.log2(lows[idx]), np.log2(highs[idx]))
    s = np.where(_neg(rng, n, sign), -1.0, 1.0)
    return (s * 2.0**e).astype(dtype)


def _special(
    rng: np.random.Generator, dtype: np.dtype, n: int, sign: int, kind: str
) -> np.ndarray:
    """`n` values of `kind` by bit assembly: fixed bits | random payload | sign."""
    fi = np.finfo(dtype)
    u = bits_dtype(dtype)
    exp = ((1 << (fi.bits - 1 - fi.nmant)) - 1) << fi.nmant
    quiet = 1 << (fi.nmant - 1)
    # kind -> (fixed bits, payload range or None)
    fixed, payload = {
        "subnormal": (0, (1, 1 << fi.nmant)),
        "zero": (0, None),
        "qnan": (exp | quiet, (0, quiet)),
        "snan": (exp, (1, quiet)),
        "inf": (exp, None),
    }[kind]
    bits = np.full(n, fixed, dtype=u)
    if payload:
        bits |= rng.integers(*payload, size=n, dtype=u)
    bits |= _neg(rng, n, sign).astype(u) << (fi.bits - 1)
    return from_bits(bits)


class Rand:
    """A seeded generator of float inputs.

    Hashes and compares by the seed, so a test can cache on it. Each call
    continues one stream.

    :param seed: the seed, as for `numpy.random.default_rng`.
    """

    # The signs to draw.
    POSITIVE = 1
    NEGATIVE = -1
    MIXED = 0

    def __init__(self, *seed: int) -> None:
        self.seed = seed
        self._gen = np.random.default_rng(list(seed))

    def __repr__(self) -> str:
        return f"Rand{self.seed}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Rand) and self.seed == other.seed

    def __hash__(self) -> int:
        return hash(self.seed)

    def __call__(
        self,
        dtype: npt.DTypeLike,
        binades: Binades,
        n: int,
        *,
        sign: int = MIXED,
        subnormal: float = 0.0,
        zero: float = 0.0,
        qnan: float = 0.0,
        snan: float = 0.0,
        inf: float = 0.0,
    ) -> FloatArray:
        """`n` shuffled values: log-uniform normals, each keyword its block share.

        :param dtype: float32 or float64.
        :param binades: (lo, hi) magnitude ranges for the normals, as from
            :func:`binades`. ``lo == 0`` clamps to the smallest subnormal.
        :param n: number of values.
        :param sign: `POSITIVE`, `NEGATIVE` or `MIXED`.
        :param subnormal: share of ``n`` that is subnormal; ``zero``, ``qnan``,
            ``snan`` and ``inf`` likewise.
        """
        rng, dt = self._gen, _float_dtype(dtype)
        fracs = {
            "subnormal": subnormal,
            "zero": zero,
            "qnan": qnan,
            "snan": snan,
            "inf": inf,
        }
        parts = []
        used = 0
        for kind, frac in fracs.items():
            k = min(round(frac * n), n - used) if frac > 0.0 else 0
            if k > 0:
                parts.append(_special(rng, dt, k, sign, kind))
                used += k
        if used < n:
            parts.append(_normals(rng, dt, n - used, binades, sign))
        out = np.concatenate(parts).astype(dt, copy=False)
        rng.shuffle(out)
        return out
