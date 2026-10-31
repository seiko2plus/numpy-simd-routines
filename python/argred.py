"""Per binade, the float nearest a grid point: trig reduction worst cases.

All arithmetic is exact in integers: a value v is held as v * mod, with
mod = 2*pi*2**1300 (pi truncated), so "mod 1" becomes ``% mod``.
"""

from __future__ import annotations

import enum
import functools
import math
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = ["PI_1300", "Argred", "Grid"]

#: floor(pi * 2**1300)
PI_1300 = int(
    "3243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c8"
    "9452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b547091"
    "79216d5d98979fb1bd1310ba698dfb5ac2ffd72dbd01adfb7b8e1afed6a267e9"
    "6ba7c9045f12c7f9924a19947b3916cf70801f2e2858efc16636920d871574e6"
    "9a458fea3f4933d7e0d95748f728eb658718bcd5882154aee7b54a41dc25a59b"
    "59c30d",
    16,
)

# Below 2**1024 the truncation shifts x*n/pi by < 2**-276, far under the
# ~2**-62 gap of the closest double, so the champions are exact.
_PI_BITS = 1300


class Grid(enum.Enum):
    """Reduction grid (n, h): grid points are (k + h/2)*pi/n."""

    PI = (1, 0)  # k*pi -- sin roots
    PI_HALF = (1, 1)  # (k+1/2)*pi -- cos roots
    PI_16_HALF = (16, 1)  # (k+1/2)*pi/16 -- pi/16 seams
    PI_128 = (128, 0)  # k*pi/128 -- f32 Extended table centres
    PI_128_HALF = (128, 1)  # f32 Extended table seams
    PI_256 = (256, 0)  # k*pi/256 -- f64 Extended table centres
    PI_256_HALF = (256, 1)  # f64 Extended table seams

    @property
    def n(self) -> int:
        return self.value[0]

    @property
    def h(self) -> int:
        return self.value[1]


def _cdiv(a: int, b: int) -> int:
    """ceil(a / b)."""
    return -(-a // b)


def _linear(grid: Grid, e: int, p: int) -> tuple[int, int, int]:
    """(a, b, mod): x*n/pi - h/2 == (m*a - b)/mod for x = m * 2**(e-p)."""
    return (2 * grid.n) << (_PI_BITS + e - p), grid.h * PI_1300, 2 * PI_1300


def _distance(m: int, a: int, b: int, mod: int) -> int:
    """mod times the distance of (m*a - b)/mod to the nearest integer."""
    r = (m * a - b) % mod
    return min(r, mod - r)


def _min_frac(a: int, b: int, n: int, mod: int) -> int:
    """The t in 0 <= t < n minimizing (b - t*a) % mod, exactly.

    Lefèvre's algorithm (Lefèvre, Muller, Tisserand, "Toward Correctly Rounded
    Transcendentals", IEEE Trans. Comput. 47(11), 1998), adapted from
    ``AlgoLefevre`` in CORE-MATH's ``src/binary64/sin/sin.sage`` (MIT).
    """
    x = a % mod
    y = mod - x
    d = b % mod
    u = v = 1
    t = 0
    while x > 0:
        if d >= x:
            m = min(d // x, (n - 1 - t) // v)  # every record reachable with step v
            d -= m * x
            t += m * v
            if d >= x:
                break
        if v > n - 1 - t:
            break
        if y > x:  # refine the above-step below x so the below-ladder can extend
            k = _cdiv(y - x, x)
            y -= k * x
            u += k * v
        # x - i*y stays a valid step only while positive
        i = min(_cdiv(x - d, y), _cdiv(x, y) - 1)
        if i <= 0:
            break
        x -= i * y
        v += i * u
    return t


def _binade_has_point(e: int, grid: Grid, emax: int) -> bool:
    """Whether [2**(e-1), 2**e) holds a positive grid point."""
    if not 3 - emax <= e <= emax:
        return False
    a, b, mod = _linear(grid, e, 1)
    k = _cdiv(a - b, mod)  # first k with x >= 2**(e-1)
    return k * mod < 2 * a - b


@functools.cache
def _champion(
    e: int, grid: Grid, precision: int, emax: int, slices: int = 1, i: int = 0
) -> int:
    """Mantissa offset t of the float closest to a `grid` point in slice `i` of
    [2**(e-1), 2**e) cut into `slices` equal parts; the float is
    (2**(precision-1) + t) * 2**(e-precision)."""
    if not _binade_has_point(e, grid, emax):
        raise ValueError(
            f"binade [{2 ** (e - 1)}, {2**e}) has no {grid.name} grid point"
        )
    a, b, mod = _linear(grid, e, precision)
    lead = 1 << (precision - 1)
    n = lead // slices
    t0 = i * n
    base = (lead + t0) * a - b
    below = t0 + _min_frac(a, -base, n, mod)
    above = t0 + _min_frac(-a, base, n, mod)
    t = min(below, above, key=lambda t: _distance(lead + t, a, b, mod))
    assert t0 <= t < t0 + n
    # A true champion sits within ~2**-(p-1) of the grid, far under this slack.
    # A slice can lie far from every grid point, so only the binade's is checked.
    assert slices > 1 or _distance(lead + t, a, b, mod) << precision < mod << 13
    return t


class Argred:
    """Worst-case reduction inputs for one float type and grid.

    :param dtype: float32 or float64.
    :param grid: the reduction grid.
    :param precision: mantissa bits incl. the implicit one; defaults to dtype's.
    :param emax: top binade exponent; defaults to dtype's. A smaller precision
        or emax gives a toy format, whose floats are exact in dtype.
    """

    def __init__(
        self,
        dtype: npt.DTypeLike,
        grid: Grid = Grid.PI,
        *,
        precision: int | None = None,
        emax: int | None = None,
    ) -> None:
        self.grid = grid
        # DTypeLike widens to non-float dtypes; narrow to floating for finfo/.type.
        self.dtype = cast("np.dtype[np.floating[Any]]", np.dtype(dtype))
        fi = np.finfo(self.dtype)
        fp, femax = int(fi.nmant) + 1, int(fi.maxexp)
        self.precision = fp if precision is None else precision
        self._emax = femax if emax is None else emax
        if self.precision > fp or self._emax > femax:
            raise ValueError(f"precision or emax beyond {self.dtype.name}'s")

    def _floats(self, e: int, t: list[int]) -> np.ndarray:
        """(2**(p-1) + t) * 2**(e-p), in dtype."""
        p = self.precision
        m = np.array(t, self.dtype) + self.dtype.type(1 << (p - 1))
        return np.ldexp(m, e - p)

    def valid_binades(self) -> Iterator[int]:
        """Binade exponents whose [2**(e-1), 2**e) contains a grid point."""
        return (
            e
            for e in range(3 - self._emax, self._emax + 1)
            if _binade_has_point(e, self.grid, self._emax)
        )

    def worst_case(self, e: int) -> float:
        """Closest float to a grid point in binade `e` (from `valid_binades`).

        :param e: binade exponent, from :meth:`valid_binades`.
        """
        t = _champion(e, self.grid, self.precision, self._emax)
        return math.ldexp((1 << (self.precision - 1)) + t, e - self.precision)

    def worst_cases(self, e: int, slices: int) -> np.ndarray:
        """Closest float to a grid point in each of `slices` equal parts of
        binade `e`. One of them is :meth:`worst_case`.

        :param e: binade exponent, from :meth:`valid_binades`.
        :param slices: a power of two, at most the floats in a binade.
        """
        if slices & (slices - 1) or not 0 < slices < 1 << self.precision:
            raise ValueError(f"slices must be a power of two, got {slices}")
        t = [
            _champion(e, self.grid, self.precision, self._emax, slices, i)
            for i in range(slices)
        ]
        return self._floats(e, t)

    def nearest(self, e: int) -> np.ndarray:
        """Every grid point that rounds to nearest into binade `e`, rounded.

        The closest float to the grid is one of them or an end of the binade.

        :param e: binade exponent, from :meth:`valid_binades`.
        """
        p = self.precision
        a, b, mod = _linear(self.grid, e, p)
        lead = 1 << (p - 1)
        # point k is (k*mod + b)/a ulps of binade e; it rounds into the binade
        # iff lead - 1/4 <= that < 2*lead - 1/2
        k0 = _cdiv((4 * lead - 1) * a - 4 * b, 4 * mod)
        k1 = _cdiv((8 * lead - 2) * a - 4 * b, 4 * mod)
        t = [(2 * (k * mod + b) + a) // (2 * a) - lead for k in range(k0, k1)]
        return self._floats(e, t)
