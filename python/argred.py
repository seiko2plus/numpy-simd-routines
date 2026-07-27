"""Per binade, the float nearest a grid point: trig reduction worst cases."""

from __future__ import annotations

import contextlib
import enum
import functools
import math
from collections.abc import Iterator
from fractions import Fraction
from typing import Any, cast

import numpy as np
import numpy.typing as npt

__all__ = ["Argred", "Grid"]

# floor(pi * 2**1300).
_PI_1300 = int(
    "3243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c8"
    "9452821e638d01377be5466cf34e90c6cc0ac29b7c97c50dd3f84d5b5b547091"
    "79216d5d98979fb1bd1310ba698dfb5ac2ffd72dbd01adfb7b8e1afed6a267e9"
    "6ba7c9045f12c7f9924a19947b3916cf70801f2e2858efc16636920d871574e6"
    "9a458fea3f4933d7e0d95748f728eb658718bcd5882154aee7b54a41dc25a59b"
    "59c30d",
    16,
)

# 2**1300 / _PI_1300 == 1/pi, exactly representable as a rational.
_INV_PI = Fraction(1 << 1300, _PI_1300)
_INV_2PI = _INV_PI / 2  # 1/(2*pi)


class Grid(enum.Enum):
    """Reduction grid (inv, offset): grid points are (k + offset)/inv."""

    PI = (_INV_PI, Fraction(0))  # k*pi -- sin roots
    PI_HALF = (_INV_PI, Fraction(1, 2))  # (k+1/2)*pi -- cos roots
    TWO_PI = (_INV_2PI, Fraction(0))  # k*2*pi -- full period

    @property
    def inv(self) -> Fraction:
        return self.value[0]

    @property
    def offset(self) -> Fraction:
        return self.value[1]


def _frac(q: Fraction) -> Fraction:
    """Fractional part in [0, 1)."""
    return q - math.floor(q)


def _ceil(q: Fraction) -> int:
    return -int(-q // 1)


def _min_frac(a: Fraction, b: Fraction, n: int) -> tuple[Fraction, int]:
    """Exact min of {b - t*a mod 1} over 0 <= t < n; returns (d, t)."""
    x = _frac(a)
    y = 1 - x
    d = _frac(b)
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
            k = _ceil((y - x) / x)
            y -= k * x
            u += k * v
        # x - i*y stays a valid step only while positive
        i = min(_ceil((x - d) / y), _ceil(x / y) - 1)
        if i <= 0:
            break
        x -= i * y
        v += i * u
    return d, t


def _grid_distance(t: int, e: int, inv: Fraction, offset: Fraction, p: int) -> Fraction:
    """Distance of x*inv to the grid Z + offset (x = binade e, mantissa t)."""
    x = Fraction((1 << (p - 1)) + t) * Fraction(2) ** (e - p)
    r = _frac(x * inv - offset)
    return min(r, 1 - r)


def _binade_candidate(
    e: int, inv: Fraction, offset: Fraction, above: bool, p: int
) -> int:
    """Mantissa offset minimizing x*inv's distance below (or above) the grid."""
    a = _frac(Fraction(2) ** (e - p) * inv)
    base = Fraction(2) ** (e - 1) * inv - offset
    b = _frac(-base)
    if above:
        a, b = _frac(-a), _frac(-b)
    _, t = _min_frac(a, b, 1 << (p - 1))
    return t


def _binade_has_point(e: int, inv: Fraction, offset: Fraction, emax: int) -> bool:
    """Whether [2**(e-1), 2**e) holds a positive grid point (k + offset)/inv."""
    if not 1 <= e <= emax:
        return False
    k = _ceil(Fraction(2) ** (e - 1) * inv - offset)  # first k with x >= 2**(e-1)
    return k + offset < Fraction(2) ** e * inv


@functools.cache
def _champion_bits(e: int, grid: Grid, precision: int, emax: int) -> int:
    """IEEE bits of the float in [2**(e-1), 2**e) closest to a `grid` point."""
    inv, offset = grid.inv, grid.offset
    if not _binade_has_point(e, inv, offset, emax):
        raise ValueError(
            f"binade [{2 ** (e - 1)}, {2**e}) has no {grid.name} grid point"
        )
    t = min(
        (
            _binade_candidate(e, inv, offset, above, precision)
            for above in (False, True)
        ),
        key=lambda t: _grid_distance(t, e, inv, offset, precision),
    )
    # A true champion sits within ~2**-(p-1) of the grid, far under this slack.
    assert 0 <= t < 1 << (precision - 1)
    assert _grid_distance(t, e, inv, offset, precision) < Fraction(2) ** (
        13 - precision
    )
    return ((e + emax - 2) << (precision - 1)) | t


class Argred:
    """Worst-case reduction inputs for one float type and grid."""

    def __init__(self, dtype: npt.DTypeLike, grid: Grid = Grid.PI) -> None:
        self.grid = grid
        # DTypeLike widens to non-float dtypes; narrow to floating for finfo/.type.
        self.dtype = cast("np.dtype[np.floating[Any]]", np.dtype(dtype))
        fi = np.finfo(self.dtype)
        self.precision = int(fi.nmant) + 1  # incl. implicit bit
        self._emax = int(fi.maxexp)
        self._uint = np.dtype(f"uint{self.dtype.itemsize * 8}")

    def valid_binades(self) -> Iterator[int]:
        """Binade exponents whose [2**(e-1), 2**e) contains a grid point."""
        inv, offset = self.grid.inv, self.grid.offset
        return (
            e
            for e in range(1, self._emax + 1)
            if _binade_has_point(e, inv, offset, self._emax)
        )

    def worst_case(self, e: int) -> float:
        """Closest float to a grid point in binade `e` (from `valid_binades`)."""
        bits = _champion_bits(e, self.grid, self.precision, self._emax)
        return float(np.array(bits, self._uint).view(self.dtype))

    def grid_points(self, e: int, budget: int | None = None) -> list[float]:
        """Nearest float to each grid point in binade `e`; `budget` samples evenly."""
        inv, offset = self.grid.inv, self.grid.offset
        if not 1 <= e <= self._emax:
            return []
        klo = _ceil(Fraction(2) ** (e - 1) * inv - offset)
        khi = _ceil(Fraction(2) ** e * inv - offset)  # exclusive
        if budget is not None and khi - klo > budget:
            span = khi - 1 - klo
            ks: range | list[int] = (
                [klo]
                if budget < 2
                else [klo + span * i // (budget - 1) for i in range(budget)]
            )
        else:
            ks = range(klo, khi)
        pts = []
        to_dtype = self.dtype.type
        with np.errstate(over="ignore"):
            for k in ks:
                x = (k + offset) / inv  # exact grid position
                if x <= 0:
                    continue
                # overflow past this type's finite max: no window there, skip.
                with contextlib.suppress(OverflowError):
                    xr = to_dtype(float(x))
                    if np.isfinite(xr):
                        pts.append(float(xr))
        return pts
