"""argred against a published worst case and brute force."""

import math
from fractions import Fraction

import numpy as np
from numpy_sr.argred import PI_1300, Argred, Grid
from pytest import mark

_PI = Fraction(PI_1300, 1 << 1300)


def _distance(x: Fraction, grid: Grid) -> Fraction:
    """Distance of x*n/pi - h/2 to the nearest integer."""
    r = x * grid.n / _PI - Fraction(grid.h, 2)
    r -= math.floor(r)
    return min(r, 1 - r)


def _round(x: Fraction, p: int) -> Fraction:
    """x rounded to nearest at `p` bits."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    e += x >= Fraction(2) ** e
    ulp = Fraction(2) ** (e - p)
    return round(x / ulp) * ulp


def test_muller_worst_case():
    # The double closest to a multiple of pi/2 (Muller, "Elementary Functions").
    x = 6381956970095103 * 2.0**797
    e = int(np.frexp(x)[1])
    assert Argred(np.float64, Grid.PI_HALF).worst_case(e) == x


@mark.parametrize("precision", range(4, 11))
@mark.parametrize("grid", list(Grid), ids=lambda g: g.name)
def test_champion_brute_force(grid, precision):
    """At a toy precision every mantissa, and every slice, can be tried."""
    p = precision
    ar = Argred(np.float64, grid, precision=p, emax=128)
    lead = 1 << (p - 1)
    for e in ar.valid_binades():
        ulp = Fraction(2) ** (e - p)
        dist = {m: _distance(m * ulp, grid) for m in range(lead, 2 * lead)}
        for slices in (1, 4):
            n = lead // slices
            want = [
                math.ldexp(min(range(m, m + n), key=dist.__getitem__), e - p)
                for m in range(lead, 2 * lead, n)
            ]
            got = ar.worst_cases(e, slices).tolist()
            assert got == want, f"binade {e}, {slices} slices"
            if slices == 1:
                assert ar.worst_case(e) == want[0], f"binade {e}"


@mark.parametrize("precision", range(4, 11))
@mark.parametrize("grid", list(Grid), ids=lambda g: g.name)
def test_nearest_brute_force(grid, precision):
    """Every grid point below a small binade's top, rounded one by one."""
    ar = Argred(np.float64, grid, precision=precision, emax=16)
    for e in ar.valid_binades():
        if grid.n * 2.0**e > 1 << 12:
            break
        lo, hi = Fraction(2) ** (e - 1), Fraction(2) ** e
        k1 = math.floor(hi * grid.n / _PI) + 1
        pts = (
            _round((k + Fraction(grid.h, 2)) * _PI / grid.n, precision)
            for k in range(k1)
        )
        want = [float(x) for x in pts if lo <= x < hi]
        assert ar.nearest(e).tolist() == want, f"binade {e}"
