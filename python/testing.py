"""Fractional-ULP error vs the exact value; correct rounding scores ~0.5.

``ulp.cpp`` maps each float to an integer key that orders like the float
(sign and magnitude to two's complement). Then:

.. code-block:: text

   error = key(computed) - key(ref) + residual
         = (computed - ref) + (ref - exact)
         = computed - exact          (in ULP)

Special cases: NaN against NaN and inf against the same inf score 0. Any
other mismatch of NaN or inf scores ``+inf``. f32 lanes are widened to f64 for
the arithmetic. The comparator is itself compiled per target and runs on the
best one the CPU supports.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy_sr as sr


def _operands(computed: Any, ref: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(computed, ref, residual)` broadcast; `ref` may be a `(ref, residual)` pair."""
    ref, residual = ref if isinstance(ref, tuple) and len(ref) == 2 else (ref, 0.0)
    a = np.asarray(computed)
    # sr.ulp_error is dtype-exact (SafeArg); only layout is its problem.
    a, b = np.broadcast_arrays(a, np.asarray(ref).astype(a.dtype, copy=False))
    return a, b, np.broadcast_to(np.asarray(residual, dtype=np.float64), a.shape)


def ulp_error(computed: Any, ref: Any) -> np.ndarray:
    """Signed fractional-ULP error.

    :param computed: the result to score.
    :param ref: the oracle's ``(ref, residual)`` tuple, or a plain array
        (residual 0) for comparisons that aren't against MPFR.
    """
    a, b, res = _operands(computed, ref)
    return sr.ulp_error(a, b, res if res.any() else None)


def _num(v: Any) -> float | str:
    """JSON-safe number: non-finite floats become tokens strict JSON accepts."""
    f = float(v)
    if math.isfinite(f):
        return f
    return "nan" if math.isnan(f) else ("inf" if f > 0 else "-inf")


def _fmt(v: np.ndarray) -> dict[str, Any]:
    """A float element as both its decimal value and exact IEEE bit pattern."""
    bits = int(sr.utils.bits(v))
    return {"value": _num(v), "bits": f"0x{bits:0{v.dtype.itemsize * 2}x}"}


def measure(
    computed: Any,
    ref: Any,
    maxulp: float,
    *,
    x: Any,
    label: str = "",
    limit: int = 16,
    **meta: Any,
) -> dict[str, Any]:
    """Compare and build a record without asserting; extra kwargs go in it.

    :param computed: the result to score.
    :param ref: as for :func:`ulp_error`.
    :param maxulp: the bound, in ULP.
    :param x: the input, or a tuple with one array per argument; shown for the
        worst elements.
    :param label: names the record in failures.
    :param limit: number of worst elements kept.
    :param meta: extra fields stored in the record.
    """
    a, b, res = (np.atleast_1d(v) for v in _operands(computed, ref))
    if not a.size:
        raise ValueError("no elements to compare")
    xs = [
        np.broadcast_to(np.asarray(v).astype(a.dtype, copy=False), a.shape)
        for v in (x if isinstance(x, tuple) else (x,))
    ]
    err = ulp_error(a, (b, res))
    abserr = np.abs(err)
    abserr[np.isnan(abserr)] = np.inf  # NaN would compare false against maxulp
    finite = np.isfinite(abserr)
    n_over = int(np.count_nonzero(abserr > maxulp))  # +inf mismatches count here
    n = abserr.size
    cand = np.arange(n)
    if 0 < limit < n:  # a full sort would dominate `measure`
        kth = np.partition(abserr, n - limit)[n - limit]
        cand = np.flatnonzero(abserr >= kth)
    worst = cand[np.argsort(-abserr[cand], kind="stable")[:limit]]
    return {
        "label": label,
        "dtype": str(a.dtype),
        "n": n,
        "maxulp": float(maxulp),
        "max_error": _num(abserr.max()),
        "mean_error": float(abserr[finite].mean()) if finite.any() else 0.0,
        "n_over": n_over,
        "passed": n_over == 0,
        "worst": [
            {
                "x": [_fmt(v[i]) for v in xs],
                "actual": _fmt(a[i]),
                "expected": _fmt(b[i]),
                "residual": float(res[i]),
                "error": _num(err[i]),
            }
            for i in map(int, worst)
        ],
        **meta,
    }


def _args(xs: list[dict[str, Any]]) -> str:
    """A record's inputs as `value (bits)`, comma-separated."""
    return ", ".join(f"{v['value']} ({v['bits']})" for v in xs)


def check(record: dict[str, Any]) -> None:
    """Fail with the offending inputs unless `measure`'s record passed.

    :param record: from :func:`measure`.
    """
    __tracebackhide__ = True
    if record["passed"]:
        return
    rows = [
        f"  x={_args(o['x'])}  "
        f"got={o['actual']['value']}  ref={o['expected']['value']}  "
        f"residual={o['residual']:+.4f}  err={o['error']}"
        for o in record["worst"]
        # _num() returns a string only for a non-finite (always a violation).
        if isinstance(o["error"], str) or abs(o["error"]) > record["maxulp"]
    ]
    on = " ".join(str(record[k]) for k in ("target", "accuracy") if k in record)
    label = f"{record['label']} on {on}" if on else record["label"]
    head = (
        f"{label}: {record['n_over']}/{record['n']} over "
        f"{record['maxulp']} ULP (max {record['max_error']}, {record['dtype']})"
    )
    raise AssertionError("\n".join([head, *rows]))


def assert_max_ulp(computed: Any, ref: Any, maxulp: float, **kw: Any) -> dict[str, Any]:
    """Assert every element scores <= `maxulp`; `kw` as for `measure`.

    :param computed: the result to score.
    :param ref: as for :func:`ulp_error`.
    :param maxulp: the bound, in ULP.
    :param kw: keyword arguments of :func:`measure`; ``x`` is required.
    """
    __tracebackhide__ = True
    record = measure(computed, ref, maxulp, **kw)
    check(record)
    return record


def assert_bits(got: np.ndarray, want: np.ndarray, x: np.ndarray, msg: str):
    """Fail unless `got` and `want` are bit-for-bit identical.

    :param got: the result.
    :param want: the expected result.
    :param x: the input, or a tuple with one array per argument; shown for the
        first mismatch.
    :param msg: prefix of the failure message.
    """
    __tracebackhide__ = True
    bad = sr.utils.bits(got) != sr.utils.bits(want)
    if not bad.size:
        raise ValueError("no elements to compare")
    if bad.any():
        i = int(np.argmax(bad))
        xs = ", ".join(
            f"{v[i]!r} (0x{int(sr.utils.bits(v)[i]):x})"
            for v in (x if isinstance(x, tuple) else (x,))
        )
        raise AssertionError(
            f"{msg}: {int(bad.sum())}/{bad.size} bit mismatches; first at "
            f"x={xs} "
            f"got {got[i]!r} want {want[i]!r}"
        )
