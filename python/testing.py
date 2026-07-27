"""Fractional-ULP error vs the exact value; correct rounding scores ~0.5."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy_sr as sr

Sink = Callable[[dict[str, Any]], None]

# float dtype -> same-width signed / unsigned int, for bit reinterpretation.
_UNSIGNED = {np.dtype(np.float32): np.uint32, np.dtype(np.float64): np.uint64}


def ulp_error(computed: Any, ref: Any) -> np.ndarray:
    """Signed fractional-ULP error; non-finite mismatches score +inf."""
    ref, residual = _unpack_ref(ref)
    a = np.asarray(computed)
    # sr.ulp_error is dtype-exact (SafeArg); only layout is its problem.
    b = np.asarray(ref).astype(a.dtype, copy=False)
    a, b = np.broadcast_arrays(a, b)
    res = np.asarray(residual, dtype=np.float64)
    r = np.broadcast_to(res, a.shape) if res.ndim or res != 0.0 else None
    return sr.ulp_error(a, b, r)


def _unpack_ref(ref: Any) -> tuple[Any, Any]:
    """Split `ref` into `(ref, residual)`; a bare reference gets 0.0."""
    if isinstance(ref, tuple) and len(ref) == 2:
        return ref
    return ref, 0.0


def _num(v: Any) -> float | str:
    """JSON-safe number: non-finite floats become tokens strict JSON accepts."""
    f = float(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f


def _fmt(v: np.ndarray) -> dict[str, Any]:
    """A float element as both its decimal value and exact IEEE bit pattern."""
    dt = v.dtype
    bits = int(v.view(_UNSIGNED[dt]))
    return {"value": _num(v), "bits": f"0x{bits:0{dt.itemsize * 2}x}"}


def _offenders(
    x: np.ndarray,
    computed: np.ndarray,
    ref: np.ndarray,
    residual: np.ndarray,
    err: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(err), kind="stable")[:limit]
    return [
        {
            "x": _fmt(x[i]),
            "actual": _fmt(computed[i]),
            "expected": _fmt(ref[i]),
            "residual": float(residual[i]),
            "error": _num(err[i]),
        }
        for i in map(int, order)
    ]


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
    """Compare and build a record without asserting; extra kwargs go in it."""
    ref, residual = _unpack_ref(ref)
    a = np.asarray(computed)
    b = np.asarray(ref).astype(a.dtype, copy=False)
    a, b = np.atleast_1d(*np.broadcast_arrays(a, b))
    x = np.broadcast_to(np.asarray(x).astype(a.dtype, copy=False), a.shape)
    res = np.broadcast_to(np.asarray(residual, dtype=np.float64), a.shape)

    err = ulp_error(a, (b, res))
    abserr = np.abs(err)
    n = int(abserr.size)
    finite = np.isfinite(abserr)
    n_over = int(np.count_nonzero(abserr > maxulp))  # +inf mismatches count here

    record = {
        "label": label,
        "dtype": str(a.dtype),
        "n": n,
        "maxulp": float(maxulp),
        "max_error": _num(abserr.max()) if n else 0.0,
        "mean_error": float(abserr[finite].mean()) if finite.any() else 0.0,
        "n_over": n_over,
        "passed": n_over == 0,
        "worst": _offenders(x, a, b, res, err, limit),
    }
    record.update(meta)
    return record


def _exceeds(error: float | str, maxulp: float) -> bool:
    # _num() returns a string only for a non-finite (always a violation).
    return isinstance(error, str) or abs(error) > maxulp


def _format_failure(record: dict[str, Any]) -> str:
    head = (
        f"{record['label']}: {record['n_over']}/{record['n']} over "
        f"{record['maxulp']} ULP (max {record['max_error']}, {record['dtype']})"
    )
    rows = [
        f"  x={o['x']['value']} ({o['x']['bits']})  "
        f"got={o['actual']['value']}  ref={o['expected']['value']}  "
        f"residual={o['residual']:+.4f}  err={o['error']}"
        for o in record["worst"]
        if _exceeds(o["error"], record["maxulp"])
    ]
    return "\n".join([head, *rows])


def assert_max_ulp(
    computed: Any,
    ref: Any,
    maxulp: float,
    *,
    x: Any,
    label: str = "",
    limit: int = 16,
    sink: Sink | None = None,
    **meta: Any,
) -> dict[str, Any]:
    """Assert every element scores <= `maxulp`; `sink` sees it either way."""
    __tracebackhide__ = True
    record = measure(
        computed,
        ref,
        maxulp,
        x=x,
        label=label,
        limit=limit,
        **meta,
    )
    if sink is not None:
        sink(record)
    if not record["passed"]:
        raise AssertionError(_format_failure(record))
    return record


def assert_bits(got: np.ndarray, want: np.ndarray, x: np.ndarray, msg: str):
    """Fail unless `got` and `want` are bit-for-bit identical."""
    __tracebackhide__ = True
    bad = sr.ieee754.bits(got) != sr.ieee754.bits(want)
    if bad.any():
        i = int(np.argmax(bad))
        raise AssertionError(
            f"{msg}: {int(bad.sum())}/{bad.size} bit mismatches; first at "
            f"x={x[i]!r} (0x{int(sr.ieee754.bits(x)[i]):x}) "
            f"got {got[i]!r} want {want[i]!r}"
        )
