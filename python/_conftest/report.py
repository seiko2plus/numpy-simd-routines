"""The ULP JSON payload for the HTML viewer: one display-ready row per group."""

import math

_OFFENDER_LIMIT = 10
_SENTINEL = 1e300  # finite stand-in for non-finite sort keys
# `testing._num` turns non-finite floats into these tokens; every float is finite.
_TOKENS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _num(v: float | str) -> float:
    return _TOKENS[v] if isinstance(v, str) else v


def _mag(v: float | str) -> float:
    """`|v|` as a rank; non-finite (nan included) outranks any finite error."""
    return math.inf if isinstance(v, str) else abs(v)


def _ulp(v: float | str) -> str:
    return v if isinstance(v, str) else f"{v:.4g}"


def _finite(f: float) -> float:
    """A JSON-safe sort key: non-finite collapses to a large finite sentinel."""
    return f if math.isfinite(f) else math.copysign(_SENTINEL, f)


def _ratio(max_error: float | str, maxulp: float) -> float:
    """Bar length + worst-first rank; non-finite error ranks above any bound."""
    m = _mag(max_error)
    if math.isinf(m):
        return math.inf
    if math.isinf(maxulp) or m == 0:
        return 0.0
    return m / maxulp if maxulp else math.inf


def _val(v: dict) -> dict:
    """An offender operand as its shortest round-tripping text plus IEEE bits."""
    text = v["value"] if isinstance(v["value"], str) else repr(v["value"])
    return {"text": text, "bits": v["bits"]}


def _offenders(recs: list) -> list:
    """Worst offenders across every test in the group, highest |error| first."""
    flat = sorted(
        ((r, o) for r in recs for o in r["worst"]),
        key=lambda t: _mag(t[1]["error"]),
        reverse=True,
    )
    return [
        {
            "test": r["label"],
            "x": [_val(v) for v in o["x"]],
            "actual": _val(o["actual"]),
            "expected": _val(o["expected"]),
            "residual": f"{o['residual']:+.4f}",
            "error": _ulp(o["error"]),
            "over": math.isinf(m := _mag(o["error"])) or m > r["maxulp"],
        }
        for r, o in flat[:_OFFENDER_LIMIT]
    ]


def _test(label: str, recs: list) -> dict:
    """One test's calls folded into its per-test entry."""
    n = sum(r["n"] for r in recs)
    lo, hi = min(r["maxulp"] for r in recs), max(r["maxulp"] for r in recs)
    worst = max(recs, key=lambda r: _ratio(r["max_error"], r["maxulp"]))
    return {
        "label": label,
        "bound": _ulp(lo) if lo == hi else f"{_ulp(lo)}–{_ulp(hi)}",  # noqa: RUF001
        "maxErr": _ulp(worst["max_error"]),
        "mean": _ulp(sum(r["mean_error"] * r["n"] for r in recs) / n if n else 0.0),
        "over": f"{sum(r['n_over'] for r in recs)}/{n}",
        "passed": all(r["passed"] for r in recs),
    }


def _row(recs: list) -> dict:
    """One aggregate table row folding every test that hit this combination."""
    n = sum(r["n"] for r in recs)
    n_over = sum(r["n_over"] for r in recs)
    mean = sum(r["mean_error"] * r["n"] for r in recs) / n if n else 0.0
    lo, hi = min(r["maxulp"] for r in recs), max(r["maxulp"] for r in recs)
    tests = sorted(
        recs, key=lambda r: _ratio(r["max_error"], r["maxulp"]), reverse=True
    )
    worst = tests[0]
    ratio = _ratio(worst["max_error"], worst["maxulp"])
    by_test: dict = {}
    for r in tests:
        by_test.setdefault(r["label"], []).append(r)
    return {
        **{k: recs[0][k] for k in ("operation", "target", "fma", "dtype", "accuracy")},
        "status": "pass" if all(r["passed"] for r in recs) else "fail",
        "tests": len(by_test),
        # en dash is a deliberate range separator in the rendered report
        "bound": _ulp(lo) if lo == hi else f"{_ulp(lo)}–{_ulp(hi)}",  # noqa: RUF001
        "boundSort": _finite(hi),
        "maxErr": _ulp(worst["max_error"]),
        "maxErrNum": _finite(_num(worst["max_error"])),
        "barPct": min(1.0, ratio) * 100 if math.isfinite(ratio) else 100.0,
        "barOver": ratio > 1,
        "mean": _ulp(mean),
        "meanNum": mean,
        "over": f"{n_over}/{n}",
        "nOver": n_over,
        "n": n,
        "search": " ".join(by_test).lower(),
        "perTest": [_test(label, rs) for label, rs in by_test.items()],
        "offenders": _offenders(recs),
    }


def serialize(records: list) -> dict:
    """Rows grouped by (target, operation, dtype, accuracy), plus facet lists."""
    dims = ("target", "operation", "dtype", "accuracy")
    groups: dict = {}
    for r in records:
        groups.setdefault(tuple(r[d] for d in dims), []).append(r)
    rows = [_row(recs) for recs in groups.values()]

    # Facet lists in display order (first appearance); the viewer just renders.
    def seen(field, opts=None):
        values = list(dict.fromkeys(r[field] for r in rows))
        return values if opts is None else [o for o in opts if o in values]

    filter_by = {
        "operation": seen("operation"),
        "target": seen("target"),
        "fma": seen("fma", ["FMA", "non-FMA"]),
        "dtype": seen("dtype"),
        "accuracy": seen("accuracy"),
        "status": seen("status", ["pass", "fail"]),
    }
    return {"filter_by": filter_by, "records": rows}
