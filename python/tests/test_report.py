"""`report.serialize`: rows per group and non-finite errors."""

import json

from numpy_sr._conftest.report import serialize
from pytest import mark


def _rec(**kw) -> dict:
    """A passing AVX2 High float32 sin record; `kw` overrides fields."""
    return {
        "label": "test_x",
        "operation": "sin",
        "target": "AVX2",
        "fma": "FMA",
        "dtype": "float32",
        "accuracy": "High",
        "n": 100,
        "n_over": 0,
        "max_error": 0.5,
        "mean_error": 0.1,
        "maxulp": 1.0,
        "passed": True,
        "worst": [],
    } | kw


def test_groups():
    """One row per (target, operation, dtype, accuracy); a failed record fails it."""
    out = serialize(
        [
            _rec(),
            _rec(target="SSE2"),
            _rec(target="SSE2", n_over=1, max_error=1.5, passed=False),
            _rec(accuracy="Low", maxulp=2.0),
        ]
    )
    assert [(r["target"], r["accuracy"], r["status"]) for r in out["records"]] == [
        ("AVX2", "High", "pass"),
        ("SSE2", "High", "fail"),
        ("AVX2", "Low", "pass"),
    ]


@mark.parametrize("token", ["nan", "inf"])
def test_non_finite(token):
    """A nan or inf error is over any bound and still serializes as strict JSON."""
    out = serialize([_rec(n_over=1, max_error=token, passed=False)])
    json.dumps(out, allow_nan=False)
    row = out["records"][0]
    assert (row["maxErr"], row["barOver"]) == (token, True)
