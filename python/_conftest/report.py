"""Run reporting and the ULP JSON payload for the HTML viewer."""

import json
import math
import time
from functools import partial

import pytest

from .options import selected_accuracies, selected_targets
from .stash import BOOK_KEY, REPORT_KEY
from .targets import TARGETS
from .xdist import resolve_seed, worker_output

_OFFENDER_LIMIT = 10
_SENTINEL = 1e300  # finite stand-in for non-finite sort keys
_TOKENS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def pytest_report_header(config: pytest.Config) -> list:
    """The selection echoed at the top and bottom of every run."""
    targets = [
        f"{name} (reference)" if getattr(target, "IS_REFERENCE", False) else name
        for name, target in selected_targets(config).items()
    ]
    accuracies = [a.name for a in selected_accuracies(config)]
    return [
        f"targets:    {', '.join(targets)}",
        f"accuracies: {', '.join(accuracies)}",
        f"seed:       {resolve_seed(config)}",
    ]


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    terminalreporter.write_sep("-", "selection")
    for line in pytest_report_header(config):
        terminalreporter.write_line(line)

    book = config.stash.get(BOOK_KEY, None)
    lines = book.report_lines() if book is not None else []
    if lines:
        terminalreporter.write_sep("-", "worst-case discoveries")
        for line in lines:
            terminalreporter.write_line(line)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    book = config.stash.get(BOOK_KEY, None)
    records = config.stash.get(REPORT_KEY, [])
    config.stash[REPORT_KEY] = []  # drained; guard against a double dump

    out = worker_output(config)
    if out is not None:
        # The controller owns every write; ship the results and touch no files.
        out["npsr_records"] = records
        out["npsr_book"] = book.deltas() if book is not None else {}
        out["npsr_restamped"] = book.restamped if book is not None else {}
        return

    if book is not None:
        book.flush()  # persist any book that gained a worse case

    if records:
        head = {"seed": resolve_seed(config), "created": time.time()}
        text = json.dumps(head | _serialize_records(records), indent=2)
        (config.rootpath / "ulp-report.json").write_text(text)


def _pnum(v: float | str) -> float:
    """Back to a float for computation; `testing._num` tokenizes non-finite."""
    return _TOKENS[v] if isinstance(v, str) else float(v)


def _mag(v: float | str) -> float:
    """`|v|` as a rank; non-finite (nan included) outranks any finite error."""
    f = _pnum(v)
    return abs(f) if math.isfinite(f) else math.inf


def _fmt(v: float | str, fmt) -> str:
    """`fmt` applied, except to non-finite values: those keep their token."""
    f = _pnum(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return fmt(f)


_fmt_ulp = partial(_fmt, fmt="{:.4g}".format)  # ULP magnitude, ~4 sig figs
_fmt_val = partial(_fmt, fmt=repr)  # operand: shortest round-tripping decimal


def _finite(f: float) -> float:
    """A JSON-safe sort key: non-finite collapses to a large finite sentinel."""
    return f if math.isfinite(f) else math.copysign(_SENTINEL, f)


def _ratio(max_error: float | str, maxulp: float) -> float:
    """Bar length + worst-first rank; non-finite error ranks above any bound."""
    m = _mag(max_error)
    if math.isinf(m):
        return math.inf
    return 0.0 if math.isinf(maxulp) else m / maxulp


def _test_name(label: str) -> str:
    """`file.py::test_luck[High-...]` -> `test_luck` (the group carries params)."""
    return label.split("::")[-1].split("[")[0]


def _val(v: dict) -> dict:
    """An offender operand as text plus its raw IEEE bits."""
    return {"text": _fmt_val(v["value"]), "bits": v["bits"]}


def _merge_offenders(recs: list) -> list:
    """Worst offenders across every test in the group, highest |error| first."""
    flat = [(r, o) for r in recs for o in r["worst"]]
    flat.sort(key=lambda t: _mag(t[1]["error"]), reverse=True)
    return [
        {
            "test": _test_name(r["label"]),
            "x": _val(o["x"]),
            "actual": _val(o["actual"]),
            "expected": _val(o["expected"]),
            "residual": f"{o['residual']:+.4f}",
            "error": _fmt_ulp(o["error"]),
            "over": math.isinf(m := _mag(o["error"])) or m > r["maxulp"],
        }
        for r, o in flat[:_OFFENDER_LIMIT]
    ]


def _row(target, operation, dtype, accuracy, recs, have_fma) -> dict:
    """One aggregate table row folding every test that hit this combination."""
    n = sum(r["n"] for r in recs)
    n_over = sum(r["n_over"] for r in recs)
    mean = sum(_pnum(r["mean_error"]) * r["n"] for r in recs) / n if n else 0.0
    lo, hi = min(r["maxulp"] for r in recs), max(r["maxulp"] for r in recs)
    tests = sorted(
        recs, key=lambda r: _ratio(r["max_error"], r["maxulp"]), reverse=True
    )
    worst = tests[0]
    ratio = _ratio(worst["max_error"], worst["maxulp"])
    return {
        "operation": operation,
        "target": target,
        "fma": "FMA" if have_fma else "non-FMA",
        "dtype": dtype,
        "accuracy": accuracy,
        "status": "pass" if all(r["passed"] for r in recs) else "fail",
        "tests": len(recs),
        # en dash is a deliberate range separator in the rendered report
        "bound": _fmt_ulp(lo) if lo == hi else f"{_fmt_ulp(lo)}–{_fmt_ulp(hi)}",  # noqa: RUF001
        "boundSort": _finite(hi),
        "maxErr": _fmt_ulp(worst["max_error"]),
        "maxErrNum": _finite(_pnum(worst["max_error"])),
        "barPct": min(1.0, ratio) * 100 if math.isfinite(ratio) else 100.0,
        "barOver": ratio > 1,
        "mean": _fmt_ulp(mean),
        "meanNum": mean,
        "over": f"{n_over}/{n}",
        "nOver": n_over,
        "n": n,
        "search": " ".join(r["label"] for r in recs).lower(),
        "perTest": [
            {
                "label": r["label"],
                "bound": _fmt_ulp(r["maxulp"]),
                "maxErr": _fmt_ulp(r["max_error"]),
                "mean": _fmt_ulp(r["mean_error"]),
                "over": f"{r['n_over']}/{r['n']}",
                "passed": r["passed"],
            }
            for r in tests
        ],
        "offenders": _merge_offenders(recs),
    }


def _serialize_records(records) -> dict:
    # Rows are grouped and ordered by these four dims.
    dims = ("target", "operation", "dtype", "accuracy")
    groups: dict = {}
    for r in records:
        groups.setdefault(tuple(r[d] for d in dims), []).append(r)
    # first-appearance order, which for each dim matches record order
    order = {d: list(dict.fromkeys(k[i] for k in groups)) for i, d in enumerate(dims)}

    # TARGETS (not sr.targets) so the numpy reference mock resolves too.
    fma_of = {t: TARGETS[t].HAVE_FMA for t in order["target"]}
    rows = [
        _row(target, op, dtype, acc, recs, fma_of[target])
        for (target, op, dtype, acc), recs in groups.items()
    ]

    def present(field, opts):
        seen = {r[field] for r in rows}
        return [o for o in opts if o in seen]

    # Facet lists in display order; the viewer just renders and toggles them.
    filter_by = {
        "operation": order["operation"],
        "target": order["target"],
        "fma": present("fma", ["FMA", "non-FMA"]),
        "dtype": order["dtype"],
        "accuracy": order["accuracy"],
        "status": present("status", ["pass", "fail"]),
    }
    return {"filter_by": filter_by, "records": rows}
