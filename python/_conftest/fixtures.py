"""Fixtures and per-target/-accuracy parametrization."""

import pathlib
import zlib
from collections.abc import ValuesView

import numpy as np
import numpy_sr as sr
import pytest

from .book import WorstBook
from .options import selected_accuracies, selected_targets
from .stash import BOOK_KEY, REPORT_KEY
from .targets import ACCURACIES, TARGETS
from .xdist import resolve_seed


@pytest.fixture(scope="session")
def targets(request: pytest.FixtureRequest) -> ValuesView:
    """Every selected target at once, for tests that compare backends."""
    return selected_targets(request.config).values()


@pytest.fixture(scope="session")
def accuracies(request: pytest.FixtureRequest) -> list:
    """Every selected accuracy at once, for tests that sweep them internally."""
    return selected_accuracies(request.config)


@pytest.fixture(params=list(TARGETS.values()), ids=list(TARGETS))
def target(request: pytest.FixtureRequest):
    """The backend under test; CLI selection deselects the rest at collection."""
    return request.param


@pytest.fixture(
    params=list(ACCURACIES.values()), ids=[a.name for a in ACCURACIES.values()]
)
def accuracy(request: pytest.FixtureRequest):
    """The `sr.Accuracy` under test; unselected ones deselected at collection."""
    return request.param


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Per-test PRNG; the salt drops the target id: all targets, same data."""
    node = request.node
    base, sep, bracket = node.name.partition("[")
    segments = bracket.rstrip("]").split("-") if sep else []
    if "target" in request.fixturenames:
        target = request.getfixturevalue("target")
        target_id = target.__name__.rsplit(".", 1)[-1]
        segments = [s for s in segments if s != target_id]
    parts = [node.path.name, base, *segments]
    seed = resolve_seed(request.config)
    return np.random.default_rng([seed, zlib.crc32("|".join(parts).encode())])


@pytest.fixture
def worst_book(request: pytest.FixtureRequest) -> "WorstBook":
    """The session's frozen worst-case book, for tests that replay it."""
    return request.config.stash[BOOK_KEY]


@pytest.fixture
def report_ulp(request: pytest.FixtureRequest):
    """`assert_max_ulp` bound to the session's report and worst-case book."""
    nodeid = request.node.nodeid
    records = request.config.stash[REPORT_KEY]
    book = request.config.stash[BOOK_KEY]
    collect = request.config.getoption("--collect-worst")
    path, _, rest = nodeid.partition("::")
    label = f"{pathlib.Path(path).name}::{rest}" if rest else path

    # operation/target/accuracy default to these; only what the test pulled in
    global_kw = {
        a: request.getfixturevalue(a) if a in request.fixturenames else None
        for a in ("operation", "target", "accuracy")
    }

    def sink(record: dict) -> None:
        records.append(record)

    def check(
        x,
        computed,
        ref,
        maxulp,
        *,
        store_worst=None,
        refresh_key=None,
        operation=None,
        target=None,
        accuracy=None,
    ):
        # Explicit rather than **kw: a stray keyword must fail, not vanish.
        __tracebackhide__ = True
        given = {"operation": operation, "target": target, "accuracy": accuracy}
        kw = {}
        for k, v in given.items():
            kw[k] = v = global_kw[k] if v is None else v
            if v is None:
                raise TypeError(
                    f"{nodeid}: no `{k}` fixture in scope — "
                    f"pass {k}=... to report_ulp()"
                )

        target = kw["target"]
        target_name = getattr(target, "__name__", str(target)).rsplit(".", 1)[-1]
        is_reference = getattr(target, "IS_REFERENCE", False)
        a = np.asarray(computed)
        # default: f64 only. Every f32 input is covered by test_exhaustive, which
        # opts back in with store_worst=True; sampling f32 elsewhere burns the cap.
        store = a.dtype == np.float64 if store_worst is None else store_worst
        # Every write to the book goes through here, so `is_reference` is enforced
        # once: only a non-reference target's error belongs in the book.
        if collect and not is_reference and (store or refresh_key is not None):
            xb = np.broadcast_to(np.asarray(x).astype(a.dtype, copy=False), a.shape)
            err = np.abs(sr.testing.ulp_error(a, ref))
            xr, er = xb.ravel(), np.asarray(err).ravel()
            if store:
                axes = (kw["operation"], kw["accuracy"].name, str(a.dtype), target_name)
                book.fold(
                    WorstBook.key(
                        *axes[:3], target.HAVE_FMA, WorstBook.source(label, *axes)
                    ),
                    xr,
                    er,
                )
            if refresh_key is not None:
                # a replay measures every stored input: stamp the book with what
                # this kernel actually does, up or down
                book.refresh(refresh_key, xr, er)

        meta = {
            "x": x,
            "label": label,
            "target": target_name,
            "accuracy": kw["accuracy"].name,
            "operation": kw["operation"],
        }
        # A reference target is not the oracle (MPFR is): recorded, not asserted.
        if is_reference:
            record = sr.testing.measure(computed, ref, maxulp, **meta)
            sink(record)
            return record
        return sr.testing.assert_max_ulp(computed, ref, maxulp, sink=sink, **meta)

    return check
