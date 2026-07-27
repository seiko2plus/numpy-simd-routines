"""Reusable frozen worst-case gate; an op opts in via `register(op, bound)`."""

from collections.abc import Callable

import numpy as np
import numpy_sr as sr
import pytest
from numpy_sr._conftest.book import WorstBook
from numpy_sr._conftest.stash import BOOK_KEY
from numpy_sr._conftest.targets import ACCURACIES, TARGETS

# op name -> bound(target, dtype, accuracy) -> max |ULP|. Domains register here.
_OP_BOUNDS: dict[str, Callable] = {}


def register(operation: str, bound: Callable) -> None:
    """Enroll `operation`; `bound(target, dtype, accuracy)` gives its max |ULP|."""
    _OP_BOUNDS[operation] = bound


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """One case per book file, against every target that book applies to.

    `target`/`accuracy` are parametrized here rather than taken from their
    fixtures so the key's own axes pick the combinations; the CLI selection
    still deselects the rest, which reads those same param names.
    """
    if "worst_key" not in metafunc.fixturenames:
        return
    cases = []
    for key in metafunc.config.stash[BOOK_KEY].all_keys():
        operation, acc, dtype, have_fma, source = WorstBook.parse(key)
        accuracy = ACCURACIES.get(acc)
        if accuracy is None:  # a book for an accuracy this build dropped
            continue
        for name, target in TARGETS.items():
            if have_fma != target.HAVE_FMA:
                continue
            if dtype == "float64" and not target.HAVE_FLOAT64:
                continue
            cases.append(
                pytest.param(
                    target,
                    accuracy,
                    key,
                    id=f"{name}-{accuracy.name}-{operation}-{dtype}-{source}",
                )
            )
    metafunc.parametrize("target,accuracy,worst_key", cases)


def test_worst_book(request, report_ulp, worst_book, target, accuracy, worst_key):
    """Replay one frozen book; a violation aborts the session."""
    operation, _, dtype, _, _ = WorstBook.parse(worst_key)
    bound = _OP_BOUNDS.get(operation)
    if bound is None:
        pytest.skip(f"{operation} has a book but no registered bound")
    x = worst_book.replay(worst_key)
    if not x.size:
        pytest.skip("empty book")

    try:
        report_ulp(
            x,
            getattr(target, operation)(x, accuracy=accuracy),
            getattr(sr.mpfr, operation)(x),
            bound(target, np.dtype(dtype).type, accuracy),
            operation=operation,
            target=target,
            accuracy=accuracy,
            # never folds new inputs; `refresh_key` re-stamps this book's errors
            # under --collect-worst, and only for a non-reference target
            store_worst=False,
            refresh_key=worst_key,
        )
    except AssertionError:
        # shouldstop is the channel xdist propagates back from a worker; a
        # pytest.exit() would abort one worker and leave the others running.
        request.session.shouldstop = "frozen worst-case regression"
        raise
