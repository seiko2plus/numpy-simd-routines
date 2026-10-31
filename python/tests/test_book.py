"""`WorstBook`: the CSV round trip, folding and the bound."""

import pathlib

import numpy as np
from numpy_sr import utils
from numpy_sr._conftest.book import WorstBook
from pytest import mark

_KEY = WorstBook.key("sin", "High", "float32", True, "test_x")
_CSV = "mpfr/sin/high_float32_fma_test_x.csv"


def _book(tmp_path: pathlib.Path) -> WorstBook:
    return WorstBook(tmp_path / "mpfr", {"sin": 1})


def _fold(book: WorstBook, x, err) -> None:
    """Fold float32 `x` into the `_KEY` book; the ref is `x`, its residual 0."""
    x = np.asarray(x, np.float32)
    rows = book.rows(_KEY, (x,), (x, np.zeros(x.size)), np.asarray(err, np.float64))
    book.fold(_KEY, rows, 1.0)


def _row(x: float, err: float) -> str:
    """The CSV row `_fold` stores for `x`."""
    bits = int(utils.bits(np.float32(x)))
    return f"{bits},{bits},0.0,{err!r}"


def _write(tmp_path: pathlib.Path, text: str) -> None:
    """Store `text` as the `_KEY` book."""
    path = tmp_path / _CSV
    path.parent.mkdir(parents=True)
    path.write_text(text)


def _lines(tmp_path: pathlib.Path) -> list[str]:
    return (tmp_path / _CSV).read_text().splitlines()


@mark.parametrize("dtype", [np.float32, np.float64])
def test_round_trip(tmp_path, dtype):
    """Input, ref and residual bits survive a flush and a reload, specials too."""
    tiny, big = np.finfo(dtype).smallest_subnormal, np.finfo(dtype).max
    x = np.array([-0.0, 0.0, tiny, -tiny, big, -big, 1.0], dtype)
    ref = x[::-1].copy()
    res = np.array([-0.0, 0.0, 5e-324, -5e-324, 1 / 3, -0.5, 1e-300])
    key = WorstBook.key("sin", "High", np.dtype(dtype).name, True, "test_x")
    book = _book(tmp_path)
    book.fold(key, book.rows(key, (x,), (ref, res), np.linspace(0.1, 0.7, 7)), 1.0)
    book.flush()

    reloaded = _book(tmp_path)
    (got,), (got_ref, got_res) = reloaded.replay(key)
    order = np.argsort(utils.bits(x))  # replay is bits ascending
    assert np.array_equal(utils.bits(got), utils.bits(x)[order])
    assert np.array_equal(utils.bits(got_ref), utils.bits(ref)[order])
    assert np.array_equal(utils.bits(got_res), utils.bits(res)[order])
    assert reloaded.bound(key) == 1.0


def test_no_header(tmp_path):
    """A book without a ``# maxulp=`` line has no bound, so the gate fails."""
    _write(tmp_path, "")
    book = _book(tmp_path)
    assert book.bound(_KEY) is None
    assert book.replay(_KEY)[0][0].size == 0


def test_fold_dedupes(tmp_path):
    """A fold keeps an input's worst error and drops non-finite inputs and errors."""
    book = _book(tmp_path)
    _fold(book, [1.0, 1.0, np.nan, 2.0, 3.0], [0.2, 0.9, 5.0, np.inf, 0.1])
    book.flush()
    assert _lines(tmp_path)[1:] == [_row(1.0, 0.9), _row(3.0, 0.1)]


def test_fold_supersedes(tmp_path):
    """A run's first fold replaces a stored error, so a kernel fix can lower it."""
    _write(tmp_path, f"# maxulp=1.0\n{_row(1.0, 2.0)}\n")
    book = _book(tmp_path)
    _fold(book, [1.0], [0.5])
    book.flush()
    assert _lines(tmp_path)[1:] == [_row(1.0, 0.5)]
