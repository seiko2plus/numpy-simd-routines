"""``python/conftest.py`` holds the hooks, fixtures and the target registry.
The book and the report have their own modules in ``python/_conftest/``.

A run
-----

1. **Configure.** The plugin parses ``--target``, ``--accuracy``, ``--oracle``,
   ``--seed``, ``--worst-dir`` and ``--worst-collect``, and loads one book
   per ``--worst-dir`` lazily from ``<dir>/<oracle>/``.
2. **Collect.** The ``target``, ``npsr`` and ``accuracy`` fixtures
   parametrize over every target and accuracy. Collection then deselects
   whatever the CLI did not select, moves the ``test_worst_book`` cases to the
   front, and skips ``slow``/``exhaustive`` tests unless a ``-m`` was given.
3. **Gate.** ``test_worst_book`` replays every book on every matching target.
   If one fails, the session stops: a regression on a known hard input is
   enough to see.
4. **Tests.** Each test computes inputs, gets the reference from the oracle and
   passes both with the target's op to ``test``, which calls the op,
   attaches the result to the test's report, asserts the bound, and (with
   ``--worst-collect``) attaches the worst inputs for the book too.
5. **Finish.** The main process (the controller, under xdist) folds what the
   reports carried into the book, writes ``doc/testing/ulp-report.json`` and,
   with ``--worst-collect``, rewrites the books that changed.

Terms
-----

- **Target**: an NPSR, oracle or reference target, see
  :mod:`numpy_sr.target`.
- **Accuracy**: the kernel's precision profile, ``sr.Accuracy.High`` or
  ``Low``. Each test module defines its own bound per accuracy and dtype.
- **ULP error**: the signed distance from the computed value to the exact
  value, in units in the last place, as a float. A correctly rounded result
  scores at most 0.5.
- **FMA / non-FMA**: NPSR targets differ in whether they have native FMA
  (``HAVE_FMA``). The kernels take different paths for each, so books are
  kept per FMA class.

Target and accuracy selection
-----------------------------

``TARGETS`` keeps the targets the host supports, without the oracles: the
NPSR targets, then the references. ``numpy`` isn't built into the extension, so
it's added there. ``TARGETS`` is the registry ``--target`` chooses from;
``NPSR_TARGETS`` is its non-reference part.

The ``target`` and ``accuracy`` fixtures parametrize over all of ``TARGETS``
and every accuracy, not over the selection; ``npsr`` is ``target`` without
the references, over ``NPSR_TARGETS``. Tests that go through the ``test``
fixture take ``target``, the others take ``npsr``; a test can't take both.
``pytest_collection_modifyitems`` then deselects the items whose ``target``
or ``accuracy`` parameter wasn't selected. That keeps the test ids the same
on every run, and unselected targets don't show up as skips.

Tests that compare all targets in one pass (to share one oracle call) use the
session fixtures ``targets`` and ``accuracies``, which return the selection
itself.

Reference targets
~~~~~~~~~~~~~~~~~

Selected next to NPSR targets (``--target avx2,numpy``), they show up side by
side in the ULP report. Their ULP errors are reported but never fail a test,
and never enter the book. The bit-exact tests take ``npsr``, so they don't
run on them.

The worst-case book
-------------------

Random tests find a different set of hard inputs on each run. The book keeps
the hardest ones found so far and replays them on every run, so an input that
was hard once is tested from then on.

Files
~~~~~

.. code-block:: text

   <dir>/<oracle>/<op>/<accuracy>_<dtype>_<fma|nonfma>_<source>.csv

- ``<dir>``: a ``--worst-dir`` (default ``python/tests/worst``) or the
  ``--worst-collect`` directory.
- ``<oracle>``: the ``--oracle`` the references come from. A run reads and
  writes only that oracle's books.
- ``<op>``: the operation (``sin``). It can't contain ``_``, because the key
  is split on ``_``.
- ``<accuracy>``, ``<dtype>``, ``<fma|nonfma>``: the axes the error depends
  on. All targets in one FMA class share a book.
- ``<source>``: the name of the test function that found the inputs, for
  example ``test_exhaustive`` or ``test_worst_adaptive``. All its
  parametrizations share the file.

The first line is ``# maxulp=<bound>``, the bound the test passed to
``test``. Each row after it is ``<bits>,<ref>,<residual>,<error>``:
the input's and the reference's IEEE bits as unsigned decimal integers (exact,
and fast for ``np.loadtxt``), the reference's residual and the input's
``|ULP|`` error, both exact. Rows are sorted worst first. A book holds
at most 10 000 rows for f64 and 1 000 for f32.

In code, a book is a key, ``<op>_<accuracy>_<dtype>_<fma>_<source>``. See
``WorstBook.key``/``parse`` in ``python/_conftest/book.py``.

The gate
~~~~~~~~

``python/tests/test_worst.py`` is shared by all operations.
``test_worst_book`` gets, from ``pytest_generate_tests`` here, one case per
book of each ``--worst-dir`` and per target in the registry that matches the
book's FMA class (and has f64, for an f64 book). The CLI
selection then deselects the rest, as for any other test. Each case replays
the book's inputs against its stored reference with the book's ``maxulp``.
Several directories can hold the same book, an archive of old cases for
example; each is replayed on its own, and the ids of the ``i``-th directory
after the first end in ``-dir<i>``.

The gate runs first. If a case fails it stops the session, so the rest of the
report is missing. To get a full report anyway:
``spin test -- -k "not test_worst_book"``.

A book without a ``maxulp`` line fails.

Collecting
~~~~~~~~~~

Without ``--worst-collect`` the books are only read. ``--worst-collect DIR``
writes the books under ``DIR``, and two things update them:

- **fold** (from ``test``): the measured inputs are merged into the book,
  which keeps the worst rows by error, up to the cap. The first time a key is
  touched in a run, the new measurement replaces the stored error of
  the same input, and the first fold's ``maxulp`` replaces the stored one.
  After that, measurements of the same input keep the maximum, and so does
  ``maxulp``, because several targets share one key.
- **replay** (from the gate, through ``replay``): only if ``DIR`` is also a
  ``--worst-dir``. On each selected target of the book's FMA class, every
  stored input is measured again and folded in before any other fold,
  without a ``maxulp``, so the gate stamps the book with what the current
  kernel actually does.
  Without this, errors would only go up: after a kernel fix the book would
  keep the old values, and the cap would push out inputs that
  are still hard to keep ones that aren't anymore.

At the end of the session, only the books whose rows or ``maxulp`` changed
are rewritten. The summary lists each one
with its max error before and after, and the worst new input if it got one.

References never go into the book, and neither do non-finite inputs or
errors.

Workflow
~~~~~~~~

A run with ``--worst-collect`` always changes the books a little: the inputs
are random, so the recorded errors differ slightly from run to run. Collect
only for the kernels you changed, and commit only their books with the change.

Collect with the gate in the run. The worst cases are known, so the gate
checks them first and stops a failing run early, without re-running every
test; while collecting, it also updates the stored cases (the replay above).
``-m slow``, ``-k "not test_worst_book"`` and a selection without
``test_worst.py`` leave the gate out; ``-m all`` keeps it.

The books are kept per FMA class, not per target, so collecting needs just one
FMA and one non-FMA target. A contribution has to include both, the non-FMA
books too. Pick the two targets with ``--target`` rather than running every
target, which only makes the run slower:

.. code-block:: sh

   spin test --worst-collect python/tests/worst -m all -j auto \
       --operation tan --target sse4,avx2
   git diff --stat python/tests/worst
   git add python/tests/worst/mpfr/tan

A book that has gone down is the fix showing up. One that went up needs a look
before you commit.

.. note::

   On an arch whose targets all have FMA, the non-FMA books can't be
   collected. There's no solution for that yet.

Design choices
~~~~~~~~~~~~~~

- **The book accumulates.** Every run draws new inputs from a new seed, and
  one test function runs as many items, on several workers. A snapshot per
  run or per item would keep only what that run or item found, so the book
  merges them all into one file per key instead.
- **Keep the worst, not all.** The cap is 10 000 rows for f64 and 1 000 for
  f32, which the exhaustive tests already cover in full. ``_top`` picks them
  with ``argpartition`` and keeps them sorted by bits, for lookups; the file
  is written worst first.
- **A replay is a fold without ``maxulp``.** The gate passes ``replay``, its
  book and key, to ``test``. Folding and replaying exclude each other, so one
  parameter selects between them. Replays are applied before every other
  fold, so their measurements count as the first touch of the key.
- **CSV.** It's readable and diffable in a commit. JSON would save only a few
  lines of code and no space: compact JSON is a little larger than the CSV,
  and indented JSON about a third larger. ``.npz`` is binary.
- **No library.** pytest-regressions compares snapshots for equality and
  can't accumulate. pandas, polars and duckdb are too big, and have no
  FreeBSD wheels.

Markers
-------

``slow`` and ``exhaustive`` tests get a skip marker at collection unless
pytest got an ``-m`` expression. ``spin test -m slow`` passes ``-m slow``,
which runs only those tests. ``-m all`` passes an expression that keeps the
regular tests too. A pytest ``-m`` expression after ``--`` works too, but not
together with spin's ``-m``.

xdist
-----

``test`` puts its record and book rows in the test's ``user_properties``,
which xdist ships with every report to the controller. There
``pytest_runtest_logreport`` collects them and ``pytest_sessionfinish``
writes the files. ``pytest_configure_node`` sends every worker the same seed.

This is the same path with or without xdist: only the main process writes,
so there's nothing to merge from the workers. The properties have to be
types execnet can send, so the book rows go as bytes, not numpy arrays.
They also end up in a ``--junitxml`` file, which gets large with
``--worst-collect``.

A gate failure sets ``session.shouldstop``, which xdist sends back from the
worker and which stops the other workers as well. ``pytest.exit()`` would stop
only the worker it ran on.

The ULP report
--------------

``pytest_sessionfinish`` writes ``doc/testing/ulp-report.json`` (outside a
checkout, ``ulp-report.json`` in the rootdir). ``_conftest/report.py`` groups
the records by target, operation, dtype and accuracy. Each row has the bound,
the max and mean error, the count over the bound, a per-test breakdown and the
10 worst inputs with their bits.
``spin docs`` renders it on the ULP report page of the testing docs, if it
exists. The page uses the vendored Tabulator in ``doc/_static/tabulator/``.

The grouping and formatting stay in Python. The page only renders the JSON,
so the viewer library can be replaced without touching the report.
"""

# Absolute imports: `spin test` runs pytest --pyargs from site-packages.

import argparse
import json
import pathlib
import time
import zlib
from typing import cast

import numpy as np
import numpy_sr as sr
import pytest
from numpy_sr._conftest.book import WorstBook
from numpy_sr._conftest.report import serialize


class _Reference:
    """A module as a reference target: swallows `accuracy`, ignores errstate."""

    IS_REFERENCE = True

    def __init__(self, name: str, module, **flags) -> None:
        self.__name__, self.module = name, module
        vars(self).update(flags)

    def __getattr__(self, op: str):
        fn = getattr(self.module, op)
        call = np.errstate(all="ignore")(lambda *xs, accuracy=None: fn(*xs))
        # what `_target_of` reads off a bound op
        call.__name__, call.__module__ = op, self.__name__
        return call


DTYPES = ["float32", "float64"]
# The oracles return tuples, so they aren't targets.
_SUPPORTED = {n: t for n, t in sr.targets.items() if t.IS_SUPPORTED}
ORACLES: dict = {n: t for n, t in _SUPPORTED.items() if t.IS_ORACLE}
NPSR_TARGETS: dict = {
    n: t for n, t in _SUPPORTED.items() if not (t.IS_ORACLE or t.IS_REFERENCE)
}
# `sr.targets` already holds the compiled references, so only numpy is manual.
TARGETS: dict = (
    NPSR_TARGETS
    | {n: t for n, t in _SUPPORTED.items() if t.IS_REFERENCE}
    | {
        "numpy": _Reference(
            "numpy", np, HAVE_FLOAT64=True, DTYPES=tuple(DTYPES), HAVE_FMA=True
        )
    }
)
ACCURACIES = {a.name.lower(): a for a in sr.Accuracy}
OPERATIONS = list(sr.target.OPERATIONS)
# Inputs per signature (operations.h); each has the op's dtype.
ARITY = {"Unary": 1, "Binary": 2}
# the books the gate replays, one per --worst-dir; the one --worst-collect writes
BOOKS_KEY = pytest.StashKey[list[WorstBook]]()
COLLECT_KEY = pytest.StashKey[WorstBook | None]()
_PROPS: list = []  # the user_properties of every call report

# Skipped unless opted into with `-m`.
_MARKERS = {
    "exhaustive": "full sweeps, minutes per op; run with -m exhaustive",
    "slow": "heavy worst-case generation, seconds-to-minutes; run with -m slow",
}

# `--<arg>` -> (choices by name, default). Each is also a test parameter name.
_ARG_DEFAULT = {
    "target": (TARGETS, list(NPSR_TARGETS.values())),
    "accuracy": (ACCURACIES, list(sr.Accuracy)),
    "operation": ({n: n for n in OPERATIONS}, OPERATIONS),
    "dtype": ({n: n for n in DTYPES}, DTYPES),
}


def _target_name(target) -> str:
    return target.__name__.rsplit(".", 1)[-1]


def _target_of(func):
    """The `TARGETS` entry `func` is an op of, from its module name."""
    name = func.__module__.rsplit(".", 1)[-1]
    if name not in TARGETS:
        raise TypeError(
            f"{func.__module__}.{func.__name__} is not an op of {list(TARGETS)}"
        )
    return TARGETS[name]


def _csv_choice(choices: dict, label: str):
    """An argparse `type` splitting on commas and validating each name."""
    by_lower = {n.lower(): v for n, v in choices.items()}

    def parse(arg: str) -> list:
        out = []
        for raw in filter(None, (s.strip() for s in arg.split(","))):
            if raw.lower() not in by_lower:
                raise argparse.ArgumentTypeError(
                    f"unknown {label} {raw!r}; choose from {', '.join(choices)}"
                )
            out.append(by_lower[raw.lower()])
        if not out:
            raise argparse.ArgumentTypeError(
                f"no {label} given; choose from {', '.join(choices)}"
            )
        return list(dict.fromkeys(out))  # --target avx2,avx2 runs it once

    return parse


def _split_ids(name: str) -> tuple[str, list[str]]:
    """`test_x[SSE4-Low-sin]` -> `("test_x", ["SSE4", "Low", "sin"])`."""
    base, sep, bracket = name.partition("[")
    return base, bracket.rstrip("]").split("-") if sep else []


def _oracle_name(config: pytest.Config) -> str:
    """The parsed ``--oracle``; it has a default."""
    return cast("str", config.getoption("--oracle"))


def _selected(config: pytest.Config, axis: str) -> list:
    """The parsed ``--<axis>`` list; each axis option has a default."""
    return cast("list", config.getoption(f"--{axis}"))


def _seed(arg: str) -> int:
    """`--seed 42` or `--seed 0x2a`."""
    try:
        return int(arg, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {arg!r}") from None


def _dirs(arg: str) -> list[pathlib.Path]:
    """`--worst-dir a,b` -> its paths, in order, each once."""
    return list(dict.fromkeys(pathlib.Path(s) for s in arg.split(",") if s))


def _source_root(config: pytest.Config) -> pathlib.Path:
    """The repository root, not site-packages."""
    # `spin test` roots pytest in site-packages; walk up to the source ancestor.
    start = pathlib.Path(config.rootpath)
    return next(
        (p for p in (start, *start.parents) if (p / "python/tests").is_dir()), start
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    for axis, (choices, default) in _ARG_DEFAULT.items():
        names = [n for n, v in choices.items() if v in default]
        parser.addoption(
            f"--{axis}",
            default=default,
            metavar="NAMES",
            type=_csv_choice(choices, axis),
            help=f"{axis} names to run, comma-separated (default: {', '.join(names)})",
        )
    parser.addoption(
        "--oracle",
        default="mpfr",
        choices=list(ORACLES),
        help="the one oracle every test scores against, a single name (default: mpfr)",
    )
    parser.addoption(
        "--seed",
        default=int(np.random.SeedSequence().generate_state(1)[0]),
        metavar="INT",
        type=_seed,
        help="root seed for every random stream "
        "(default: drawn fresh, echoed at the end of the run)",
    )
    parser.addoption(
        "--worst-dir",
        metavar="DIRS",
        type=_dirs,
        help="directories of worst-case books the gate replays, comma-separated "
        "(default: python/tests/worst)",
    )
    parser.addoption(
        "--worst-collect",
        metavar="DIR",
        type=pathlib.Path,
        help="fold measured inputs into the worst-case books under DIR "
        "(off by default: the books are replayed as a gate but never rewritten)",
    )


def pytest_configure(config: pytest.Config) -> None:
    _PROPS.clear()  # from an earlier session in this process
    arity = {op: ARITY[sig] for op, sig in sr.target.SIGNATURES.items()}
    oracle, cwd = _oracle_name(config), config.invocation_params.dir
    explicit = config.getoption("--worst-dir")
    # the source tree's books, not the build-install copy; it may have none
    dirs = explicit or [_source_root(config) / "python/tests/worst"]
    collect = config.getoption("--worst-collect")
    book = WorstBook(cwd / collect / oracle, arity) if collect else None
    config.stash[COLLECT_KEY] = book
    books = []
    for d in dirs:
        root = cwd / d / oracle
        if explicit and not (cwd / d).is_dir():
            raise pytest.UsageError(f"--worst-dir: no such directory: {cwd / d}")
        # the collected dir is replayed through the same book, so a replay
        # re-stamps it
        same = book and root.resolve() == book.root.resolve()
        books.append(book if same else WorstBook(root, arity))
        # e.g. one level too deep: the gate would find nothing and skip
        if explicit and not same and not books[-1].all_keys():
            raise pytest.UsageError(f"--worst-dir: no books under {root}")
    config.stash[BOOKS_KEY] = books
    for name, doc in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {doc}")
    wi = getattr(config, "workerinput", None)
    if wi is None:
        return
    if "npsr_seed" in wi:
        config.option.seed = wi["npsr_seed"]
    else:
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(
                "xdist controller is not running the numpy_sr plugin: workers "
                "seed independently, and neither the ULP report nor the "
                "worst-case book is kept"
            ),
            stacklevel=2,
        )


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """The gate: one case per book file, against every target it applies to.

    `target`/`accuracy` are parametrized here rather than taken from their
    fixtures so the key's own axes pick the combinations. The CLI selection
    still deselects the rest, which reads those same param names, `operation`
    and `dtype` included.

    It also rejects a test that takes both `target` and `npsr`.
    """
    if {"target", "npsr"} <= set(metafunc.fixturenames):
        pytest.fail(
            f"{metafunc.function.__name__}: takes both `target` and `npsr`; pick one",
            pytrace=False,
        )
    if "worst_key" not in metafunc.fixturenames:
        return
    books, cases = metafunc.config.stash[BOOKS_KEY], []
    for i, book in enumerate(books):
        tail = f"-dir{i}" if i else ""  # the first dir keeps the plain ids
        for key in book.all_keys():
            operation, acc, dtype, have_fma, source = WorstBook.parse(key)
            accuracy = ACCURACIES.get(acc)
            if accuracy is None:  # a book for an accuracy this build dropped
                continue
            for name, target in TARGETS.items():
                if have_fma != target.HAVE_FMA:
                    continue
                if dtype not in target.DTYPES:
                    continue
                cases.append(
                    pytest.param(
                        target,
                        accuracy,
                        operation,
                        dtype,
                        book,
                        key,
                        id=f"{name}-{accuracy.name}-{operation}-{dtype}-{source}{tail}",
                    )
                )
    if not cases:
        roots = ", ".join(str(b.root) for b in books)
        pytest.skip(f"no worst-case books under {roots}")
    metafunc.parametrize("target,accuracy,operation,dtype,worst_book,worst_key", cases)


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    # The fixtures parametrize over the whole registry; deselect (not skip, which
    # the full reference set would flood) whatever the CLI did not select.
    allowed = {axis: set(_selected(config, axis)) for axis in _ARG_DEFAULT}
    kept, dropped = [], []
    for item in items:
        params = getattr(item, "callspec", None)
        params = dict(params.params) if params else {}
        if "dtype" in params:
            params["dtype"] = np.dtype(params["dtype"]).name
        ok = all(params[k] in v for k, v in allowed.items() if k in params)
        if "npsr" in params:
            ok = ok and params["npsr"] in allowed["target"]
        t = params.get("target", params.get("npsr"))
        if t is not None and "dtype" in params:
            ok = ok and params["dtype"] in t.DTYPES
        (kept if ok else dropped).append(item)
    config.hook.pytest_deselected(items=dropped)
    items[:] = kept

    # The frozen gate guards the whole suite: hoist it whatever file it lives in.
    items.sort(key=lambda it: getattr(it, "originalname", it.name) != "test_worst_book")
    if config.option.markexpr:
        return
    for item in items:
        for mark in _MARKERS:
            if mark in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason=f"{mark}; opt in with -m {mark}")
                )


def pytest_report_header(config: pytest.Config) -> list:
    """The selection echoed at the top and bottom of every run."""
    targets = [
        f"{_target_name(t)} (reference)" if t.IS_REFERENCE else _target_name(t)
        for t in _selected(config, "target")
    ]
    return [
        f"targets:    {', '.join(targets)}",
        f"accuracies: {', '.join(a.name for a in _selected(config, 'accuracy'))}",
        f"operations: {', '.join(_selected(config, 'operation'))}",
        f"dtypes:     {', '.join(_selected(config, 'dtype'))}",
        f"oracle:     {_oracle_name(config)}",
        f"seed:       {config.getoption('--seed')}",
    ]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":  # every phase carries the same properties
        _PROPS.extend(report.user_properties)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    terminalreporter.write_sep("-", "selection")
    for line in pytest_report_header(config):
        terminalreporter.write_line(line)
    book = config.stash[COLLECT_KEY]
    lines = book.report_lines() if book else []
    if lines:
        terminalreporter.write_sep("-", "worst-case discoveries")
        for line in lines:
            terminalreporter.write_line(line)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config, book = session.config, session.config.stash[COLLECT_KEY]
    if hasattr(config, "workerinput"):  # its reports carry everything
        return
    if book:
        # Replays first: they re-stamp every stored row, and a fold landing
        # first would leave them at their old errors.
        for name, args in sorted(_PROPS, key=lambda p: p[0] != "npsr_replay"):
            if name in ("npsr_replay", "npsr_fold"):
                book.fold(*args)
        book.flush()
    records = [v for n, v in _PROPS if n == "npsr_record"]
    if records:
        head = {
            "seed": config.getoption("--seed"),
            "oracle": _oracle_name(config),
            "created": time.time(),
        }
        text = json.dumps(head | serialize(records), indent=2)
        out = _source_root(config) / "doc/testing"
        if not out.is_dir():  # not run from a checkout
            out = config.rootpath
        (out / "ulp-report.json").write_text(text)


# optionalhook: the specs live in xdist, which need not be installed.
@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Controller -> worker: the one seed every random stream derives from."""
    node.workerinput["npsr_seed"] = node.config.getoption("--seed")


@pytest.fixture(scope="session")
def targets(request: pytest.FixtureRequest) -> list:
    """Every selected target at once, for tests that compare backends.

    ``--target`` selects only NPSR and reference targets, never the oracle;
    use the ``oracle`` fixture for it.
    """
    return _selected(request.config, "target")


@pytest.fixture(scope="session")
def accuracies(request: pytest.FixtureRequest) -> list:
    """Every selected accuracy at once, for tests that sweep them internally."""
    return _selected(request.config, "accuracy")


@pytest.fixture(scope="session")
def oracle(request: pytest.FixtureRequest) -> sr.target.Oracle:
    """The ``--oracle`` the tests score against; ops return ``(ref, residual)``."""
    return ORACLES[_oracle_name(request.config)]


@pytest.fixture(params=list(TARGETS.values()), ids=list(TARGETS))
def target(request: pytest.FixtureRequest):
    """The backend under test; CLI selection deselects the rest at collection.

    Only NPSR and reference targets; the oracle is never a ``target``.
    """
    return request.param


@pytest.fixture(params=list(NPSR_TARGETS.values()), ids=list(NPSR_TARGETS))
def npsr(request: pytest.FixtureRequest):
    """An NPSR target under test: ``target`` without the references."""
    return request.param


@pytest.fixture(params=list(sr.Accuracy), ids=[a.name for a in sr.Accuracy])
def accuracy(request: pytest.FixtureRequest):
    """The `sr.Accuracy` under test; unselected ones deselected at collection."""
    return request.param


@pytest.fixture
def rand(request: pytest.FixtureRequest) -> sr.utils.Rand:
    """Per-test `sr.utils.Rand`, seeded with ``(--seed, crc32(salt))``.

    The salt is the file name, the test name and its parameter ids, with the
    target id removed. So:

    - every target gets the same data in a test, and a test can cache the
      inputs and the oracle's reference once for all of them, keyed on ``rand``;
    - the data doesn't depend on which worker runs the test or in which order;
    - rerunning with the printed seed reproduces a failure.
    """
    node = request.node
    base, segments = _split_ids(node.name)
    for name in ("target", "npsr"):
        if name in request.fixturenames:
            target_id = _target_name(request.getfixturevalue(name))
            segments = [s for s in segments if s != target_id]
    salt = zlib.crc32("|".join([node.path.name, base, *segments]).encode())
    return sr.utils.Rand(cast("int", request.config.getoption("--seed")), salt)


@pytest.fixture
def test(request: pytest.FixtureRequest):
    """`sr.testing.assert_max_ulp` bound to the session's report and book.

    Every accuracy test goes through it. Returns ``check(func, args, ref,
    maxulp, *, accuracy, replay=None)``, which calls
    ``func(*args, accuracy=accuracy)``, builds a record with
    `sr.testing.measure` (size, max and mean error, count over the bound, the
    16 worst inputs), adds it to the session's report, asserts the bound
    (references only record) and returns ``func``'s output.

    - ``func``: an op of a target, ``target.sin``; the target and the
      operation are read off it.
    - ``args``: the input, or a tuple with one array per argument.
    - ``ref``: what the oracle returns, ``(ref, residual)``.
    - ``maxulp``: as for `sr.testing.measure`.
    - ``accuracy``: the `sr.Accuracy` passed to ``func``.
    - ``replay``: under ``--worst-collect``, an NPSR target's ``(args,
      |error|)`` pairs and ``maxulp`` are folded into this test's book. With
      the gate's ``(book, key)``, they re-stamp that book instead if it's the
      collected one; the gate only, see `the book <The worst-case book_>`_.

    Bit-exact checks don't go through it. They use
    ``sr.testing.assert_bits(got, want, x, msg)``, which compares the IEEE bit
    patterns, so ``-0.0``, NaN payloads and signs all count.
    """
    config, node = request.config, request.node
    book = config.stash[COLLECT_KEY]
    label = node.originalname

    def check(func, args, ref, maxulp, *, accuracy, replay=None):
        target, operation = _target_of(func), func.__name__
        xs = args if isinstance(args, tuple) else (args,)
        a = np.asarray(func(*xs, accuracy=accuracy))
        # set after the call, so a kernel error keeps this frame
        __tracebackhide__ = True
        # Only a non-reference target's error belongs in the book. An archive
        # book is only replayed; the collected one is re-stamped.
        if book and not target.IS_REFERENCE and (not replay or replay[0] is book):
            xr = tuple(
                np.broadcast_to(
                    np.asarray(v).astype(a.dtype, copy=False), a.shape
                ).ravel()
                for v in xs
            )
            _, b, res = sr.testing._operands(a, ref)
            rr = (b.ravel(), res.ravel())
            er = np.abs(sr.testing.ulp_error(a, (b, res))).ravel()
            if replay:
                # a replay measures every stored input: stamp the book with what
                # this kernel actually does, up or down
                name, key, bound = "npsr_replay", replay[1], None
            else:
                name, bound = "npsr_fold", float(maxulp)
                key = WorstBook.key(
                    operation, accuracy.name, str(a.dtype), target.HAVE_FMA, label
                )
            node.user_properties.append(
                (name, (key, book.rows(key, xr, rr, er), bound))
            )

        record = sr.testing.measure(
            a,
            ref,
            maxulp,
            x=xs,
            label=label,
            target=_target_name(target),
            fma="FMA" if target.HAVE_FMA else "non-FMA",
            accuracy=accuracy.name,
            operation=operation,
        )
        node.user_properties.append(("npsr_record", record))
        # A reference target is not the oracle: recorded, not asserted.
        if not target.IS_REFERENCE:
            sr.testing.check(record)
        return a

    return check
