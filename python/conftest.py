"""``python/conftest.py`` holds the hooks, fixtures and the target registry.

A run
-----

1. **Configure.** The plugin parses ``--target``, ``--accuracy``, ``--oracle`` and
   ``--seed``.
2. **Collect.** The ``target``, ``npsr`` and ``accuracy`` fixtures
   parametrize over every target and accuracy. Collection then deselects
   whatever the CLI did not select, and skips ``slow``/``exhaustive`` tests
   unless a ``-m`` was given.
3. **Tests.** Each test computes inputs, gets the reference from the oracle and
   passes both with the target's op to ``test``, which calls the op and
   asserts the bound.

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
  (``HAVE_FMA``). The kernels take different paths for each.

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

Selected next to NPSR targets (``--target avx2,numpy``), they run beside them.
Their ULP errors never fail a test. The bit-exact tests take ``npsr``, so they
don't run on them.

Markers
-------

``slow`` and ``exhaustive`` tests get a skip marker at collection unless
pytest got an ``-m`` expression. ``spin test -m slow`` passes ``-m slow``,
which runs only those tests. ``-m all`` passes an expression that keeps the
regular tests too. A pytest ``-m`` expression after ``--`` works too, but not
together with spin's ``-m``.

xdist
-----

``pytest_configure_node`` sends every worker the same seed.
"""

# Absolute imports: `spin test` runs pytest --pyargs from site-packages.

import argparse
import zlib
from typing import cast

import numpy as np
import numpy_sr as sr
import pytest


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


def pytest_configure(config: pytest.Config) -> None:
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
                "seed independently"
            ),
            stacklevel=2,
        )


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Reject a test that takes both `target` and `npsr`."""
    if {"target", "npsr"} <= set(metafunc.fixturenames):
        pytest.fail(
            f"{metafunc.function.__name__}: takes both `target` and `npsr`; pick one",
            pytrace=False,
        )


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


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    terminalreporter.write_sep("-", "selection")
    for line in pytest_report_header(config):
        terminalreporter.write_line(line)


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
    """`sr.testing.assert_max_ulp` bound to the test.

    Every accuracy test goes through it. Returns ``check(func, args, ref,
    maxulp, *, accuracy)``, which calls
    ``func(*args, accuracy=accuracy)``, builds a record with
    `sr.testing.measure` (size, max and mean error, count over the bound, the
    16 worst inputs), asserts the bound
    (references only record) and returns ``func``'s output.

    - ``func``: an op of a target, ``target.sin``; the target and the
      operation are read off it.
    - ``args``: the input, or a tuple with one array per argument.
    - ``ref``: what the oracle returns, ``(ref, residual)``.
    - ``maxulp``: as for `sr.testing.measure`.
    - ``accuracy``: the `sr.Accuracy` passed to ``func``.

    Bit-exact checks don't go through it. They use
    ``sr.testing.assert_bits(got, want, x, msg)``, which compares the IEEE bit
    patterns, so ``-0.0``, NaN payloads and signs all count.
    """
    node = request.node
    label = node.originalname

    def check(func, args, ref, maxulp, *, accuracy):
        target, operation = _target_of(func), func.__name__
        xs = args if isinstance(args, tuple) else (args,)
        a = np.asarray(func(*xs, accuracy=accuracy))
        # set after the call, so a kernel error keeps this frame
        __tracebackhide__ = True
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
        # A reference target is not the oracle: recorded, not asserted.
        if not target.IS_REFERENCE:
            sr.testing.check(record)
        return a

    return check
