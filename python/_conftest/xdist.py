"""xdist glue: one seed per run, worker results merged back on the controller."""

from typing import cast

import pytest

from .stash import BOOK_KEY, REPORT_KEY


def worker_output(config: pytest.Config) -> dict | None:
    """The channel back to the controller; `None` unless this is a worker."""
    return getattr(config, "workeroutput", None)


def resolve_seed(config: pytest.Config) -> int:
    """The run's root seed: the controller's, else this process's own default."""
    wi = getattr(config, "workerinput", None)
    if wi is not None and "npsr_seed" in wi:
        return int(wi["npsr_seed"])
    return cast(int, config.getoption("--seed"))


def warn_unmerged(config: pytest.Config) -> None:
    """A worker whose controller lacks this plugin: its results are dropped."""
    wi = getattr(config, "workerinput", None)
    if wi is not None and "npsr_seed" not in wi:
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(
                "xdist controller is not running the numpy_sr plugin: workers "
                "seed independently, and neither the ULP report nor the "
                "worst-case book is kept"
            ),
            stacklevel=2,
        )


# optionalhook: the specs live in xdist, which need not be installed.
@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Controller -> worker: the one seed every random stream derives from."""
    node.workerinput["npsr_seed"] = resolve_seed(node.config)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """Absorb a finished worker's report records and book deltas."""
    out = getattr(node, "workeroutput", None) or {}  # a crashed node sends none
    config = node.config
    config.stash[REPORT_KEY].extend(out.get("npsr_records", []))
    config.stash[BOOK_KEY].absorb(
        out.get("npsr_book", {}), out.get("npsr_restamped", {})
    )
