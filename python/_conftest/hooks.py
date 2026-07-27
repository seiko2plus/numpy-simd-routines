"""Session setup and collection hooks: stash, markers, selection, ordering."""

import pathlib

import pytest

from .book import WorstBook
from .options import selected_accuracies, selected_targets
from .stash import BOOK_KEY, REPORT_KEY
from .xdist import warn_unmerged


def _book_dir(config: pytest.Config) -> pathlib.Path:
    """The committed `python/tests/worst`, not the build-install copy."""
    # `spin test` roots pytest in site-packages; walk up to the source ancestor.
    start = pathlib.Path(config.rootpath)
    for p in (start, *start.parents):
        if (p / "python" / "tests").is_dir():
            return p / "python" / "tests" / "worst"
    return start / "python" / "tests" / "worst"


def pytest_configure(config: pytest.Config) -> None:
    config.stash[REPORT_KEY] = []
    config.stash[BOOK_KEY] = WorstBook(_book_dir(config))
    warn_unmerged(config)
    config.addinivalue_line(
        "markers",
        "exhaustive: full f32 sweep, minutes per op; run with -m exhaustive",
    )
    config.addinivalue_line(
        "markers",
        "slow: heavy worst-case generation, seconds-to-minutes; run with -m slow",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # The fixtures parametrize over the whole registry; deselect (not skip, which
    # the full reference set would flood) whatever the CLI did not select.
    sel_targets = set(selected_targets(config).values())
    sel_accuracies = set(selected_accuracies(config))
    kept, dropped = [], []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        keep = ("target" not in params or params["target"] in sel_targets) and (
            "accuracy" not in params or params["accuracy"] in sel_accuracies
        )
        (kept if keep else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept

    # The frozen gate guards the whole suite: hoist it whatever file it lives in.
    items.sort(key=lambda it: getattr(it, "originalname", it.name) != "test_worst_book")
    if config.option.markexpr:
        return
    for item in items:
        for mark in ["exhaustive", "slow"]:
            if mark in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason=f"{mark}; opt in with -m {mark}")
                )
