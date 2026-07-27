"""Pytest plugin for numpy_sr; the re-exports below are the whole job."""

# Absolute imports: `spin test` runs pytest --pyargs from site-packages.

from numpy_sr._conftest.fixtures import (
    accuracies,
    accuracy,
    report_ulp,
    rng,
    target,
    targets,
    worst_book,
)
from numpy_sr._conftest.hooks import (
    pytest_collection_modifyitems,
    pytest_configure,
)
from numpy_sr._conftest.options import pytest_addoption
from numpy_sr._conftest.report import (
    pytest_report_header,
    pytest_sessionfinish,
    pytest_terminal_summary,
)
from numpy_sr._conftest.xdist import (
    pytest_configure_node,
    pytest_testnodedown,
)

__all__ = [
    "accuracies",
    "accuracy",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_configure_node",
    "pytest_report_header",
    "pytest_sessionfinish",
    "pytest_terminal_summary",
    "pytest_testnodedown",
    "report_ulp",
    "rng",
    "target",
    "targets",
    "worst_book",
]
