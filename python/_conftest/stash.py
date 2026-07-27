"""Session stash keys; one module so writer and readers share key identity."""

import pytest

from .book import WorstBook

# One record per ULP comparison (pass or fail), drained into the JSON report.
REPORT_KEY = pytest.StashKey[list]()

# The frozen worst-case book (tests/worst/*.csv), folded during the run.
BOOK_KEY = pytest.StashKey[WorstBook]()
