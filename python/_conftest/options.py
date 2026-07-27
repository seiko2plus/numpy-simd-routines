"""The `--target`/`--accuracy`/`--seed`/`--collect-worst` selection axes."""

import argparse
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from .targets import ACCURACIES, DEFAULT_ACCURACIES, DEFAULT_TARGETS, TARGETS


def _csv_choice(choices: dict, label: str) -> Callable[[str], list]:
    """An argparse `type` splitting on commas and validating each name."""

    def parse(arg: str) -> list:
        out = []
        for raw in arg.split(","):
            key = raw.strip().lower()
            if not key:
                continue
            if key not in choices:
                raise argparse.ArgumentTypeError(
                    f"unknown {label} {raw.strip()!r}; choose from {', '.join(choices)}"
                )
            out.append(choices[key])
        return out

    return parse


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--target",
        default=DEFAULT_TARGETS,
        metavar="NAMES",
        type=_csv_choice({n.lower(): n for n in TARGETS}, "target"),
        help="restrict targets; comma-separated "
        f"(default: {', '.join(DEFAULT_TARGETS)})",
    )
    parser.addoption(
        "--accuracy",
        default=DEFAULT_ACCURACIES,
        metavar="NAMES",
        type=_csv_choice(ACCURACIES, "accuracy"),
        help="restrict accuracies; comma-separated "
        f"(default: {', '.join(a.name.lower() for a in DEFAULT_ACCURACIES)})",
    )
    parser.addoption(
        "--seed",
        default=int(np.random.SeedSequence().generate_state(1)[0]),
        metavar="INT",
        type=lambda s: int(s, 0),
        help="root seed for every random stream "
        "(default: drawn fresh, echoed at the end of the run)",
    )
    parser.addoption(
        "--collect-worst",
        action="store_true",
        default=False,
        help="fold measured inputs into the worst-case book "
        "(off by default: the book is replayed as a gate but never rewritten)",
    )


def selected_targets(config: pytest.Config) -> dict:
    # the dict drops repeats from e.g. --target avx2,avx2
    selected = cast(list, config.getoption("--target"))
    return {n: TARGETS[n] for n in selected}


def selected_accuracies(config: pytest.Config) -> list:
    selected = cast(list, config.getoption("--accuracy"))
    return list(dict.fromkeys(selected))
