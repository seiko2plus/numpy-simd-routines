"""The selectable target and accuracy registries."""

from collections.abc import Callable

import numpy as np
import numpy_sr as sr


class _TargetMock(type):
    """A module as a reference target: swallows `accuracy`, ignores errstate."""

    IS_REFERENCE = True

    def __new__(mcls, name: str, module, flags: dict) -> type:
        return super().__new__(mcls, name, (), {"MODULE": module, **flags})

    def __getattr__(cls, op: str) -> Callable:
        fn = getattr(cls.MODULE, op)

        @np.errstate(all="ignore")
        def call(*args, accuracy=None):
            return fn(*args)

        return call


# `sr.targets` already holds the compiled references, so only numpy is manual.
# Excluded: what this CPU cannot run, and the MPFR oracle (returns tuples).
TARGETS: dict = {
    name: target
    for name, target in sr.targets.items()
    if target.IS_SUPPORTED and not target.IS_ORACLE
} | {
    "numpy": _TargetMock("numpy", np, {"HAVE_FLOAT64": True, "HAVE_FMA": True}),
}

# References stay selectable via --target but out of the default run.
DEFAULT_TARGETS: list = [
    name
    for name, target in TARGETS.items()
    if not getattr(target, "IS_REFERENCE", False)
]

ACCURACIES: dict = {a.name.lower(): a for a in sr.Accuracy}
DEFAULT_ACCURACIES: list = list(sr.Accuracy)
