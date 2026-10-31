"""numpy_sr.target against the submodules it types."""

import numpy_sr as sr
from pytest import mark


@mark.parametrize("name", sorted(sr.targets))
def test_operations(name):
    """Every target binds exactly meson's op list (OPERATIONS), or none if this
    CPU can't run it."""
    backend = sr.targets[name]
    ops = sorted(n for n in dir(backend) if not n.startswith("_"))
    ops = [n for n in ops if callable(getattr(backend, n))]
    want = sr.target.OPERATIONS if backend.IS_SUPPORTED else ()
    assert tuple(ops) == want
