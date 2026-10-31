# Expand tests

Getting a new kernel `npsr::HWY_NAMESPACE::<Op>(prec, v)` in `npsr/` tested
takes three phases: bind it to Python, test it, then collect its worst
cases. `<op>` is the op's Python name and `<Op>` its kernel name.

## 1. Bind it to Python

1. `python/meson.build`: add `'<op>': '<Signature>'` to `signatures`, where
   `<Signature>` is one of the signatures in `operations.h`.
2. `meson_options.txt`: add `'<op>'` to the default of `operation`.

That's all the binding needs. The build adds `Operation::k<Op>` (meson
passes the list to the compiler as `NPSR_OPERATIONS`), generates the op's
TU, holding every target, from `npsr-op.tmpl.cpp`, and the typed `<op>` of
`Target` and `Oracle` in `numpy_sr.target`.

The name is the Python name, the book directory and the name of the NumPy
ufunc the `numpy` reference calls, so use the NumPy name, without `_`.
`operation-inl.h` binds `sr::<Op>(prec, v...)`, one vector per input of the
signature, so the kernel needs that signature.

`mpfr.cpp` and `svml.cpp` bind every op through `mpfr_<op>` and
`__svml_<op>f16`, `__svml_<op>8` and their `_ha` variants. The SVML
subproject builds the op's `svml_z0_<op>_*.s` files from the same
`operation` list; if SVML has no files for the op, configure fails.

## 2. Test it

A new test module is only needed when the op needs a new test strategy.
Otherwise, add the op to the existing module that already tests its family.

The op's tests need at least:

- a bound per accuracy and dtype;
- accuracy tests that call `test(target.<op>, x, sr.mpfr.<op>(x), bound,
  accuracy=...)`. The bound is stored in the book, and the gate replays the
  book with it;
- an `exhaustive` sweep, see [Test strategies](strategies.md);
- bit-exact tests for what must hold exactly: specials, symmetry, tiny
  inputs, and FP flags with `sr.FPEnv`.

`python/utils.py` has the input generators (`Rand`, `binades`,
`float_range`, `ulp_neighbors`, `both_signs`).

For a new module:

- write its strategy in the module docstring and add it to
  [Test strategies](strategies.md) with an `automodule`;
- add it to the `tests/` `install_sources` in `python/meson.build`. Tests
  that aren't installed don't run.

## 3. Collect the worst cases

Create the books and commit them:

```sh
spin test --worst-collect python/tests/worst -m all -j auto
git add python/tests/worst/mpfr/<op>
```
