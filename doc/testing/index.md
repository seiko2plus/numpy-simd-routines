# Testing

The NPSR (numpy-simd-routines) tests are part C++, part Python. C++ compiles
the kernels, once for each SIMD target, into a Python extension, and Python
drives the tests.

## Requirements

- A C++17 compiler.
- MPFR (`libmpfr-dev` on Debian/Ubuntu, `mpfr` on FreeBSD and Homebrew).
- Python 3.12 or newer, with the packages in `test_requirements.txt`:

  ```sh
  pip install -r test_requirements.txt
  ```

The other requirements come as meson subprojects: Highway, and SVML, which
serves only as a reference target for comparison.

## Floating-point environment

The kernels assume the default rounding mode, round to nearest, and the
tests run only under it. The ULP bounds don't hold under the directed modes
(upward, downward, toward zero), and those modes aren't tested.

## Running

```sh
spin build              # meson build into ./build, installs into ./build-install
spin test               # builds if needed, then runs the suite
spin test -j auto       # in parallel, one xdist worker per core
```

The suite is driven by [spin](https://github.com/scientific-python/spin), a
developer tool for scientific Python projects, installed with
`test_requirements.txt`. `spin test` builds the package if needed, then
runs pytest, and the table below lists the options it takes. spin's other
commands (`spin build`, `spin docs`, ...) have their own options;
see `spin --help`.

| Option | Meaning |
|---|---|
| `--target avx2,sse4` | Only these targets (case-insensitive). Default: every NPSR target this CPU supports. The {ref}`reference targets <testing/api/numpy_sr.conftest:Reference targets>` run only when named. The oracle can't be selected; use `--oracle`. |
| `--operation sin,cos` | Only these operations' tests. Default: all. |
| `--dtype float32,float64` | Only these dtypes' tests. Default: both. |
| `--accuracy high,low` | Only these accuracy profiles (case-insensitive). Default: all. |
| `--oracle mpfr` | The oracle the tests score against. Takes a single name, not a list. Default: `mpfr`. |
| `--seed N` | Root seed for all random data, any Python integer literal (`12345`, `0x1f`). Default: a fresh one, printed in the header and the summary. See the {py:func}`rand <numpy_sr.conftest.rand>` fixture. |
| `-m slow`, `-m exhaustive`, `-m all` | Marker-gated tests, see {ref}`Markers <testing/api/numpy_sr.conftest:Markers>`. Comma-separated or repeated: `-m slow,exhaustive`. |
| `-t numpy_sr.tests.<module>::<test>` | Run one module, class or test. |
| `-j N` / `-j auto` | xdist workers. |
| `-v`, `--verbose` | Verbose output. |
| `--no-build` | Don't build before running the tests. |
| `-C DIR`, `--build-dir DIR` | Meson build directory, installed into `DIR-install`. Default: `build`, or `$SPIN_BUILD_DIR`. |

Anything after `--` goes to pytest unchanged:

```sh
spin test --target avx2 -- -k "float64" -x
```

For details on how a run works, the reference targets, markers and
the ULP report, see [`numpy_sr.conftest`](api/numpy_sr.conftest.md).

## Reporting

Every run writes `doc/testing/ulp-report.json`. For each target,
operation, dtype and accuracy, it holds the bound, the max and mean ULP
error, the number of inputs over the bound, and the worst inputs with their
bits. The terminal summary repeats the targets, accuracies and seed of the
run.

`spin docs` renders the JSON, if it exists, on the [ULP report](ulp-report.md)
page.

## Going further

- [ULP report](ulp-report.md): the report of the last run.
- [Python](python.md): the `numpy_sr` API, one page per module.
  - [`numpy_sr.target`](api/numpy_sr.target.md): the extension, its targets
    and the MPFR oracle.
  - [`numpy_sr.testing`](api/numpy_sr.testing.md): the ULP comparator.
  - [`numpy_sr.conftest`](api/numpy_sr.conftest.md): the pytest plugin:
    a run, terms, selection and the reference targets, fixtures, markers,
    xdist and the ULP report.
- [Test strategies](strategies.md): how each kind of math function is tested.
- [Expand tests](expand-tests.md): every file a new kernel
  touches to get tested.

```{toctree}
:hidden:

ulp-report
python
strategies
expand-tests
```
