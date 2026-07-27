py = import('python').find_installation()
# install_sources flattens to basenames: tests/__init__.py would clobber the
# package __init__.py if shared with this call, so tests get their own.
py.install_sources(
  [
    '__init__.py',
    'argred.py',
    'fpenv.py',
    'ieee754.py',
    'testing.py',
    'conftest.py',
    '_numpy_sr.pyi',  # type stub for the compiled extension (sibling of the .so)
  ],
  subdir: 'numpy_sr',
)
# conftest.py is a thin aggregator; its implementation lives here, one module
# per concern (install_sources flattens to basenames, so its own subdir).
py.install_sources(
  [
    '_conftest/__init__.py',
    '_conftest/book.py',
    '_conftest/fixtures.py',
    '_conftest/hooks.py',
    '_conftest/options.py',
    '_conftest/report.py',
    '_conftest/stash.py',
    '_conftest/targets.py',
    '_conftest/xdist.py',
  ],
  subdir: 'numpy_sr/_conftest',
)
py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_trig.py',
    'tests/test_worst.py',
  ],
  subdir: 'numpy_sr/tests',
)
subdir('_numpy_sr')

