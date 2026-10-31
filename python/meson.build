py = import('python').find_installation(pure: false)

# The signature (operations.h) of every operation -Doperation can name.
signatures = {
  'sin': 'Unary',
  'cos': 'Unary',
}
operations = {}
foreach op : get_option('operation')
  if op not in signatures
    error('unknown operation @0@; known: @1@'.format(op, ', '.join(signatures.keys())))
  endif
  operations += {op: signatures[op]}
endforeach

# install_sources flattens to basenames: tests/__init__.py would clobber the
# package __init__.py if shared with this call, so tests get their own.
py.install_sources(
  [
    '__init__.py',
    'argred.py',
    'fpenv.py',
    'target.py',
    'testing.py',
    'utils.py',
    'conftest.py',
    '_numpy_sr.pyi',  # type stub for the compiled extension (sibling of the .so)
  ],
  subdir: 'numpy_sr',
)
# install_sources flattens to basenames, so _conftest gets its own subdir.
py.install_sources(
  [
    '_conftest/__init__.py',
    '_conftest/report.py',
  ],
  subdir: 'numpy_sr/_conftest',
)
py.install_sources(
  [
    'tests/__init__.py',
    'tests/test_argred.py',
    'tests/test_report.py',
    'tests/test_target.py',
    'tests/test_trig.py',
  ],
  subdir: 'numpy_sr/tests',
)
subdir('_numpy_sr')

