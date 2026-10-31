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

py.install_sources(
  [
    '__init__.py',
    'fpenv.py',
    'target.py',
    'utils.py',
    '_numpy_sr.pyi',  # type stub for the compiled extension (sibling of the .so)
  ],
  subdir: 'numpy_sr',
)
subdir('_numpy_sr')

