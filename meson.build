
project(
  'npsr',
  'cpp',
  version: '0.1.0',
  meson_version: '>= 1.3.0',
  license: 'BSD-3-Clause',
  default_options: [
    'cpp_std=c++17',
    'buildtype=release',
  ],
)

npsr_inc = include_directories('.')

# Builds the `numpy_sr` Python package: one foreach_target TU per operation
# (python/_numpy_sr/<op>.cpp) linked into a single pybind11 extension that
# exposes a submodule per runnable Highway target.
if get_option('tests').disable_auto_if(meson.is_subproject()).allowed()
  subdir('python')
endif
