project(
  'svml',
  'c',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1',
)

# npsr's -Doperation, yielded (meson_options.txt).
operations = get_option('operation')
if operations.length() == 0
  error('svml: -Doperation is required')
endif
svml_srcs = []
foreach op : operations
  foreach v : ['d_ha', 'd_la', 's_ha', 's_la']
    svml_srcs += files('linux/avx512/svml_z0_@0@_@1@.s'.format(op, v))
  endforeach
endforeach

svml_lib = static_library(
  'svml',
  svml_srcs,
  pic: true,
)

svml_dep = declare_dependency(link_with: svml_lib)
meson.override_dependency('svml', svml_dep)
