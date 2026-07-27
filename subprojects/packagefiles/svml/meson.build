project(
  'svml',
  'c',
  license: 'BSD-3-Clause',
  meson_version: '>=1.1',
)

svml_srcs = files(
  'linux/avx512/svml_z0_cos_d_ha.s',
  'linux/avx512/svml_z0_cos_d_la.s',
  'linux/avx512/svml_z0_cos_s_ha.s',
  'linux/avx512/svml_z0_cos_s_la.s',
  'linux/avx512/svml_z0_sin_d_ha.s',
  'linux/avx512/svml_z0_sin_d_la.s',
  'linux/avx512/svml_z0_sin_s_ha.s',
  'linux/avx512/svml_z0_sin_s_la.s',
)

svml_lib = static_library(
  'svml',
  svml_srcs,
  pic: true,
)

svml_dep = declare_dependency(link_with: svml_lib)
meson.override_dependency('svml', svml_dep)
