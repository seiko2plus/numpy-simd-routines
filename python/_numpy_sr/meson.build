hwy_proj = subproject(
  'highway',
  default_options: [
    'contrib=disabled',
    'examples=disabled',
    # 'tests' alone still pulls googletest; 'test_standalone' cuts it too.
    'tests=disabled',
    'test_standalone=true',
  ],
)
hwy_dep = hwy_proj.get_variable('hwy_dep')

cpp = meson.get_compiler('cpp')
mpfr_dep = dependency('mpfr', required: false)
mpfr_rpath = ''
if mpfr_dep.found()
  mpfr_rpath = mpfr_dep.get_variable(pkgconfig: 'libdir', default_value: '')
else
  mpfr_dep = cpp.find_library('mpfr', has_headers: ['mpfr.h'])
endif


pybind11_dep = dependency('pybind11')

# Intel SVML: an AVX-512 comparison reference. numpy/SVML ships x86-64 Linux
# assembly only; elsewhere the target binds no ops (NPSR_HAVE_SVML, target.h).
svml_enabled = (
  host_machine.system() == 'linux' and host_machine.cpu_family() == 'x86_64'
)
svml_link = []
svml_cpp_args = ['-DNPSR_HAVE_SVML=0']
if svml_enabled
  svml_dep = subproject('svml').get_variable('svml_dep')
  # Isolated in its own -mavx512f library so the module still loads at baseline
  # ISA; these kernels only run when explicitly called (runtime-guarded).
  svml_link = [static_library(
    'npsr_svml',
    'svml.cpp',
    include_directories: npsr_inc,
    # NPSR_HAVE_SVML must match module.cpp's view of TargetSVML::kIsEnabled.
    cpp_args: ['-mavx512f', '-DNPSR_HAVE_SVML=1'],
    gnu_symbol_visibility: 'hidden',
    # hwy_dep: target.h reads HWY_AVX3 for the ISA gate (headers only).
    dependencies: [pybind11_dep, hwy_dep, svml_dep],
    pic: true,
  )]
  svml_cpp_args = ['-DNPSR_HAVE_SVML=1']
endif

# One foreach_target TU per op: SIMD variants build in parallel and touching
# one op recompiles only that op.
op_sources = [
  'sin.cpp',
  'cos.cpp',
]

py.extension_module(
  '_numpy_sr',
  ['module.cpp', 'mpfr.cpp', 'ulp.cpp'] + op_sources,
  include_directories: npsr_inc,
  cpp_args: svml_cpp_args,
  dependencies: [pybind11_dep, hwy_dep, mpfr_dep],
  link_with: svml_link,
  install_rpath: mpfr_rpath,
  # pybind11's namespace is hidden; matching it keeps our own glue types from
  # out-scoping their members (PYBIND11_MODULE still exports PyInit_).
  gnu_symbol_visibility: 'hidden',
  subdir: 'numpy_sr',
  install: true,
)
