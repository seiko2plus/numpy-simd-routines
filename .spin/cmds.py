import os
import pathlib
import shutil
import sys

import click
import spin.cmds.meson as _meson

curdir = pathlib.Path(__file__).parent
rootdir = curdir.parent
toolsdir = rootdir / "tools"
sys.path.insert(0, str(toolsdir))

PACKAGE = "numpy_sr"

# Registered in `python/conftest.py`; both are skipped unless opted into.
MARKERS = ("slow", "exhaustive")

LINT_PATHS = ("python", ".spin", "tools/sollya")


def _markexpr(markers):
    """`--marker slow,exhaustive` -> pytest's `-m "slow or exhaustive"`.

    The conftest only arms its skip-the-marked gate while pytest's mark
    expression is empty, so `all` hands over a tautology: it disarms the gate
    without narrowing the selection."""
    names = list(
        dict.fromkeys(n for spec in markers for n in spec.split(",") if n.strip())
    )
    choices = (*MARKERS, "all")
    unknown = [n for n in names if n not in choices]
    if unknown:
        raise click.BadParameter(
            f"unknown marker(s): {', '.join(unknown)}; pick from {', '.join(choices)}"
        )
    if not names:
        return None
    if "all" in names:
        return f"{MARKERS[0]} or not {MARKERS[0]}"
    return " or ".join(names)


def _is_path(arg):
    return not arg.startswith("-") and (
        arg.endswith(".py") or "::" in arg or os.sep in arg
    )


def _as_path(arg, build_dir):
    """`numpy_sr.tests.test_trig[::node]` -> its installed file. A `--pyargs`
    module selection loads conftest.py too late to register --target & co,
    so a selection has to reach pytest as a path. Returns None if unresolved."""
    if not (arg == PACKAGE or arg.startswith(f"{PACKAGE}.")):
        return None
    module, sep, node = arg.partition("::")
    base = pathlib.Path(_meson._get_site_packages(build_dir), *module.split("."))
    for candidate in (base.with_suffix(".py"), base):
        if candidate.exists():
            return f"{candidate}{sep}{node}"
    return None


def _selection(pytest_args, tests, build_dir):
    """Resolve test selections to paths and re-add the `--pyargs numpy_sr` that
    upstream drops as soon as any pytest argument is given -- without it pytest
    collects the whole of site-packages."""
    args = tuple(_as_path(a, build_dir) or a for a in pytest_args)
    resolved = _as_path(tests, build_dir) if tests else None
    if resolved:
        args, tests = (*args, resolved), None
    if tests or any(_is_path(a) for a in args):
        return args, tests
    return ("--pyargs", PACKAGE, *args), tests


@click.command(
    help=f"""🔧 Run tests

    Wraps `spin.cmds.meson.test`, exposing the suite's own pytest options
    directly and keeping collection rooted at `--pyargs {PACKAGE}`:

    \b
      spin test --target avx2,sse4
      spin test --accuracy high --seed 12345
      spin test --operation sin --dtype float32
      spin test --marker slow

    Plain pytest options still go after `--`:

    \b
      spin test --target sse2 -- -k trig -x
    """
)
@click.argument("pytest_args", nargs=-1)
@click.option(
    "--target",
    metavar="NAMES",
    help="Restrict targets; comma-separated (default: every supported NPSR "
    "target; references run only when named).",
)
@click.option(
    "--accuracy",
    metavar="NAMES",
    help="Restrict accuracy profiles; comma-separated (default: high,low).",
)
@click.option(
    "--operation",
    metavar="NAMES",
    help="Restrict operations; comma-separated (default: all).",
)
@click.option(
    "--dtype",
    metavar="NAMES",
    help="Restrict dtypes; comma-separated (default: float32,float64).",
)
@click.option(
    "--oracle",
    metavar="NAME",
    help="The oracle the tests score against; a single name, not a list "
    "(default: mpfr).",
)
@click.option(
    "--seed",
    metavar="INT",
    help="Root seed for every random stream (default: drawn fresh, echoed at "
    "the end of the run).",
)
@click.option(
    "--marker",
    "-m",
    "markers",
    metavar="NAMES",
    multiple=True,
    help="Opt into marker-gated tests, comma-separated or repeated: "
    f"{', '.join(MARKERS)}, or `all` to add them to the regular run. Naming "
    "one runs only its tests (default: neither, both are skipped).",
)
@click.option(
    "-j",
    "n_jobs",
    metavar="N_JOBS",
    default="1",
    help="Number of xdist workers; `auto` for all cores.",
)
@click.option(
    "--tests",
    "-t",
    metavar="TESTS",
    help=f"Module, class or function to run, e.g. {PACKAGE}.tests.test_trig.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
@_meson.build_option
@_meson.build_dir_option
@click.pass_context
def test(
    ctx,
    *,
    pytest_args,
    target,
    accuracy,
    operation,
    dtype,
    oracle,
    seed,
    markers,
    n_jobs,
    tests,
    verbose,
    build,
    build_dir,
):
    suite_args = []
    for name, value in (
        ("--target", target),
        ("--accuracy", accuracy),
        ("--operation", operation),
        ("--dtype", dtype),
        ("--oracle", oracle),
        ("--seed", seed),
    ):
        if value is not None:
            suite_args += [name, value]

    markexpr = _markexpr(markers)
    if markexpr:
        if any(a.startswith("-m") for a in pytest_args):
            raise click.UsageError(
                "--marker clashes with the `-m` handed to pytest after `--`; "
                "keep just one of them."
            )
        suite_args += ["-m", markexpr]

    pytest_args, tests = _selection(tuple(suite_args) + pytest_args, tests, build_dir)
    ctx.invoke(
        _meson.test,
        pytest_args=pytest_args,
        n_jobs=n_jobs,
        tests=tests,
        verbose=verbose,
        build=build,
        build_dir=build_dir,
    )


def _build(ctx, *, meson_args, build_dir, clean, **kwargs):
    # spin hands MESON_ARGS to `meson setup` only on a fresh build dir.
    configured = os.path.isdir(os.path.join(build_dir, "meson-info"))
    if meson_args and configured and not clean:
        _meson._run([*_meson._meson_cli(), "configure", build_dir, *meson_args])
    ctx.invoke(
        _meson.build, meson_args=meson_args, build_dir=build_dir, clean=clean, **kwargs
    )


build = click.Command(
    "build",
    callback=click.pass_context(_build),
    params=_meson.build.params,
    help=f"""{_meson.build.help}

    Unlike upstream, MESON_ARGS also apply to an already configured build
    dir, e.g. to narrow what the tests build:

    \b
      spin build -- -Doperation=sin
    """,
)


@click.command(help="Generate sollya c++/python based files")
@click.option("-f", "--force", is_flag=True, help="Force regenerate all files")
def sollya(*, force):
    import sollya  # type: ignore[import]

    sollya.main(force=force)


# TODO: lint the C++ too (clang-format), once npsr and python/_numpy_sr are clean.
@click.command(help="🧹 Run ruff and pyright")
@click.option("--fix", is_flag=True, help="Apply ruff's fixes and format in place")
@click.pass_context
def lint(ctx, *, fix):
    ruff = ("check", "--fix"), ("format",)
    if not fix:
        ruff = ("check",), ("format", "--check")
    cmds = [("ruff", *args, *LINT_PATHS) for args in ruff] + [("pyright",)]
    missing = sorted({c[0] for c in cmds if shutil.which(c[0]) is None})
    if missing:
        raise click.ClickException(f"not installed: {', '.join(missing)}")
    # pyright resolves numpy_sr and the generated _ops.py from build-install/.
    ctx.invoke(build)
    failed = [c for c in cmds if _meson._run(c, sys_exit=False).returncode]
    if failed:
        raise click.ClickException(
            f"failed: {', '.join(' '.join(c[:2]) for c in failed)}"
        )
