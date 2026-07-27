import os
import pathlib
import sys

import click
import spin.cmds.meson as _meson

curdir = pathlib.Path(__file__).parent
rootdir = curdir.parent
toolsdir = rootdir / "tools"
sys.path.insert(0, str(toolsdir))

PACKAGE = "numpy_sr"

# Registered in `python/_conftest/hooks.py`; both are skipped unless opted into.
MARKERS = ("slow", "exhaustive")


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
      spin test --collect-worst
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
    help="Restrict ISA targets; comma-separated (default: every supported "
    "target except references).",
)
@click.option(
    "--accuracy",
    metavar="NAMES",
    help="Restrict accuracy profiles; comma-separated (default: high,low).",
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
    "--collect-worst",
    is_flag=True,
    default=False,
    help="Fold measured inputs into the worst-case book instead of only "
    "replaying it as a gate.",
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
    seed,
    markers,
    collect_worst,
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
        ("--seed", seed),
    ):
        if value is not None:
            suite_args += [name, value]
    if collect_worst:
        suite_args.append("--collect-worst")

    markexpr = _markexpr(markers)
    if markexpr:
        if any(a == "-m" or a.startswith("-m") for a in pytest_args):
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


@click.command(help="Generate sollya c++/python based files")
@click.option("-f", "--force", is_flag=True, help="Force regenerate all files")
def sollya(*, force):
    import sollya  # type: ignore[import]

    sollya.main(force=force)


# Markers in tools/ulp_report/index.html -> what replaces them in the bundle.
# Tabulator stays a CDN <script>/<link>, so only our own css/js/data are inlined.
_CSS_LINK = '<link rel="stylesheet" href="viewer.css">'
_DATA_SLOT = '<script type="application/json" id="ulp-data"></script>'
_SIDECAR = '<script src="../../ulp-report.js"></script>'
_VIEWER = '<script src="viewer.js"></script>'


def _replace_once(html, marker, repl):
    if html.count(marker) != 1:
        raise SystemExit(f"index.html: expected exactly one {marker!r}")
    return html.replace(marker, repl)


def _bundle(json_text):
    viewdir = toolsdir / "ulp_report"
    html = (viewdir / "index.html").read_text()
    # "</" inside JSON strings would close the inline script tag early.
    data = json_text.replace("</", "<\\/")
    html = _replace_once(
        html, _CSS_LINK, f"<style>\n{(viewdir / 'viewer.css').read_text()}</style>"
    )
    html = _replace_once(
        html,
        _DATA_SLOT,
        f'<script type="application/json" id="ulp-data">{data}</script>',
    )
    html = _replace_once(html, _SIDECAR, "")
    return _replace_once(
        html, _VIEWER, f"<script>\n{(viewdir / 'viewer.js').read_text()}</script>"
    )


@click.command(help="Bundle the ULP JSON report + viewer into one self-contained HTML")
@click.argument("json_path", required=False, type=click.Path(path_type=pathlib.Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("ulp-report.html"),
    show_default=True,
    help="Output HTML path",
)
@click.option(
    "--open/--no-open",
    "show",
    default=True,
    show_default=True,
    help="Open the result in the default browser (--no-open for CI)",
)
def ulp_report(json_path, output, show):
    import webbrowser

    json_path = json_path or rootdir / "ulp-report.json"
    if not json_path.exists():
        raise SystemExit(
            f"{json_path}: not found. Run the test suite first, or pass a report path."
        )
    json_text = json_path.read_text()
    output.write_text(_bundle(json_text))
    print(f"{output} ({output.stat().st_size:,} bytes)")
    # Sidecar for the live index.html over file://, where fetch() is CORS-blocked.
    sidecar = json_path.with_suffix(".js")
    sidecar.write_text(f"window.NPSR_ULP_DATA = {json_text};\n")
    print(f"{sidecar} ({sidecar.stat().st_size:,} bytes)")
    if show:
        webbrowser.open(output.resolve().as_uri())
