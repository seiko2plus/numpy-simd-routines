import os
import pathlib
import sys

import click
import spin.cmds.meson as _meson

curdir = pathlib.Path(__file__).parent
rootdir = curdir.parent
toolsdir = rootdir / "tools"
sys.path.insert(0, str(toolsdir))


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
