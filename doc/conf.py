import inspect
import pathlib

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.domains import ObjType
from sphinx.domains.python import PyFunction, PythonDomain
from sphinx.util.docutils import SphinxDirective

project = "NumPy SIMD Routines"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
]

root_doc = "index"
exclude_patterns = ["_build"]

myst_heading_anchors = 3
autosectionlabel_prefix_document = True

# Theme and options follow NumPy's doc/source/conf.py.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/numpy/numpy-simd-routines",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    # The ULP report table needs the width.
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
        "testing/ulp-report": [],
    },
}
html_static_path = ["_static"]
html_css_files = ["npsr.css"]
html_context = {"default_mode": "light"}
html_copy_source = False

autodoc_default_options = {"members": True, "undoc-members": True}


class PyFixture(PyFunction):
    """``py:fixture``: a function requested by name, so shown without arguments."""

    option_spec = PyFunction.option_spec | {
        "scope": directives.unchanged,
        "parametrized": directives.flag,
    }

    def needs_arglist(self):
        return False

    def get_signature_prefix(self, sig):
        words = self.options.get("scope", "").split()
        if "parametrized" in self.options:
            words.append("parametrized")
        prefix = []
        for word in [*words, "fixture"]:
            prefix += [nodes.Text(word), addnodes.desc_sig_space()]
        return prefix


def add_markers_and_fixtures(app, what, name, obj, options, lines):
    """Put the markers in the Markers section and the fixtures in their own."""
    from _pytest.fixtures import FixtureFunctionDefinition

    if what != "module" or name != "numpy_sr.conftest":
        return
    markers = []
    for marker, doc in obj._MARKERS.items():
        markers += [f".. py:decorator:: {marker}", "   :module: pytest.mark", ""]
        markers += [f"   {doc}", ""]
    start = lines.index("Markers") + 3
    lines[start:start] = markers
    lines += ["", "Fixtures", "--------"]
    for fixture, fdef in vars(obj).items():
        if not isinstance(fdef, FixtureFunctionDefinition):
            continue
        marker = fdef._fixture_function_marker
        lines += ["", f".. py:fixture:: {fixture}"]
        if marker.scope != "function":
            lines += [f"   :scope: {marker.scope}"]
        if marker.params is not None:
            lines += ["   :parametrized:"]
        lines += [""]
        doc = inspect.getdoc(fdef._get_wrapped_function()) or ""
        lines += [f"   {line}" for line in doc.splitlines()]


def hide_ops_bases(app, name, obj, options, bases):
    """Target and Oracle document their ops; ``numpy_sr._ops`` stays private."""
    bases[:] = [b for b in bases if b.__module__ != "numpy_sr._ops"]


ULP_REPORT_HTML = """\
<div class="ulp-report">
<p id="ulp-meta" class="meta"></p>
<section id="ulp-filters" class="filters"></section>
<div id="ulp-grid"></div>
<script type="application/json" id="ulp-data">{data}</script>
</div>
"""


class UlpReport(SphinxDirective):
    """``ulp-report``: render the JSON report the test suite writes."""

    required_arguments = 1

    def run(self):
        rel, path = self.env.relfn2path(self.arguments[0])
        # Rebuild the page whenever the report changes, appears or goes away.
        self.env.note_dependency(rel)
        try:
            text = pathlib.Path(path).read_text()
        except FileNotFoundError:
            note = nodes.note()
            note += nodes.paragraph(
                text=f"No {rel} yet: run spin test, then spin docs again."
            )
            return [note]
        # "</" inside JSON strings would close the inline script tag early.
        data = text.replace("</", "<\\/")
        return [nodes.raw("", ULP_REPORT_HTML.format(data=data), format="html")]


def add_ulp_viewer(app, pagename, templatename, context, doctree):
    """Load the viewer only on the pages that embed a report."""
    if 'id="ulp-data"' not in context.get("body", ""):
        return
    app.add_css_file("tabulator/tabulator.min.css")
    app.add_css_file("ulp_report.css")
    app.add_js_file("tabulator/tabulator.min.js", loading_method="defer")
    app.add_js_file("ulp_report.js", loading_method="defer")


def setup(app):
    PythonDomain.object_types["fixture"] = ObjType("fixture", "func", "obj")
    app.add_directive_to_domain("py", "fixture", PyFixture)
    app.connect("autodoc-process-docstring", add_markers_and_fixtures)
    app.connect("autodoc-process-bases", hide_ops_bases)
    app.add_directive("ulp-report", UlpReport)
    app.connect("html-page-context", add_ulp_viewer)
