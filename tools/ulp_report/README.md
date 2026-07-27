# ULP report viewer

Static viewer for the fractional-ULP JSON report the pytest suite writes
(`ulp-report.json` at the repo root). `spin ulp-report` bundles that JSON into a
single HTML file and also writes a `ulp-report.js` sidecar so `index.html` works
over `file://`.

- **Local:** run the test suite, then `spin ulp-report` to refresh the sidecar
  and open `index.html` directly — no server. A report living elsewhere
  (custom `$NPSR_ULP_JSON`, CI download) can be dragged onto the page instead.
- **CI:** `spin ulp-report` (optionally `<json> -o <out.html>`) emits one HTML
  artifact with the JSON inlined.

The JSON is a render manifest: `python/_conftest/report.py::_serialize_records` does
every format, ULP ratio, mean and sort-key up front, so each entry in `records`
is already one table row of display strings (plus numeric aids for sorting).

The page is just [Tabulator](https://tabulator.info) loaded from a CDN:
`viewer.js` maps `payload.records` to columns and lets the grid handle sort and
per-column filters — no framework, no build, no vendored file. **The CDN means
the page needs network access to render** (both `index.html` and the bundled
artifact).

- **Filters:** facet pills above the table (one group per categorical column)
  drive `table.setFilter`, keeping the grid header clean; numeric columns sort
  by their `*Num` field so `inf`/`1e+100` rank correctly.
- **Detail:** click a row to expand its per-test breakdown and worst offenders
  (value + IEEE bits), built from the row's own `perTest`/`offenders` — the
  only custom markup, styled by `viewer.css`. `renderVertical: "basic"` renders
  all rows (no virtual scroll) so print and Ctrl-F see everything.

To pin or upgrade Tabulator, edit the two CDN `<script>`/`<link>` URLs in
`index.html`.
