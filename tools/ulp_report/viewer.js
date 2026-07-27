/* NPSR ULP report viewer. Takes the JSON and prints it with Tabulator (CDN).
 *
 * The payload is a render manifest (python/conftest.py::_serialize_records did
 * every format, ratio, mean and sort-key): each record is one table row, with
 * display strings for text and numeric fields for sorting. Filtering is by
 * facet pills above the table (clean header, like the old viewer); a row click
 * expands the per-test + worst-offender detail the manifest already carries.
 *
 * Data sources, strongest first (see index.html):
 *   1. inline <script type="application/json" id="ulp-data">  (spin ulp-report)
 *   2. window.NPSR_ULP_DATA  (../../ulp-report.js sidecar)
 *   3. drag-and-drop / file picker
 */

"use strict";

const FACETS = ["operation", "target", "fma", "dtype", "accuracy", "status"];

/* Accepts the .json report or its `window.NPSR_ULP_DATA = {...};` sidecar. */
function parseText(text) {
  const body = text.trim()
    .replace(/^window\.NPSR_ULP_DATA\s*=/, "")
    .replace(/;$/, "");
  return JSON.parse(body);
}

/* A right-aligned monospace numeric column: sort by `numField`, show the
 * precomputed display string in `dispField`. */
const numCol = (title, numField, dispField, minWidth) => ({
  title, field: numField, hozAlign: "right", sorter: "number", cssClass: "mono", minWidth,
  formatter: (cell) => cell.getRow().getData()[dispField],
});

const COLUMNS = [
  { title: "op", field: "operation", minWidth: 60 },
  { title: "target", field: "target", minWidth: 130 },
  { title: "fma", field: "fma", minWidth: 90,
    formatter: (cell) => `<span class="badge fma">${cell.getValue()}</span>` },
  { title: "dtype", field: "dtype", minWidth: 90 },
  { title: "accuracy", field: "accuracy", minWidth: 100 },
  { title: "tests", field: "tests", hozAlign: "right", sorter: "number", cssClass: "mono", minWidth: 70 },
  numCol("bound", "boundSort", "bound", 90),
  { title: "max error", field: "maxErrNum", sorter: "number", cssClass: "bar-cell",
    width: 160, widthGrow: 0, // pinned narrow so the bar stays compact
    formatter: (cell) => {
      const d = cell.getRow().getData();
      return `<div class="bar-wrap"><span class="mono">${d.maxErr}</span>` +
        `<span class="bar"><i class="${d.barOver ? "over" : ""}" ` +
        `style="width:${Math.min(100, d.barPct)}%"></i></span></div>`;
    } },
  numCol("mean", "meanNum", "mean", 85),
  numCol("over", "nOver", "over", 85),
  { title: "status", field: "status", minWidth: 90,
    formatter: (cell) => {
      const v = cell.getValue();
      return `<span class="badge ${v}">${v === "pass" ? "✓ pass" : "✗ fail"}</span>`;
    } },
];

/* ---- facet pills (above the table; Tabulator keeps a clean header) ---- */

function buildFilters(payload, table) {
  const box = document.getElementById("filters");
  box.replaceChildren();
  const active = {};
  for (const f of FACETS) active[f] = new Set(payload.filter_by[f]);
  const apply = () => table.setFilter((row) => FACETS.every((f) => active[f].has(row[f])));

  for (const f of FACETS) {
    const pills = document.createElement("div");
    pills.className = "pills";
    for (const v of payload.filter_by[f]) {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "pill";
      pill.textContent = v;
      pill.setAttribute("aria-pressed", "true");
      pill.addEventListener("click", () => {
        active[f].has(v) ? active[f].delete(v) : active[f].add(v);
        pill.setAttribute("aria-pressed", String(active[f].has(v)));
        apply();
      });
      pills.append(pill);
    }
    const facet = document.createElement("div");
    facet.className = "facet";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = f;
    facet.append(name, pills);
    box.append(facet);
  }
}

/* ---- expandable detail (per-test + worst offenders), built from the row ---- */

/* value on top, IEEE bits under it (old two-line layout keeps columns narrow). */
const operand = (o) => `<span class="v">${o.text}</span><span class="bits">${o.bits}</span>`;

function detailHTML(d) {
  const perTest = d.perTest.map((t) =>
    `<tr class="${t.passed ? "" : "over"}"><td>${t.label}</td>` +
    `<td class="num">${t.bound}</td><td class="num">${t.maxErr}</td>` +
    `<td class="num">${t.mean}</td><td class="num">${t.over}</td>` +
    `<td>${t.passed ? "✓" : "✗"}</td></tr>`).join("");
  const offenders = d.offenders.map((o) =>
    `<tr class="${o.over ? "over" : ""}"><td>${o.test}</td>` +
    `<td>${operand(o.x)}</td><td>${operand(o.actual)}</td><td>${operand(o.expected)}</td>` +
    `<td class="num">${o.residual}</td><td class="num">${o.error}</td></tr>`).join("");
  return (
    `<table class="sub"><thead><tr><th>test</th><th class="num">bound</th>` +
    `<th class="num">max error</th><th class="num">mean</th><th class="num">over</th>` +
    `<th>status</th></tr></thead><tbody>${perTest}</tbody></table>` +
    `<div class="sub-h">worst offenders (all tests)</div>` +
    `<table class="sub"><thead><tr><th>test</th><th>x</th><th>actual</th>` +
    `<th>expected</th><th class="num">residual</th><th class="num">error</th></tr></thead>` +
    `<tbody>${offenders}</tbody></table>`
  );
}

/* Toggle the detail panel directly on the row element. (Tabulator's
 * `row.reformat()` does not re-run `rowFormatter` in v6, so we manage the DOM
 * here; a sort/filter rebuilds the row and collapses it, which is fine.) */
function toggleDetail(row) {
  const el = row.getElement();
  const open = el.querySelector(":scope > .detail");
  if (open) { open.remove(); return; }
  const holder = document.createElement("div");
  holder.className = "detail";
  holder.innerHTML = detailHTML(row.getData());
  el.appendChild(holder);
}

function show(payload, source) {
  document.getElementById("meta").textContent =
    `seed ${payload.seed} · ${new Date(payload.created * 1000).toLocaleString()} · ${source}`;
  document.getElementById("dropzone").hidden = true;

  const table = new Tabulator("#grid", {
    data: payload.records,
    layout: "fitColumns",
    columns: COLUMNS,
    initialSort: [{ column: "maxErrNum", dir: "desc" }],
    renderVertical: "basic", // ~56 rows: render them all (print- and find-friendly)
    selectableRows: false, // no row selection: kill Tabulator's #bbb/#9abcea highlights
  });
  // Clicks inside an open detail panel (e.g. selecting text) must not collapse it.
  table.on("rowClick", (e, row) => {
    if (e.target.closest(".detail")) return;
    toggleDetail(row);
  });
  table.on("tableBuilt", () => buildFilters(payload, table));
}

function loadFile(file) {
  file.text().then((t) => show(parseText(t), file.name))
    .catch((e) => alert(`Could not parse ${file.name}: ${e.message}`));
}

const slot = document.getElementById("ulp-data").textContent.trim();
if (slot) show(JSON.parse(slot), "embedded");
else if (window.NPSR_ULP_DATA) show(window.NPSR_ULP_DATA, "ulp-report.js sidecar");
else document.getElementById("dropzone").hidden = false;

document.getElementById("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});
document.body.addEventListener("dragover", (e) => e.preventDefault());
document.body.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
