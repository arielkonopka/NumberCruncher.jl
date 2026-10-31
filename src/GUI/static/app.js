// NumberCruncher browser GUI. Talks to the JSON API in src/GUI/GUI.jl.
"use strict";

const $ = (id) => document.getElementById(id);
const state = { data: null, dataset: null, status: null, phase: null, polling: null };

// ------------------------------------------------------------------ helpers
async function api(path, payload, opts = {}) {
  const init = payload === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": opts.contentType || "application/json" },
    body: opts.raw ? payload : JSON.stringify(payload),
  };
  const res = await fetch(path, init);
  const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) {
    const err = new Error(body.error || `HTTP ${res.status}`);
    err.exists = body.exists;
    throw err;
  }
  return body;
}

function say(el, text, kind = "") {
  el = typeof el === "string" ? $(el) : el;
  el.textContent = text;
  el.className = "msg " + kind;
}

const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–"
  : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2) : x.toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function layout(extra = {}) {
  const text = cssVar("--text"), grid = cssVar("--plot-grid");
  const axis = { gridcolor: grid, zerolinecolor: grid, color: text, automargin: true };
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: text, size: 12 }, margin: { l: 50, r: 16, t: 10, b: 40 },
    xaxis: { ...axis }, yaxis: { ...axis },
    legend: { orientation: "h", y: -0.18 }, hovermode: "closest",
  }, extra);
}
const COLORS = { real: "#2f6fdb", est: "#e07b25", ok: "#1f8f55", traj: "#8a94a8", last: "#c43d3d" };
const plotCfg = { responsive: true, displaylogo: false };

// Saves the file only after confirming an overwrite.
async function saveWithOverwrite(path, payload, msgEl) {
  try {
    return await api(path, payload);
  } catch (e) {
    if (e.exists && confirm(`${e.message}. Overwrite it?`)) return api(path, { ...payload, overwrite: true });
    throw e;
  }
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll("section.tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + name));
  if (name === "results") renderResults();
  if (name === "phase" && state.phase) renderPhase();
  window.dispatchEvent(new Event("resize"));
}
document.querySelectorAll("#src-tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#src-tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".source").forEach((s) => s.classList.toggle("active", s.id === "src-" + b.dataset.src));
  say("src-msg", "");
}));

// ------------------------------------------------------------------ data
async function loadData(selectFile) {
  const d = await api("/api/data");
  state.data = d;
  $("data-dir").textContent = d.data_dir;
  const tbody = $("files-table").querySelector("tbody");
  tbody.innerHTML = d.files.map((f) => f.error
    ? `<tr><td>${esc(f.name)}</td><td colspan="4" class="neg">${esc(f.error)}</td></tr>`
    : `<tr class="clickable" data-file="${esc(f.name)}"><td>${esc(f.name)}</td><td class="num">${f.rows}</td>
       <td>${esc(f.columns.join(", "))}</td><td>${esc(f.first)}</td><td>${esc(f.last)}</td></tr>`).join("")
    || `<tr><td colspan="5" class="hint">No CSV files yet. Use “Get data”.</td></tr>`;
  tbody.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => pickFile(tr.dataset.file)));

  const fileSel = $("ds-file");
  const current = selectFile || fileSel.value || d.dataset?.file;
  fileSel.innerHTML = d.files.filter((f) => !f.error).map((f) => `<option>${esc(f.name)}</option>`).join("");
  if (current && d.files.some((f) => f.name === current)) fileSel.value = current;
  fillColumns(d.dataset && d.dataset.file === fileSel.value ? d.dataset.column : null);

  $("yh-interval").innerHTML = d.yahoo_intervals.map((i) => `<option>${i}</option>`).join("");
  if (!$("wx-vars").children.length) {
    $("wx-vars").innerHTML = d.meteo_variables.map((v) =>
      `<label class="inline"><input type="checkbox" value="${v}" ${v === "temperature_2m" || v === "pressure_msl" ? "checked" : ""}> ${v}</label>`).join("");
  }
  if (d.fetch && d.fetch.status !== "idle") showFetch(d.fetch);
  if (d.dataset && !state.dataset) showDataset(d.dataset);
}

function fillColumns(preferred) {
  const f = state.data.files.find((x) => x.name === $("ds-file").value);
  const cols = f ? f.columns : [];
  const guess = preferred || ["adjclose", "close", "open", "x", "temperature_2m"].find((c) => cols.includes(c))
    || cols.find((c) => !["t", "timestamp", "volume"].includes(c)) || cols[0];
  $("ds-column").innerHTML = cols.map((c) => `<option ${c === guess ? "selected" : ""}>${esc(c)}</option>`).join("");
  suggestTransform();
}

function suggestTransform() {
  const c = ($("ds-column").value || "").toLowerCase();
  const priceLike = /close|open|high|low|price|sp500|djia|nasdaq|^[a-z]{3,}$/.test(c) && !/temperature|humidity|pressure|wind|cloud|precip|^x$|^y$|^z$/.test(c);
  $("ds-transform").value = priceLike ? "logdiff" : "raw";
}

function pickFile(name) {
  $("ds-file").value = name;
  fillColumns();
  $("tab-data").scrollIntoView({ behavior: "smooth" });
}

$("ds-file").addEventListener("change", () => fillColumns());
$("ds-column").addEventListener("change", suggestTransform);
$("ds-apply").addEventListener("click", async () => {
  say("ds-msg", "Preparing…");
  try {
    const d = await api("/api/dataset", {
      file: $("ds-file").value, column: $("ds-column").value,
      transform: $("ds-transform").value, noise: parseFloat($("ds-noise").value) || 0,
    });
    showDataset(d);
    say("ds-msg", "Dataset ready. Next: Run or Phase space.", "ok");
  } catch (e) { say("ds-msg", e.message, "err"); }
});

const TRANSFORM_NAMES = { raw: "raw values", diff: "first differences", logdiff: "log returns" };
function showDataset(d) {
  state.dataset = d;
  const label = `${d.file} · ${d.column} · ${TRANSFORM_NAMES[d.transform]}${d.noise > 0 ? ` · noise ${d.noise}` : ""}`;
  $("dataset-chip").textContent = `${label} (${d.length} values)`;
  $("ds-summary").innerHTML = `<div class="stats">
    ${[["values", d.length], ["mean", fmt(d.mean)], ["std", fmt(d.std)], ["min", fmt(d.min)], ["max", fmt(d.max)]]
      .map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("")}
    </div><div class="hint">${esc(d.first)} → ${esc(d.last)}</div>`;
  $("ds-preview").style.display = "block";
  Plotly.react("ds-preview", [{ x: d.preview.labels, y: d.preview.values, type: "scattergl", mode: "lines",
    line: { color: COLORS.real, width: 1 }, name: d.column }], layout({ showlegend: false }), plotCfg);
  $("run-n").max = Math.floor(d.length / 2);
}

// --- importers
$("up-file").addEventListener("change", () => {
  const f = $("up-file").files[0];
  if (f && !$("up-name").value) $("up-name").value = f.name.replace(/[^A-Za-z0-9_.-]/g, "_").replace(/^[^A-Za-z0-9]+/, "");
});
$("up-go").addEventListener("click", async () => {
  const f = $("up-file").files[0];
  if (!f) return say("src-msg", "Choose a CSV file first.", "err");
  const name = $("up-name").value.trim();
  const send = (overwrite) => api(`/api/data/upload?name=${encodeURIComponent(name)}&overwrite=${overwrite}`, f,
    { raw: true, contentType: "text/csv" });
  say("src-msg", "Uploading…");
  try {
    let r;
    try { r = await send(false); }
    catch (e) { if (e.exists && confirm(`${e.message}. Overwrite it?`)) r = await send(true); else throw e; }
    say("src-msg", `Saved ${r.file}: ${r.rows} rows.`, "ok");
    await loadData(r.file);
  } catch (e) { say("src-msg", e.message, "err"); }
});

const safeName = (s) => s.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "").toLowerCase();
const autoName = () => {
  if (!$("yh-name").dataset.touched) $("yh-name").value = `${safeName($("yh-symbol").value) || "symbol"}_${$("yh-interval").value || "1d"}.csv`;
  if (!$("fr-name").dataset.touched) $("fr-name").value = `fred_${safeName($("fr-series").value) || "series"}.csv`;
};
["yh-name", "fr-name"].forEach((id) => $(id).addEventListener("input", () => { $(id).dataset.touched = "1"; }));
["yh-symbol", "yh-interval", "fr-series"].forEach((id) => $(id).addEventListener("input", autoName));
$("yh-interval").addEventListener("change", autoName);

async function runImport(payload, button) {
  button.disabled = true;
  say("src-msg", "Downloading…");
  try {
    const r = await saveWithOverwrite("/api/data/import", payload);
    say("src-msg", `Saved ${r.file}: ${r.rows} rows.`, "ok");
    await loadData(r.file);
  } catch (e) { say("src-msg", e.message, "err"); }
  button.disabled = false;
}
$("yh-go").addEventListener("click", () => runImport({ source: "yahoo", symbol: $("yh-symbol").value,
  interval: $("yh-interval").value, name: $("yh-name").value.trim() }, $("yh-go")));
$("fr-go").addEventListener("click", () => runImport({ source: "fred", series: $("fr-series").value,
  name: $("fr-name").value.trim() }, $("fr-go")));
$("wx-go").addEventListener("click", () => runImport({ source: "openmeteo",
  latitude: parseFloat($("wx-lat").value), longitude: parseFloat($("wx-lon").value),
  start_date: $("wx-start").value, end_date: $("wx-end").value,
  variables: [...$("wx-vars").querySelectorAll("input:checked")].map((i) => i.value),
  name: $("wx-name").value.trim() }, $("wx-go")));

$("lz-go").addEventListener("click", async () => {
  const u0 = $("lz-u0").value.split(",").map((x) => parseFloat(x));
  if (u0.length !== 3 || u0.some(Number.isNaN)) return say("src-msg", "x₀, y₀, z₀ must be three numbers.", "err");
  $("lz-go").disabled = true;
  say("src-msg", "Integrating…");
  try {
    const r = await saveWithOverwrite("/api/data/lorenz", {
      name: $("lz-name").value.trim(), N: parseInt($("lz-N").value), dt: parseFloat($("lz-dt").value),
      Ttr: parseFloat($("lz-ttr").value), sigma: parseFloat($("lz-sigma").value),
      rho: parseFloat($("lz-rho").value), beta: parseFloat($("lz-beta").value), x0: u0[0], y0: u0[1], z0: u0[2],
    });
    say("src-msg", `Saved ${r.file}: ${r.rows} rows.`, "ok");
    await loadData(r.file);
  } catch (e) { say("src-msg", e.message, "err"); }
  $("lz-go").disabled = false;
});

$("cc-go").addEventListener("click", async () => {
  if (state.data && !state.data.key_present) return say("cc-status", "No API key in config/cryptocompare_key.txt.", "err");
  try {
    const r = await saveWithOverwrite("/api/data/fetch", { start_date: $("cc-start").value, name: $("cc-name").value.trim() });
    showFetch(r);
    pollFetch();
  } catch (e) { say("cc-status", e.message, "err"); }
});
function showFetch(f) {
  say("cc-status", `${f.file}: ${f.message}`, f.status === "failed" ? "err" : f.status === "done" ? "ok" : "");
}
async function pollFetch() {
  const d = await api("/api/data");
  showFetch(d.fetch);
  if (d.fetch.status === "running") setTimeout(pollFetch, 2000);
  else await loadData(d.fetch.file);
}

// ------------------------------------------------------------------ run
$("run-start").addEventListener("click", async () => {
  if (!state.dataset) return say("run-msg", "Choose a dataset on the Data tab first.", "err");
  const ks = $("run-ks").value.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!ks.length || ks.some((k) => !Number.isInteger(k) || k < 1)) return say("run-msg", "k values must be positive integers.", "err");
  try {
    const s = await api("/api/run", { n_points: parseInt($("run-n").value), ks,
      msize: parseInt($("run-msize").value), window: parseInt($("run-window").value),
      seed: parseInt($("run-seed").value) || 1 });
    state.status = s;
    say("run-msg", "");
    renderRun();
    startPolling();
  } catch (e) { say("run-msg", e.message, "err"); }
});
$("run-stop").addEventListener("click", async () => {
  await api("/api/run/stop", {});
  say("run-msg", "Stopping after the reconstructions in progress…", "warn");
});

function startPolling() {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    try {
      state.status = await api("/api/run/status");
      renderRun();
      if ($("tab-results").classList.contains("active")) renderResults();
      if (state.status.status !== "running") clearInterval(state.polling);
    } catch (e) { say("run-msg", e.message, "err"); }
  }, 1500);
}

function renderRun() {
  const s = state.status;
  if (!s || s.status === "idle") return;
  const running = s.status === "running";
  $("run-start").disabled = running;
  $("run-stop").disabled = !running;
  $("run-bar").style.width = `${(100 * s.done / s.total).toFixed(1)}%`;
  const rate = s.done > 0 ? s.elapsed / s.done : null;
  const eta = running && rate ? `${Math.round(rate * (s.total - s.done))} s` : "–";
  $("run-stats").innerHTML = [["status", s.status], ["done", `${s.done} / ${s.total}`], ["failed", s.failed],
    ["elapsed", `${Math.round(s.elapsed)} s`], ["remaining", eta]]
    .map(([l, v]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="l">${l}</div></div>`).join("");
  if (s.message) say("run-msg", s.message, s.status === "failed" ? "err" : "warn");
  else if (s.status === "done") say("run-msg", "Finished. See the Results tab.", "ok");
}

// ------------------------------------------------------------------ results
function renderResults() {
  const s = state.status;
  if (!s || s.status === "idle" || !s.results.length) {
    $("res-info").textContent = s && s.status === "running" ? "Waiting for the first points…" : "No run yet.";
    return;
  }
  const p = s.params, d = s.dataset;
  $("res-info").innerHTML = `${esc(d.file)} · ${esc(d.column)} · ${TRANSFORM_NAMES[d.transform]}${d.noise > 0 ? ` · noise ${d.noise}` : ""}
    — ${s.results.length} of ${s.total} points, max delay ${p.msize}, window ${p.window}${s.status === "running" ? " (running…)" : ""}`;

  const m = s.metrics;
  const bestKnn = m.models.filter((r) => !r.baseline).reduce((a, b) => (b.mae < a.mae ? b : a));
  $("res-table").innerHTML = `<thead><tr><th>Model</th><th class="num">MAE</th><th class="num">RMSE</th>
      <th class="num">Skill</th><th class="num">Hit rate</th><th class="num">p (hit vs 50%)</th><th class="num">Correlation</th></tr></thead><tbody>` +
    m.models.map((r) => `<tr class="${r.baseline ? "baseline" : ""} ${r === bestKnn ? "best" : ""}">
      <td>${esc(r.model)}${r.model === m.best_baseline ? " (best baseline)" : ""}</td>
      <td class="num">${fmt(r.mae)}</td><td class="num">${fmt(r.rmse)}</td>
      <td class="num ${r.baseline ? "" : r.skill > 0 ? "pos" : "neg"}">${fmt(r.skill)}</td>
      <td class="num">${r.hit === null ? "–" : r.hit.toFixed(1) + "%"}</td>
      <td class="num">${r.hit_p === null ? "–" : r.hit_p < 1e-4 ? "<0.0001" : r.hit_p.toFixed(4)}</td><td class="num">${fmt(r.corr)}</td></tr>`).join("") +
    `</tbody>`;
  const n = m.models.length ? s.results.length : 0;
  const ci = n ? (196 * Math.sqrt(0.25 / n)).toFixed(1) : "–";
  $("res-table").insertAdjacentHTML("beforeend",
    `<caption style="caption-side:bottom;text-align:left" class="hint">With ${n} points, hit rates within roughly 50 ± ${ci}% are indistinguishable from a coin flip; p is the exact two-sided binomial test.</caption>`);

  const kSel = $("res-k");
  const prev = kSel.value;
  kSel.innerHTML = p.ks.map((k) => `<option value="${k}">k = ${k}</option>`).join("");
  kSel.value = p.ks.map(String).includes(prev) ? prev : String(bestKnn.model.replace("k = ", ""));
  drawResultCharts();

  $("res-bym").innerHTML = `<thead><tr><th>m</th><th class="num">points</th><th class="num">MAE (k=${kSel.value})</th>
      <th class="num">persistence MAE</th><th class="num">zero MAE</th><th class="num">hit rate</th></tr></thead><tbody>` +
    m.by_m.map((g) => { const sc = g.knn[kSel.value];
      return `<tr><td>${g.m}</td><td class="num">${g.n}</td><td class="num">${fmt(sc.mae)}</td>
        <td class="num">${fmt(g.persistence_mae)}</td><td class="num">${fmt(g.zero_mae)}</td>
        <td class="num">${sc.hit === null ? "–" : sc.hit.toFixed(1) + "%"}</td></tr>`; }).join("") + `</tbody>`;
}

function drawResultCharts() {
  const s = state.status, k = $("res-k").value;
  const res = s.results;
  const x = res.map((r) => r.label);
  const real = res.map((r) => r.real), est = res.map((r) => r.est[k]);
  const traces = [
    { x, y: real, name: "real", mode: "lines+markers", line: { color: COLORS.real, width: 1.5 }, marker: { size: 4 } },
    { x, y: est, name: `estimate (k=${k})`, mode: "lines+markers", line: { color: COLORS.est, width: 1.5, dash: "dot" }, marker: { size: 4 } },
  ];
  if (s.raw) traces.push({ x, y: res.map((r) => r.last), name: "persistence", mode: "lines",
    line: { color: COLORS.traj, width: 1 } });
  Plotly.react("res-series", traces, layout(), plotCfg);
  Plotly.react("res-hist", [{ x: est.map((e, i) => e - real[i]), type: "histogram", name: "estimate − real",
    marker: { color: COLORS.est }, nbinsx: 30 }], layout({ showlegend: false, bargap: 0.05,
    xaxis: { title: "estimate − real", color: cssVar("--text"), gridcolor: cssVar("--plot-grid") } }), plotCfg);
  const models = s.metrics.models;
  Plotly.react("res-bars", [{ x: models.map((r) => r.model), y: models.map((r) => r.mae), type: "bar",
    marker: { color: models.map((r) => (r.baseline ? COLORS.traj : COLORS.est)) } }],
    layout({ showlegend: false, yaxis: { title: "MAE", color: cssVar("--text"), gridcolor: cssVar("--plot-grid") } }), plotCfg);
}
$("res-k").addEventListener("change", renderResults);

$("res-csv").addEventListener("click", () => {
  const s = state.status;
  if (!s || !s.results.length) return;
  const ks = s.params.ks;
  const head = ["offset", "label", "m", "taus", "theiler", "real", "last", "window_mean", ...ks.map((k) => `est_k${k}`)];
  const lines = s.results.map((r) => [r.offset, `"${r.label}"`, r.m, `"${r.taus.join(" ")}"`, r.theiler, r.real, r.last,
    r.window_mean, ...ks.map((k) => r.est[k])].join(","));
  const blob = new Blob([[head.join(","), ...lines].join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob),
    download: `results_${s.dataset.file.replace(/\.csv$/, "")}_${s.dataset.column}.csv` });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ------------------------------------------------------------------ phase space
$("ph-go").addEventListener("click", async () => {
  if (!state.dataset) return say("ph-msg", "Choose a dataset on the Data tab first.", "err");
  $("ph-go").disabled = true;
  say("ph-msg", "Reconstructing…");
  try {
    state.phase = await api("/api/phase", { offset: parseInt($("ph-offset").value), msize: parseInt($("ph-msize").value),
      window: parseInt($("ph-window").value), k: parseInt($("ph-k").value), seed: parseInt($("run-seed").value) || 1 });
    const ph = state.phase;
    const opts = ph.taus.map((t, j) => `<option value="${j}">s(n+${t})${j === ph.searched_index - 1 ? " — unknown" : ""}</option>`).join("");
    ["ph-x", "ph-y", "ph-z"].forEach((id, i) => { $(id).innerHTML = opts; $(id).value = String(Math.min(i, ph.m - 1)); });
    // put the unknown coordinate on the last axis so the estimate's spread is visible
    const axes = ph.m >= 3 ? "ph-z" : ph.m === 2 ? "ph-y" : "ph-x";
    $(axes).value = String(ph.searched_index - 1);
    if (ph.m >= 2) {
      const others = [...Array(ph.m).keys()].filter((j) => j !== ph.searched_index - 1);
      $("ph-x").value = String(others[0]);
      if (ph.m >= 3) $("ph-y").value = String(others[1]);
    }
    say("ph-msg", "");
    renderPhase();
  } catch (e) { say("ph-msg", e.message, "err"); }
  $("ph-go").disabled = false;
});
["ph-x", "ph-y", "ph-z", "ph-3d"].forEach((id) => $(id).addEventListener("change", renderPhase));

function renderPhase() {
  const ph = state.phase;
  if (!ph) return;
  const u = ph.searched_index - 1;
  $("ph-info").innerHTML = [["dimension m", ph.m], ["delays τ", ph.taus.join(", ")], ["Theiler w", ph.theiler],
    ["points", ph.n], ["estimate", fmt(ph.estimate[u])], ["real", fmt(ph.real[u])], ["target", ph.label]]
    .map(([l, v]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="l">${l}</div></div>`).join("");

  const three = $("ph-3d").checked && ph.m >= 3;
  $("ph-z").disabled = !three;
  const ix = +$("ph-x").value, iy = +$("ph-y").value, iz = +$("ph-z").value;
  const one = ph.m === 1;
  const pick = (cols, j) => cols[j];
  const pts = (cols) => one ? { x: cols[0].map((_, i) => i), y: cols[0] }
    : three ? { x: pick(cols, ix), y: pick(cols, iy), z: pick(cols, iz) } : { x: pick(cols, ix), y: pick(cols, iy) };
  const single = (v) => one ? { x: [0], y: [v[0]] } : three ? { x: [v[ix]], y: [v[iy]], z: [v[iz]] } : { x: [v[ix]], y: [v[iy]] };
  const type = three ? "scatter3d" : "scattergl";
  const traces = [
    { ...pts(ph.trajectory), type, mode: three ? "lines" : "markers", name: "trajectory",
      line: { color: COLORS.traj, width: 1 }, marker: { color: COLORS.traj, size: 2, opacity: 0.35 }, opacity: three ? 0.5 : 1 },
    { ...pts(ph.neighbours), type, mode: "markers", name: `neighbours (k=${ph.neighbours[0].length})`,
      marker: { color: COLORS.real, size: three ? 3 : 6 } },
    { ...pts(ph.successors), type, mode: "markers", name: "successors", marker: { color: COLORS.ok, size: three ? 3 : 6 } },
    { ...single(ph.last), type, mode: "markers", name: "last known point", marker: { color: COLORS.last, size: three ? 6 : 11, symbol: "x" } },
    { ...single(ph.estimate), type, mode: "markers", name: "estimate", marker: { color: COLORS.est, size: three ? 6 : 12, symbol: "diamond" } },
    { ...single(ph.real), type, mode: "markers", name: "real next point", marker: { color: "#7a3fd1", size: three ? 6 : 12, symbol: "circle-open", line: { width: 2 } } },
  ];
  const lbl = (j) => `s(n+${ph.taus[j]})${j === u ? " (unknown)" : ""}`;
  const text = cssVar("--text"), grid = cssVar("--plot-grid");
  const ax = (title) => ({ title, color: text, gridcolor: grid, zerolinecolor: grid, backgroundcolor: "rgba(0,0,0,0)" });
  const lay = three
    ? layout({ scene: { xaxis: ax(lbl(ix)), yaxis: ax(lbl(iy)), zaxis: ax(lbl(iz)) }, margin: { l: 0, r: 0, t: 0, b: 0 } })
    : layout({ xaxis: ax(one ? "n" : lbl(ix)), yaxis: ax(one ? lbl(0) : lbl(iy)) });
  Plotly.react("ph-plot", traces, lay, plotCfg);
}

// ------------------------------------------------------------------ start
$("wx-end").value = new Date(Date.now() - 7 * 86400e3).toISOString().slice(0, 10);
autoName();
(async () => {
  try {
    await loadData();
    const s = await api("/api/run/status");
    if (s.status !== "idle") { state.status = s; renderRun(); if (s.status === "running") startPolling(); }
  } catch (e) { say("ds-msg", "Cannot reach the NumberCruncher server: " + e.message, "err"); }
})();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (state.dataset) showDataset(state.dataset);
  if (state.status) renderResults();
  if (state.phase) renderPhase();
});
