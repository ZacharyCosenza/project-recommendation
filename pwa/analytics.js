// Analytics tab: model status and retraining, the event map with its card, calibration, and the events table.
import { api, escapeHtml } from "/pwa/api.js";

const $ = (id) => document.getElementById(id);
// Charts don't pan, zoom or show hover boxes (awkward on a touchscreen); the map still opens a card on tap.
const PLOT_CONFIG = { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false };
let plotly = null;
let page = 1;
let cardUrl = null;
let started = false;
let chartsStale = true;
let following = false;

// Charts only change after a retrain or a search, so the tab keeps them until one of those happens.
export function markAnalyticsStale() {
  chartsStale = true;
}

function loadPlotly() {
  plotly ??= new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/vendor/plotly.min.js";
    s.onload = () => resolve(window.Plotly);
    s.onerror = reject;
    document.head.append(s);
  });
  return plotly;
}

async function draw(el, figure, height, { tappable = false } = {}) {
  const Plotly = await loadPlotly();
  const layout = { ...figure.layout, autosize: true, height, dragmode: false };
  delete layout.width;
  for (const axis of ["xaxis", "yaxis"]) layout[axis] = { ...layout[axis], fixedrange: true };
  const data = figure.data.map((trace) => ({ ...trace, hoverinfo: "none" }));
  await Plotly.react(el, data, layout, { ...PLOT_CONFIG, staticPlot: !tappable });
  return el;
}

const pct = (p) => (p == null ? "—" : `${Math.round(p * 100)}%`);

async function loadModel() {
  const { model, retrain } = await api("/api/model");
  $("model-status").textContent = model
    ? `Model trained on ${model.n_events} events (${model.label_counts["1"]} 👍, ${model.label_counts["-1"]} 👎), `
      + `${model.trained_at.slice(0, 16).replace("T", " ")} UTC.`
    : "Model not trained yet.";
  $("retrain").disabled = retrain.state === "running";
  $("retrain").textContent = retrain.state === "running" ? "Retraining…" : "Retrain";
  return retrain;
}

async function retrain() {
  $("retrain-banner").innerHTML = "";
  $("retrain").disabled = true;
  $("retrain").textContent = "Retraining…";
  await api("/api/retrain", { method: "POST" }).catch(() => {});
  followRetrain();
}

async function followRetrain() {
  // One watcher at a time, however often the tab is opened while a retrain runs.
  if (following) return;
  following = true;
  let state;
  do {
    await new Promise((r) => setTimeout(r, 3000));
    state = await loadModel();
  } while (state.state === "running");
  following = false;
  const kind = state.state === "error" ? "warning" : "success";
  $("retrain-banner").innerHTML = `<p class="notice ${kind}">${escapeHtml(state.message)}</p>`;
  markAnalyticsStale();
  showAnalytics();
}

async function loadCharts() {
  // The current charts stay up while the new ones load, then update in place.
  chartsStale = false;
  $("map-status").textContent = "Updating the map (the first time after new events takes ~20s)…";
  const [map, cal] = await Promise.all([api("/api/map"), api("/api/calibration")]);
  $("map-status").textContent = map.figure ? "" : map.reason;
  if (map.figure) {
    const el = await draw($("map"), map.figure, Math.min($("map").offsetWidth, 560), { tappable: true });
    el.removeAllListeners?.("plotly_click");
    el.on("plotly_click", (d) => { const url = d.points[0]?.customdata; if (url) showCard(url); });
  }
  if (cal.figure) {
    $("calibration-caption").textContent = `Each event's P(like) here comes from a model trained without that event's `
      + `label (${cal.n} events, ${cal.folds}-fold). Points on the dashed line mean a 30% P(like) really is liked about 30% `
      + `of the time. Brier score ${cal.brier.toFixed(3)} (always guessing the like rate scores ${cal.baseline.toFixed(3)}; `
      + "lower is better).";
    await draw($("calibration"), cal.figure, 360);
  } else {
    $("calibration-caption").textContent = "Retrain the model to see how well its P(like) matches what you actually liked.";
  }
}

async function showCard(url) {
  cardUrl = url;
  const e = await api(`/api/event?url=${encodeURIComponent(url)}`);
  if (cardUrl !== url) return;
  const link = /^https?:\/\//.test(url) ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">Open event page ↗</a>` : "";
  const buttons = [["👍 Like", 1], ["👎 Dislike", -1], ["Clear", 0]]
    .map(([name, v]) => `<button data-label="${v}" class="${e.label === v && v !== 0 ? "on" : ""}">${name}</button>`).join("");
  $("map-card").className = "panel";
  $("map-card").innerHTML = `<div class="muted"><b>${escapeHtml(e.theme)}</b> · ${escapeHtml(e.city)} · P(like) ${pct(e.p_like)}</div>
    <div class="text">${e.highlight ?? escapeHtml(e.description)}</div>
    <p class="muted">${e.highlight ? "Blue words push P(like) up, red words push it down. Hover or long-press a word for its value."
      : e.p_like == null ? "Word colouring appears once a model is trained." : "Colouring words…"}</p>
    <label class="field">Context <input id="card-context" value="${escapeHtml(e.context)}"></label>
    ${link}<div class="label-buttons">${buttons}</div>`;
  $("card-context").addEventListener("change", async (ev) => {
    await api("/api/events", { method: "PATCH", body: { url, context: ev.target.value } });
    showCard(url);
  });
  $("map-card").querySelectorAll("[data-label]").forEach((b) => b.addEventListener("click", async () => {
    await api("/api/events", { method: "PATCH", body: { url, label: Number(b.dataset.label) } });
    showCard(url);
  }));
  if (!e.highlight && e.p_like != null) {
    setTimeout(() => { if (cardUrl === url) showCard(url); }, 3000);
  }
}

function eventRow(e) {
  const row = document.createElement("div");
  row.className = "event-row";
  const link = /^https?:\/\//.test(e.url) ? `<a href="${escapeHtml(e.url)}" target="_blank" rel="noopener">Open ↗</a>` : "";
  row.innerHTML = `<div class="row-head">
      <span class="bar"><span style="width:${(e.p_like ?? 0) * 100}%"></span></span>${pct(e.p_like)}
      <span><b>${escapeHtml(e.theme)}</b> · ${escapeHtml(e.city)}</span>
      <span class="row-actions"><button class="like ${e.label === 1 ? "on" : ""}">👍</button>
        <button class="dislike ${e.label === -1 ? "on" : ""}">👎</button></span></div>
    <div class="desc clamp">${escapeHtml(e.description)}</div>
    <input placeholder="Context" value="${escapeHtml(e.context)}">${link}`;
  row.querySelector(".desc").addEventListener("click", (ev) => ev.target.classList.toggle("clamp"));
  row.querySelector("input").addEventListener("change", (ev) =>
    api("/api/events", { method: "PATCH", body: { url: e.url, context: ev.target.value } }));
  // Tapping the active button clears it back to neutral.
  for (const [cls, value] of [["like", 1], ["dislike", -1]]) {
    row.querySelector(`.${cls}`).addEventListener("click", async () => {
      e.label = e.label === value ? 0 : value;
      await api("/api/events", { method: "PATCH", body: { url: e.url, label: e.label } });
      row.querySelector(".like").classList.toggle("on", e.label === 1);
      row.querySelector(".dislike").classList.toggle("on", e.label === -1);
    });
  }
  return row;
}

async function loadEvents() {
  const view = $("events-view").value;
  const params = new URLSearchParams({ view, page });
  if (view === "all") {
    if ($("events-city").value) params.set("city", $("events-city").value);
    if ($("events-label").value) params.set("label", $("events-label").value);
  }
  const data = await api(`/api/events?${params}`);
  const city = $("events-city").value;
  $("events-city").innerHTML = `<option value="">Any city</option>`
    + data.cities.map((c) => `<option ${c === city ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
  $("events-city").hidden = $("events-label").hidden = view !== "all";
  document.querySelector(".pager").hidden = view !== "all";
  $("page-no").textContent = `Page ${page}`;
  $("prev-page").disabled = page === 1;
  $("next-page").disabled = data.events.length < data.page_size;
  $("events").replaceChildren(...(data.events.length ? data.events.map(eventRow) : [Object.assign(
    document.createElement("p"), { className: "muted", textContent: "No events here." })]));
}

export async function showAnalytics() {
  if (!started) {
    started = true;
    $("retrain").addEventListener("click", retrain);
    for (const id of ["events-view", "events-city", "events-label"]) {
      $(id).addEventListener("change", () => { page = 1; loadEvents(); });
    }
    $("prev-page").addEventListener("click", () => { page -= 1; loadEvents(); });
    $("next-page").addEventListener("click", () => { page += 1; loadEvents(); });
  }
  const state = await loadModel();
  if (state.state === "running") followRetrain();
  if (chartsStale) loadCharts();
  loadEvents();
}
