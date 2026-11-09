// Search tab: the new-search form, live progress of a running search, and past runs.
import { markAnalyticsStale } from "/pwa/analytics.js";
import { api, escapeHtml } from "/pwa/api.js";

const $ = (id) => document.getElementById(id);
let estimate = { seconds_per_query: 30, cost_per_query: 0.065 };
let watching = null;
let polling = false;
let loaded = false;

function addRow(text = "", max = 5) {
  const row = document.createElement("div");
  row.className = "query-row";
  row.innerHTML = `<input class="q-text" placeholder="e.g. comedy shows" value="${escapeHtml(text)}">
    <input class="q-max" type="number" min="1" max="10" value="${max}" title="Max results">
    <button class="small" title="Remove">✕</button>`;
  row.querySelector("button").addEventListener("click", () => { row.remove(); update(); });
  row.querySelectorAll("input").forEach((i) => i.addEventListener("input", update));
  $("query-rows").append(row);
}

function queries() {
  return [...document.querySelectorAll(".query-row")]
    .map((r) => ({ text: r.querySelector(".q-text").value.trim(), max_results: Number(r.querySelector(".q-max").value) || 5 }))
    .filter((q) => q.text);
}

function update() {
  const n = queries().length;
  const minutes = Math.max(1, Math.round((n * estimate.seconds_per_query) / 60));
  $("estimate").textContent = `This calls paid APIs: roughly $${(n * estimate.cost_per_query).toFixed(2)} and ~${minutes} min `
    + `for ${n} quer${n === 1 ? "y" : "ies"}. It runs in the background, so you can leave the app.`;
  $("run-search").disabled = !$("confirm").checked || n === 0 || watching !== null;
}

function banner(kind, message) {
  $("search-banner").innerHTML = message ? `<p class="notice ${kind}">${escapeHtml(message)}</p>` : "";
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    while (true) {
      const { active } = await api("/api/search/status");
      if (active) {
        watching = active.id;
        $("search-progress").innerHTML = `<p class="muted">Searching ${escapeHtml(active.location)}: ${active.done}/${active.total}
          queries done · ${active.events} events found so far.</p>
          <div class="progress"><span style="width:${(100 * active.done) / Math.max(active.total, 1)}%"></span></div>`;
      } else {
        $("search-progress").innerHTML = "";
        if (watching !== null) {
          const result = await api(`/api/runs/${watching}`);
          banner(result.kind, result.message);
          markAnalyticsStale();
          watching = null;
          loadRuns();
        }
        break;
      }
      update();
      await new Promise((r) => setTimeout(r, 3000));
    }
  } finally {
    polling = false;
    update();
  }
}

async function loadRuns() {
  const { runs, spend, daily } = await api("/api/runs");
  const last = daily ? ` · Last daily run ${daily.day}: ${daily.status}${daily.detail ? ` (${daily.detail})` : ""}` : "";
  $("spend").textContent = `Lifetime API spend: $${spend.toFixed(2)}${last}`;
  $("runs").innerHTML = runs.length === 0 ? `<p class="muted">No runs yet.</p>` : `<table class="runs">
    <tr><th>Started</th><th>Where</th><th>Events</th><th>Cost</th><th>Status</th></tr>
    ${runs.map((r) => `<tr><td>${r.started_at.slice(0, 16).replace("T", " ")}</td><td>${escapeHtml(r.location)}</td>
      <td>${r.total_event_count ?? 0} (${r.new_event_count ?? 0} new)</td><td>$${(r.cost_usd ?? 0).toFixed(2)}</td>
      <td title="${escapeHtml(r.error ?? "")}">${r.status}</td></tr>`).join("")}</table>`;
}

async function run() {
  $("run-search").disabled = true;
  try {
    const { run_id } = await api("/api/search", { method: "POST", body: {
      location: $("location").value, date_start: $("date-start").value, date_end: $("date-end").value, queries: queries(),
    } });
    watching = run_id;
    $("confirm").checked = false;
    banner("", "");
    poll();
  } catch (err) {
    banner("error", `Couldn't start the search: ${err.message}`);
    update();
  }
}

export async function showSearch() {
  if (!loaded) {
    loaded = true;
    const d = await api("/api/search/defaults");
    estimate = d;
    d.queries.forEach((q) => addRow(q.text, q.max_results));
    $("location").value = d.location;
    $("date-start").value = d.date_start;
    $("date-end").value = d.date_end;
    $("add-query").addEventListener("click", () => { addRow(); update(); });
    $("confirm").addEventListener("change", update);
    $("run-search").addEventListener("click", run);
    update();
  }
  loadRuns();
  poll();
}
