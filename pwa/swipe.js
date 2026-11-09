// Swipe tab: one unrated event at a time. Right = like, left = dislike, up = neutral.
import { api, escapeHtml } from "/pwa/api.js";

const LABELS = { like: 1, dislike: -1, neutral: 0 };
const TINTS = { like: "33,102,172", dislike: "178,24,43", neutral: "158,158,158" };
const THRESHOLD = 0.3; // share of the card width a drag must travel to count
const FLICK_SPEED = 0.6; // px per ms; a fast short swipe also counts
const LOW_WATER = 3; // fetch more when fewer cards than this are waiting

const stack = document.getElementById("stack");
const progress = document.getElementById("progress");
const undoButton = document.getElementById("undo");
let cards = [];
let history = [];
const skipped = new Set();
let counts = { reviewed: 0, total: 0 };
let fetching = false;
let busy = false;

function showProgress() {
  progress.textContent = `${counts.reviewed} rated · ${counts.total - counts.reviewed} left`;
  undoButton.disabled = history.length === 0;
}

async function refill() {
  if (fetching || cards.length >= LOW_WATER) return;
  fetching = true;
  try {
    const exclude = [...cards, ...history].map((c) => c.url).concat([...skipped]);
    const query = exclude.map((u) => `exclude=${encodeURIComponent(u)}`).join("&");
    const data = await api(`/api/queue?${query}`);
    const held = new Set(cards.map((c) => c.url));
    cards.push(...data.cards.filter((c) => !held.has(c.url)));
    counts = { reviewed: data.reviewed, total: data.total };
  } finally {
    fetching = false;
  }
  render();
}

function cardElement(card) {
  const el = document.createElement("article");
  el.className = "card";
  const score = card.p_like == null ? "" : ` · P(like) ${Math.round(card.p_like * 100)}%`;
  const link = /^https?:\/\//.test(card.url)
    ? `<a class="open" href="${escapeHtml(card.url)}" target="_blank" rel="noopener">Open event page ↗</a>` : "";
  el.innerHTML = `
    <div class="tint"></div>
    <div class="stamp like">LIKE</div><div class="stamp dislike">NOPE</div><div class="stamp neutral">MEH</div>
    <div class="meta"><b>${escapeHtml(card.theme)}</b> · ${escapeHtml(card.city)}${score}</div>
    <div class="text">${card.highlight ?? escapeHtml(card.description)}</div>
    <div class="note">${card.highlight ? "Blue words push P(like) up, red words push it down." : "Colouring words…"}</div>
    ${link}`;
  return el;
}

function render() {
  showProgress();
  stack.replaceChildren();
  if (cards.length === 0) {
    if (fetching) return;
    stack.innerHTML = skipped.size
      ? `<div class="empty"><p>Only skipped events are left.</p><button class="small" id="unskip">Show the ${skipped.size} skipped</button></div>`
      : `<div class="empty"><p><b>All caught up.</b></p><p>Run a search to find more events to rate.</p></div>`;
    document.getElementById("unskip")?.addEventListener("click", () => { skipped.clear(); refill(); });
    return;
  }
  if (cards[1]) {
    const behind = cardElement(cards[1]);
    behind.classList.add("behind");
    stack.append(behind);
  }
  const top = cardElement(cards[0]);
  stack.append(top);
  attachDrag(top);
  if (!cards[0].highlight && cards[0].p_like != null) waitForHighlight(cards[0]);
}

async function waitForHighlight(card) {
  while (cards[0] === card && !card.highlight) {
    await new Promise((r) => setTimeout(r, 2000));
    const data = await api(`/api/explanation?url=${encodeURIComponent(card.url)}`).catch(() => null);
    if (data) {
      card.highlight = data.highlight;
      if (cards[0] === card) render();
    }
  }
}

function direction(dx, dy) {
  if (dy < 0 && -dy > Math.abs(dx)) return "neutral";
  return dx >= 0 ? "like" : "dislike";
}

function feedback(el, dir, strength) {
  el.querySelector(".tint").style.cssText = `opacity:1;background:rgba(${TINTS[dir]},${0.22 * strength})`;
  el.querySelectorAll(".stamp").forEach((s) => { s.style.opacity = s.classList.contains(dir) ? strength : 0; });
}

function attachDrag(el) {
  let start = null;
  el.addEventListener("pointerdown", (e) => {
    if (busy || e.target.closest("a")) return;
    el.setPointerCapture(e.pointerId);
    el.classList.remove("settling");
    start = { x: e.clientX, y: e.clientY, t: performance.now() };
  });
  el.addEventListener("pointermove", (e) => {
    if (!start) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    el.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx / 20}deg)`;
    const dist = direction(dx, dy) === "neutral" ? -dy : Math.abs(dx);
    feedback(el, direction(dx, dy), Math.min(dist / (el.offsetWidth * THRESHOLD), 1));
  });
  const release = (e) => {
    if (!start) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    const dir = direction(dx, dy);
    const dist = dir === "neutral" ? -dy : Math.abs(dx);
    const speed = dist / Math.max(performance.now() - start.t, 1);
    start = null;
    if (dist > el.offsetWidth * THRESHOLD || (speed > FLICK_SPEED && dist > 50)) {
      rate(dir, { dx, dy });
    } else {
      el.classList.add("settling");
      el.style.transform = "";
      feedback(el, dir, 0);
    }
  };
  el.addEventListener("pointerup", release);
  el.addEventListener("pointercancel", release);
}

async function rate(dir, from = { dx: 0, dy: 0 }) {
  if (busy || cards.length === 0) return;
  busy = true;
  const card = cards[0];
  const el = stack.lastElementChild;
  const w = stack.offsetWidth;
  const to = dir === "neutral" ? `translate(${from.dx}px, ${-1.5 * stack.offsetHeight}px)`
    : `translate(${(dir === "like" ? 1.5 : -1.5) * w}px, ${from.dy}px) rotate(${dir === "like" ? 20 : -20}deg)`;
  feedback(el, dir, 1);
  el.classList.add("flying");
  el.style.transform = to;
  el.style.opacity = "0";
  try {
    await api("/api/label", { method: "POST", body: { url: card.url, label: LABELS[dir] } });
    cards.shift();
    history.push(card);
    counts.reviewed += 1;
  } catch (err) {
    alert(`Couldn't save that rating: ${err.message}`);
  }
  await new Promise((r) => setTimeout(r, 250));
  busy = false;
  render();
  refill();
}

async function undo() {
  if (busy || history.length === 0) return;
  busy = true;
  const card = history.pop();
  try {
    await api("/api/undo", { method: "POST", body: { url: card.url } });
    cards.unshift(card);
    counts.reviewed -= 1;
  } catch (err) {
    history.push(card);
    alert(`Couldn't undo: ${err.message}`);
  }
  busy = false;
  render();
}

function skip() {
  if (busy || cards.length === 0) return;
  skipped.add(cards.shift().url);
  render();
  refill();
}

export function startSwipe() {
  document.getElementById("like").addEventListener("click", () => rate("like"));
  document.getElementById("dislike").addEventListener("click", () => rate("dislike"));
  document.getElementById("neutral").addEventListener("click", () => rate("neutral"));
  document.getElementById("skip").addEventListener("click", skip);
  undoButton.addEventListener("click", undo);
  document.addEventListener("keydown", (e) => {
    if (!document.getElementById("tab-swipe").classList.contains("active")) return;
    const actions = { ArrowRight: () => rate("like"), ArrowLeft: () => rate("dislike"),
                      ArrowUp: () => rate("neutral"), ArrowDown: skip, Backspace: undo };
    if (actions[e.key]) {
      e.preventDefault();
      actions[e.key]();
    }
  });
  refill();
}
