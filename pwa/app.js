import { api } from "/pwa/api.js";
import { showAnalytics } from "/pwa/analytics.js";
import { showSearch } from "/pwa/search.js";
import { startSwipe } from "/pwa/swipe.js";

const SHOW = { search: showSearch, analytics: showAnalytics };

if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");

document.querySelectorAll(".tabbar button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabbar button").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${button.dataset.tab}`));
    SHOW[button.dataset.tab]?.();
  });
});

// Keeps the server awake while the app is on screen; it stops itself about 10 minutes after the app closes.
const { heartbeat_seconds } = await api("/api/config");
setInterval(() => {
  if (document.visibilityState === "visible") api("/api/heartbeat").catch(() => {});
}, heartbeat_seconds * 1000);

startSwipe();
