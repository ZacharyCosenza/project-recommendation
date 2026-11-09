// Minimal service worker: registering one is what lets Android install the app. Push handlers go here next.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
