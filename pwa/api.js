// fetch wrapper: JSON in and out, and a lapsed sign-in sends you back to the login page.
export async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (response.status === 401) {
    location.href = "/login?next=/";
    throw new Error("signed out");
  }
  if (!response.ok) throw new Error(`${method} ${path}: ${response.status}`);
  return response.headers.get("content-type")?.includes("json") ? response.json() : null;
}

export function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
