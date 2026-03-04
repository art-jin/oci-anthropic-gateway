function qs(name) {
  const p = new URLSearchParams(window.location.search);
  return p.get(name);
}

function esc(v) {
  if (v === null || v === undefined) return "";
  return String(v)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtJson(v) {
  return JSON.stringify(v, null, 2);
}

async function api(path) {
  const r = await fetch(`/debug/api${path}`);
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`HTTP ${r.status}: ${text}`);
  }
  return r.json();
}

function badgeClass(flag) {
  if (flag === true || flag === 1) return "badge ok";
  if (flag === false || flag === 0) return "badge warn";
  return "badge";
}
