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

function getDebugAuthToken() {
  const fromQuery = qs("auth_token");
  if (fromQuery) {
    const t = String(fromQuery).trim();
    if (t) {
      localStorage.setItem("debugAuthToken", t);
      return t;
    }
  }
  return localStorage.getItem("debugAuthToken") || "";
}

function clearDebugAuthToken() {
  localStorage.removeItem("debugAuthToken");
}

function ensureDebugAuthToken() {
  const existing = getDebugAuthToken();
  if (existing) return existing;
  return "";
}

function requestDebugAuthTokenInteractively(options = {}) {
  const force = Boolean(options && options.force);
  return new Promise((resolve) => {
    const existing = ensureDebugAuthToken();
    if (existing) {
      if (!force) {
        resolve(existing);
        return;
      }
      clearDebugAuthToken();
    }
    const current = document.getElementById("debug-auth-overlay");
    if (current) current.remove();

    const old = document.getElementById("debug-auth-overlay");
    if (old) old.remove();

    const overlay = document.createElement("div");
    overlay.id = "debug-auth-overlay";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.background = "rgba(15, 23, 42, 0.45)";
    overlay.style.zIndex = "99999";
    overlay.style.display = "flex";
    overlay.style.alignItems = "center";
    overlay.style.justifyContent = "center";

    const card = document.createElement("div");
    card.style.width = "min(92vw, 420px)";
    card.style.background = "#fff";
    card.style.border = "1px solid #d1d5db";
    card.style.borderRadius = "10px";
    card.style.padding = "16px";
    card.style.boxShadow = "0 10px 30px rgba(0,0,0,0.2)";

    const title = document.createElement("div");
    title.textContent = "Debug UI Authentication";
    title.style.fontSize = "15px";
    title.style.fontWeight = "600";
    title.style.marginBottom = "10px";

    const desc = document.createElement("div");
    desc.textContent = "Enter bearer token to access debug timeline:";
    desc.style.fontSize = "13px";
    desc.style.color = "#4b5563";
    desc.style.marginBottom = "10px";

    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = "DEBUG_UI_AUTH_TOKEN";
    input.style.width = "100%";
    input.style.boxSizing = "border-box";
    input.style.padding = "8px 10px";
    input.style.border = "1px solid #cbd5e1";
    input.style.borderRadius = "6px";
    input.style.marginBottom = "10px";

    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.justifyContent = "flex-end";
    row.style.gap = "8px";

    const submit = document.createElement("button");
    submit.textContent = "Continue";
    submit.style.padding = "7px 12px";
    submit.style.border = "1px solid #2563eb";
    submit.style.background = "#2563eb";
    submit.style.color = "#fff";
    submit.style.borderRadius = "6px";
    submit.style.cursor = "pointer";

    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    cancel.style.padding = "7px 12px";
    cancel.style.border = "1px solid #cbd5e1";
    cancel.style.background = "#fff";
    cancel.style.color = "#111827";
    cancel.style.borderRadius = "6px";
    cancel.style.cursor = "pointer";

    const finish = (token) => {
      overlay.remove();
      resolve(token || "");
    };

    submit.addEventListener("click", () => {
      const token = (input.value || "").trim();
      if (!token) return;
      localStorage.setItem("debugAuthToken", token);
      finish(token);
    });
    cancel.addEventListener("click", () => finish(""));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submit.click();
    });

    row.appendChild(cancel);
    row.appendChild(submit);
    card.appendChild(title);
    card.appendChild(desc);
    card.appendChild(input);
    card.appendChild(row);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    input.focus();
  });
}

function installDebugAuthToolbar() {
  if (document.getElementById("debug-auth-toolbar")) return;
  const toolbar = document.createElement("div");
  toolbar.id = "debug-auth-toolbar";
  toolbar.style.position = "fixed";
  toolbar.style.right = "12px";
  toolbar.style.bottom = "12px";
  toolbar.style.zIndex = "99998";
  toolbar.style.display = "flex";
  toolbar.style.alignItems = "center";
  toolbar.style.gap = "8px";
  toolbar.style.padding = "8px 10px";
  toolbar.style.border = "1px solid #d1d5db";
  toolbar.style.borderRadius = "999px";
  toolbar.style.background = "rgba(255, 255, 255, 0.94)";
  toolbar.style.boxShadow = "0 4px 12px rgba(0,0,0,0.12)";

  const status = document.createElement("span");
  status.style.fontSize = "12px";
  status.style.color = "#4b5563";

  const setBtn = document.createElement("button");
  setBtn.textContent = "Set Token";
  setBtn.style.fontSize = "12px";
  setBtn.style.padding = "5px 8px";
  setBtn.style.border = "1px solid #2563eb";
  setBtn.style.background = "#2563eb";
  setBtn.style.color = "#fff";
  setBtn.style.borderRadius = "999px";
  setBtn.style.cursor = "pointer";

  const clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  clearBtn.style.fontSize = "12px";
  clearBtn.style.padding = "5px 8px";
  clearBtn.style.border = "1px solid #cbd5e1";
  clearBtn.style.background = "#fff";
  clearBtn.style.color = "#111827";
  clearBtn.style.borderRadius = "999px";
  clearBtn.style.cursor = "pointer";

  const refreshStatus = () => {
    status.textContent = ensureDebugAuthToken() ? "Auth: token set" : "Auth: missing";
  };

  setBtn.addEventListener("click", async () => {
    await requestDebugAuthTokenInteractively({ force: true });
    refreshStatus();
    location.reload();
  });
  clearBtn.addEventListener("click", () => {
    clearDebugAuthToken();
    refreshStatus();
    location.reload();
  });

  toolbar.appendChild(status);
  toolbar.appendChild(setBtn);
  toolbar.appendChild(clearBtn);
  document.body.appendChild(toolbar);
  refreshStatus();
}

async function api(path, options = {}) {
  const doFetch = async (token) => {
    const headers = { ...(options.headers || {}) };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    return fetch(`/debug/api${path}`, { ...options, headers });
  };

  let token = ensureDebugAuthToken();
  if (!token) {
    token = await requestDebugAuthTokenInteractively();
  }
  let r = await doFetch(token);
  if ((r.status === 401 || r.status === 403) && token) {
    clearDebugAuthToken();
    token = await requestDebugAuthTokenInteractively();
    if (token) {
      r = await doFetch(token);
    }
  }

  if (!r.ok) {
    const text = await r.text();
    throw new Error(`HTTP ${r.status}: ${text}`);
  }
  // Handle empty response for DELETE requests
  const contentLength = r.headers.get('content-length');
  if (contentLength === '0' || r.status === 204) {
    return null;
  }
  return r.json();
}

window.getDebugAuthToken = getDebugAuthToken;
window.ensureDebugAuthToken = ensureDebugAuthToken;
window.requestDebugAuthTokenInteractively = requestDebugAuthTokenInteractively;
window.installDebugAuthToolbar = installDebugAuthToolbar;

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", installDebugAuthToolbar);
} else {
  installDebugAuthToolbar();
}

function badgeClass(flag) {
  if (flag === true || flag === 1) return "badge ok";
  if (flag === false || flag === 0) return "badge warn";
  return "badge";
}
