// Shared helpers: theme toggle, brand mark, icons, toasts.
"use strict";

const HV = (() => {
  const THEME_KEY = "hv-theme";

  function storageGet(k) {
    try { return localStorage.getItem(k); } catch { return null; }
  }
  function storageSet(k, v) {
    try { localStorage.setItem(k, v); } catch { /* private mode: fine */ }
  }

  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
    else delete document.documentElement.dataset.theme;
  }
  applyTheme(storageGet(THEME_KEY));

  function isDark() {
    const t = document.documentElement.dataset.theme;
    if (t) return t === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function toggleTheme() {
    const next = isDark() ? "light" : "dark";
    applyTheme(next);
    storageSet(THEME_KEY, next);
    syncThemeButtons();
  }

  // Line icons (24x24, stroke currentColor).
  const P = {
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    play: '<circle cx="12" cy="12" r="10"/><path d="M10 8l6 4-6 4z"/>',
    settings: '<path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    check: '<path d="M20 6L9 17l-5-5"/>',
    x: '<path d="M18 6L6 18M6 6l12 12"/>',
    camera: '<path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>',
    cameraOff: '<path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.7 0H14a2 2 0 0 1 2 2v3.3l1 1L23 7v10M1 1l22 22"/>',
    lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    heartHand: '<path d="M19 14c1.5-1.5 3-3.2 3-5.5A5.5 5.5 0 0 0 12 5 5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4 3 5.5l7 7z"/>',
    sliders: '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z"/><circle cx="12" cy="12" r="3"/>',
    eyeClosed: '<path d="M2 10c2.5 3 6 5 10 5s7.5-2 10-5M6 13.5L4.5 16M18 13.5l1.5 2.5M12 15v3"/>',
    tilt: '<path d="M12 3v3M5.6 5.6l2 2M3 12h3"/><circle cx="13" cy="13" r="7" transform="rotate(-20 13 13)"/><path d="M10.5 12h.01M15.5 11h.01"/>',
    message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>',
    replay: '<path d="M1 4v6h6"/><path d="M3.5 15a9 9 0 1 0 2.1-9.4L1 10"/>',
    arrowRight: '<path d="M5 12h14M12 5l7 7-7 7"/>',
    arrowLeft: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
    volume: '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14"/>',
    shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
  };
  function icon(name, cls = "") {
    return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ""}</svg>`;
  }

  // Brand mark: a face with one eye closed (the confirming blink) and a smile.
  const MARK = `<svg class="brand-mark" viewBox="0 0 40 40" aria-hidden="true">
    <circle cx="20" cy="20" r="19" fill="var(--accent-soft)"/>
    <circle cx="14" cy="17" r="2.2" fill="var(--accent)"/>
    <path d="M23.5 17.5q2.5 -2 5 0" stroke="var(--accent)" stroke-width="2.2" fill="none" stroke-linecap="round"/>
    <path d="M13 24q7 6 14 0" stroke="var(--accent)" stroke-width="2.4" fill="none" stroke-linecap="round"/>
  </svg>`;

  function syncThemeButtons() {
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
      b.innerHTML = icon(isDark() ? "sun" : "moon");
      b.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
    });
  }

  function init() {
    document.querySelectorAll("[data-brand-mark]").forEach((el) => (el.outerHTML = MARK));
    document.querySelectorAll("[data-icon]").forEach((el) => {
      el.outerHTML = icon(el.dataset.icon, el.className);
    });
    document.querySelectorAll("[data-theme-toggle]").forEach((b) => b.addEventListener("click", toggleTheme));
    syncThemeButtons();
  }

  function toast(text, ms = 3500) {
    let area = document.querySelector(".toast-area");
    if (!area) {
      area = document.createElement("div");
      area.className = "toast-area";
      area.setAttribute("role", "status");
      document.body.appendChild(area);
    }
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = text;
    area.appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  function esc(x) {
    return String(x ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  }

  // Detect whether the local HeadVoice app is serving this page.
  async function localApp() {
    try {
      const r = await fetch("/api/state", { cache: "no-store" });
      return r.ok;
    } catch {
      return false;
    }
  }

  // One WebSocket to the local app, with reconnect.
  function connect(onMessage) {
    let ws = null;
    const api = {
      send(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); },
    };
    (function open() {
      ws = new WebSocket(`ws://${location.host}/ws`);
      ws.onmessage = (e) => onMessage(JSON.parse(e.data));
      ws.onclose = () => setTimeout(open, 1000);
    })();
    return api;
  }

  document.addEventListener("DOMContentLoaded", init);
  return { icon, toast, esc, localApp, connect, storageGet, storageSet, isDark };
})();
