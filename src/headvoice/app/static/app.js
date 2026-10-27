// Session screen. Speaks only when the server sends "speak".
"use strict";

const $ = (id) => document.getElementById(id);
let snap = null;
let calibActive = false;
let lastCalibLabel = null;

const conn = HV.connect(onMessage);
const speak = Session.speaker($("speaking"), $("speaking-text"), () => conn.send({ type: "speech_ended" }));

function onMessage(msg) {
  if (msg.type === "speak") speak(msg.text);
  else if (msg.type === "notice") HV.toast(friendly(msg.text));
  else if (msg.type === "snapshot") render(msg);
}

// Agent notices in plain words for the user; details stay in the helper panel.
function friendly(text) {
  if (text === "Please hold still") return "Please hold still for a moment.";
  if (text === "Recalibration suggested") return "A quick recalibration may help. A helper can press C.";
  return text;
}

// Status chips, camera window, guidance -------------------------------------------------
const STATE_TEXT = {
  idle: ["Looking for you", "neutral"], tracking: ["Listening", "ok"], highlighting: ["Choosing", "ok"],
  armed: ["Hold the blink", "ok"], speaking: ["Speaking", "ok"], recovering: ["Lost you, waiting", "warn"],
};
function chip(id, text, tone) {
  const el = $(id);
  el.className = `pill ${tone}`;
  el.lastElementChild.textContent = text;
}

function qualityTone(s) {
  const f = s.flags || [];
  if (!s.camera_ok || f.includes("frozen_feed") || f.includes("no_face")) return "bad";
  if (s.quality < 0.5) return "warn";
  return "ok";
}

function guidance(s) {
  const f = s.flags || [];
  if (!s.camera_ok) return ["The camera is not giving a picture. A helper can pick another camera in setup.", true];
  if (f.includes("frozen_feed")) return ["The camera picture is frozen. If you use a phone as a webcam, open its app.", true];
  if (f.includes("no_face")) return ["Move so your face is in the camera picture.", true];
  if (f.includes("no_neutral")) return ["Hold still and look at the screen. Learning your resting position.", false];
  if (f.includes("turned_away")) return ["Face the screen to keep going.", true];
  if (f.includes("low_light")) return ["It is a little dark. More light on your face will help.", true];
  if (!s.running) return ["Ready. A helper presses Start, or the S key.", false];
  if (s.state === "armed") return ["Keep your eyes closed until the ring fills.", false];
  if (s.state === "highlighting") return ["Hold a long blink to say it. Two quick blinks cancel.", false];
  if (s.menu.layout === "binary_scan") return ["Tilt left or right to narrow down. Hold a long blink to speak.", false];
  return ["Tilt to choose. Hold a long blink to speak. A long blink with nothing chosen shows more messages.", false];
}

function render(s) {
  snap = s;
  document.body.classList.toggle("running", s.running);
  $("start-btn").innerHTML = s.running ? "Stop <kbd>S</kbd>" : "Start <kbd>S</kbd>";
  $("start-btn").className = s.running ? "btn btn-secondary" : "btn btn-primary";

  const [stText, stTone] = s.running ? STATE_TEXT[s.state] || [s.state, "neutral"] : ["Not started", "neutral"];
  chip("chip-state", stText, stTone);
  const q = qualityTone(s);
  chip("chip-signal", q === "ok" ? "Signal good" : q === "warn" ? "Signal weak" : "No signal", q);
  chip("chip-online", s.online ? "Online" : "Offline", s.online ? "ok" : "neutral");

  const pip = $("pip");
  pip.className = `pip q-${q}`;
  $("pip-empty").hidden = s.camera_ok;
  $("pip-status").textContent = q === "ok" ? "You are in view" : q === "warn" ? "Signal weak" : "Not in view";

  Session.renderMenu($("menu"), s.menu, s.armed_progress, s.state);
  const [g, warn] = guidance(s);
  $("guide-text").textContent = g;
  $("guide").classList.toggle("warn", warn);

  const t = s.task;
  $("task").hidden = !t || t.practice;
  if (t && !t.practice) $("task").textContent = t.target ? `Please say: ${t.text} (${t.i + 1} of ${t.n})` : "Task complete. Thank you.";

  renderCalibration(s);
  if (!$("drawer").hidden) renderDrawer(s);
}

// Calibration (helper key C) -------------------------------------------------------------
let audio = null;
function tone(freq) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = audio.createOscillator();
    const g = audio.createGain();
    o.frequency.value = freq || 440;
    g.gain.value = freq ? 0.15 : 0;
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + 0.15);
  } catch { /* prompts are also on screen */ }
}
function openConsent() {
  $("consent-cloud").hidden = $("consent-cloud-note").hidden = !(snap && snap.online);
  $("consent").hidden = false;
}
function closeConsent() { $("consent").hidden = true; }
function startCalibration(upload) {
  closeConsent();
  tone(0); // silent: unlocks audio on this user gesture
  calibActive = true;
  conn.send({ type: "calibrate", upload });
}
$("consent-local").onclick = () => startCalibration(false);
$("consent-cloud").onclick = () => startCalibration(true);
$("consent-cancel").onclick = closeConsent;

function renderCalibration(s) {
  const c = s.calibration;
  $("calibration").hidden = !c;
  if (!c) {
    lastCalibLabel = null;
    if (calibActive && s.calibration_result) calibActive = false;
    return;
  }
  $("calib-step").textContent = `Step ${c.step} of ${c.of}`;
  $("calib-prompt").textContent = c.prompt;
  $("calib-bar").style.width = `${Math.round((c.progress || 0) * 100)}%`;
  if (c.label !== lastCalibLabel) {
    if (c.label === "closed") tone(440);
    else if (lastCalibLabel === "closed") tone(880);
    lastCalibLabel = c.label;
  }
}

// Helper drawer -------------------------------------------------------------------------
const fmt = (v, d = 1) => (v === null || v === undefined ? "--" : typeof v === "number" ? v.toFixed(d) : HV.esc(v));
const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;

function renderDrawer(s) {
  const f = s.features || {};
  $("features").innerHTML =
    row("Frames per second", fmt(s.fps)) +
    row("Turn / nod / tilt", `${fmt(f.yaw)} / ${fmt(f.pitch)} / ${fmt(f.roll)} deg`) +
    row("Eyes open (L / R)", `${fmt(f.eye_open_l, 2)} / ${fmt(f.eye_open_r, 2)}`) +
    row("Signal quality", `${fmt(f.quality, 2)} ${HV.esc((f.quality_flags || []).join(", "))}`) +
    row("Landmarks", HV.esc(f.landmark_source || "--")) +
    row("Low-light boost", f.clahe_applied ? "on" : "off") +
    row("Camera", s.camera_index === null || s.camera_index === undefined ? "--" : `Camera ${s.camera_index + 1}`);
  const p = s.params;
  $("params").innerHTML =
    row("Layout", HV.esc(s.layout)) + row("Profile", HV.esc(p.profile)) +
    row("Long blink", `${p.long_blink_ms} ms`) + row("Tilt sensitivity", fmt(p.tilt_scale, 2)) +
    row("Highlight timeout", `${p.highlight_timeout_ms} ms`) + row("Undo", HV.esc(p.undo_gesture.replace("_", " "))) +
    row("Adaptation", HV.esc(s.policy)) + row("Condition", HV.esc(s.condition));
  if (document.activeElement !== $("condition")) $("condition").value = s.condition;
  if (document.activeElement !== $("layout-select")) $("layout-select").value = s.layout;
  $("layout-note").textContent = s.layout_locked
    ? "Layout chosen by a helper: automatic adaptation will not change it."
    : "Choosing a layout here keeps it: automatic adaptation will not change it this session.";

  const d = s.agent_decisions;
  $("agent").innerHTML = !d.length
    ? `<p class="fine">${s.policy === "none" ? "Adaptation is off." : "No decisions yet. Adaptation checks in about once a minute, or sooner when something changes."}</p>`
    : d.slice().reverse().map((x) => {
        const acts = x.actions.map((a, i) =>
          `${HV.esc(a.tool)} ${HV.esc(JSON.stringify(a.args))} <span class="${x.applied[i] ? "applied" : "rejected"}">${x.applied[i] ? "applied" : "rejected: " + HV.esc(x.rejected || "")}</span>`).join("; ");
        const ev = Object.entries(x.evidence).filter(([, v]) => v !== null && v !== undefined)
          .map(([k, v]) => `${k}=${HV.esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(" ");
        const out = x.outcome_after
          ? `<div class="out">after: ${Object.entries(x.outcome_after).map(([k, v]) => `${k}=${v}`).join(" ")}</div>` : "";
        return `<div class="decision"><b>${fmt(x.t, 0)} s, ${HV.esc(x.policy)}, ${HV.esc(x.trigger)}</b>${acts}<div class="why">${HV.esc(x.reason)}</div><div class="ev">${ev}</div>${out}</div>`;
      }).join("");

  $("events").innerHTML = s.events.slice().reverse().map((e) => {
    const extra = Object.entries(e).filter(([k]) => k !== "type" && k !== "t")
      .map(([k, v]) => `${k}=${typeof v === "number" ? fmt(v, 0) : HV.esc(JSON.stringify(v))}`).join(" ");
    return `<li>${fmt(e.t, 1)} ${HV.esc(e.type)} ${extra}</li>`;
  }).join("");
}

function toggleDrawer(open) {
  const d = $("drawer");
  d.hidden = open === undefined ? !d.hidden : !open;
  if (!d.hidden && snap) renderDrawer(snap);
}
$("drawer-btn").onclick = () => toggleDrawer();
$("drawer-close").onclick = () => toggleDrawer(false);

// Controls ------------------------------------------------------------------------------
function startStop() {
  Session.unlockSpeech(); // the click or key press also unlocks browser speech
  conn.send({ type: snap && snap.running ? "stop" : "start" });
}
$("start-btn").onclick = startStop;
$("recenter").onclick = () => { conn.send({ type: "neutral" }); HV.toast("Resting position updated"); };
$("calibrate").onclick = () => { toggleDrawer(false); openConsent(); };
$("task-start").onclick = () => conn.send({ type: "task_start", n: 10 });
$("task-skip").onclick = () => conn.send({ type: "task_skip" });
$("layout-select").onchange = (e) => {
  conn.send({ type: "layout", layout: e.target.value });
  HV.toast("Layout changed. It will stay as chosen.");
};
$("condition").onchange = (e) =>
  fetch("api/condition", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ condition: e.target.value }) });

// Camera window: shown at the top right, can be hidden (V).
function setPip(visible) {
  $("pip").hidden = !visible;
  $("pip-show").hidden = visible;
  document.body.classList.toggle("pip-hidden", !visible);
  $("pip-img").src = visible ? "preview.mjpg" : "";
  HV.storageSet("hv-pip", visible ? "1" : "0");
}
$("pip-hide").onclick = () => setPip(false);
$("pip-show").onclick = () => setPip(true);
setPip(HV.storageGet("hv-pip") !== "0");
if (HV.storageGet("hv-onboarded") === "1") $("setup-link").hidden = true;

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (!$("consent").hidden) {
    if (k === "y" && snap && snap.online) startCalibration(true);
    else if (k === "l") startCalibration(false);
    else if (k === "escape") closeConsent();
    return;
  }
  if (k === "s") startStop();
  else if (k === "i") toggleDrawer();
  else if (k === "n") $("recenter").click();
  else if (k === "c") openConsent();
  else if (k === "v") setPip($("pip").hidden);
  else if (k === "t") conn.send({ type: "task_start", n: 10 });
  else if (k === "k") conn.send({ type: "task_skip" });
  else if (k === "escape") {
    if (snap && snap.calibration) conn.send({ type: "calibrate_cancel" });
    else toggleDrawer(false);
  }
});
