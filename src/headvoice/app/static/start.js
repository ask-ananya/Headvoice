// Onboarding: welcome, camera check, optional calibration, practice.
"use strict";

const LABELS = ["Welcome", "Camera check", "Calibration", "Practice"];
let step = 1;
let snap = null;
let calibStarted = false;
let practiceStarted = false;
const done = { hi: false, say1: false, say2: false };
const $ = (id) => document.getElementById(id);

const conn = HV.connect(onMessage);
const speak = Session.speaker($("speaking"), $("speaking-text"), () => conn.send({ type: "speech_ended" }));

function show(n) {
  step = n;
  document.querySelectorAll(".step").forEach((s) => (s.hidden = Number(s.dataset.step) !== n));
  [...$("stepper").children].forEach((li, i) => {
    li.className = i + 1 < n ? "done" : i + 1 === n ? "current" : "";
  });
  $("step-label").textContent = `Step ${n} of 4: ${LABELS[n - 1]}`;
  $("cam-preview").src = n === 2 ? "/preview.mjpg" : "";
  if (n === 2) loadCameras();
  if (n === 4 && !practiceStarted) {
    Session.unlockSpeech();
    conn.send({ type: "practice" });
    practiceStarted = true;
  }
  window.scrollTo({ top: 0 });
}

document.querySelectorAll("[data-next]").forEach((b) => b.addEventListener("click", () => show(step + 1)));
document.querySelectorAll("[data-back]").forEach((b) => b.addEventListener("click", () => show(step - 1)));

// Camera step -------------------------------------------------------------------------
async function loadCameras() {
  const sel = $("cam-select");
  sel.innerHTML = "<option>Looking for cameras...</option>";
  try {
    const r = await (await fetch("/api/cameras")).json();
    const name = { in_use: "in use by HeadVoice", live: "working", frozen: "picture frozen", no_frames: "no picture" };
    sel.innerHTML = r.cameras
      .map((c) => `<option value="${c.index}" ${c.index === r.current ? "selected" : ""} ${c.status === "frozen" || c.status === "no_frames" ? "disabled" : ""}>Camera ${c.index + 1} (${name[c.status] || c.status})</option>`)
      .join("");
  } catch {
    sel.innerHTML = "<option>Could not list cameras</option>";
  }
}
$("cam-refresh").addEventListener("click", loadCameras);
$("cam-select").addEventListener("change", async (e) => {
  await fetch("/api/camera", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ index: Number(e.target.value) }) });
  HV.toast("Switching camera...");
});
$("recenter").addEventListener("click", () => {
  conn.send({ type: "neutral" });
  HV.toast("Resting position updated");
});

function renderCamera(s) {
  const flags = s.flags || [];
  const camOk = s.camera_ok && !flags.includes("frozen_feed");
  const face = camOk && !flags.includes("no_face");
  const neutral = face && !flags.includes("no_neutral");
  $("chk-camera").classList.toggle("ok", camOk);
  $("chk-face").classList.toggle("ok", face);
  $("chk-neutral").classList.toggle("ok", neutral);
  $("cam-empty").hidden = camOk;
  let hint = "";
  if (!s.camera_ok) hint = "No picture. Pick another camera below.";
  else if (flags.includes("frozen_feed")) hint = "The picture is frozen. If you use a phone as a webcam, open its app, or pick another camera.";
  else if (flags.includes("no_face")) hint = "Move so your whole face is in the picture.";
  else if (flags.includes("low_light")) hint = "It is a little dark. More light on your face will help.";
  else if (flags.includes("no_neutral")) hint = "Hold still and look at the screen for a moment.";
  else if (flags.includes("turned_away")) hint = "Face the screen.";
  else hint = "All set.";
  $("cam-hint").textContent = hint;
  $("cam-next").disabled = !neutral;
}

// Calibration step --------------------------------------------------------------------
let audio = null;
function tone(freq) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const o = audio.createOscillator();
    const g = audio.createGain();
    o.frequency.value = freq;
    g.gain.value = 0.15;
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + 0.15);
  } catch { /* no audio: prompts are also on screen */ }
}
function startCalibration(upload) {
  tone(0); // unlock audio on this click
  calibStarted = true;
  conn.send({ type: "calibrate", upload });
  $("calib-choices").hidden = true;
  $("calib-live").hidden = false;
  $("calib-skip").hidden = true;
}
$("calib-local").addEventListener("click", () => startCalibration(false));
$("calib-cloud").addEventListener("click", () => startCalibration(true));

let lastLabel = null;
const LAYOUT_NAME = { four_way: "four directions (up, down, left, right)", binary_scan: "left and right scanning", two_choice: "two choices, left and right" };
function renderCalibration(s) {
  $("calib-cloud").hidden = !s.online;
  const c = s.calibration;
  if (c) {
    $("calib-step").textContent = `Step ${c.step} of ${c.of}`;
    $("calib-prompt").textContent = c.prompt;
    $("calib-bar").style.width = `${Math.round((c.progress || 0) * 100)}%`;
    if (c.label !== lastLabel) {
      if (c.label === "closed") tone(440);
      else if (lastLabel === "closed") tone(880);
      lastLabel = c.label;
    }
  } else if (calibStarted && s.calibration_result) {
    calibStarted = false;
    const r = s.calibration_result;
    $("calib-live").hidden = true;
    const box = $("calib-result");
    box.hidden = false;
    box.innerHTML = r.ok
      ? `<h3>Done. Your layout: ${HV.esc(LAYOUT_NAME[s.layout] || s.layout)}</h3><p>Long blink: ${HV.esc(s.params.long_blink_ms)} ms. HeadVoice will keep adjusting as you go.</p>`
      : `<h3>We could not finish calibration</h3><p>HeadVoice will use its standard settings, which work for most people. You can calibrate again later with the C key.</p>`;
    $("calib-next").hidden = false;
  }
}

// Practice step -----------------------------------------------------------------------
function renderPractice(s) {
  if (!practiceStarted) return;
  Session.renderMenu($("practice-menu"), s.menu, s.armed_progress, s.state);
  const t = s.task;
  if (t && t.practice) {
    $("practice-target").textContent = t.target ? `Goal: say "${t.text}"` : "Both messages said. Well done!";
    if (t.i >= 1) done.say1 = true;
    if (t.i >= 2 || !t.target) done.say2 = true;
  }
  if (s.menu.highlighted !== null && s.menu.highlighted !== undefined) done.hi = true;
  $("chk-hi").classList.toggle("ok", done.hi);
  $("chk-say1").classList.toggle("ok", done.say1);
  $("chk-say2").classList.toggle("ok", done.say2);
}
$("finish").addEventListener("click", () => HV.storageSet("hv-onboarded", "1"));

function onMessage(msg) {
  if (msg.type === "speak") speak(msg.text);
  else if (msg.type === "notice") HV.toast(msg.text);
  else if (msg.type === "snapshot") {
    snap = msg;
    if (step === 2) renderCamera(msg);
    if (step === 3) renderCalibration(msg);
    if (step === 4) renderPractice(msg);
  }
}

show(1);
