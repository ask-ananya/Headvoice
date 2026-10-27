// Landing page: hosted or local mode, and the animated "choosing a message" preview.
"use strict";

(async () => {
  const local = await HV.localApp();
  if (!local) {
    // Hosted site: the app itself runs on the user's computer.
    document.body.classList.add("hosted");
    const set = (id, prop, v) => { document.getElementById(id)[prop] = v; };
    set("primary-action", "href", "download.html");
    set("primary-title", "textContent", "Get HeadVoice");
    set("primary-text", "textContent", "Free for Windows. Runs on your own computer with any webcam.");
    set("secondary-action", "href", "#how");
    set("secondary-title", "textContent", "See how it works");
    set("secondary-text", "textContent", "Three steps: look, tilt, blink.");
    set("demo-title", "textContent", "See it on real recordings");
    set("demo-text", "textContent", "Replay recorded sessions on AWS and see how each interface setup performs, including the adaptation agent's decisions and the numbers behind them.");
    const demo = document.getElementById("demo-action");
    demo.href = document.body.dataset.judgeUrl || "judge";
    demo.firstChild.textContent = "Open the replay page ";
  } else if (HV.storageGet("hv-onboarded") === "1") {
    // Returning local user: lead with the messages.
    const a = document.getElementById("primary-action");
    const b = document.getElementById("secondary-action");
    a.parentNode.insertBefore(b, a);
    b.classList.add("lead");
  }
})();

// Preview: highlight a row, fill the ring (the long blink), mark it spoken, repeat.
(() => {
  const rows = [...document.querySelectorAll("#preview-list li")];
  const ring = document.getElementById("ring-fill");
  const status = document.getElementById("preview-status");
  const LEN = 94.25;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const order = [0, 2, 1, 3];
  let k = 0;

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  function setRing(f) { ring.style.strokeDashoffset = String(LEN * (1 - f)); }

  async function cycle() {
    while (true) {
      const row = rows[order[k % order.length]];
      k++;
      rows.forEach((r) => r.classList.remove("hi", "spoken"));
      setRing(0);
      status.textContent = "Tilt to choose, hold a blink to speak.";
      await wait(1200);
      row.classList.add("hi");
      status.textContent = `${row.dataset.dir}: "${row.firstElementChild.textContent}" is highlighted.`;
      await wait(1100);
      status.textContent = "Holding a long blink...";
      if (reduced) {
        setRing(1);
        await wait(700);
      } else {
        for (let i = 1; i <= 20; i++) { setRing(i / 20); await wait(35); }
      }
      row.classList.remove("hi");
      row.classList.add("spoken");
      status.textContent = `Spoken: "${row.firstElementChild.textContent}"`;
      await wait(2200);
    }
  }
  cycle();
})();
