// Shared by the session screen and onboarding practice: message cards, blink ring, speech.
"use strict";

const Session = (() => {
  const CUE = { up: "Tilt up", down: "Tilt down", left: "Tilt left", right: "Tilt right" };
  const ARROW = { up: "\u2191", down: "\u2193", left: "\u2190", right: "\u2192" };

  // Renders the menu into `el`. Rebuilds only when items or layout change, so CSS
  // transitions run on highlight changes instead of being reset every snapshot.
  function renderMenu(el, menu, armedProgress, state) {
    const key = menu.layout + "|" + menu.items.join(",");
    if (el.dataset.key !== key) {
      el.dataset.key = key;
      el.className = `menu menu-${menu.layout}`;
      el.innerHTML = "";
      menu.items.forEach((id, i) => {
        const card = document.createElement("div");
        card.className = "msg";
        card.dataset.index = String(i);
        if (menu.layout === "four_way") card.style.gridArea = menu.slots[i];
        card.innerHTML = `
          <svg class="msg-ring" aria-hidden="true"><rect class="msg-ring-fill" x="0" y="0" width="100%" height="100%" pathLength="100" rx="30" ry="30"/></svg>
          <span class="msg-text"></span><span class="msg-cue"></span>`;
        card.querySelector(".msg-text").textContent = menu.texts[i];
        el.appendChild(card);
      });
    }
    const [lo, hi] = menu.range;
    const narrowed = menu.layout === "binary_scan" && hi - lo < menu.items.length;
    const mid = lo + Math.ceil((hi - lo) / 2);
    [...el.children].forEach((card, i) => {
      const isHi = menu.highlighted === i;
      card.classList.toggle("hi", isHi);
      card.classList.toggle("armed", isHi && state === "armed");
      card.classList.toggle("out", narrowed && (i < lo || i >= hi));
      card.classList.toggle("in", narrowed && i >= lo && i < hi);
      const cue = card.querySelector(".msg-cue");
      if (menu.layout === "binary_scan") {
        const inRange = i >= lo && i < hi;
        cue.textContent = !inRange ? "" : i < mid ? `${ARROW.left} Left half` : `${ARROW.right} Right half`;
      } else {
        const slot = menu.slots[i];
        cue.textContent = `${ARROW[slot]} ${CUE[slot]}`;
      }
      const fill = card.querySelector(".msg-ring-fill");
      fill.style.strokeDashoffset = String(100 * (1 - (isHi && state === "armed" ? armedProgress : 0)));
    });
  }

  // Speech: only when the server sends "speak". Reports back when finished.
  function speaker(overlay, textEl, onDone) {
    let speaking = false;
    return function speak(text) {
      speaking = true;
      textEl.textContent = text;
      overlay.hidden = false;
      const done = () => {
        if (!speaking) return;
        speaking = false;
        overlay.hidden = true;
        onDone();
      };
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.95;
        u.onend = done;
        u.onerror = done;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        setTimeout(done, 8000); // never stay stuck if the voice engine does not report the end
      } else {
        setTimeout(done, 2500);
      }
    };
  }

  // Browsers need a user gesture before speech: call this from a click or key press.
  function unlockSpeech() {
    if ("speechSynthesis" in window) window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  }

  return { renderMenu, speaker, unlockSpeech };
})();
