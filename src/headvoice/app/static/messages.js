// My messages: add and remove the user's own phrases (stored on this computer only).
"use strict";

const $ = (id) => document.getElementById(id);
const GROUPS = { needs: "Needs", feelings: "Feelings", people: "People", responses: "Responses" };

async function load() {
  const r = await (await fetch("api/messages")).json();
  renderMine(r.custom);
  renderBuiltin(r.builtin);
}

function renderMine(items) {
  $("mine-count").textContent = String(items.length);
  $("empty").hidden = items.length > 0;
  $("mine").innerHTML = items
    .map((m) => `<li><span class="t">${HV.esc(m.text)}</span>
      <button class="btn btn-ghost" data-remove="${HV.esc(m.id)}" aria-label="Remove ${HV.esc(m.text)}">${HV.icon("x")} Remove</button></li>`)
    .join("");
}

function renderBuiltin(items) {
  $("builtin-count").textContent = String(items.length);
  const by = {};
  items.forEach((p) => (by[p.group] = by[p.group] || []).push(p.text));
  $("builtin").innerHTML = Object.entries(by)
    .map(([g, texts]) => `<h3>${HV.esc(GROUPS[g] || g)}</h3><div class="chips-wrap">${texts.map((t) => `<span class="pill neutral">${HV.esc(t)}</span>`).join("")}</div>`)
    .join("");
}

$("text").addEventListener("input", (e) => {
  $("count").textContent = `${e.target.value.length} / 80`;
  $("error").textContent = "";
});

$("add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  const r = await fetch("api/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const body = await r.json();
  if (!body.ok) {
    $("error").textContent = body.error;
    return;
  }
  $("text").value = "";
  $("count").textContent = "0 / 80";
  HV.toast(`Added "${body.message.text}"`);
  load();
  $("text").focus();
});

$("mine").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-remove]");
  if (!btn) return;
  const li = btn.closest("li");
  await fetch(`api/messages/${encodeURIComponent(btn.dataset.remove)}`, { method: "DELETE" });
  HV.toast(`Removed "${li.querySelector(".t").textContent}"`);
  load();
});

load();
