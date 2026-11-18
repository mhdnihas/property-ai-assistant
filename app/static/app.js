(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const chat = $("#chat");
  const form = $("#composer");
  const input = $("#input");
  const send = $("#send");
  const welcome = $("#welcome");
  const statusEl = $("#status");
  const { render, escapeHtml } = window.SafeMarkdown;

  const MAX_HISTORY = 12;
  let history = [];      // [{role, content}]
  let busy = false;
  let controller = null;

  // ------------------------------------------------------------------ status
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const h = await res.json();
      const ok = h.status === "ok" && h.llm_configured;
      statusEl.className = "status " + (ok ? "ok" : "err");
      statusEl.querySelector(".status-text").textContent = ok ? `${h.documents.toLocaleString()} docs indexed` : "AI not configured";
      statusEl.title = ok ? `Models: ${h.models.join(", ")}` : "OPENROUTER_API_KEY missing on server";
    } catch {
      statusEl.className = "status err";
      statusEl.querySelector(".status-text").textContent = "Offline";
    }
  }

  // ------------------------------------------------------------------ rendering
  const scrollDown = () => { chat.scrollTop = chat.scrollHeight; };

  function addUserMessage(text) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chat.appendChild(el);
    scrollDown();
  }

  function addBotMessage() {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = '<div class="bubble"><div class="answer"><div class="typing"><span></span><span></span><span></span></div></div></div>';
    chat.appendChild(el);
    scrollDown();
    return el.querySelector(".bubble");
  }

  const money = (v) => (v == null ? "" : `${Number(v).toLocaleString()} SAR`);

  function renderSources(bubble, sources) {
    if (!sources.length) return;
    const seen = new Map();
    for (const s of sources) {          // group chunks of the same page
      if (!seen.has(s.url)) seen.set(s.url, { ...s, nums: [s.n] });
      else seen.get(s.url).nums.push(s.n);
    }
    const items = [...seen.values()].map((s) => {
      const facts = s.kind === "listing"
        ? [s.district, s.city, s.bedrooms != null ? `${s.bedrooms} bd` : "", money(s.price)].filter(Boolean).join(" · ")
        : s.kind.replace("_", " ");
      return `<a class="source" href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">
        <div class="src-head"><span class="num">${s.nums.join(", ")}</span>
        <span class="badge ${s.source}">${s.source === "darglobal" ? "DarGlobal" : "Wasalt"}</span></div>
        <div class="src-title">${escapeHtml(s.title)}</div>
        <div class="src-facts">${escapeHtml(facts)}</div></a>`;
    }).join("");
    const details = document.createElement("details");
    details.className = "sources";
    details.innerHTML = `<summary>Sources (${seen.size})</summary><div class="source-list">${items}</div>`;
    bubble.appendChild(details);
  }

  // ------------------------------------------------------------------ SSE over fetch
  async function* readEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = "message", data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) yield { event, data: JSON.parse(data) };
      }
    }
  }

  async function ask(question) {
    if (busy || !question.trim()) return;
    busy = true;
    send.disabled = true;
    welcome?.remove();
    addUserMessage(question);
    const bubble = addBotMessage();
    const answerEl = bubble.querySelector(".answer");

    let answer = "";
    let sources = [];
    let pending = false;
    const paint = () => {
      pending = false;
      answerEl.innerHTML = render(answer, sources);
      scrollDown();
    };

    controller = new AbortController();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history: history.slice(-MAX_HISTORY) }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const msg = body.error || (Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail) || `Request failed (${res.status})`;
        throw new Error(msg);
      }
      for await (const { event, data } of readEvents(res)) {
        if (event === "sources") sources = data.sources;
        else if (event === "token") {
          answer += data.t;
          if (!pending) { pending = true; requestAnimationFrame(paint); }
        } else if (event === "error") {
          throw new Error(data.message);
        } else if (event === "done") {
          paint();
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = `${data.model || "model"} · ${data.elapsed_s}s`;
          bubble.appendChild(meta);
        }
      }
      paint();
      renderSources(bubble, sources);
      history.push({ role: "user", content: question }, { role: "assistant", content: answer });
    } catch (err) {
      if (err.name !== "AbortError") {
        if (!answer) answerEl.innerHTML = "";
        const note = document.createElement("div");
        note.className = "error-note";
        note.textContent = err.message || "Something went wrong.";
        bubble.appendChild(note);
      }
    } finally {
      busy = false;
      send.disabled = false;
      controller = null;
      scrollDown();
      input.focus();
    }
  }

  // ------------------------------------------------------------------ events
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    autosize();
    ask(q);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });

  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.addEventListener("input", autosize);

  document.getElementById("suggestions")?.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) ask(btn.textContent.trim());
  });

  $("#new-chat").addEventListener("click", () => {
    controller?.abort();
    history = [];
    chat.innerHTML = "";
    chat.appendChild(welcome);
    input.focus();
  });

  checkHealth();
  input.focus();
})();
