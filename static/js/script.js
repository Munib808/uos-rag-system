(function () {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send-btn");
  const resetBtn = document.getElementById("reset-btn");
  const chipList = document.getElementById("chip-list");

  let isSending = false;

  // ---- auto-grow textarea -------------------------------------------------
  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }
  input.addEventListener("input", autoGrow);

  // Enter to send, Shift+Enter for newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // ---- helpers -------------------------------------------------------------
  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // Minimal markdown-ish formatting: **bold**, newlines, "- " bullet lists
  function formatAnswer(text) {
    let safe = escapeHtml(text);
    safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    const lines = safe.split("\n");
    let html = "";
    let inList = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (/^[-•]\s+/.test(trimmed)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${trimmed.replace(/^[-•]\s+/, "")}</li>`;
      } else {
        if (inList) { html += "</ul>"; inList = false; }
        if (trimmed.length) html += `<p>${trimmed}</p>`;
      }
    }
    if (inList) html += "</ul>";
    return html || `<p>${safe}</p>`;
  }

  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg user";
    row.innerHTML = `
      <div class="avatar">YOU</div>
      <div class="bubble"><p>${escapeHtml(text)}</p></div>
    `;
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function addTypingIndicator() {
    const row = document.createElement("div");
    row.className = "msg assistant";
    row.id = "typing-row";
    row.innerHTML = `
      <div class="avatar">UOS</div>
      <div class="bubble">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    `;
    messagesEl.appendChild(row);
    scrollToBottom();
  }

  function removeTypingIndicator() {
    const row = document.getElementById("typing-row");
    if (row) row.remove();
  }

  function addAssistantMessage(answer, sources, isError) {
    const row = document.createElement("div");
    row.className = "msg assistant";

    const bubbleClass = isError ? "bubble error-bubble" : "bubble";
    let sourcesHtml = "";

    if (!isError && sources && sources.length) {
      const cards = sources
        .map((s) => {
          const tag = [s.source, s.page ? `p.${s.page}` : null].filter(Boolean).join(" · ");
          return `<div class="source-card"><span class="tag">${escapeHtml(tag || "Handbook")}</span>${escapeHtml(s.excerpt)}${s.excerpt.length >= 400 ? "…" : ""}</div>`;
        })
        .join("");
      sourcesHtml = `
        <button class="sources-toggle" type="button">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          View ${sources.length} source passage${sources.length > 1 ? "s" : ""}
        </button>
        <div class="sources-panel">${cards}</div>
      `;
    }

    row.innerHTML = `
      <div class="avatar">UOS</div>
      <div class="${bubbleClass}">
        ${formatAnswer(answer)}
        ${sourcesHtml}
      </div>
    `;

    messagesEl.appendChild(row);

    const toggle = row.querySelector(".sources-toggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        row.querySelector(".sources-panel").classList.toggle("open");
      });
    }

    scrollToBottom();
  }

  // ---- send flow ------------------------------------------------------------
  async function sendMessage(text) {
    if (isSending || !text.trim()) return;
    isSending = true;
    sendBtn.disabled = true;

    addUserMessage(text);
    input.value = "";
    autoGrow();
    addTypingIndicator();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      removeTypingIndicator();

      if (!res.ok) {
        addAssistantMessage(data.error || "Something went wrong.", null, true);
      } else {
        addAssistantMessage(data.answer, data.sources, false);
      }
    } catch (err) {
      removeTypingIndicator();
      addAssistantMessage(
        "Couldn't reach the server. Check your connection and try again.",
        null,
        true
      );
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  chipList.addEventListener("click", (e) => {
    const btn = e.target.closest(".chip");
    if (!btn) return;
    sendMessage(btn.dataset.q);
  });

  resetBtn.addEventListener("click", async () => {
    await fetch("/api/reset", { method: "POST" });
    messagesEl.innerHTML = "";
    addAssistantMessage(
      "Conversation cleared. Ask me anything about hostel fees, facilities, rules, or applications.",
      null,
      false
    );
  });

  input.focus();
})();
