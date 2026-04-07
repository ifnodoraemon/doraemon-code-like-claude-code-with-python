const state = {
  executionMode: "turn",
  maxWorkers: 2,
  currentSessionId: null,
  isStreaming: false,
  taskGraph: [],
  readyTasks: [],
  workerAssignments: {},
  toolCatalog: [],
  messages: [],
  sessions: [],
};

const els = {
  sessions: document.getElementById("sessions"),
  messages: document.getElementById("messages"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  form: document.getElementById("chat-form"),
  newChat: document.getElementById("new-chat"),
  modeTurn: document.getElementById("mode-turn"),
  modeOrchestrate: document.getElementById("mode-orchestrate"),
  maxWorkers: document.getElementById("max-workers"),
  workerLabel: document.getElementById("worker-label"),
  chatBadge: document.getElementById("chat-badge"),
  composerMode: document.getElementById("composer-mode"),
  summaryMode: document.getElementById("summary-mode"),
  summaryTasks: document.getElementById("summary-tasks"),
  summaryWorkers: document.getElementById("summary-workers"),
  taskCountBadge: document.getElementById("task-count-badge"),
  workerCountBadge: document.getElementById("worker-count-badge"),
  readyCountBadge: document.getElementById("ready-count-badge"),
  toolCountBadge: document.getElementById("tool-count-badge"),
  taskGraph: document.getElementById("task-graph"),
  workerAssignments: document.getElementById("worker-assignments"),
  readyTasks: document.getElementById("ready-tasks"),
  tools: document.getElementById("tools"),
};

function generateId() {
  return Math.random().toString(36).slice(2, 11);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setExecutionMode(mode) {
  state.executionMode = mode;
  els.modeTurn.classList.toggle("active", mode === "turn");
  els.modeOrchestrate.classList.toggle("active", mode === "orchestrate");
  els.maxWorkers.disabled = mode !== "orchestrate";
  els.workerLabel.textContent =
    mode === "orchestrate"
      ? `${state.maxWorkers} worker slots`
      : "Direct agent turn";
  els.chatBadge.textContent = mode === "orchestrate" ? `orchestrate x${state.maxWorkers}` : "single turn";
  els.composerMode.textContent =
    mode === "orchestrate" ? `workers: ${state.maxWorkers}` : "single agent turn";
  els.summaryMode.textContent = mode === "orchestrate" ? "Lead Runtime" : "Direct Turn";
  els.prompt.placeholder =
    mode === "orchestrate"
      ? "Describe the goal you want the lead runtime to decompose..."
      : "Ask Doraemon Code anything...";
}

function renderSessions() {
  if (!state.sessions.length) {
    els.sessions.innerHTML = '<div class="muted">No saved sessions.</div>';
    return;
  }

  els.sessions.innerHTML = state.sessions
    .map(
      (session) => `
        <button class="session-button ${session.id === state.currentSessionId ? "active" : ""}" data-session-id="${escapeHtml(session.id)}">
          <div class="session-title">${escapeHtml(session.name || "Untitled Session")}</div>
          <div class="session-meta">${escapeHtml(String(session.message_count))} messages</div>
        </button>
      `,
    )
    .join("");

  for (const button of els.sessions.querySelectorAll("[data-session-id]")) {
    button.addEventListener("click", () => {
      state.currentSessionId = button.getAttribute("data-session-id");
      renderSessions();
    });
  }
}

function renderMessages() {
  if (!state.messages.length) {
    els.messages.classList.add("empty");
    els.messages.innerHTML = `
      <div class="empty-state">
        <div class="empty-mark">RT</div>
        <h4>Choose a direct turn or orchestrate a goal</h4>
        <p>
          Orchestration will materialize a task graph, assign worker roles, and return a lead summary.
          Single turn keeps the direct chat loop.
        </p>
      </div>
    `;
    return;
  }

  els.messages.classList.remove("empty");
  els.messages.innerHTML = state.messages
    .map((message) => {
      const toolStrip =
        message.tool_calls && message.tool_calls.length
          ? `<div class="tool-strip">Tools: ${message.tool_calls
              .map((toolCall) => escapeHtml(toolCall.name || "tool"))
              .join(", ")}</div>`
          : "";
      return `
        <article class="message-row ${escapeHtml(message.role)}">
          ${message.role === "user" ? "" : '<div class="message-avatar">AI</div>'}
          <div class="message-bubble">
            ${message.meta ? `<div class="message-meta">${escapeHtml(message.meta)}</div>` : ""}
            <div class="message-content">${escapeHtml(message.content || "")}</div>
            ${toolStrip}
          </div>
          ${message.role === "user" ? '<div class="message-avatar">You</div>' : ""}
        </article>
      `;
    })
    .join("");

  els.messages.scrollTop = els.messages.scrollHeight;
}

function taskCount(nodes) {
  return nodes.reduce((count, node) => count + 1 + taskCount(node.children || []), 0);
}

function renderTaskTree(nodes, level = 0) {
  return nodes
    .map((node) => {
      const flags = [];
      if (node.ready) flags.push('<span class="pill">ready</span>');
      if (node.assigned_agent) flags.push(`<span class="pill">${escapeHtml(node.assigned_agent)}</span>`);
      return `
        <div class="task-card" style="margin-left:${level * 14}px">
          <div class="task-title">${escapeHtml(node.title)}</div>
          <div class="task-id">${escapeHtml(node.id)}</div>
          <div class="task-flags">
            <span class="pill status-pill ${escapeHtml(node.status)}">${escapeHtml(node.status)}</span>
            ${flags.join("")}
          </div>
        </div>
        ${renderTaskTree(node.children || [], level + 1)}
      `;
    })
    .join("");
}

function renderTaskPanels() {
  const taskTotal = taskCount(state.taskGraph);
  els.summaryTasks.textContent = String(taskTotal);
  els.summaryWorkers.textContent = String(Object.keys(state.workerAssignments).length);
  els.taskCountBadge.textContent = `${taskTotal} tasks`;
  els.workerCountBadge.textContent = `${Object.keys(state.workerAssignments).length} mappings`;
  els.readyCountBadge.textContent = `${state.readyTasks.length} ready`;
  els.toolCountBadge.textContent = `${state.toolCatalog.length} tools`;

  els.taskGraph.innerHTML = state.taskGraph.length
    ? renderTaskTree(state.taskGraph)
    : "No tasks yet. Run orchestration to persist a task graph.";

  const assignments = Object.entries(state.workerAssignments);
  els.workerAssignments.innerHTML = assignments.length
    ? assignments
        .map(
          ([taskId, assignment]) => `
            <div class="assignment-card">
              <div class="assignment-role">${escapeHtml(assignment.role)}</div>
              <div class="task-id">${escapeHtml(taskId)}</div>
              <div class="assignment-session">${escapeHtml(assignment.worker_session_id)}</div>
              <div class="assignment-tools">
                ${assignment.allowed_tool_names
                  .map((toolName) => `<span class="pill">${escapeHtml(toolName)}</span>`)
                  .join("")}
              </div>
            </div>
          `,
        )
        .join("")
    : "Worker assignments will appear after orchestration runs.";

  els.readyTasks.innerHTML = state.readyTasks.length
    ? state.readyTasks
        .map(
          (task) => `
            <div class="task-card">
              <div class="task-title">${escapeHtml(task.title)}</div>
              <div class="task-id">${escapeHtml(task.id)}</div>
            </div>
          `,
        )
        .join("")
    : "No ready tasks at the moment.";

  els.tools.innerHTML = state.toolCatalog.length
    ? state.toolCatalog
        .slice(0, 12)
        .map(
          (tool) => `
            <div class="tool-card">
              <div class="tool-name">${escapeHtml(tool.name)}</div>
              <div class="tool-description">${escapeHtml(tool.description || "")}</div>
              <div class="tool-flags">
                <span class="pill">${escapeHtml(tool.policy.capability_group || "uncategorized")}</span>
                <span class="pill">${escapeHtml(tool.policy.sandbox || "workspace_read")}</span>
                <span class="pill">${tool.policy.requires_approval ? "approval required" : "no approval"}</span>
              </div>
            </div>
          `,
        )
        .join("")
    : "Tool catalog is loading.";
}

async function fetchSessions() {
  try {
    const response = await fetch("/api/sessions");
    if (!response.ok) return;
    state.sessions = await response.json();
    renderSessions();
  } catch (error) {
    console.error("Failed to fetch sessions", error);
  }
}

async function fetchTasks() {
  try {
    const [treeResponse, readyResponse] = await Promise.all([
      fetch("/api/tasks?project=default&mode=build"),
      fetch("/api/tasks?project=default&mode=build&ready_only=true"),
    ]);

    if (treeResponse.ok) {
      const data = await treeResponse.json();
      state.taskGraph = data.tree || [];
    }
    if (readyResponse.ok) {
      const data = await readyResponse.json();
      state.readyTasks = data.tasks || [];
    }

    renderTaskPanels();
  } catch (error) {
    console.error("Failed to fetch tasks", error);
  }
}

async function fetchTools() {
  try {
    const response = await fetch("/api/tools?mode=build");
    if (!response.ok) return;
    const data = await response.json();
    state.toolCatalog = data.tools || [];
    renderTaskPanels();
  } catch (error) {
    console.error("Failed to fetch tools", error);
  }
}

function addMessage(message) {
  state.messages.push({
    id: generateId(),
    timestamp: Date.now(),
    ...message,
  });
  renderMessages();
}

function setStreaming(active) {
  state.isStreaming = active;
  els.send.disabled = active;
  els.prompt.disabled = active;
  els.send.textContent = active ? "..." : "Send";
}

async function submitPrompt(event) {
  event.preventDefault();
  const prompt = els.prompt.value.trim();
  if (!prompt || state.isStreaming) return;

  addMessage({
    role: "user",
    content: prompt,
    meta: state.executionMode === "orchestrate" ? `orchestrate x${state.maxWorkers}` : "turn",
  });

  els.prompt.value = "";
  setStreaming(true);

  const assistantMessage = {
    id: generateId(),
    role: "assistant",
    content: "",
    meta: state.executionMode === "orchestrate" ? "lead runtime" : "",
    timestamp: Date.now(),
    tool_calls: [],
  };
  state.messages.push(assistantMessage);
  renderMessages();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: prompt,
        session_id: state.currentSessionId,
        project: "default",
        execution_mode: state.executionMode,
        max_workers: state.executionMode === "orchestrate" ? state.maxWorkers : 1,
      }),
    });

    if (!response.ok) throw new Error("Network response was not ok");
    const reader = response.body?.getReader();
    if (!reader) throw new Error("Missing response stream");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        const payload = part.slice(6);
        if (payload === "[DONE]") continue;

        const data = JSON.parse(payload);
        const lastMessage = state.messages[state.messages.length - 1];
        if (!lastMessage || lastMessage.role !== "assistant") continue;

        if (data.type === "orchestration") {
          lastMessage.content = data.content || "";
          lastMessage.meta = data.result?.success ? "orchestration completed" : "orchestration blocked";
          state.workerAssignments = data.result?.worker_assignments || {};
          state.taskGraph = data.task_graph || [];
          renderTaskPanels();
        } else {
          if (data.content) {
            lastMessage.content = `${lastMessage.content || ""}${data.content}`;
          }
          if (data.tool_calls) {
            lastMessage.tool_calls = data.tool_calls;
          }
          if (data.error) {
            lastMessage.meta = data.error;
          }
        }
        renderMessages();
      }
    }

    await Promise.all([fetchSessions(), fetchTasks()]);
  } catch (error) {
    addMessage({
      role: "system",
      content: `Error: ${error}`,
    });
  } finally {
    setStreaming(false);
  }
}

function initEvents() {
  els.form.addEventListener("submit", submitPrompt);
  els.newChat.addEventListener("click", () => {
    state.messages = [];
    state.currentSessionId = null;
    state.workerAssignments = {};
    renderMessages();
    renderSessions();
    renderTaskPanels();
  });
  els.modeTurn.addEventListener("click", () => setExecutionMode("turn"));
  els.modeOrchestrate.addEventListener("click", () => setExecutionMode("orchestrate"));
  els.maxWorkers.addEventListener("input", () => {
    state.maxWorkers = Number(els.maxWorkers.value);
    if (state.executionMode === "orchestrate") {
      els.workerLabel.textContent = `${state.maxWorkers} worker slots`;
      els.chatBadge.textContent = `orchestrate x${state.maxWorkers}`;
      els.composerMode.textContent = `workers: ${state.maxWorkers}`;
    }
  });
}

async function bootstrap() {
  initEvents();
  setExecutionMode("turn");
  renderMessages();
  renderSessions();
  renderTaskPanels();
  await Promise.all([fetchSessions(), fetchTasks(), fetchTools()]);
}

bootstrap();
