import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import {
    Bot,
    Boxes,
    Loader2,
    Network,
    Plus,
    Send,
    Settings,
    Sparkles,
    Terminal,
    Wrench,
} from 'lucide-react'

const generateId = () => Math.random().toString(36).slice(2, 11)

type ExecutionMode = 'turn' | 'orchestrate'

interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content?: string
    tool_calls?: Array<{ name?: string; arguments?: Record<string, unknown> }>
    timestamp: number
    meta?: string
}

interface Session {
    id: string
    name: string
    message_count: number
    updated_at: number
}

interface SessionDetail {
    id: string
    name: string | null
    messages: Array<{
        role: 'user' | 'assistant' | 'system' | 'tool'
        content?: string
        tool_calls?: Array<{ name?: string; arguments?: Record<string, unknown> }>
    }>
}

interface TaskNode {
    id: string
    title: string
    status: string
    ready?: boolean
    assigned_agent?: string | null
    children?: TaskNode[]
}

interface ToolCatalogItem {
    name: string
    description: string
    policy: {
        capability_group?: string | null
        requires_approval?: boolean
        sandbox?: string
    }
}

interface OrchestrationResult {
    success: boolean
    summary: string
    worker_assignments?: Record<
        string,
        {
            role: string
            worker_session_id: string
            allowed_tool_names: string[]
        }
    >
}

function App() {
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState<Message[]>([])
    const [isStreaming, setIsStreaming] = useState(false)
    const [sessions, setSessions] = useState<Session[]>([])
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
    const [executionMode, setExecutionMode] = useState<ExecutionMode>('turn')
    const [maxWorkers, setMaxWorkers] = useState(2)
    const [taskGraph, setTaskGraph] = useState<TaskNode[]>([])
    const [readyTasks, setReadyTasks] = useState<TaskNode[]>([])
    const [workerAssignments, setWorkerAssignments] = useState<
        Record<string, { role: string; worker_session_id: string; allowed_tool_names: string[] }>
    >({})
    const [tools, setTools] = useState<ToolCatalogItem[]>([])
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    useEffect(() => {
        void Promise.all([fetchSessions(), fetchTasks(), fetchTools()])
    }, [])

    useEffect(() => {
        if (!currentSessionId || isStreaming) {
            return
        }
        setWorkerAssignments({})
        void Promise.all([fetchSessionDetail(currentSessionId), fetchTasks()])
    }, [currentSessionId, isStreaming])

    const fetchSessions = async () => {
        try {
            const res = await fetch('/api/sessions')
            if (!res.ok) return
            const data = await res.json()
            setSessions(data)
        } catch (err) {
            console.error('Failed to fetch sessions', err)
        }
    }

    const fetchTasks = async () => {
        try {
            const [treeRes, readyRes] = await Promise.all([
                fetch('/api/tasks?project=default&mode=build'),
                fetch('/api/tasks?project=default&mode=build&ready_only=true'),
            ])
            if (treeRes.ok) {
                const data = await treeRes.json()
                setTaskGraph(data.tree || [])
            }
            if (readyRes.ok) {
                const data = await readyRes.json()
                setReadyTasks(data.tasks || [])
            }
        } catch (err) {
            console.error('Failed to fetch tasks', err)
        }
    }

    const fetchTools = async () => {
        try {
            const res = await fetch('/api/tools?mode=build')
            if (!res.ok) return
            const data = await res.json()
            setTools(data.tools || [])
        } catch (err) {
            console.error('Failed to fetch tools', err)
        }
    }

    const mapSessionMessage = (
        message: SessionDetail['messages'][number],
        index: number,
    ): Message => ({
        id: `${index}-${message.role}-${generateId()}`,
        role:
            message.role === 'tool'
                ? 'system'
                : message.role === 'system'
                  ? 'system'
                  : message.role,
        content: message.content,
        tool_calls: message.tool_calls,
        timestamp: Date.now() + index,
        meta: message.role === 'tool' ? 'tool result' : undefined,
    })

    const fetchSessionDetail = async (sessionId: string) => {
        try {
            const res = await fetch(`/api/sessions/${sessionId}`)
            if (!res.ok) return
            const data: SessionDetail = await res.json()
            setMessages((data.messages || []).map(mapSessionMessage))
        } catch (err) {
            console.error('Failed to fetch session detail', err)
        }
    }

    const resetConversation = () => {
        setMessages([])
        setCurrentSessionId(null)
        setTaskGraph([])
        setReadyTasks([])
        setWorkerAssignments({})
    }

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault()
        if (!input.trim() || isStreaming) return

        const prompt = input
        const userMessage: Message = {
            id: generateId(),
            role: 'user',
            content: prompt,
            timestamp: Date.now(),
            meta: executionMode === 'orchestrate' ? `orchestrate x${maxWorkers}` : 'turn',
        }

        setMessages((prev) => [...prev, userMessage])
        setInput('')
        setIsStreaming(true)

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: prompt,
                    session_id: currentSessionId,
                    project: 'default',
                    execution_mode: executionMode,
                    max_workers: executionMode === 'orchestrate' ? maxWorkers : 1,
                }),
            })

            if (!response.ok) throw new Error('Network response was not ok')

            const reader = response.body?.getReader()
            const decoder = new TextDecoder()

            if (!reader) return

            const assistantMessage: Message = {
                id: generateId(),
                role: 'assistant',
                content: '',
                timestamp: Date.now(),
                meta: executionMode === 'orchestrate' ? 'coordinator runtime' : undefined,
            }

            setMessages((prev) => [...prev, assistantMessage])

            const handleSsePayload = (payload: string) => {
                if (payload === '[DONE]') {
                    return true
                }

                try {
                    const data = JSON.parse(payload)
                    if (data.session_id) {
                        setCurrentSessionId(data.session_id)
                    }
                    if (data.type === 'orchestration') {
                        const result = data.result as OrchestrationResult
                        setMessages((prev) => {
                            const next = [...prev]
                            const last = next[next.length - 1]
                            if (last?.role === 'assistant') {
                                last.content = data.content || result.summary
                                last.meta = result.success
                                    ? 'orchestration completed'
                                    : 'orchestration blocked'
                            }
                            return next
                        })
                        setTaskGraph(data.task_graph || [])
                        setWorkerAssignments(result.worker_assignments || {})
                        return false
                    }

                    setMessages((prev) => {
                        const next = [...prev]
                        const last = next[next.length - 1]
                        if (last?.role !== 'assistant') return next

                        if (data.content) {
                            last.content = `${last.content || ''}${data.content}`
                        }
                        if (data.tool_calls) {
                            last.tool_calls = data.tool_calls
                        }
                        if (data.error) {
                            if (!last.content) {
                                last.content = `Error: ${data.error}`
                            }
                            last.meta = data.error
                        }
                        return next
                    })
                } catch (err) {
                    console.error('Failed to parse SSE payload', err)
                }

                return false
            }

            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()
                buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

                const events = buffer.split('\n\n')
                buffer = events.pop() || ''

                let shouldStop = false
                for (const eventChunk of events) {
                    const payload = eventChunk
                        .split('\n')
                        .filter((line) => line.startsWith('data: '))
                        .map((line) => line.slice(6))
                        .join('\n')

                    if (!payload) continue
                    if (handleSsePayload(payload)) {
                        shouldStop = true
                        break
                    }
                }

                if (shouldStop) {
                    await reader.cancel()
                    break
                }

                if (done) {
                    const payload = buffer
                        .split('\n')
                        .filter((line) => line.startsWith('data: '))
                        .map((line) => line.slice(6))
                        .join('\n')

                    if (payload) {
                        handleSsePayload(payload)
                    }
                    break
                }
            }

            await Promise.all([fetchTasks(), fetchSessions()])
        } catch (error) {
            console.error('Chat error', error)
            setMessages((prev) => [
                ...prev,
                {
                    id: generateId(),
                    role: 'system',
                    content: `Error: ${error}`,
                    timestamp: Date.now(),
                },
            ])
        } finally {
            setIsStreaming(false)
        }
    }

    return (
        <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.12),_transparent_32%),radial-gradient(circle_at_bottom_right,_rgba(249,115,22,0.12),_transparent_28%),linear-gradient(180deg,_#020617_0%,_#0f172a_48%,_#111827_100%)] text-slate-100">
            <div className="mx-auto flex min-h-screen max-w-[1700px] gap-4 px-4 py-4 lg:px-6">
                <aside className="hidden w-72 shrink-0 rounded-[28px] border border-white/10 bg-slate-950/80 p-4 shadow-2xl shadow-black/30 backdrop-blur xl:flex xl:flex-col">
                    <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-cyan-500/15 text-cyan-300">
                            <Terminal size={20} />
                        </div>
                        <div>
                            <div className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">Agent Runtime</div>
                            <h1 className="text-lg font-semibold text-white">Doraemon Code</h1>
                        </div>
                    </div>

                    <button
                        onClick={resetConversation}
                        className="mt-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-sm text-slate-200 transition hover:bg-white/10"
                    >
                        <Plus size={16} /> New Chat
                    </button>

                    <div className="mt-6">
                        <div className="mb-2 text-xs uppercase tracking-[0.24em] text-slate-500">Recent Sessions</div>
                        <div className="space-y-2">
                            {sessions.map((session) => (
                                <button
                                    key={session.id}
                                    onClick={() => setCurrentSessionId(session.id)}
                                    className={`w-full rounded-2xl border px-3 py-3 text-left text-sm transition ${
                                        currentSessionId === session.id
                                            ? 'border-cyan-400/40 bg-cyan-500/10 text-white'
                                            : 'border-white/5 bg-black/10 text-slate-400 hover:border-white/10 hover:bg-white/5'
                                    }`}
                                >
                                    <div className="truncate font-medium">{session.name || 'Untitled Session'}</div>
                                    <div className="mt-1 text-xs text-slate-500">
                                        {session.message_count} messages
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4">
                        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white">
                            <Sparkles size={16} className="text-amber-300" />
                            Execution
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => setExecutionMode('turn')}
                                className={`rounded-2xl px-3 py-2 text-sm transition ${
                                    executionMode === 'turn'
                                        ? 'bg-cyan-500 text-slate-950'
                                        : 'bg-slate-900/70 text-slate-300 hover:bg-slate-800'
                                }`}
                            >
                                Single Turn
                            </button>
                            <button
                                onClick={() => setExecutionMode('orchestrate')}
                                className={`rounded-2xl px-3 py-2 text-sm transition ${
                                    executionMode === 'orchestrate'
                                        ? 'bg-orange-400 text-slate-950'
                                        : 'bg-slate-900/70 text-slate-300 hover:bg-slate-800'
                                }`}
                            >
                                Orchestrate
                            </button>
                        </div>

                        <label className="mt-4 block text-xs uppercase tracking-[0.2em] text-slate-500">
                            Max Workers
                        </label>
                        <input
                            type="range"
                            min={1}
                            max={4}
                            value={maxWorkers}
                            onChange={(event) => setMaxWorkers(Number(event.target.value))}
                            disabled={executionMode !== 'orchestrate'}
                            className="mt-2 w-full accent-orange-400 disabled:opacity-40"
                        />
                        <div className="mt-2 text-sm text-slate-300">
                            {executionMode === 'orchestrate' ? `${maxWorkers} worker slots` : 'Uses direct agent turn'}
                        </div>
                    </div>

                    <div className="mt-auto rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-slate-400">
                        <div className="mb-2 flex items-center gap-2 text-slate-200">
                            <Settings size={16} />
                            Runtime Notes
                        </div>
                        <p>Lead/worker orchestration is agentic-first. Worker roles and tool scope are assigned at runtime, not by a fixed flow chart.</p>
                    </div>
                </aside>

                <main className="flex min-w-0 flex-1 flex-col gap-4">
                    <section className="rounded-[30px] border border-white/10 bg-slate-950/70 p-4 shadow-2xl shadow-black/20 backdrop-blur">
                        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                            <div>
                                <div className="text-xs uppercase tracking-[0.28em] text-cyan-300/70">Live Session</div>
                                <h2 className="mt-2 text-2xl font-semibold text-white">Agentic Workspace</h2>
                                <p className="mt-2 max-w-2xl text-sm text-slate-400">
                                    Single-turn chat and orchestration now share the same runtime. Task graph state, worker assignments, and tool surface are visible while you work.
                                </p>
                            </div>

                            <div className="grid gap-3 sm:grid-cols-3">
                                <SummaryCard
                                    icon={<Bot size={16} />}
                                    label="Mode"
                                    value={executionMode === 'orchestrate' ? 'Coordinated Runtime' : 'Direct Turn'}
                                />
                                <SummaryCard
                                    icon={<Boxes size={16} />}
                                    label="Tasks"
                                    value={String(countTasks(taskGraph))}
                                />
                                <SummaryCard
                                    icon={<Network size={16} />}
                                    label="Workers"
                                    value={String(Object.keys(workerAssignments).length)}
                                />
                            </div>
                        </div>
                    </section>

                    <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_420px]">
                        <section className="flex min-h-[640px] flex-col rounded-[30px] border border-white/10 bg-slate-950/70 shadow-2xl shadow-black/20 backdrop-blur">
                            <div className="border-b border-white/10 px-5 py-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div>
                                        <div className="text-xs uppercase tracking-[0.22em] text-slate-500">Conversation</div>
                                        <div className="mt-1 text-lg font-medium text-white">Prompt and Runtime Output</div>
                                    </div>
                                    <div className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-400">
                                        {executionMode === 'orchestrate' ? `orchestrate x${maxWorkers}` : 'single turn'}
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto px-4 py-5">
                                {messages.length === 0 ? (
                                    <div className="flex h-full flex-col items-center justify-center text-center">
                                        <div className="flex h-20 w-20 items-center justify-center rounded-[28px] bg-cyan-500/10 text-cyan-300">
                                            <Terminal size={34} />
                                        </div>
                                        <h3 className="mt-6 text-2xl font-semibold text-white">Choose a direct turn or orchestrate a goal</h3>
                                        <p className="mt-3 max-w-xl text-sm text-slate-400">
                                            Orchestration will persist a task graph, assign execution profiles, and return a coordinator summary. Single-turn mode keeps the classic direct chat loop.
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-4">
                                        {messages.map((message) => (
                                            <article
                                                key={message.id}
                                                className={`mx-auto flex max-w-4xl gap-4 ${
                                                    message.role === 'user' ? 'justify-end' : ''
                                                }`}
                                            >
                                                {message.role !== 'user' && (
                                                    <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-cyan-500/15 text-cyan-300">
                                                        <Bot size={16} />
                                                    </div>
                                                )}

                                                <div
                                                    className={`max-w-[85%] rounded-[26px] border px-5 py-4 ${
                                                        message.role === 'user'
                                                            ? 'border-cyan-400/30 bg-cyan-500/90 text-slate-950'
                                                            : message.role === 'system'
                                                              ? 'border-red-400/20 bg-red-500/10 text-red-100'
                                                              : 'border-white/10 bg-slate-900/80 text-slate-100'
                                                    }`}
                                                >
                                                    {message.meta && (
                                                        <div className="mb-2 text-[11px] uppercase tracking-[0.22em] text-slate-400">
                                                            {message.meta}
                                                        </div>
                                                    )}
                                                    <div className="whitespace-pre-wrap text-sm leading-6">{message.content}</div>
                                                    {message.tool_calls && message.tool_calls.length > 0 && (
                                                        <div className="mt-4 rounded-2xl border border-cyan-400/20 bg-black/30 p-3 text-xs text-cyan-200">
                                                            Tools: {message.tool_calls.map((toolCall) => toolCall.name || 'tool').join(', ')}
                                                        </div>
                                                    )}
                                                </div>

                                                {message.role === 'user' && (
                                                    <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-slate-700 text-xs font-semibold text-white">
                                                        You
                                                    </div>
                                                )}
                                            </article>
                                        ))}
                                        <div ref={messagesEndRef} />
                                    </div>
                                )}
                            </div>

                            <div className="border-t border-white/10 px-4 py-4">
                                <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
                                    <div className="rounded-[28px] border border-white/10 bg-black/20 p-3">
                                        <div className="mb-3 flex flex-wrap gap-2">
                                            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400">
                                                project: default
                                            </span>
                                            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400">
                                                {executionMode === 'orchestrate' ? `workers: ${maxWorkers}` : 'single agent turn'}
                                            </span>
                                        </div>
                                        <div className="flex gap-3">
                                            <input
                                                type="text"
                                                value={input}
                                                onChange={(event) => setInput(event.target.value)}
                                                placeholder={
                                                    executionMode === 'orchestrate'
                                                        ? 'Describe the goal you want the coordinator to decompose...'
                                                        : 'Ask Doraemon Code anything...'
                                                }
                                                className="min-w-0 flex-1 rounded-2xl border border-white/10 bg-slate-900/90 px-4 py-4 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-400/40"
                                                disabled={isStreaming}
                                            />
                                            <button
                                                type="submit"
                                                disabled={!input.trim() || isStreaming}
                                                className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-400 text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
                                            >
                                                {isStreaming ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                                            </button>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </section>

                        <aside className="grid min-h-[640px] gap-4 lg:grid-cols-2 xl:grid-cols-1">
                            <Panel
                                icon={<Network size={16} className="text-orange-300" />}
                                title="Task Graph"
                                subtitle={`${countTasks(taskGraph)} tasks persisted`}
                            >
                                {taskGraph.length === 0 ? (
                                    <EmptyState text="No tasks yet. Run orchestration to materialize a task graph." />
                                ) : (
                                    <div className="space-y-2">{renderTaskTree(taskGraph)}</div>
                                )}
                            </Panel>

                            <Panel
                                icon={<Boxes size={16} className="text-emerald-300" />}
                                title="Worker Assignments"
                                subtitle={`${Object.keys(workerAssignments).length} active mappings`}
                            >
                                {Object.entries(workerAssignments).length === 0 ? (
                                    <EmptyState text="Execution-profile assignments will appear after orchestration runs." />
                                ) : (
                                    <div className="space-y-3">
                                        {Object.entries(workerAssignments).map(([taskId, assignment]) => (
                                            <div key={taskId} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                                                <div className="flex items-center justify-between gap-3">
                                                    <div className="text-sm font-medium text-white">{assignment.role}</div>
                                                    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{taskId}</div>
                                                </div>
                                                <div className="mt-2 text-xs text-slate-400">{assignment.worker_session_id}</div>
                                                <div className="mt-3 flex flex-wrap gap-2">
                                                    {assignment.allowed_tool_names.map((toolName) => (
                                                        <span
                                                            key={`${taskId}-${toolName}`}
                                                            className="rounded-full border border-white/10 px-2 py-1 text-[11px] text-slate-300"
                                                        >
                                                            {toolName}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </Panel>

                            <Panel
                                icon={<Sparkles size={16} className="text-cyan-300" />}
                                title="Ready Queue"
                                subtitle={`${readyTasks.length} dependency-ready tasks`}
                            >
                                {readyTasks.length === 0 ? (
                                    <EmptyState text="No ready tasks at the moment." />
                                ) : (
                                    <div className="space-y-2">
                                        {readyTasks.map((task) => (
                                            <div key={task.id} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                                                <div className="text-sm font-medium text-white">{task.title}</div>
                                                <div className="mt-1 text-xs text-slate-500">{task.id}</div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </Panel>

                            <Panel
                                icon={<Wrench size={16} className="text-violet-300" />}
                                title="Tool Surface"
                                subtitle={`${tools.length} tools visible in build mode`}
                            >
                                {tools.length === 0 ? (
                                    <EmptyState text="No tool catalog loaded." />
                                ) : (
                                    <div className="space-y-2">
                                        {tools.slice(0, 12).map((tool) => (
                                            <div key={tool.name} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                                                <div className="flex items-start justify-between gap-3">
                                                    <div>
                                                        <div className="text-sm font-medium text-white">{tool.name}</div>
                                                        <div className="mt-1 text-xs text-slate-400">{tool.description}</div>
                                                    </div>
                                                    <div className="rounded-full border border-white/10 px-2 py-1 text-[11px] text-slate-400">
                                                        {tool.policy.capability_group || 'uncategorized'}
                                                    </div>
                                                </div>
                                                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-400">
                                                    <span className="rounded-full border border-white/10 px-2 py-1">
                                                        {tool.policy.sandbox || 'workspace_read'}
                                                    </span>
                                                    <span className="rounded-full border border-white/10 px-2 py-1">
                                                        {tool.policy.requires_approval ? 'approval required' : 'no approval'}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </Panel>
                        </aside>
                    </div>
                </main>
            </div>
        </div>
    )
}

function SummaryCard({
    icon,
    label,
    value,
}: {
    icon: ReactNode
    label: string
    value: string
}) {
    return (
        <div className="rounded-3xl border border-white/10 bg-black/20 px-4 py-3">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-slate-500">
                {icon}
                {label}
            </div>
            <div className="mt-2 text-lg font-semibold text-white">{value}</div>
        </div>
    )
}

function Panel({
    icon,
    title,
    subtitle,
    children,
}: {
    icon: React.ReactNode
    title: string
    subtitle: string
    children: ReactNode
}) {
    return (
        <section className="rounded-[28px] border border-white/10 bg-slate-950/70 p-4 shadow-2xl shadow-black/20 backdrop-blur">
            <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                    <div className="flex items-center gap-2 text-sm font-medium text-white">
                        {icon}
                        {title}
                    </div>
                    <div className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-500">{subtitle}</div>
                </div>
            </div>
            {children}
        </section>
    )
}

function EmptyState({ text }: { text: string }) {
    return <div className="rounded-2xl border border-dashed border-white/10 p-4 text-sm text-slate-500">{text}</div>
}

function renderTaskTree(nodes: TaskNode[], level = 0): React.ReactNode[] {
    return nodes.flatMap((node) => {
        const badgeColor =
            node.status === 'completed'
                ? 'bg-emerald-500/15 text-emerald-200'
                : node.status === 'blocked'
                  ? 'bg-red-500/15 text-red-200'
                  : node.status === 'in_progress'
                    ? 'bg-amber-500/15 text-amber-200'
                    : 'bg-slate-700/50 text-slate-300'

        const item = (
            <div
                key={node.id}
                style={{ marginLeft: `${level * 14}px` }}
                className="rounded-2xl border border-white/10 bg-black/20 p-3"
            >
                <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-white">{node.title}</div>
                        <div className="mt-1 text-xs text-slate-500">{node.id}</div>
                    </div>
                    <span className={`rounded-full px-2 py-1 text-[11px] uppercase tracking-[0.16em] ${badgeColor}`}>
                        {node.status}
                    </span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-400">
                    {node.ready && <span className="rounded-full border border-white/10 px-2 py-1">ready</span>}
                    {node.assigned_agent && (
                        <span className="rounded-full border border-white/10 px-2 py-1">{node.assigned_agent}</span>
                    )}
                </div>
            </div>
        )

        return [item, ...(node.children ? renderTaskTree(node.children, level + 1) : [])]
    })
}

function countTasks(nodes: TaskNode[]): number {
    return nodes.reduce((count, node) => count + 1 + countTasks(node.children || []), 0)
}

export default App
