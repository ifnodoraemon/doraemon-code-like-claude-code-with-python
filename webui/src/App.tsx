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

import {
    filterMessagesForRange,
    getNextSessionIndex,
    isMessageRangeLoaded,
} from './lib/sessionWindow'

const generateId = () => Math.random().toString(36).slice(2, 11)

type ExecutionMode = 'turn' | 'orchestrate'
type RunViewMode = 'session' | 'run'
const DEFAULT_MESSAGE_WINDOW = 200
type ToolCallSummary = { name?: string; arguments?: Record<string, unknown> }
type WorkerAssignment = {
    role: string
    worker_session_id: string
    allowed_tool_names: string[]
}

interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content?: string
    tool_calls?: ToolCallSummary[]
    timestamp: number
    meta?: string
    session_index?: number
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
        tool_calls?: ToolCallSummary[]
    }>
    message_count?: number
    message_offset?: number
    has_more_messages?: boolean
    orchestration_state?: OrchestrationRun
    orchestration_runs?: OrchestrationRun[]
    active_orchestration_run_id?: string | null
}

interface SessionWindowSnapshot {
    offset: number
    count: number
    hasMore: boolean
    messages: Message[]
    runs: OrchestrationRun[]
    preferredRunId: string | null
}

interface RunDetailResponse {
    run: OrchestrationRun
}

interface ProjectContextResponse {
    project_dir: string
    project_name: string
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
}

interface OrchestrationRun extends OrchestrationResult {
    run_id?: string
    goal?: string
    resumed_from_run_id?: string | null
    message_start_index?: number | null
    message_end_index?: number | null
    root_task_id?: string | null
    plan_id?: string | null
    executed_task_ids?: string[]
    completed_task_ids?: string[]
    failed_task_ids?: string[]
    blocked_task_id?: string | null
    task_summaries?: Record<string, string>
    task_graph?: TaskNode[]
    worker_assignments?: Record<string, WorkerAssignment>
}

function appendToolCalls(
    existing: ToolCallSummary[] | undefined,
    incoming: ToolCallSummary[] | undefined,
): ToolCallSummary[] | undefined {
    if (!incoming || incoming.length === 0) {
        return existing
    }
    return [...(existing || []), ...incoming]
}

function collectReadyTasks(nodes: TaskNode[]): TaskNode[] {
    return nodes.flatMap((node) => {
        const readyNodes =
            node.ready && node.status === 'pending'
                ? [{ id: node.id, title: node.title, status: node.status, ready: node.ready }]
                : []
        return [...readyNodes, ...collectReadyTasks(node.children || [])]
    })
}

function normalizeOrchestrationRuns(
    runs: OrchestrationRun[] | undefined,
    latest: OrchestrationRun | undefined,
): OrchestrationRun[] {
    if (runs && runs.length > 0) {
        return runs
    }
    if (latest && (latest.run_id || latest.root_task_id || latest.summary || (latest.task_graph || []).length > 0)) {
        return [latest]
    }
    return []
}

function getExplicitlySelectedRun(
    runs: OrchestrationRun[],
    selectedRunId: string | null,
): OrchestrationRun | null {
    if (!selectedRunId) {
        return null
    }
    return runs.find((run) => run.run_id === selectedRunId) || null
}

function upsertRun(existing: OrchestrationRun[], incoming: OrchestrationRun): OrchestrationRun[] {
    if (!incoming.run_id) {
        return [...existing, incoming]
    }
    const next = existing.filter((run) => run.run_id !== incoming.run_id)
    next.push(incoming)
    return next
}

function App() {
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState<Message[]>([])
    const [isStreaming, setIsStreaming] = useState(false)
    const [currentProjectDir, setCurrentProjectDir] = useState('')
    const [sessions, setSessions] = useState<Session[]>([])
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
    const [executionMode, setExecutionMode] = useState<ExecutionMode>('turn')
    const [maxWorkers, setMaxWorkers] = useState(2)
    const [orchestrationRuns, setOrchestrationRuns] = useState<OrchestrationRun[]>([])
    const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
    const [runViewMode, setRunViewMode] = useState<RunViewMode>('session')
    const [needsSessionHydration, setNeedsSessionHydration] = useState(false)
    const [messageOffset, setMessageOffset] = useState(0)
    const [messageCount, setMessageCount] = useState(0)
    const [hasMoreMessages, setHasMoreMessages] = useState(false)
    const [uiError, setUiError] = useState<string | null>(null)
    const [taskGraph, setTaskGraph] = useState<TaskNode[]>([])
    const [readyTasks, setReadyTasks] = useState<TaskNode[]>([])
    const [workerAssignments, setWorkerAssignments] = useState<Record<string, WorkerAssignment>>({})
    const [tools, setTools] = useState<ToolCatalogItem[]>([])
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const currentSessionIdRef = useRef<string | null>(null)
    const selectedRunIdRef = useRef<string | null>(null)
    const runViewModeRef = useRef<RunViewMode>('session')
    const sessionDetailRequestRef = useRef(0)
    const runDetailRequestRef = useRef(0)
    const messageOffsetRef = useRef(0)
    const activeStreamRequestRef = useRef(0)
    const activeStreamControllerRef = useRef<AbortController | null>(null)

    const selectedRun =
        orchestrationRuns.find((run) => run.run_id === selectedRunId) ||
        orchestrationRuns[orchestrationRuns.length - 1] ||
        null
    const selectedMessageRun =
        runViewMode === 'run' ? getExplicitlySelectedRun(orchestrationRuns, selectedRunId) : null
    const visibleMessages = filterMessagesForRange(messages, selectedMessageRun)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [visibleMessages])

    useEffect(() => {
        currentSessionIdRef.current = currentSessionId
    }, [currentSessionId])

    useEffect(() => {
        selectedRunIdRef.current = selectedRunId
    }, [selectedRunId])

    useEffect(() => {
        runViewModeRef.current = runViewMode
    }, [runViewMode])

    useEffect(() => {
        messageOffsetRef.current = messageOffset
    }, [messageOffset])

    useEffect(() => {
        void Promise.all([fetchTools(), fetchProjectContext()])
    }, [])

    useEffect(() => {
        void fetchSessions()
    }, [])

    useEffect(() => {
        if (runViewMode !== 'run') {
            setTaskGraph([])
            setReadyTasks([])
            setWorkerAssignments({})
            return
        }
        const selectedRun =
            orchestrationRuns.find((run) => run.run_id === selectedRunId) ||
            orchestrationRuns[orchestrationRuns.length - 1]
        const selectedTaskGraph = selectedRun?.task_graph || []
        setTaskGraph(selectedTaskGraph)
        setReadyTasks(collectReadyTasks(selectedTaskGraph))
        setWorkerAssignments(selectedRun?.worker_assignments || {})
    }, [orchestrationRuns, selectedRunId, runViewMode])

    const fetchProjectContext = async () => {
        try {
            const res = await fetch('/api/projects')
            if (!res.ok) return
            const data: ProjectContextResponse = await res.json()
            setCurrentProjectDir(data.project_dir)
        } catch (err) {
            console.error('Failed to fetch project context', err)
        }
    }

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

    const invalidateSessionDetailRequests = (sessionId: string | null = currentSessionIdRef.current) => {
        sessionDetailRequestRef.current += 1
        currentSessionIdRef.current = sessionId
    }

    const invalidateRunDetailRequests = () => {
        runDetailRequestRef.current += 1
    }

    const cancelActiveStream = () => {
        activeStreamRequestRef.current += 1
        activeStreamControllerRef.current?.abort()
        activeStreamControllerRef.current = null
        setIsStreaming(false)
    }

    const setSessionView = () => {
        runViewModeRef.current = 'session'
        selectedRunIdRef.current = null
        setRunViewMode('session')
        setSelectedRunId(null)
    }

    const setRunView = (runId: string | null) => {
        runViewModeRef.current = 'run'
        selectedRunIdRef.current = runId
        setRunViewMode('run')
        setSelectedRunId(runId)
    }

    const applySessionWindowSnapshot = (
        snapshot: SessionWindowSnapshot,
        mode: 'replace' | 'prepend' = 'replace',
    ) => {
        messageOffsetRef.current = snapshot.offset
        if (mode === 'prepend') {
            setMessages((prev) => [...snapshot.messages, ...prev])
        } else {
            setMessages(snapshot.messages)
        }
        setMessageOffset(snapshot.offset)
        setMessageCount(snapshot.count)
        setHasMoreMessages(snapshot.hasMore)
        setOrchestrationRuns(snapshot.runs)
        setSelectedRunId(snapshot.preferredRunId)
        setNeedsSessionHydration(false)
        setUiError(null)
    }

    const mapSessionMessage = (
        message: SessionDetail['messages'][number],
        index: number,
        offset: number,
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
        session_index: offset + index,
    })

    const fetchSessionDetail = async (
        sessionId: string,
        options?: {
            messageLimit?: number
            messageOffset?: number
            mode?: 'replace' | 'prepend'
            preserveRunSelection?: boolean
        },
    ): Promise<SessionWindowSnapshot | null> => {
        try {
            const requestId = sessionDetailRequestRef.current + 1
            sessionDetailRequestRef.current = requestId
            const params = new URLSearchParams()
            if (typeof options?.messageLimit === 'number') {
                params.set('message_limit', String(options.messageLimit))
            }
            if (typeof options?.messageOffset === 'number') {
                params.set('message_offset', String(options.messageOffset))
            }
            const query = params.toString()
            const res = await fetch(`/api/sessions/${sessionId}${query ? `?${query}` : ''}`)
            if (!res.ok) return null
            const data: SessionDetail = await res.json()
            if (
                requestId !== sessionDetailRequestRef.current ||
                currentSessionIdRef.current !== sessionId
            ) {
                return null
            }
            const offset = data.message_offset || 0
            const mappedMessages = (data.messages || []).map((message, index) =>
                mapSessionMessage(message, index, offset),
            )
            const runs = normalizeOrchestrationRuns(data.orchestration_runs, data.orchestration_state)
            const preferredRunId =
                (options?.preserveRunSelection ?? runViewModeRef.current === 'run')
                    ? getExplicitlySelectedRun(runs, selectedRunIdRef.current)?.run_id ||
                      data.active_orchestration_run_id ||
                      runs[runs.length - 1]?.run_id ||
                      null
                    : null
            const snapshot: SessionWindowSnapshot = {
                offset,
                count: data.message_count || mappedMessages.length,
                hasMore: Boolean(data.has_more_messages),
                messages: mappedMessages,
                runs,
                preferredRunId,
            }
            applySessionWindowSnapshot(snapshot, options?.mode)
            return snapshot
        } catch (err) {
            console.error('Failed to fetch session detail', err)
            return null
        }
    }

    const fetchRunDetail = async (sessionId: string, runId: string): Promise<OrchestrationRun | null> => {
        try {
            const requestId = runDetailRequestRef.current + 1
            runDetailRequestRef.current = requestId
            const res = await fetch(`/api/sessions/${sessionId}/runs/${runId}`)
            if (!res.ok) return null
            const data: RunDetailResponse = await res.json()
            if (
                requestId !== runDetailRequestRef.current ||
                currentSessionIdRef.current !== sessionId
            ) {
                return null
            }
            setOrchestrationRuns((prev) => upsertRun(prev, data.run))
            return data.run
        } catch (err) {
            console.error('Failed to fetch run detail', err)
            return null
        }
    }

    const clearConversationWindow = (options?: { clearRuns?: boolean }) => {
        messageOffsetRef.current = 0
        setMessages([])
        setMessageOffset(0)
        setMessageCount(0)
        setHasMoreMessages(false)
        setTaskGraph([])
        setReadyTasks([])
        setWorkerAssignments({})
        if (options?.clearRuns) {
            setOrchestrationRuns([])
            setSelectedRunId(null)
        }
    }

    const resetConversation = () => {
        cancelActiveStream()
        invalidateSessionDetailRequests(null)
        invalidateRunDetailRequests()
        runViewModeRef.current = 'session'
        selectedRunIdRef.current = null
        messageOffsetRef.current = 0
        setUiError(null)
        setMessages([])
        setCurrentSessionId(null)
        setOrchestrationRuns([])
        setSelectedRunId(null)
        setRunViewMode('session')
        setNeedsSessionHydration(false)
        setMessageOffset(0)
        setMessageCount(0)
        setHasMoreMessages(false)
        setTaskGraph([])
        setReadyTasks([])
        setWorkerAssignments({})
    }

    const startStreamingRequest = async (
        body: Record<string, unknown>,
        userMessage: Message,
        assistantMeta?: string,
    ) => {
        setUiError(null)
        const sessionIdForRequest = currentSessionIdRef.current
        if (currentSessionId && (runViewMode === 'run' || needsSessionHydration)) {
            const preparedWindow = await fetchSessionDetail(currentSessionId, {
                messageLimit: DEFAULT_MESSAGE_WINDOW,
                mode: 'replace',
                preserveRunSelection: false,
            })
            if (!preparedWindow) {
                if (currentSessionIdRef.current === sessionIdForRequest) {
                    setUiError('Failed to synchronize the latest session transcript before sending.')
                }
                return
            }
        }
        invalidateSessionDetailRequests(currentSessionIdRef.current)
        invalidateRunDetailRequests()
        const streamRequestId = activeStreamRequestRef.current + 1
        activeStreamRequestRef.current = streamRequestId
        const abortController = new AbortController()
        activeStreamControllerRef.current = abortController
        setNeedsSessionHydration(false)
        setSessionView()
        setMessages((prev) => [
            ...prev,
            {
                ...userMessage,
                session_index: getNextSessionIndex(prev, messageOffsetRef.current),
            },
        ])
        setIsStreaming(true)

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: abortController.signal,
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
                meta: assistantMeta,
            }

            setMessages((prev) => [
                ...prev,
                {
                    ...assistantMessage,
                    session_index: getNextSessionIndex(prev, messageOffsetRef.current),
                },
            ])

            const handleSsePayload = (payload: string) => {
                if (activeStreamRequestRef.current !== streamRequestId) {
                    return true
                }
                if (payload === '[DONE]') {
                    return true
                }

                try {
                    const data = JSON.parse(payload)
                    if (data.session_id) {
                        currentSessionIdRef.current = data.session_id
                        setCurrentSessionId(data.session_id)
                    }
                    if (data.type === 'orchestration') {
                        const result = data.result as OrchestrationRun
                        const run = { ...result, task_graph: data.task_graph || [] }
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
                        setOrchestrationRuns((prev) => upsertRun(prev, run))
                        if (run.run_id) {
                            setRunView(run.run_id)
                        }
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
                            last.tool_calls = appendToolCalls(last.tool_calls, data.tool_calls)
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

            await fetchSessions()
        } catch (error) {
            if (
                abortController.signal.aborted ||
                activeStreamRequestRef.current !== streamRequestId
            ) {
                return
            }
            console.error('Chat error', error)
            setUiError(`Chat error: ${error}`)
            setMessages((prev) => [
                ...prev,
                {
                    id: generateId(),
                    role: 'system',
                    content: `Error: ${error}`,
                    timestamp: Date.now(),
                    session_index: getNextSessionIndex(prev, messageOffsetRef.current),
                },
            ])
        } finally {
            if (activeStreamRequestRef.current === streamRequestId) {
                activeStreamControllerRef.current = null
                setIsStreaming(false)
            }
        }
    }

    useEffect(() => {
        if (!currentSessionId || !needsSessionHydration || isStreaming) {
            return
        }

        clearConversationWindow({ clearRuns: true })
        void (async () => {
            const snapshot = await fetchSessionDetail(currentSessionId, {
                messageLimit: DEFAULT_MESSAGE_WINDOW,
                mode: 'replace',
                preserveRunSelection: false,
            })
            if (!snapshot && currentSessionIdRef.current === currentSessionId) {
                setUiError('Failed to load session transcript.')
                setNeedsSessionHydration(false)
            }
        })()
    }, [currentSessionId, needsSessionHydration, isStreaming])

    useEffect(() => {
        if (
            runViewMode !== 'run' ||
            !selectedMessageRun ||
            !currentSessionId ||
            isStreaming ||
            isMessageRangeLoaded(
                selectedMessageRun.message_start_index,
                selectedMessageRun.message_end_index,
                messageOffset,
                messages,
            )
        ) {
            return
        }
        const startIndex = selectedMessageRun.message_start_index || 0
        const endIndex = selectedMessageRun.message_end_index || startIndex
        const runId = selectedMessageRun.run_id || null
        void (async () => {
            const snapshot = await fetchSessionDetail(currentSessionId, {
                messageOffset: startIndex,
                messageLimit: Math.max(1, endIndex - startIndex + 1),
                mode: 'replace',
                preserveRunSelection: true,
            })
            if (
                !snapshot &&
                currentSessionIdRef.current === currentSessionId &&
                runViewModeRef.current === 'run' &&
                selectedRunIdRef.current === runId
            ) {
                setUiError('Failed to load run transcript.')
            }
        })()
    }, [runViewMode, selectedMessageRun, currentSessionId, isStreaming, messageOffset, messages.length])

    useEffect(() => {
        if (
            runViewMode !== 'run' ||
            !currentSessionId ||
            !selectedRunId ||
            isStreaming
        ) {
            return
        }

        const run = orchestrationRuns.find((item) => item.run_id === selectedRunId)
        if (!run || run.task_graph) {
            return
        }

        void (async () => {
            const detailedRun = await fetchRunDetail(currentSessionId, selectedRunId)
            if (!detailedRun && currentSessionIdRef.current === currentSessionId) {
                setUiError('Failed to load orchestration run details.')
            }
        })()
    }, [runViewMode, currentSessionId, selectedRunId, orchestrationRuns, isStreaming])

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

        setInput('')
        await startStreamingRequest(
            {
                message: prompt,
                session_id: currentSessionId,
                project: currentProjectDir || 'default',
                execution_mode: executionMode,
                max_workers: executionMode === 'orchestrate' ? maxWorkers : 1,
            },
            userMessage,
            executionMode === 'orchestrate' ? 'coordinator runtime' : undefined,
        )
    }

    const handleResumeRun = async (run: OrchestrationRun) => {
        if (!currentSessionId || !run.run_id || isStreaming) return
        setExecutionMode('orchestrate')
        await startStreamingRequest(
            {
                message: '',
                session_id: currentSessionId,
                resume_run_id: run.run_id,
                project: currentProjectDir || 'default',
                execution_mode: 'orchestrate',
                max_workers: maxWorkers,
            },
            {
                id: generateId(),
                role: 'user',
                content: `Resume orchestration: ${run.goal || run.root_task_id || run.run_id}`,
                timestamp: Date.now(),
                meta: 'resume run',
            },
            'coordinator runtime',
        )
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
                                    onClick={() => {
                                        cancelActiveStream()
                                        invalidateSessionDetailRequests(session.id)
                                        invalidateRunDetailRequests()
                                        setUiError(null)
                                        setSessionView()
                                        clearConversationWindow({ clearRuns: true })
                                        setNeedsSessionHydration(true)
                                        setCurrentSessionId(session.id)
                                    }}
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
                                <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 px-4 py-3">
                                    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
                                        Project Directory
                                    </div>
                                    <div className="mt-2 break-all font-mono text-sm text-slate-200">
                                        {currentProjectDir || 'Loading...'}
                                    </div>
                                </div>
                                <div className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-500">
                                    The Web UI runs against the directory it was started from.
                                </div>
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
                                    <div className="flex flex-wrap items-center gap-2">
                                        {runViewMode === 'run' && selectedMessageRun && (
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    clearConversationWindow()
                                                    invalidateSessionDetailRequests(currentSessionId)
                                                    invalidateRunDetailRequests()
                                                    setUiError(null)
                                                    setSessionView()
                                                    if (currentSessionId) {
                                                        void (async () => {
                                                            const snapshot = await fetchSessionDetail(currentSessionId, {
                                                                messageLimit: DEFAULT_MESSAGE_WINDOW,
                                                                mode: 'replace',
                                                                preserveRunSelection: false,
                                                            })
                                                            if (!snapshot && currentSessionIdRef.current === currentSessionId) {
                                                                setUiError('Failed to load session transcript.')
                                                            }
                                                        })()
                                                    }
                                                }}
                                                className="rounded-full border border-cyan-400/30 px-3 py-1 text-xs text-cyan-200 transition hover:bg-cyan-500/10"
                                                disabled={isStreaming}
                                            >
                                                Show Session Transcript
                                            </button>
                                        )}
                                        {runViewMode === 'session' && selectedRun?.run_id && (
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    invalidateSessionDetailRequests(currentSessionId)
                                                    invalidateRunDetailRequests()
                                                    setUiError(null)
                                                    setRunView(selectedRun.run_id || null)
                                                }}
                                                className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
                                                disabled={isStreaming}
                                            >
                                                View Selected Run
                                            </button>
                                        )}
                                        <div className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-400">
                                            {runViewMode === 'run' && selectedMessageRun
                                                ? `run ${selectedMessageRun.run_id || selectedMessageRun.root_task_id || ''}`
                                                : executionMode === 'orchestrate'
                                                  ? `orchestrate x${maxWorkers}`
                                                  : 'single turn'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto px-4 py-5">
                                {uiError && (
                                    <div className="mx-auto mb-4 max-w-4xl rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                                        {uiError}
                                    </div>
                                )}
                                {visibleMessages.length === 0 ? (
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
                                        {runViewMode === 'session' && hasMoreMessages && currentSessionId && (
                                            <div className="mx-auto max-w-4xl">
                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        setUiError(null)
                                                        void (async () => {
                                                            const snapshot = await fetchSessionDetail(currentSessionId, {
                                                                messageOffset: Math.max(
                                                                    0,
                                                                    messageOffset - DEFAULT_MESSAGE_WINDOW,
                                                                ),
                                                                messageLimit: Math.min(
                                                                    DEFAULT_MESSAGE_WINDOW,
                                                                    messageOffset,
                                                                ),
                                                                mode: 'prepend',
                                                                preserveRunSelection: false,
                                                            })
                                                            if (!snapshot && currentSessionIdRef.current === currentSessionId) {
                                                                setUiError('Failed to load older messages.')
                                                            }
                                                        })()
                                                    }}
                                                    className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
                                                    disabled={isStreaming}
                                                >
                                                    Load Older Messages ({messageOffset}/{messageCount})
                                                </button>
                                            </div>
                                        )}
                                        {visibleMessages.map((message) => (
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
                                                project dir: {currentProjectDir || 'loading'}
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
                                icon={<Terminal size={16} className="text-cyan-300" />}
                                title="Runs"
                                subtitle={`${orchestrationRuns.length} orchestration attempts in session`}
                            >
                                {orchestrationRuns.length === 0 ? (
                                    <EmptyState text="Run orchestration to build session-level execution history." />
                                ) : (
                                    <div className="space-y-3">
                                        {[...orchestrationRuns].reverse().map((run) => {
                                            const isSelected = run.run_id === (selectedRun?.run_id || null)
                                            return (
                                                <div
                                                    key={run.run_id || `${run.root_task_id}-${run.summary}`}
                                                    className={`rounded-2xl border p-3 ${
                                                        isSelected
                                                            ? 'border-cyan-400/40 bg-cyan-500/10'
                                                            : 'border-white/10 bg-black/20'
                                                    }`}
                                                >
                                                    <button
                                                        onClick={() => {
                                                            invalidateSessionDetailRequests(currentSessionId)
                                                            invalidateRunDetailRequests()
                                                            setUiError(null)
                                                            setRunView(run.run_id || null)
                                                        }}
                                                        className="w-full text-left disabled:opacity-60"
                                                        disabled={isStreaming}
                                                    >
                                                        <div className="flex items-start justify-between gap-3">
                                                            <div>
                                                                <div className="text-sm font-medium text-white">
                                                                    {run.goal || run.summary || run.root_task_id || 'Run'}
                                                                </div>
                                                                <div className="mt-1 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                                                                    {run.success ? 'completed' : 'blocked'}
                                                                    {run.resumed_from_run_id ? ' · resumed' : ''}
                                                                </div>
                                                            </div>
                                                            <div className="text-[11px] text-slate-500">
                                                                {run.run_id || run.root_task_id}
                                                            </div>
                                                        </div>
                                                    </button>
                                                    {!run.success && run.run_id && currentSessionId && (
                                                        <button
                                                            onClick={() => void handleResumeRun(run)}
                                                            disabled={isStreaming}
                                                            className="mt-3 rounded-full border border-orange-400/30 px-3 py-1 text-xs text-orange-200 transition hover:bg-orange-500/10 disabled:opacity-50"
                                                        >
                                                            Resume
                                                        </button>
                                                    )}
                                                </div>
                                            )
                                        })}
                                    </div>
                                )}
                            </Panel>

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
