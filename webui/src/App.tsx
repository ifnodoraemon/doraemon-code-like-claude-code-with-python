import { useEffect, useRef, useState, type FormEvent, type ReactNode, type KeyboardEvent } from 'react'
import {
    Bot,
    Boxes,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Menu,
    Network,
    Plus,
    Search,
    Send,
    Sparkles,
    Square,
    Terminal,
    Wrench,
} from 'lucide-react'

import MessageBubble from './components/MessageBubble'
import AgentTimeline, { type TimelineEntry } from './components/AgentTimeline'
import DiffViewer from './components/DiffViewer'
import CostIndicator from './components/CostIndicator'
import MobileSessionDrawer from './components/MobileSessionDrawer'

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
    const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false)
    const [sessionSearch, setSessionSearch] = useState('')
    const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false)
    const [timeline, setTimeline] = useState<TimelineEntry[]>([])
    const messagesEndRef = useRef<HTMLDivElement>(null)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
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
        setTimeline([])
    }

    const startStreamingRequest = async (
        body: Record<string, unknown>,
        userMessage: Message,
        assistantMeta?: string,
    ) => {
        setUiError(null)
        setTimeline([])
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

                    if (data.type === 'tool_call') {
                        setTimeline((prev) => [
                            ...prev,
                            {
                                id: generateId(),
                                type: 'tool_call' as const,
                                label: data.name || 'tool_call',
                                detail: data.args ? JSON.stringify(data.args) : undefined,
                                status: 'running' as const,
                            },
                        ])
                    }

                    if (data.type === 'tool_result') {
                        setTimeline((prev) => [
                            ...prev,
                            {
                                id: generateId(),
                                type: 'tool_result' as const,
                                label: `Result: ${data.name || 'tool'}`,
                                status: data.error ? ('failed' as const) : ('completed' as const),
                                duration_ms: data.duration_ms,
                            },
                        ])
                    }

                    if (data.type === 'thinking' || data.type === 'thought') {
                        setTimeline((prev) => [
                            ...prev,
                            {
                                id: generateId(),
                                type: 'thinking' as const,
                                label: data.content || 'Thinking...',
                                status: 'completed' as const,
                                duration_ms: data.duration_ms,
                            },
                        ])
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

    const handleTextareaKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            if (!input.trim() || isStreaming) return
            void handleSubmit(event as unknown as FormEvent)
        }
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

        setInput('')
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'
        }
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

    const handleTextareaChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(event.target.value)
        const textarea = event.target
        textarea.style.height = 'auto'
        const computedStyle = window.getComputedStyle(textarea)
        const lineHeight = parseFloat(computedStyle.lineHeight) || 20
        const paddingTop = parseFloat(computedStyle.paddingTop) || 0
        const paddingBottom = parseFloat(computedStyle.paddingBottom) || 0
        const maxRows = 6
        const maxHeight = lineHeight * maxRows + paddingTop + paddingBottom
        textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`
    }

    const examplePrompts = [
        { text: '修复代码中的 Bug', mode: 'turn' as ExecutionMode },
        { text: '添加新功能', mode: 'turn' as ExecutionMode },
        { text: '拆分复杂任务并协作完成', mode: 'orchestrate' as ExecutionMode },
    ]

    return (
        <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
            <div className="mx-auto flex min-h-screen max-w-[1700px] gap-3 px-4 py-3 lg:px-6">
                <aside className="hidden w-60 shrink-0 rounded-2xl border border-white/[0.07] bg-slate-950/80 p-3 shadow-xl shadow-black/30 backdrop-blur xl:flex xl:flex-col">
                    <div className="flex items-center gap-2.5 border-b border-white/[0.07] pb-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300">
                            <Terminal size={18} />
                        </div>
                        <div>
                            <h1 className="text-base font-semibold text-white">Doraemon Code</h1>
                        </div>
                    </div>

                    <button
                        onClick={resetConversation}
                        className="mt-3 flex items-center gap-2 rounded-xl border border-white/[0.07] bg-white/5 px-3 py-2.5 text-sm text-slate-200 transition hover:bg-white/10"
                    >
                        <Plus size={15} /> 新建会话
                    </button>

                    <div className="mt-4 flex-1 overflow-y-auto">
                        <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-500">会话</div>
                        <div className="mb-2">
                            <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5">
                                <Search size={12} className="text-slate-500" />
                                <input
                                    type="text"
                                    value={sessionSearch}
                                    onChange={e => setSessionSearch(e.target.value)}
                                    placeholder="搜索..."
                                    className="flex-1 bg-transparent text-xs text-slate-100 outline-none placeholder:text-slate-500"
                                />
                            </div>
                        </div>
                        <div className="space-y-1">
                            {sessions
                                .filter(s => !sessionSearch || (s.name || 'Untitled Session').toLowerCase().includes(sessionSearch.toLowerCase()))
                                .map((session) => (
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
                                    className={`w-full rounded-xl border px-3 py-2 text-left text-sm transition ${
                                        currentSessionId === session.id
                                            ? 'border-cyan-400/30 bg-cyan-500/10 text-white'
                                            : 'border-transparent bg-transparent text-slate-400 hover:bg-white/5'
                                    }`}
                                >
                                    <div className="truncate font-medium">{session.name || '未命名会话'}</div>
                                    <div className="mt-0.5 text-xs text-slate-500">
                                        {session.message_count} 条消息
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mt-4 rounded-xl border border-white/[0.07] bg-white/[0.03] p-3">
                        <div className="mb-2 flex items-center gap-2 text-xs font-medium text-white">
                            <Sparkles size={14} className="text-amber-300" />
                            Execution
                        </div>

                        <div className="grid grid-cols-2 gap-1.5">
                            <button
                                onClick={() => setExecutionMode('turn')}
                                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
                                    executionMode === 'turn'
                                        ? 'bg-cyan-500 text-slate-950'
                                        : 'bg-slate-800/70 text-slate-400 hover:bg-slate-700/70'
                                }`}
                            >
                                直接执行
                            </button>
                            <button
                                onClick={() => setExecutionMode('orchestrate')}
                                className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
                                    executionMode === 'orchestrate'
                                        ? 'bg-orange-400 text-slate-950'
                                        : 'bg-slate-800/70 text-slate-400 hover:bg-slate-700/70'
                                }`}
                            >
                                拆分协作
                            </button>
                        </div>

                        {executionMode === 'orchestrate' && (
                            <div className="mt-3">
                                <div className="flex items-center justify-between text-[11px] text-slate-500">
                                    <span>Workers</span>
                                    <span className="font-medium text-slate-300">{maxWorkers}</span>
                                </div>
                                <input
                                    type="range"
                                    min={1}
                                    max={4}
                                    value={maxWorkers}
                                    onChange={(event) => setMaxWorkers(Number(event.target.value))}
                                    className="mt-1 w-full accent-orange-400"
                                />
                            </div>
                        )}
                    </div>
                </aside>

                <main className="flex min-w-0 flex-1 flex-col gap-3">
                    <section className="flex items-center gap-4 rounded-2xl border border-white/[0.07] bg-slate-950/70 px-4 py-2.5 backdrop-blur">
                        <button
                            type="button"
                            onClick={() => setMobileDrawerOpen(true)}
                            className="xl:hidden text-slate-400 hover:text-white"
                        >
                            <Menu size={18} />
                        </button>
                        <div className="flex items-center gap-2 text-sm text-slate-300">
                            <span className="font-mono text-xs text-slate-500">{currentProjectDir || '...'}</span>
                        </div>
                        <div className="ml-auto flex items-center gap-2">
                            <CostIndicator sessionId={currentSessionId} />
                            <SummaryCard
                                icon={<Bot size={13} />}
                                label="Mode"
                                value={executionMode === 'orchestrate' ? 'Coordinated' : 'Direct'}
                            />
                            <SummaryCard
                                icon={<Boxes size={13} />}
                                label="Tasks"
                                value={String(countTasks(taskGraph))}
                            />
                            <SummaryCard
                                icon={<Network size={13} />}
                                label="Workers"
                                value={String(Object.keys(workerAssignments).length)}
                            />
                        </div>
                    </section>

                    <div
                        className={`grid min-h-0 flex-1 gap-3 ${
                            isRightSidebarCollapsed ? 'xl:grid-cols-[minmax(0,1fr)_auto]' : 'xl:grid-cols-[minmax(0,1.4fr)_380px]'
                        }`}
                    >
                        <section className="flex min-h-[640px] flex-col rounded-2xl border border-white/[0.07] bg-slate-950/70 shadow-xl shadow-black/20 backdrop-blur">
                            <div className="border-b border-white/[0.07] px-4 py-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="text-sm font-medium text-white">Conversation</div>
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
                                                className="rounded-lg border border-cyan-400/30 px-2.5 py-1 text-xs text-cyan-200 transition hover:bg-cyan-500/10"
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
                                                className="rounded-lg border border-white/[0.07] px-2.5 py-1 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
                                                disabled={isStreaming}
                                            >
                                                View Selected Run
                                            </button>
                                        )}
                                        <div className="rounded-lg border border-white/[0.07] px-2.5 py-1 text-xs text-slate-400">
                                            {runViewMode === 'run' && selectedMessageRun
                                                ? `run ${selectedMessageRun.run_id || selectedMessageRun.root_task_id || ''}`
                                                : executionMode === 'orchestrate'
                                                  ? `orchestrate x${maxWorkers}`
                                                  : 'single turn'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex-1 overflow-y-auto px-4 py-4">
                                {uiError && (
                                    <div className="mx-auto mb-3 max-w-3xl rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2.5 text-sm text-red-100">
                                        {uiError}
                                    </div>
                                )}
                                {visibleMessages.length === 0 ? (
                                    <div className="flex h-full flex-col items-center justify-center text-center">
                                        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-cyan-500/10 text-cyan-300">
                                            <Terminal size={28} />
                                        </div>
                                        <h3 className="mt-4 text-lg font-semibold text-white">开始使用 Doraemon Code</h3>
                                        <p className="mt-2 max-w-md text-sm text-slate-400">
                                            直接执行单轮对话，或拆分协作来分解目标为协调的 worker 任务。
                                        </p>
                                        <div className="mt-6 grid gap-3 sm:grid-cols-3">
                                            {examplePrompts.map((example) => (
                                                <button
                                                    key={example.text}
                                                    type="button"
                                                    onClick={() => {
                                                        setExecutionMode(example.mode)
                                                        setInput(example.text)
                                                        if (textareaRef.current) {
                                                            textareaRef.current.focus()
                                                        }
                                                    }}
                                                    className="group rounded-xl border border-white/[0.07] bg-slate-900/60 px-4 py-3 text-left transition hover:border-cyan-400/30 hover:bg-slate-800/60"
                                                >
                                                    <div className="text-sm font-medium text-white group-hover:text-cyan-200">
                                                        {example.text}
                                                    </div>
                                                    <div className="mt-1 text-[11px] text-slate-500">
                                                        {example.mode === 'orchestrate' ? '拆分协作' : '直接执行'}
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {runViewMode === 'session' && hasMoreMessages && currentSessionId && (
                                            <div className="mx-auto max-w-3xl">
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
                                                    className="rounded-lg border border-white/[0.07] px-2.5 py-1 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
                                                    disabled={isStreaming}
                                                >
                                                    Load Older Messages ({messageOffset}/{messageCount})
                                                </button>
                                            </div>
                                        )}
                                        {visibleMessages.map((message) => (
                                            <MessageBubble key={message.id} message={message} />
                                        ))}
                                        <div ref={messagesEndRef} />
                                    </div>
                                )}
                            </div>

                            <div className="border-t border-white/[0.07] px-4 py-3">
                                <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
                                    <div className="flex gap-2 rounded-xl border border-white/[0.07] bg-black/20 p-2">
                                        <textarea
                                            ref={textareaRef}
                                            value={input}
                                            onChange={handleTextareaChange}
                                            onKeyDown={handleTextareaKeyDown}
                                            placeholder={
                                                executionMode === 'orchestrate'
                                                    ? '描述要拆分协作的目标...'
                                                    : '描述你的需求...'
                                            }
                                            rows={1}
                                            className="min-w-0 flex-1 resize-none rounded-lg bg-transparent px-3 py-2.5 text-sm leading-relaxed text-slate-100 outline-none transition placeholder:text-slate-500 focus:ring-1 focus:ring-cyan-400/30"
                                            disabled={isStreaming}
                                        />
                                        {isStreaming ? (
                                            <button
                                                type="button"
                                                onClick={cancelActiveStream}
                                                className="flex h-10 w-10 items-center justify-center rounded-lg bg-red-600 text-white transition hover:bg-red-500"
                                            >
                                                <Square size={16} fill="currentColor" />
                                            </button>
                                        ) : (
                                            <button
                                                type="submit"
                                                disabled={!input.trim()}
                                                className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500 text-white transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-40"
                                            >
                                                <Send size={18} />
                                            </button>
                                        )}
                                    </div>
                                </form>
                            </div>
                        </section>

                        <aside
                            className={`${
                                isRightSidebarCollapsed
                                    ? 'flex min-h-[640px] w-12 shrink-0 flex-col items-center rounded-2xl border border-white/[0.07] bg-slate-950/70 px-1.5 py-3 shadow-xl shadow-black/20 backdrop-blur'
                                    : 'grid min-h-[640px] gap-3 lg:grid-cols-2 xl:grid-cols-1'
                            }`}
                        >
                            <button
                                type="button"
                                onClick={() => setIsRightSidebarCollapsed((collapsed) => !collapsed)}
                                className={`flex items-center gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5 text-xs text-slate-300 transition hover:bg-white/5 ${
                                    isRightSidebarCollapsed ? 'justify-center px-0 py-2' : 'justify-center'
                                }`}
                            >
                                {isRightSidebarCollapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
                                {!isRightSidebarCollapsed && <span>Collapse</span>}
                            </button>
                            {isRightSidebarCollapsed ? (
                                <div className="mt-3 flex flex-1 flex-col items-center gap-3 text-slate-500">
                                    <Terminal size={14} />
                                    <Network size={14} />
                                    <Boxes size={14} />
                                    <Sparkles size={14} />
                                    <Wrench size={14} />
                                </div>
                            ) : (
                                <>
                            <Panel
                                icon={<Terminal size={14} className="text-cyan-300" />}
                                title="Runs"
                                subtitle={`${orchestrationRuns.length} in session`}
                            >
                                {orchestrationRuns.length === 0 ? (
                                    <EmptyState text="No runs yet. Send a prompt to start." />
                                ) : (
                                    <div className="space-y-1.5">
                                        {[...orchestrationRuns].reverse().map((run) => {
                                            const isSelected = run.run_id === (selectedRun?.run_id || null)
                                            return (
                                                <div
                                                    key={run.run_id || `${run.root_task_id}-${run.summary}`}
                                                    className={`rounded-lg border p-2 ${
                                                        isSelected
                                                            ? 'border-cyan-400/30 bg-cyan-500/10'
                                                            : 'border-white/[0.07] bg-black/20'
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
                                                        <div className="flex items-center justify-between gap-2">
                                                            <span className="truncate text-xs font-medium text-white">
                                                                {run.goal || run.summary || run.root_task_id || 'Run'}
                                                            </span>
                                                            <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${run.success ? 'bg-emerald-500/15 text-emerald-200' : 'bg-red-500/15 text-red-200'}`}>
                                                                {run.success ? 'done' : 'blocked'}
                                                            </span>
                                                        </div>
                                                    </button>
                                                    {!run.success && run.run_id && currentSessionId && (
                                                        <button
                                                            onClick={() => void handleResumeRun(run)}
                                                            disabled={isStreaming}
                                                            className="mt-1.5 rounded-md border border-orange-400/30 px-2 py-0.5 text-[11px] text-orange-200 transition hover:bg-orange-500/10 disabled:opacity-50"
                                                        >
                                                            Resume
                                                        </button>
                                                    )}
                                                </div>
                                            )
                                        })}
                                    </div>
                                )}
                                <AgentTimeline entries={timeline} />
                            </Panel>

                            <Panel
                                icon={<Network size={14} className="text-orange-300" />}
                                title="Tasks"
                                subtitle={`${countTasks(taskGraph)} total · ${readyTasks.length} ready`}
                            >
                                {taskGraph.length === 0 && readyTasks.length === 0 ? (
                                    <EmptyState text="No tasks yet. Run orchestration to plan." />
                                ) : (
                                    <div className="space-y-1.5">
                                        {readyTasks.length > 0 && (
                                            <div className="mb-1">
                                                <div className="mb-1 flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-emerald-400/70">
                                                    <Sparkles size={10} /> Ready
                                                </div>
                                                {readyTasks.map((task) => (
                                                    <div key={task.id} className="rounded-lg border border-emerald-400/15 bg-emerald-500/5 px-2 py-1.5">
                                                        <div className="text-xs font-medium text-white">{task.title}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        {taskGraph.length > 0 && renderTaskTree(taskGraph)}
                                    </div>
                                )}

                                {Object.entries(workerAssignments).length > 0 && (
                                    <div className="mt-2 border-t border-white/[0.05] pt-2">
                                        <div className="mb-1 flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-slate-500">
                                            <Boxes size={10} /> Workers
                                        </div>
                                        <div className="space-y-1">
                                            {Object.entries(workerAssignments).map(([taskId, assignment]) => (
                                                <div key={taskId} className="rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5">
                                                    <div className="flex items-center justify-between gap-1">
                                                        <span className="text-xs font-medium text-white">{assignment.role}</span>
                                                        <span className="text-[10px] text-slate-500">{taskId}</span>
                                                    </div>
                                                    <div className="mt-1 flex flex-wrap gap-0.5">
                                                        {assignment.allowed_tool_names.map((toolName) => (
                                                            <span
                                                                key={`${taskId}-${toolName}`}
                                                                className="rounded border border-white/[0.07] px-1 py-0.5 text-[10px] text-slate-400"
                                                            >
                                                                {toolName}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </Panel>

                            <CollapsiblePanel
                                icon={<Wrench size={14} className="text-violet-300" />}
                                title="Tools"
                                subtitle={`${tools.length} available`}
                                defaultCollapsed={true}
                            >
                                {tools.length === 0 ? (
                                    <EmptyState text="No tool catalog loaded." />
                                ) : (
                                    <div className="space-y-1">
                                        {tools.slice(0, 12).map((tool) => (
                                            <div key={tool.name} className="flex items-center justify-between gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5">
                                                <div className="min-w-0">
                                                    <span className="text-xs font-medium text-white">{tool.name}</span>
                                                </div>
                                                <span className="shrink-0 rounded border border-white/[0.07] px-1 py-0.5 text-[10px] text-slate-400">
                                                    {tool.policy.capability_group || 'general'}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </CollapsiblePanel>

                            <DiffViewer
                                sessionId={currentSessionId}
                                isStreaming={isStreaming}
                                onUndo={() => {
                                    if (currentSessionId) {
                                        void fetchSessionDetail(currentSessionId, {
                                            messageLimit: DEFAULT_MESSAGE_WINDOW,
                                            mode: 'replace',
                                            preserveRunSelection: false,
                                        })
                                    }
                                }}
                            />
                                 </>
                             )}
                         </aside>
                     </div>
                 </main>

                <MobileSessionDrawer
                    sessions={sessions}
                    currentSessionId={currentSessionId}
                    onSelect={(id) => {
                        cancelActiveStream()
                        invalidateSessionDetailRequests(id)
                        invalidateRunDetailRequests()
                        setUiError(null)
                        setSessionView()
                        clearConversationWindow({ clearRuns: true })
                        setNeedsSessionHydration(true)
                        setCurrentSessionId(id)
                    }}
                    onNewChat={resetConversation}
                    onClose={() => setMobileDrawerOpen(false)}
                    open={mobileDrawerOpen}
                />
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
        <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-3 py-1.5">
            <span className="text-slate-500">{icon}</span>
            <span className="text-xs text-slate-400">{label}</span>
            <span className="text-xs font-medium text-white">{value}</span>
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
        <section className="rounded-2xl border border-white/[0.07] bg-slate-950/70 p-3 shadow-lg shadow-black/20 backdrop-blur">
            <div className="mb-3 flex items-start justify-between gap-2">
                <div>
                    <div className="flex items-center gap-1.5 text-sm font-medium text-white">
                        {icon}
                        {title}
                    </div>
                    <div className="mt-0.5 text-[11px] text-slate-500">{subtitle}</div>
                </div>
            </div>
            {children}
        </section>
    )
}

function EmptyState({ text }: { text: string }) {
    return <div className="rounded-lg border border-dashed border-white/[0.07] px-2 py-2 text-[11px] text-slate-500">{text}</div>
}

function CollapsiblePanel({
    icon,
    title,
    subtitle,
    defaultCollapsed = false,
    children,
}: {
    icon: React.ReactNode
    title: string
    subtitle: string
    defaultCollapsed?: boolean
    children: ReactNode
}) {
    const [collapsed, setCollapsed] = useState(defaultCollapsed)
    return (
        <section className="rounded-2xl border border-white/[0.07] bg-slate-950/70 p-3 shadow-lg shadow-black/20 backdrop-blur">
            <button
                type="button"
                onClick={() => setCollapsed((c) => !c)}
                className="flex w-full items-center justify-between gap-2"
            >
                <div className="flex items-center gap-1.5 text-sm font-medium text-white">
                    {icon}
                    {title}
                    <span className="text-[11px] font-normal text-slate-500">{subtitle}</span>
                </div>
                <ChevronDown size={14} className={`text-slate-500 transition-transform ${collapsed ? '' : 'rotate-180'}`} />
            </button>
            {!collapsed && <div className="mt-2">{children}</div>}
        </section>
    )
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
                style={{ marginLeft: `${level * 10}px` }}
                className="rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5"
            >
                <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                        <div className="truncate text-xs font-medium text-white">{node.title}</div>
                    </div>
                    <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${badgeColor}`}>
                        {node.status}
                    </span>
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