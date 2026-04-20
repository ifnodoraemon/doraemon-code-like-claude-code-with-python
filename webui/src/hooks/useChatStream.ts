import { useCallback, useRef, useState } from 'react'
import { getNextSessionIndex } from '../lib/sessionWindow'

type ToolCallSummary = { name?: string; arguments?: Record<string, unknown> }

export interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content?: string
    tool_calls?: ToolCallSummary[]
    timestamp: number
    meta?: string
    session_index?: number
}

export interface TimelineEntry {
    id: string
    type: 'thinking' | 'tool_call' | 'tool_result' | 'response' | 'error'
    label: string
    detail?: string
    status: 'running' | 'completed' | 'failed'
    duration_ms?: number
}

const generateId = () => Math.random().toString(36).slice(2, 11)

function appendToolCalls(
    existing: ToolCallSummary[] | undefined,
    incoming: ToolCallSummary[] | undefined,
): ToolCallSummary[] | undefined {
    if (!incoming || incoming.length === 0) return existing
    return [...(existing || []), ...incoming]
}

export function useChatStream() {
    const [isStreaming, setIsStreaming] = useState(false)
    const activeStreamRequestRef = useRef(0)
    const activeStreamControllerRef = useRef<AbortController | null>(null)

    const cancel = useCallback(() => {
        activeStreamRequestRef.current += 1
        activeStreamControllerRef.current?.abort()
        activeStreamControllerRef.current = null
        setIsStreaming(false)
    }, [])

    const startStream = useCallback(
        async (params: {
            body: Record<string, unknown>
            userMessage: Message
            messages: Message[]
            messageOffset: number
            onMessagesUpdate: (updater: (prev: Message[]) => Message[]) => void
            onSessionId: (id: string) => void
            onTimelineEntry: (entry: TimelineEntry) => void
            onOrchestrationEvent: (data: Record<string, unknown>) => void
            onError: (error: string) => void
            assistantMeta?: string
        }) => {
            const {
                body,
                userMessage,
                messages,
                messageOffset,
                onMessagesUpdate,
                onSessionId,
                onTimelineEntry,
                onOrchestrationEvent,
                onError,
                assistantMeta,
            } = params

            const streamRequestId = activeStreamRequestRef.current + 1
            activeStreamRequestRef.current = streamRequestId
            const abortController = new AbortController()
            activeStreamControllerRef.current = abortController

            const nextIndex = getNextSessionIndex(messages, messageOffset)

            onMessagesUpdate((prev) => [
                ...prev,
                { ...userMessage, session_index: nextIndex },
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
                    session_index: nextIndex + 1,
                }

                onMessagesUpdate((prev) => [...prev, assistantMessage])

                const handleSsePayload = (payload: string) => {
                    if (activeStreamRequestRef.current !== streamRequestId) return true
                    if (payload === '[DONE]') return true

                    try {
                        const data = JSON.parse(payload)
                        if (data.session_id) onSessionId(data.session_id)

                        if (data.type === 'orchestration') {
                            onOrchestrationEvent(data)
                            return false
                        }

                        // Thinking events
                        if (data.type === 'thinking' || data.type === 'thought') {
                            onTimelineEntry({
                                id: generateId(),
                                type: 'thinking',
                                label: data.content || 'Thinking...',
                                status: 'completed',
                                duration_ms: data.duration_ms,
                            })
                            return false
                        }

                        // Tool call events
                        if (data.type === 'tool_call') {
                            onTimelineEntry({
                                id: generateId(),
                                type: 'tool_call',
                                label: data.name || 'tool_call',
                                detail: JSON.stringify(data.args || {}),
                                status: 'running',
                            })
                            return false
                        }

                        if (data.type === 'tool_result') {
                            onTimelineEntry({
                                id: generateId(),
                                type: 'tool_result',
                                label: `Result: ${data.name || 'tool'}`,
                                status: data.error ? 'failed' : 'completed',
                                duration_ms: data.duration_ms,
                            })
                            return false
                        }

                        onMessagesUpdate((prev) => {
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
                                if (!last.content) last.content = `Error: ${data.error}`
                                last.meta = data.error
                            }
                            return next
                        })
                    } catch {
                        // ignore parse errors
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
                        if (payload) handleSsePayload(payload)
                        break
                    }
                }
            } catch (error) {
                if (abortController.signal.aborted || activeStreamRequestRef.current !== streamRequestId) {
                    return
                }
                onError(`Chat error: ${error}`)
            } finally {
                if (activeStreamRequestRef.current === streamRequestId) {
                    activeStreamControllerRef.current = null
                    setIsStreaming(false)
                }
            }
        },
        [],
    )

    return { isStreaming, startStream, cancel }
}
