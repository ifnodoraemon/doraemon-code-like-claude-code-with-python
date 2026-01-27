import { useState, useEffect, useRef } from 'react'
import { Send, Terminal, Settings, Plus, Loader2 } from 'lucide-react'

// Simple ID generator
const generateId = () => Math.random().toString(36).substr(2, 9)

interface Message {
    id: string
    role: 'user' | 'assistant' | 'system' | 'tool'
    content?: string
    tool_calls?: any[]
    tool_call_id?: string
    name?: string
    timestamp: number
}

interface Session {
    id: string
    name: string
    message_count: number
    updated_at: number
}

function App() {
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState<Message[]>([])
    const [isStreaming, setIsStreaming] = useState(false)
    const [sessions, setSessions] = useState<Session[]>([])
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    useEffect(() => {
        fetchSessions()
    }, [])

    const fetchSessions = async () => {
        try {
            const res = await fetch('/api/sessions')
            if (res.ok) {
                const data = await res.json()
                setSessions(data)
            }
        } catch (err) {
            console.error('Failed to fetch sessions', err)
        }
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!input.trim() || isStreaming) return

        const userMsg: Message = {
            id: generateId(),
            role: 'user',
            content: input,
            timestamp: Date.now()
        }

        setMessages(prev => [...prev, userMsg])
        setInput('')
        setIsStreaming(true)

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: input,
                    session_id: currentSessionId,
                    project: 'default'
                })
            })

            if (!response.ok) throw new Error('Network response was not ok')

            const reader = response.body?.getReader()
            const decoder = new TextDecoder()

            if (!reader) return

            let currentAssistantMsg: Message = {
                id: generateId(),
                role: 'assistant',
                content: '',
                timestamp: Date.now()
            }

            setMessages(prev => [...prev, currentAssistantMsg])

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value)
                const lines = chunk.split('\n\n')

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6)
                        if (dataStr === '[DONE]') break

                        try {
                            const data = JSON.parse(dataStr)

                            setMessages(prev => {
                                const newMessages = [...prev]
                                const lastMsg = newMessages[newMessages.length - 1]

                                if (lastMsg.role === 'assistant') {
                                    if (data.content) {
                                        lastMsg.content = (lastMsg.content || '') + data.content
                                    }
                                    if (data.tool_calls) {
                                        // TODO: Handle tool calls visualization
                                        lastMsg.tool_calls = data.tool_calls
                                    }
                                }
                                return newMessages
                            })
                        } catch (e) {
                            console.error('Error parsing SSE', e)
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Chat error:', error)
            setMessages(prev => [...prev, {
                id: generateId(),
                role: 'system',
                content: `Error: ${error}`,
                timestamp: Date.now()
            }])
        } finally {
            setIsStreaming(false)
        }
    }

    return (
        <div className="flex h-screen bg-slate-900 text-slate-100 font-sans">
            {/* Sidebar */}
            <div className="w-64 border-r border-slate-700 flex flex-col bg-slate-950">
                <div className="p-4 border-b border-slate-800 flex items-center gap-2">
                    <Terminal className="text-blue-400" />
                    <h1 className="font-bold text-lg">Doraemon Code</h1>
                </div>

                <div className="p-2">
                    <button
                        onClick={() => {
                            setMessages([])
                            setCurrentSessionId(null)
                        }}
                        className="w-full flex items-center gap-2 p-2 rounded hover:bg-slate-800 transition text-sm text-slate-300"
                    >
                        <Plus size={16} /> New Chat
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-2">
                    <div className="text-xs font-semibold text-slate-500 mb-2 px-2 uppercase tracking-wider">Recent</div>
                    {sessions.map(s => (
                        <button
                            key={s.id}
                            onClick={() => setCurrentSessionId(s.id)}
                            className={`w-full text-left p-2 rounded text-sm truncate mb-1 ${currentSessionId === s.id ? 'bg-slate-800 text-white' : 'text-slate-400 hover:bg-slate-900'
                                }`}
                        >
                            {s.name || 'Untitled Session'}
                        </button>
                    ))}
                </div>

                <div className="p-4 border-t border-slate-800">
                    <button className="flex items-center gap-2 text-sm text-slate-400 hover:text-white">
                        <Settings size={16} /> Settings
                    </button>
                </div>
            </div>

            {/* Main Chat */}
            <div className="flex-1 flex flex-col">
                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-slate-500">
                            <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mb-4">
                                <Terminal size={32} className="text-blue-500" />
                            </div>
                            <p className="text-lg font-medium text-slate-300">How can I help you code today?</p>
                        </div>
                    )}

                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'justify-end' : ''
                            }`}>
                            {msg.role !== 'user' && (
                                <div className="w-8 h-8 rounded bg-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
                                    <Terminal size={16} className="text-white" />
                                </div>
                            )}

                            <div className={`rounded-2xl px-5 py-3 max-w-[80%] ${msg.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-slate-100'
                                }`}>
                                <div className="whitespace-pre-wrap">{msg.content}</div>
                                {msg.tool_calls && (
                                    <div className="mt-2 p-2 bg-slate-900 rounded text-xs font-mono text-cyan-400 border border-slate-700">
                                        Using tools: {msg.tool_calls.map(tc => tc.function.name).join(', ')}...
                                    </div>
                                )}
                            </div>

                            {msg.role === 'user' && (
                                <div className="w-8 h-8 rounded bg-slate-700 flex items-center justify-center flex-shrink-0 mt-1">
                                    <div className="text-xs font-bold">You</div>
                                </div>
                            )}
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-slate-800 bg-slate-900">
                    <div className="max-w-4xl mx-auto relative">
                        <form onSubmit={handleSubmit} className="relative">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask Doraemon Code anything..."
                                className="w-full bg-slate-800 border-none rounded-xl py-4 pl-4 pr-12 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                disabled={isStreaming}
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || isStreaming}
                                className="absolute right-2 top-2 p-2 bg-blue-600 text-white rounded-lg disabled:opacity-50 hover:bg-blue-500 transition"
                            >
                                {isStreaming ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                            </button>
                        </form>
                        <div className="text-center mt-2 text-xs text-slate-500">
                            Doraemon Code can make mistakes. Please review generated code.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default App
