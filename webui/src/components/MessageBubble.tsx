import { useState } from 'react'
import { Bot } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
// @ts-expect-error — no types for react-syntax-highlighter
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
// @ts-expect-error — no types for prism styles
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

type ToolCallSummary = { name?: string; arguments?: Record<string, unknown> }

export interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content?: string
    tool_calls?: ToolCallSummary[]
    timestamp: number
    meta?: string
}

function ToolCallCard({ toolCall }: { toolCall: ToolCallSummary }) {
    const [expanded, setExpanded] = useState(false)
    const args = toolCall.arguments || {}

    return (
        <div className="rounded-lg border border-cyan-400/20 bg-black/30">
            <button
                type="button"
                onClick={() => setExpanded(!expanded)}
                className="flex w-full items-center justify-between px-3 py-2 text-left text-xs text-cyan-200 hover:bg-cyan-500/5"
            >
                <span className="font-mono font-medium">{toolCall.name || 'tool'}</span>
                <span className="text-slate-500">{expanded ? '▾' : '▸'}</span>
            </button>
            {expanded && Object.keys(args).length > 0 && (
                <pre className="border-t border-cyan-400/10 px-3 py-2 text-[11px] text-slate-300 overflow-x-auto">
                    {JSON.stringify(args, null, 2)}
                </pre>
            )}
        </div>
    )
}

export default function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === 'user'
    const isSystem = message.role === 'system'

    return (
        <article className={`mx-auto flex max-w-3xl gap-3 ${isUser ? 'justify-end' : ''}`}>
            {!isUser && (
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300">
                    <Bot size={14} />
                </div>
            )}

            <div
                className={`max-w-[85%] rounded-2xl border px-4 py-3 ${
                    isUser
                        ? 'border-cyan-400/25 bg-cyan-500/90 text-slate-950'
                        : isSystem
                          ? 'border-red-400/20 bg-red-500/10 text-red-100'
                          : 'border-white/[0.07] bg-slate-900/80 text-slate-100'
                }`}
            >
                {message.meta && (
                    <div className="mb-1.5 text-[11px] uppercase tracking-wider text-slate-400">
                        {message.meta}
                    </div>
                )}

                {isUser || isSystem ? (
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
                ) : (
                    <div className="prose prose-invert prose-sm max-w-none prose-pre:p-0 prose-pre:bg-transparent">
                        <ReactMarkdown
                            components={{
                                code({ className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '')
                                    const inline = !match
                                    if (inline) {
                                        return (
                                            <code className="rounded bg-black/30 px-1 py-0.5 text-cyan-200" {...props}>
                                                {children}
                                            </code>
                                        )
                                    }
                                    return (
                                        <SyntaxHighlighter
                                            style={oneDark}
                                            language={match[1]}
                                            PreTag="div"
                                            className="!rounded-lg !border !border-white/[0.07] !text-xs"
                                        >
                                            {String(children).replace(/\n$/, '')}
                                        </SyntaxHighlighter>
                                    )
                                },
                            }}
                        >
                            {message.content || ''}
                        </ReactMarkdown>
                    </div>
                )}

                {message.tool_calls && message.tool_calls.length > 0 && (
                    <div className="mt-3 space-y-1.5">
                        {message.tool_calls.map((tc, i) => (
                            <ToolCallCard key={`${tc.name}-${i}`} toolCall={tc} />
                        ))}
                    </div>
                )}
            </div>

            {isUser && (
                <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-slate-700 text-[11px] font-semibold text-white">
                    You
                </div>
            )}
        </article>
    )
}
