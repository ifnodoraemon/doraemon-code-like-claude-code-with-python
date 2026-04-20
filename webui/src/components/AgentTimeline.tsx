import { Bot, FileText, Terminal, Wrench, Loader2, CheckCircle2, XCircle, Brain } from 'lucide-react'
import type { ReactNode } from 'react'

export interface TimelineEntry {
    id: string
    type: 'thinking' | 'tool_call' | 'tool_result' | 'response' | 'error'
    label: string
    detail?: string
    status: 'running' | 'completed' | 'failed'
    duration_ms?: number
}

const iconMap: Record<string, { icon: ReactNode; color: string }> = {
    thinking: { icon: <Brain size={12} />, color: 'text-violet-300' },
    tool_call: { icon: <Wrench size={12} />, color: 'text-cyan-300' },
    tool_result: { icon: <FileText size={12} />, color: 'text-slate-300' },
    response: { icon: <Bot size={12} />, color: 'text-emerald-300' },
    error: { icon: <XCircle size={12} />, color: 'text-red-300' },
}

export default function AgentTimeline({ entries }: { entries: TimelineEntry[] }) {
    if (entries.length === 0) return null

    return (
        <div className="space-y-1">
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
                <Terminal size={10} /> Agent Activity
            </div>
            {entries.map((entry) => {
                const cfg = iconMap[entry.type] || iconMap.tool_call
                return (
                    <div
                        key={entry.id}
                        className="flex items-center gap-2 rounded-lg border border-white/[0.05] bg-black/20 px-2 py-1.5 text-xs"
                    >
                        <span className={cfg.color}>{cfg.icon}</span>
                        <span className="min-w-0 flex-1 truncate text-slate-200">{entry.label}</span>
                        {entry.status === 'running' && (
                            <Loader2 size={10} className="animate-spin text-cyan-400" />
                        )}
                        {entry.status === 'completed' && (
                            <CheckCircle2 size={10} className="text-emerald-400" />
                        )}
                        {entry.status === 'failed' && <XCircle size={10} className="text-red-400" />}
                        {entry.duration_ms != null && (
                            <span className="text-[10px] text-slate-500">{entry.duration_ms}ms</span>
                        )}
                    </div>
                )
            })}
        </div>
    )
}
