import { useState } from 'react'
import { X, Search, Plus, MessageSquare } from 'lucide-react'

interface Session {
    id: string
    name: string
    message_count: number
    updated_at: number
}

interface Props {
    sessions: Session[]
    currentSessionId: string | null
    onSelect: (id: string) => void
    onNewChat: () => void
    onClose: () => void
    open: boolean
}

export default function MobileSessionDrawer({ sessions, currentSessionId, onSelect, onNewChat, onClose, open }: Props) {
    const [search, setSearch] = useState('')

    if (!open) return null
    const filtered = search
        ? sessions.filter(s => (s.name || 'Untitled').toLowerCase().includes(search.toLowerCase()))
        : sessions

    return (
        <div className="fixed inset-0 z-50 bg-slate-950/95 backdrop-blur sm:hidden">
            <div className="flex h-full flex-col p-4">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-white">Sessions</h2>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">
                        <X size={20} />
                    </button>
                </div>

                <div className="flex gap-2 mb-4">
                    <div className="flex flex-1 items-center gap-2 rounded-xl border border-white/[0.07] bg-black/20 px-3 py-2">
                        <Search size={14} className="text-slate-500" />
                        <input
                            type="text"
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                            placeholder="搜索会话..."
                            className="flex-1 bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-500"
                        />
                    </div>
                    <button
                        onClick={onNewChat}
                        className="flex items-center gap-1 rounded-xl border border-white/[0.07] bg-white/5 px-3 py-2 text-sm text-slate-200"
                    >
                        <Plus size={14} /> 新建
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto space-y-1">
                    {filtered.map(s => (
                        <button
                            key={s.id}
                            onClick={() => { onSelect(s.id); onClose() }}
                            className={`w-full rounded-xl border px-3 py-2.5 text-left transition ${
                                currentSessionId === s.id
                                    ? 'border-cyan-400/30 bg-cyan-500/10 text-white'
                                    : 'border-transparent text-slate-400 hover:bg-white/5'
                            }`}
                        >
                            <div className="flex items-center gap-2">
                                <MessageSquare size={14} className="shrink-0 text-slate-500" />
                                <div className="min-w-0 flex-1">
                                    <div className="truncate text-sm font-medium">{s.name || '未命名会话'}</div>
                                    <div className="text-xs text-slate-500">{s.message_count} 条消息</div>
                                </div>
                            </div>
                        </button>
                    ))}
                    {filtered.length === 0 && (
                        <div className="py-8 text-center text-xs text-slate-500">无匹配会话</div>
                    )}
                </div>
            </div>
        </div>
    )
}
