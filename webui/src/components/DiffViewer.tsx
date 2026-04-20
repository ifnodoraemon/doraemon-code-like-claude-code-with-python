import { useState, useEffect } from 'react'
import { GitCompare, RotateCcw, FileText, Trash2, Plus, ChevronDown, ChevronRight } from 'lucide-react'

interface FileChange {
    path: string
    before: { exists: boolean; content: string | null; size: number | null; mtime: number | null }
    after: { exists: boolean; content: string | null; size: number | null; mtime: number | null }
}

interface CheckpointDiff {
    id: string
    created_at: string | null
    prompt: string | null
    description: string
    files: FileChange[]
}

interface Props {
    sessionId: string | null
    isStreaming: boolean
    onUndo: () => void
}

function FileChangeRow({ change }: { change: FileChange }) {
    const [expanded, setExpanded] = useState(false)
    const wasCreated = !change.before.exists && change.after.exists
    const wasDeleted = change.before.exists && !change.after.exists
    const wasModified = change.before.exists && change.after.exists
    const name = change.path.split('/').pop() || change.path

    return (
        <div className="rounded-lg border border-white/[0.07] bg-black/20">
            <button
                type="button"
                onClick={() => setExpanded(!expanded)}
                className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs hover:bg-white/[0.03]"
            >
                {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                {wasCreated && <Plus size={10} className="text-emerald-400" />}
                {wasDeleted && <Trash2 size={10} className="text-red-400" />}
                {wasModified && <FileText size={10} className="text-amber-400" />}
                <span className="flex-1 truncate font-mono text-slate-200" title={change.path}>{name}</span>
                <span className="text-[10px] text-slate-500">{wasCreated ? 'new' : wasDeleted ? 'del' : 'mod'}</span>
            </button>
            {expanded && (change.before.content !== null || change.after.content !== null) && (
                <div className="border-t border-white/[0.05] px-2 py-1">
                    <pre className="max-h-40 overflow-auto text-[10px] text-slate-400 whitespace-pre-wrap">
                        {change.after.content || '(deleted)'}
                    </pre>
                </div>
            )}
        </div>
    )
}

export default function DiffViewer({ sessionId, isStreaming, onUndo }: Props) {
    const [checkpoints, setCheckpoints] = useState<CheckpointDiff[]>([])
    const [undoInProgress, setUndoInProgress] = useState(false)

    useEffect(() => {
        if (!sessionId) { setCheckpoints([]); return }
        fetch(`/api/sessions/${sessionId}/diff?include_content=false`)
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data?.checkpoints) setCheckpoints(data.checkpoints) })
            .catch(() => {})
    }, [sessionId, isStreaming])

    const totalFiles = checkpoints.reduce((n, cp) => n + cp.files.length, 0)

    const handleUndo = async () => {
        if (!sessionId || undoInProgress || isStreaming) return
        setUndoInProgress(true)
        try {
            await fetch(`/api/sessions/${sessionId}/undo`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'code' }),
            })
            onUndo()
            setCheckpoints([])
        } catch { /* ignore */ }
        finally { setUndoInProgress(false) }
    }

    if (totalFiles === 0) return null

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-sm font-medium text-white">
                    <GitCompare size={14} className="text-amber-300" />
                    Changes
                    <span className="text-[11px] font-normal text-slate-500">{totalFiles} file(s)</span>
                </div>
                <button
                    type="button"
                    onClick={handleUndo}
                    disabled={undoInProgress || isStreaming}
                    className="flex items-center gap-1 rounded border border-orange-400/30 px-1.5 py-0.5 text-[10px] text-orange-200 transition hover:bg-orange-500/10 disabled:opacity-40"
                >
                    <RotateCcw size={9} /> Undo
                </button>
            </div>
            {checkpoints.map(cp => (
                <div key={cp.id} className="space-y-1">
                    {cp.files.map((f, i) => <FileChangeRow key={`${cp.id}-${i}`} change={f} />)}
                </div>
            ))}
        </div>
    )
}
