import { useState, useEffect } from 'react'
import { DollarSign } from 'lucide-react'

interface CostData {
    total_tokens: number
    total_cost_usd: number
    sessions_count: number
}

export default function CostIndicator({ sessionId }: { sessionId: string | null }) {
    const [cost, setCost] = useState<CostData | null>(null)

    useEffect(() => {
        if (!sessionId) return
        fetch(`/api/sessions/${sessionId}`)
            .then(r => r.ok ? r.json() : null)
            .then(data => {
                if (!data) return
                const tokens = data.metadata?.total_tokens || 0
                const estimatedCost = tokens * 0.00001
                setCost({ total_tokens: tokens, total_cost_usd: estimatedCost, sessions_count: 1 })
            })
            .catch(() => {})
    }, [sessionId])

    if (!cost || cost.total_tokens === 0) return null

    return (
        <div className="flex items-center gap-2 rounded-lg border border-white/[0.07] bg-black/20 px-2 py-1.5 text-xs">
            <DollarSign size={10} className="text-amber-300" />
            <span className="text-slate-400">{(cost.total_cost_usd).toFixed(4)} USD</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">{cost.total_tokens.toLocaleString()} tokens</span>
        </div>
    )
}
