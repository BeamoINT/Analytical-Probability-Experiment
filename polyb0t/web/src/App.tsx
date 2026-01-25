import { useState, useEffect, useCallback } from 'react'
import Dashboard from './components/Dashboard'

interface DashboardData {
  timestamp: string
  system: {
    cpu_percent: number
    memory_percent: number
    disk_percent: number
    uptime_hours: number
    python_version: string
  }
  trading: {
    is_live: boolean
    placing_orders: boolean
    balance_usd: number
    available_balance: number
    position_count: number
    total_exposure: number
    trades_today: number
    signals_today: number
  }
  ai: {
    is_ready: boolean
    model_version: string | null
    training_examples: number
    labeled_examples: number
    last_training: string | null
    next_training: string | null
  }
  moe: {
    total_experts: number
    active_experts: number
    suspended_experts: number
    probation_experts: number
    deprecated_experts: number
    untrained_experts: number
    state_counts: Record<string, number>
    top_experts: Array<{
      expert_id: string
      domain: string
      state: string
      profit_pct: number
      trades: number
      win_rate: number
    }>
    gating_accuracy: number
    training_cycles: number
  }
  meta_controller: {
    total_mixtures: number
    resolved_mixtures: number
    total_profit_pct: number
    win_rate: number
    top_mixtures: Array<[string, number]>
    synergy_insights: Array<{
      pair: string
      trades: number
      synergy: number
    }>
  }
  arbitrage: {
    is_enabled: boolean
    total_trades: number
    win_rate: number
    total_profit: number
  }
  recent_activity: Array<{
    type: string
    timestamp: string
    market_id: string
    side: string
    size_usd: number
    price: number
  }>
  errors: Array<{ message: string }>
}

function App() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [darkMode, setDarkMode] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch('/api/dashboard')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
      setLastUpdate(new Date())
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch')
    } finally {
      setLoading(false)
    }
  }, [])

  // Initial fetch and WebSocket setup
  useEffect(() => {
    fetchData()

    // Try WebSocket for real-time updates
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`)

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'status_update' && msg.data) {
          setData(msg.data)
          setLastUpdate(new Date())
        }
      } catch (e) {
        console.error('WebSocket parse error:', e)
      }
    }

    ws.onerror = () => {
      console.log('WebSocket error, falling back to polling')
    }

    ws.onclose = () => {
      console.log('WebSocket closed, using polling')
    }

    // Fallback polling every 10 seconds
    const interval = setInterval(fetchData, 10000)

    return () => {
      ws.close()
      clearInterval(interval)
    }
  }, [fetchData])

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#0a0a0a',
        color: '#00ff88',
        fontSize: '1.5rem',
      }}>
        Loading PolyB0T Dashboard...
      </div>
    )
  }

  return (
    <Dashboard
      data={data}
      error={error}
      lastUpdate={lastUpdate}
      onRefresh={fetchData}
      darkMode={darkMode}
      onToggleTheme={() => setDarkMode(!darkMode)}
    />
  )
}

export default App
