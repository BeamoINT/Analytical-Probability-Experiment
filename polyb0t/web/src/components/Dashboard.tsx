import { useState } from 'react'
import './Dashboard.css'

interface DashboardProps {
  data: any
  error: string | null
  lastUpdate: Date | null
  onRefresh: () => void
  darkMode: boolean
  onToggleTheme: () => void
}

function Dashboard({ data, error, lastUpdate, onRefresh, darkMode }: DashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'experts' | 'meta'>('overview')

  if (error) {
    return (
      <div className="dashboard error-state">
        <div className="error-container">
          <h1>Connection Error</h1>
          <p>{error}</p>
          <button onClick={onRefresh}>Retry</button>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="dashboard loading-state">
        <p>No data available</p>
      </div>
    )
  }

  const { system, trading, ai, moe, meta_controller, arbitrage } = data

  return (
    <div className={`dashboard ${darkMode ? 'dark' : 'light'}`}>
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1 className="logo">PolyB0T</h1>
          <span className="version">v0.2.0</span>
        </div>
        <nav className="tabs">
          <button
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`tab ${activeTab === 'experts' ? 'active' : ''}`}
            onClick={() => setActiveTab('experts')}
          >
            Experts ({moe.total_experts})
          </button>
          <button
            className={`tab ${activeTab === 'meta' ? 'active' : ''}`}
            onClick={() => setActiveTab('meta')}
          >
            Meta-Controller
          </button>
        </nav>
        <div className="header-right">
          <span className="last-update">
            Updated: {lastUpdate?.toLocaleTimeString() || 'Never'}
          </span>
          <button className="refresh-btn" onClick={onRefresh}>
            Refresh
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="main">
        {activeTab === 'overview' && (
          <div className="overview">
            {/* Status Cards */}
            <div className="status-row">
              <StatusCard
                title="Trading Mode"
                value={trading.is_live ? 'LIVE' : trading.placing_orders ? 'DRY RUN' : 'OFF'}
                status={trading.is_live ? 'success' : 'warning'}
              />
              <StatusCard
                title="AI Status"
                value={ai.is_ready ? 'READY' : 'TRAINING'}
                status={ai.is_ready ? 'success' : 'warning'}
              />
              <StatusCard
                title="Balance"
                value={`$${trading.balance_usd?.toFixed(2) || '0.00'}`}
                subtitle={`Available: $${trading.available_balance?.toFixed(2) || '0.00'}`}
              />
              <StatusCard
                title="Positions"
                value={trading.position_count.toString()}
                subtitle={`Exposure: $${trading.total_exposure?.toFixed(2) || '0.00'}`}
              />
            </div>

            {/* System Health */}
            <div className="card">
              <h2>System Health</h2>
              <div className="metrics-grid">
                <Metric label="CPU" value={`${system.cpu_percent?.toFixed(1)}%`} />
                <Metric label="Memory" value={`${system.memory_percent?.toFixed(1)}%`} />
                <Metric label="Disk" value={`${system.disk_percent?.toFixed(1)}%`} />
                <Metric label="Uptime" value={`${system.uptime_hours?.toFixed(1)}h`} />
              </div>
            </div>

            {/* MoE Summary */}
            <div className="card">
              <h2>Mixture of Experts</h2>
              <div className="expert-summary">
                <div className="expert-states">
                  <StateIndicator state="active" count={moe.active_experts} />
                  <StateIndicator state="suspended" count={moe.suspended_experts} />
                  <StateIndicator state="probation" count={moe.probation_experts} />
                  <StateIndicator state="deprecated" count={moe.deprecated_experts} />
                  <StateIndicator state="untrained" count={moe.untrained_experts} />
                </div>
                <div className="gating-info">
                  <span>Gating Accuracy: {(moe.gating_accuracy * 100).toFixed(1)}%</span>
                  <span>Training Cycles: {moe.training_cycles}</span>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="card">
              <h2>Today's Activity</h2>
              <div className="metrics-grid">
                <Metric label="Trades" value={trading.trades_today.toString()} />
                <Metric label="Signals" value={trading.signals_today.toString()} />
                <Metric
                  label="Arbitrage"
                  value={arbitrage.is_enabled ? 'ON' : 'OFF'}
                  status={arbitrage.is_enabled ? 'success' : 'muted'}
                />
                <Metric
                  label="Arb Win Rate"
                  value={`${(arbitrage.win_rate * 100).toFixed(0)}%`}
                />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'experts' && (
          <div className="experts-view">
            <div className="card full-width">
              <h2>All Experts ({moe.total_experts})</h2>
              <div className="experts-grid">
                {moe.top_experts?.map((expert: any) => (
                  <ExpertCard key={expert.expert_id} expert={expert} />
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'meta' && (
          <div className="meta-view">
            <div className="card">
              <h2>Meta-Controller Status</h2>
              <div className="metrics-grid">
                <Metric label="Total Mixtures" value={meta_controller.total_mixtures.toString()} />
                <Metric label="Resolved" value={meta_controller.resolved_mixtures.toString()} />
                <Metric
                  label="Win Rate"
                  value={`${(meta_controller.win_rate * 100).toFixed(1)}%`}
                  status={meta_controller.win_rate > 0.5 ? 'success' : 'warning'}
                />
                <Metric
                  label="Total Profit"
                  value={`${(meta_controller.total_profit_pct * 100).toFixed(1)}%`}
                  status={meta_controller.total_profit_pct > 0 ? 'success' : 'error'}
                />
              </div>
            </div>

            <div className="card">
              <h2>Top Mixtures</h2>
              <div className="mixtures-list">
                {meta_controller.top_mixtures?.map(([mix, count]: [string, number], i: number) => (
                  <div key={i} className="mixture-item">
                    <span className="mixture-name">{mix}</span>
                    <span className="mixture-count">{count} uses</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h2>Expert Synergies</h2>
              <div className="synergies-list">
                {meta_controller.synergy_insights?.slice(0, 10).map((s: any, i: number) => (
                  <div key={i} className="synergy-item">
                    <span className="synergy-pair">{s.pair}</span>
                    <span className="synergy-trades">{s.trades} trades</span>
                    <span className={`synergy-score ${s.synergy > 0 ? 'positive' : 'negative'}`}>
                      {s.synergy.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

// Sub-components

function StatusCard({ title, value, subtitle, status }: {
  title: string
  value: string
  subtitle?: string
  status?: 'success' | 'warning' | 'error'
}) {
  return (
    <div className={`status-card ${status || ''}`}>
      <div className="status-title">{title}</div>
      <div className="status-value">{value}</div>
      {subtitle && <div className="status-subtitle">{subtitle}</div>}
    </div>
  )
}

function Metric({ label, value, status }: {
  label: string
  value: string
  status?: 'success' | 'warning' | 'error' | 'muted'
}) {
  return (
    <div className={`metric ${status || ''}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  )
}

function StateIndicator({ state, count }: { state: string; count: number }) {
  return (
    <div className={`state-indicator ${state}`}>
      <span className="state-count">{count}</span>
      <span className="state-label">{state}</span>
    </div>
  )
}

function ExpertCard({ expert }: { expert: any }) {
  const profitClass = expert.profit_pct > 0 ? 'positive' : expert.profit_pct < 0 ? 'negative' : ''
  
  return (
    <div className={`expert-card ${expert.state}`}>
      <div className="expert-header">
        <span className="expert-domain">{expert.domain}</span>
        <span className={`expert-state-badge ${expert.state}`}>{expert.state}</span>
      </div>
      <div className="expert-metrics">
        <div className={`expert-profit ${profitClass}`}>
          {(expert.profit_pct * 100).toFixed(1)}%
        </div>
        <div className="expert-stats">
          <span>{expert.trades} trades</span>
          <span>{(expert.win_rate * 100).toFixed(0)}% win</span>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
