"""Trade Post-Mortem Analysis and Logging.

This module provides comprehensive logging for every trade outcome,
enabling diagnosis of loss sources and strategy improvement.

Key Metrics Captured:
- Slippage (Execution vs Mid-market)
- Order Book Depth at time of trade
- Rule ID that triggered the entry
- Latency metrics
- Market conditions at entry/exit
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TradePostMortem:
    """Complete post-mortem data for a trade."""
    
    # Trade Identification
    trade_id: str
    intent_id: str
    token_id: str
    market_id: str
    
    # Entry Details
    entry_time: datetime
    entry_side: str  # BUY or SELL
    entry_signal_edge: float  # Edge at time of signal
    entry_mid_price: float  # Mid-price at signal generation
    entry_best_bid: float
    entry_best_ask: float
    entry_spread_bps: int
    entry_bid_depth_usd: float  # Depth at top 5 levels
    entry_ask_depth_usd: float
    entry_rule_id: str  # Which rule/signal triggered
    entry_confidence: float
    
    # Execution Details
    exec_time: datetime | None = None
    exec_price: float = 0.0
    exec_size_usd: float = 0.0
    exec_slippage_bps: int = 0  # Execution vs mid at signal time
    exec_latency_ms: int = 0  # Time from signal to execution
    
    # Exit Details (if closed)
    exit_time: datetime | None = None
    exit_side: str = ""
    exit_price: float = 0.0
    exit_mid_price: float = 0.0
    exit_spread_bps: int = 0
    exit_bid_depth_usd: float = 0.0
    exit_ask_depth_usd: float = 0.0
    exit_reason: str = ""
    exit_slippage_bps: int = 0
    
    # Outcome
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    hold_time_minutes: float = 0.0
    is_winner: bool = False
    
    # Market Conditions
    market_volume_24h: float = 0.0
    market_momentum_1h: float = 0.0
    market_momentum_24h: float = 0.0
    orderbook_imbalance_entry: float = 0.0
    orderbook_imbalance_exit: float = 0.0
    
    # Diagnosis
    loss_cause: str = ""  # Diagnosed cause of loss
    lessons: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "intent_id": self.intent_id,
            "token_id": self.token_id,
            "market_id": self.market_id,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_side": self.entry_side,
            "entry_signal_edge": self.entry_signal_edge,
            "entry_mid_price": self.entry_mid_price,
            "entry_spread_bps": self.entry_spread_bps,
            "entry_bid_depth_usd": self.entry_bid_depth_usd,
            "entry_ask_depth_usd": self.entry_ask_depth_usd,
            "entry_rule_id": self.entry_rule_id,
            "entry_confidence": self.entry_confidence,
            "exec_price": self.exec_price,
            "exec_size_usd": self.exec_size_usd,
            "exec_slippage_bps": self.exec_slippage_bps,
            "exec_latency_ms": self.exec_latency_ms,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "exit_slippage_bps": self.exit_slippage_bps,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "hold_time_minutes": self.hold_time_minutes,
            "is_winner": self.is_winner,
            "loss_cause": self.loss_cause,
            "lessons": self.lessons,
        }


class PostMortemAnalyzer:
    """Analyzes trades and diagnoses loss causes.
    
    Loss Categories:
    1. SLIPPAGE_EXCEEDED_EDGE - Entry slippage ate the edge
    2. THIN_MARKET - Traded in illiquid market
    3. LATENCY_REVERSION - Price moved against us during execution delay
    4. FALSE_EDGE - Model edge was wrong (market was right)
    5. ADVERSE_SELECTION - We were picked off by informed traders
    6. MOMENTUM_REVERSAL - Entered at the top of a move
    7. SPREAD_EXCEEDED_EDGE - Spread was wider than our edge
    8. STOP_LOSS_HIT - Hit stop loss (could be any underlying cause)
    """
    
    def __init__(self, db_path: str = "data/postmortem.db") -> None:
        """Initialize post-mortem analyzer."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_postmortems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    intent_id TEXT,
                    token_id TEXT,
                    market_id TEXT,
                    entry_time TIMESTAMP,
                    entry_side TEXT,
                    entry_signal_edge REAL,
                    entry_mid_price REAL,
                    entry_spread_bps INTEGER,
                    entry_bid_depth_usd REAL,
                    entry_ask_depth_usd REAL,
                    entry_rule_id TEXT,
                    entry_confidence REAL,
                    exec_price REAL,
                    exec_size_usd REAL,
                    exec_slippage_bps INTEGER,
                    exec_latency_ms INTEGER,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    exit_reason TEXT,
                    exit_slippage_bps INTEGER,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    hold_time_minutes REAL,
                    is_winner INTEGER,
                    loss_cause TEXT,
                    lessons TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_postmortem_token
                ON trade_postmortems(token_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_postmortem_loss_cause
                ON trade_postmortems(loss_cause)
            """)
    
    def diagnose_loss(self, pm: TradePostMortem) -> str:
        """Diagnose the cause of a losing trade.
        
        Args:
            pm: Trade post-mortem data.
            
        Returns:
            Loss cause category.
        """
        if pm.is_winner:
            return "WINNER"
        
        lessons = []
        
        # 1. Check if slippage exceeded edge
        entry_slippage_pct = pm.exec_slippage_bps / 10000.0
        edge_pct = abs(pm.entry_signal_edge)
        
        if entry_slippage_pct > edge_pct * 0.5:
            lessons.append(f"Slippage {pm.exec_slippage_bps}bps consumed {entry_slippage_pct/edge_pct*100:.0f}% of edge")
            pm.lessons = lessons
            return "SLIPPAGE_EXCEEDED_EDGE"
        
        # 2. Check for thin market
        total_depth = pm.entry_bid_depth_usd + pm.entry_ask_depth_usd
        if total_depth < 100:  # Less than $100 depth
            lessons.append(f"Thin market: only ${total_depth:.0f} depth")
            pm.lessons = lessons
            return "THIN_MARKET"
        
        # 3. Check if spread exceeded edge
        spread_pct = pm.entry_spread_bps / 10000.0
        if spread_pct > edge_pct:
            lessons.append(f"Spread {pm.entry_spread_bps}bps exceeded edge {edge_pct*10000:.0f}bps")
            pm.lessons = lessons
            return "SPREAD_EXCEEDED_EDGE"
        
        # 4. Check for latency/reversion
        if pm.exec_latency_ms > 5000:  # More than 5 seconds
            lessons.append(f"High latency: {pm.exec_latency_ms}ms execution delay")
            pm.lessons = lessons
            return "LATENCY_REVERSION"
        
        # 5. Check for momentum reversal (entered at top)
        if pm.market_momentum_1h > 0.1 and pm.entry_side == "BUY":
            lessons.append("Bought after 10%+ 1h pump - momentum reversal")
            pm.lessons = lessons
            return "MOMENTUM_REVERSAL"
        
        # 6. Check for adverse selection (price moved against immediately)
        if pm.hold_time_minutes < 10 and pm.pnl_pct < -5:
            lessons.append("Quick loss suggests adverse selection (informed trader)")
            pm.lessons = lessons
            return "ADVERSE_SELECTION"
        
        # 7. Stop loss hit
        if "stop" in (pm.exit_reason or "").lower():
            lessons.append("Stop-loss triggered")
            pm.lessons = lessons
            return "STOP_LOSS_HIT"
        
        # 8. Default: false edge (model was wrong)
        lessons.append("Model edge did not materialize - market was right")
        pm.lessons = lessons
        return "FALSE_EDGE"
    
    def record_postmortem(self, pm: TradePostMortem) -> None:
        """Record a post-mortem to the database.
        
        Args:
            pm: Trade post-mortem data.
        """
        # Diagnose if not already done
        if not pm.loss_cause:
            pm.loss_cause = self.diagnose_loss(pm)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trade_postmortems (
                        trade_id, intent_id, token_id, market_id,
                        entry_time, entry_side, entry_signal_edge, entry_mid_price,
                        entry_spread_bps, entry_bid_depth_usd, entry_ask_depth_usd,
                        entry_rule_id, entry_confidence,
                        exec_price, exec_size_usd, exec_slippage_bps, exec_latency_ms,
                        exit_time, exit_price, exit_reason, exit_slippage_bps,
                        pnl_usd, pnl_pct, hold_time_minutes, is_winner,
                        loss_cause, lessons
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pm.trade_id, pm.intent_id, pm.token_id, pm.market_id,
                    pm.entry_time.isoformat() if pm.entry_time else None,
                    pm.entry_side, pm.entry_signal_edge, pm.entry_mid_price,
                    pm.entry_spread_bps, pm.entry_bid_depth_usd, pm.entry_ask_depth_usd,
                    pm.entry_rule_id, pm.entry_confidence,
                    pm.exec_price, pm.exec_size_usd, pm.exec_slippage_bps, pm.exec_latency_ms,
                    pm.exit_time.isoformat() if pm.exit_time else None,
                    pm.exit_price, pm.exit_reason, pm.exit_slippage_bps,
                    pm.pnl_usd, pm.pnl_pct, pm.hold_time_minutes, 1 if pm.is_winner else 0,
                    pm.loss_cause, ",".join(pm.lessons),
                ))
                
            # Log structured post-mortem
            log_level = logging.INFO if pm.is_winner else logging.WARNING
            logger.log(
                log_level,
                f"TRADE POST-MORTEM: {'✅ WIN' if pm.is_winner else '❌ LOSS'} "
                f"${pm.pnl_usd:+.2f} ({pm.pnl_pct:+.1f}%)",
                extra={
                    "trade_id": pm.trade_id[:12],
                    "token_id": pm.token_id[:12],
                    "entry_edge": pm.entry_signal_edge,
                    "entry_spread_bps": pm.entry_spread_bps,
                    "exec_slippage_bps": pm.exec_slippage_bps,
                    "hold_time_min": pm.hold_time_minutes,
                    "loss_cause": pm.loss_cause,
                    "lessons": pm.lessons,
                },
            )
                
        except Exception as e:
            logger.error(f"Failed to record post-mortem: {e}")
    
    def get_loss_breakdown(self, days: int = 30) -> dict[str, dict[str, float]]:
        """Get breakdown of losses by cause.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            Dict of loss cause -> stats.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT loss_cause, COUNT(*) as count, 
                           SUM(pnl_usd) as total_pnl,
                           AVG(pnl_pct) as avg_pnl_pct
                    FROM trade_postmortems
                    WHERE recorded_at > datetime('now', ?)
                    AND is_winner = 0
                    GROUP BY loss_cause
                    ORDER BY total_pnl ASC
                """, (f'-{days} days',)).fetchall()
                
                return {
                    row[0]: {
                        "count": row[1],
                        "total_pnl_usd": row[2],
                        "avg_pnl_pct": row[3],
                    }
                    for row in rows
                }
                
        except Exception as e:
            logger.error(f"Failed to get loss breakdown: {e}")
            return {}
    
    def get_rule_performance(self, days: int = 30) -> dict[str, dict[str, float]]:
        """Get performance by entry rule.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            Dict of rule_id -> performance stats.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT entry_rule_id, 
                           COUNT(*) as total_trades,
                           SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                           SUM(pnl_usd) as total_pnl,
                           AVG(pnl_pct) as avg_pnl_pct
                    FROM trade_postmortems
                    WHERE recorded_at > datetime('now', ?)
                    GROUP BY entry_rule_id
                    ORDER BY total_pnl DESC
                """, (f'-{days} days',)).fetchall()
                
                return {
                    row[0]: {
                        "total_trades": row[1],
                        "wins": row[2],
                        "win_rate": row[2] / row[1] if row[1] > 0 else 0,
                        "total_pnl_usd": row[3],
                        "avg_pnl_pct": row[4],
                    }
                    for row in rows
                }
                
        except Exception as e:
            logger.error(f"Failed to get rule performance: {e}")
            return {}


# Singleton
_postmortem_analyzer: PostMortemAnalyzer | None = None


def get_postmortem_analyzer() -> PostMortemAnalyzer:
    """Get singleton post-mortem analyzer."""
    global _postmortem_analyzer
    if _postmortem_analyzer is None:
        _postmortem_analyzer = PostMortemAnalyzer()
    return _postmortem_analyzer
