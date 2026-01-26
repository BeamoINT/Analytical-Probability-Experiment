"""Backtesting Engine for strategy evaluation.

Provides comprehensive backtesting with:
- Historical data replay from market_snapshots
- Realistic execution simulation
- Slippage and orderbook modeling
- Full position and P&L tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from polyb0t.backtest.analytics import PerformanceAnalytics, PerformanceMetrics
from polyb0t.backtest.slippage import SlippageModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    
    # Execution settings
    slippage_enabled: bool = True
    commission_pct: float = 0.001  # 0.1% commission
    
    # Position limits
    max_position_pct: float = 0.15  # 15% max per trade
    max_total_exposure_pct: float = 0.60  # 60% max total exposure
    
    # Data settings
    min_data_points: int = 100  # Minimum snapshots to run
    

@dataclass
class BacktestTrade:
    """Record of a single trade."""
    
    trade_id: str
    token_id: str
    market_id: str
    side: str  # BUY or SELL
    
    entry_time: datetime
    entry_price: float
    size: float
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    profit_usd: float = 0.0
    profit_pct: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    
    # Signal info
    signal_strength: float = 0.0
    expert_id: str = ""
    
    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None
    
    @property
    def is_profitable(self) -> bool:
        return self.profit_pct > 0


@dataclass
class BacktestPosition:
    """Current open position."""
    
    token_id: str
    market_id: str
    side: str
    size: float
    entry_price: float
    entry_time: datetime
    trade_id: str
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    # Configuration
    config: BacktestConfig
    strategy_name: str
    
    # Summary metrics
    metrics: PerformanceMetrics
    
    # Detailed data
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[datetime] = field(default_factory=list)
    
    # Per-category breakdown
    category_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    # Expert performance
    expert_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "initial_capital": self.config.initial_capital,
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "total_trades": len(self.trades),
            "final_equity": self.equity_curve[-1] if self.equity_curve else self.config.initial_capital,
        }


class BacktestEngine:
    """Main backtesting engine.
    
    Replays historical market data and simulates strategy execution
    with realistic slippage and commission modeling.
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtest engine.
        
        Args:
            config: Backtest configuration.
        """
        self.config = config
        self.slippage = SlippageModel()
        self.analytics = PerformanceAnalytics()
        
        # State
        self._cash = config.initial_capital
        self._positions: Dict[str, BacktestPosition] = {}
        self._trades: List[BacktestTrade] = []
        self._trade_counter = 0
        
        # Equity tracking
        self._equity_curve: List[float] = [config.initial_capital]
        self._equity_timestamps: List[datetime] = []
        
        logger.info(f"BacktestEngine initialized: {config.start_date} to {config.end_date}")
    
    def run(self, strategy, data: List[Dict[str, Any]]) -> BacktestResult:
        """Run backtest with given strategy and data.
        
        Args:
            strategy: Strategy implementing BacktestStrategy protocol.
            data: List of market snapshot dicts, sorted by timestamp.
            
        Returns:
            BacktestResult with all metrics and trades.
        """
        logger.info(f"Starting backtest with {len(data)} data points")
        
        if len(data) < self.config.min_data_points:
            logger.warning(f"Not enough data points: {len(data)} < {self.config.min_data_points}")
            return self._empty_result(strategy.name)
        
        # Filter data to date range
        filtered_data = self._filter_data(data)
        if len(filtered_data) < self.config.min_data_points:
            logger.warning(f"Not enough data in date range")
            return self._empty_result(strategy.name)
        
        logger.info(f"Processing {len(filtered_data)} snapshots in date range")
        
        # Process each snapshot chronologically
        for i, snapshot in enumerate(filtered_data):
            self._process_snapshot(snapshot, strategy)
            
            # Record equity periodically
            if i % 100 == 0 or i == len(filtered_data) - 1:
                equity = self._calculate_equity(snapshot)
                self._equity_curve.append(equity)
                ts = snapshot.get("timestamp") or snapshot.get("created_at")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        ts = datetime.utcnow()
                self._equity_timestamps.append(ts)
        
        # Close any remaining positions at last price
        self._close_all_positions(filtered_data[-1])
        
        # Calculate performance metrics
        metrics = self.analytics.compute(
            trades=self._trades,
            equity_curve=self._equity_curve,
            initial_capital=self.config.initial_capital,
        )
        
        # Calculate per-category metrics
        category_metrics = self._calculate_category_metrics()
        
        # Calculate per-expert metrics
        expert_metrics = self._calculate_expert_metrics()
        
        result = BacktestResult(
            config=self.config,
            strategy_name=strategy.name,
            metrics=metrics,
            trades=self._trades,
            equity_curve=self._equity_curve,
            equity_timestamps=self._equity_timestamps,
            category_metrics=category_metrics,
            expert_metrics=expert_metrics,
        )
        
        logger.info(
            f"Backtest complete: {len(self._trades)} trades, "
            f"final equity ${self._equity_curve[-1]:,.2f}, "
            f"return {metrics.total_return_pct:+.2f}%"
        )
        
        return result
    
    def _filter_data(self, data: List[Dict]) -> List[Dict]:
        """Filter data to config date range."""
        filtered = []
        
        for snapshot in data:
            ts = snapshot.get("timestamp") or snapshot.get("created_at")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    continue
            
            if self.config.start_date <= ts <= self.config.end_date:
                filtered.append(snapshot)
        
        # Sort by timestamp
        filtered.sort(key=lambda x: x.get("timestamp") or x.get("created_at", ""))
        
        return filtered
    
    def _process_snapshot(self, snapshot: Dict, strategy) -> None:
        """Process a single market snapshot."""
        token_id = snapshot.get("token_id", "")
        
        # Update existing positions
        self._update_positions(snapshot)
        
        # Check for exit signals on existing positions
        if token_id in self._positions:
            exit_signal = strategy.should_exit(snapshot, self._positions[token_id])
            if exit_signal:
                self._close_position(token_id, snapshot)
        
        # Check for entry signals
        if token_id not in self._positions:
            signal = strategy.generate_signal(snapshot)
            if signal and self._can_open_position(signal.size):
                self._open_position(snapshot, signal)
    
    def _update_positions(self, snapshot: Dict) -> None:
        """Update position mark-to-market."""
        token_id = snapshot.get("token_id", "")
        
        if token_id in self._positions:
            pos = self._positions[token_id]
            price = snapshot.get("price", pos.current_price)
            pos.current_price = price
            
            if pos.side == "BUY":
                pos.unrealized_pnl = (price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.size
    
    def _can_open_position(self, size: float) -> bool:
        """Check if we can open a new position."""
        # Check cash
        if size > self._cash:
            return False
        
        # Check total exposure
        total_exposure = sum(p.size for p in self._positions.values())
        equity = self._cash + total_exposure
        
        if (total_exposure + size) / equity > self.config.max_total_exposure_pct:
            return False
        
        # Check per-position limit
        if size / equity > self.config.max_position_pct:
            return False
        
        return True
    
    def _open_position(self, snapshot: Dict, signal) -> None:
        """Open a new position."""
        token_id = snapshot.get("token_id", "")
        market_id = snapshot.get("market_id", "")
        price = snapshot.get("price", 0)
        
        # Apply slippage
        if self.config.slippage_enabled:
            slippage = self.slippage.estimate_slippage(
                size=signal.size,
                price=price,
                side=signal.side,
                spread=snapshot.get("spread", 0.02),
            )
            if signal.side == "BUY":
                price += slippage
            else:
                price -= slippage
        else:
            slippage = 0
        
        # Commission
        commission = signal.size * self.config.commission_pct
        
        # Create position
        self._trade_counter += 1
        trade_id = f"bt_{self._trade_counter}"
        
        ts = snapshot.get("timestamp") or snapshot.get("created_at")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                ts = datetime.utcnow()
        
        position = BacktestPosition(
            token_id=token_id,
            market_id=market_id,
            side=signal.side,
            size=signal.size,
            entry_price=price,
            entry_time=ts,
            current_price=price,
            trade_id=trade_id,
        )
        
        self._positions[token_id] = position
        self._cash -= signal.size + commission
        
        # Create trade record
        trade = BacktestTrade(
            trade_id=trade_id,
            token_id=token_id,
            market_id=market_id,
            side=signal.side,
            entry_time=ts,
            entry_price=price,
            size=signal.size,
            signal_strength=signal.confidence,
            expert_id=getattr(signal, 'expert_id', ''),
            slippage_cost=slippage * signal.size / price if price > 0 else 0,
            commission_cost=commission,
        )
        
        self._trades.append(trade)
    
    def _close_position(self, token_id: str, snapshot: Dict) -> None:
        """Close an existing position."""
        if token_id not in self._positions:
            return
        
        pos = self._positions[token_id]
        price = snapshot.get("price", pos.current_price)
        
        # Apply slippage
        if self.config.slippage_enabled:
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            slippage = self.slippage.estimate_slippage(
                size=pos.size,
                price=price,
                side=exit_side,
                spread=snapshot.get("spread", 0.02),
            )
            if exit_side == "SELL":
                price -= slippage
            else:
                price += slippage
        
        # Commission
        commission = pos.size * self.config.commission_pct
        
        # Calculate P&L
        if pos.side == "BUY":
            profit_usd = (price - pos.entry_price) * pos.size / pos.entry_price
        else:
            profit_usd = (pos.entry_price - price) * pos.size / pos.entry_price
        
        profit_usd -= commission
        profit_pct = profit_usd / pos.size * 100 if pos.size > 0 else 0
        
        # Get timestamp
        ts = snapshot.get("timestamp") or snapshot.get("created_at")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                ts = datetime.utcnow()
        
        # Update trade record
        for trade in self._trades:
            if trade.trade_id == pos.trade_id:
                trade.exit_time = ts
                trade.exit_price = price
                trade.profit_usd = profit_usd
                trade.profit_pct = profit_pct
                trade.commission_cost += commission
                break
        
        # Update cash
        self._cash += pos.size + profit_usd
        
        # Remove position
        del self._positions[token_id]
    
    def _close_all_positions(self, last_snapshot: Dict) -> None:
        """Close all remaining positions at end of backtest."""
        for token_id in list(self._positions.keys()):
            self._close_position(token_id, last_snapshot)
    
    def _calculate_equity(self, snapshot: Dict) -> float:
        """Calculate current total equity."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        positions_value = sum(p.size for p in self._positions.values())
        return self._cash + positions_value + unrealized
    
    def _calculate_category_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics broken down by category."""
        # Group trades by category (from market_id prefix or explicit category)
        category_trades: Dict[str, List[BacktestTrade]] = {}
        
        for trade in self._trades:
            # Extract category from market_id or use 'unknown'
            category = "unknown"
            if hasattr(trade, 'category') and trade.category:
                category = trade.category
            
            if category not in category_trades:
                category_trades[category] = []
            category_trades[category].append(trade)
        
        # Calculate metrics per category
        result = {}
        for category, trades in category_trades.items():
            if trades:
                result[category] = self.analytics.compute_from_trades(trades)
        
        return result
    
    def _calculate_expert_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Calculate metrics broken down by expert."""
        expert_trades: Dict[str, List[BacktestTrade]] = {}
        
        for trade in self._trades:
            expert_id = trade.expert_id or "unknown"
            
            if expert_id not in expert_trades:
                expert_trades[expert_id] = []
            expert_trades[expert_id].append(trade)
        
        result = {}
        for expert_id, trades in expert_trades.items():
            if trades:
                result[expert_id] = self.analytics.compute_from_trades(trades)
        
        return result
    
    def _empty_result(self, strategy_name: str) -> BacktestResult:
        """Return empty result when backtest can't run."""
        return BacktestResult(
            config=self.config,
            strategy_name=strategy_name,
            metrics=PerformanceMetrics(),
            trades=[],
            equity_curve=[self.config.initial_capital],
            equity_timestamps=[],
        )
