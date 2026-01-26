"""Performance analytics for backtesting.

Provides comprehensive performance metrics including:
- Return metrics (total, CAGR, monthly)
- Risk metrics (Sharpe, Sortino, max drawdown)
- Trade statistics (win rate, profit factor, avg trade)
- Distribution analysis
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Return metrics
    total_return_pct: float = 0.0
    total_return_usd: float = 0.0
    cagr: float = 0.0  # Compound annual growth rate
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    volatility: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L statistics
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pct: float = 0.0
    
    # Exposure
    avg_exposure_pct: float = 0.0
    max_exposure_pct: float = 0.0
    time_in_market_pct: float = 0.0
    
    # Time metrics
    avg_hold_time_hours: float = 0.0
    avg_win_hold_time: float = 0.0
    avg_loss_hold_time: float = 0.0
    
    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0
    
    # Monthly returns
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return_pct": self.total_return_pct,
            "total_return_usd": self.total_return_usd,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pct": self.avg_trade_pct,
            "best_trade_pct": self.best_trade_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "avg_hold_time_hours": self.avg_hold_time_hours,
        }


class PerformanceAnalytics:
    """Calculate comprehensive performance metrics."""
    
    def __init__(self, risk_free_rate: float = 0.05):
        """Initialize analytics.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
        """
        self.risk_free_rate = risk_free_rate
    
    def compute(
        self,
        trades: List,
        equity_curve: List[float],
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Compute all performance metrics.
        
        Args:
            trades: List of BacktestTrade objects.
            equity_curve: List of equity values over time.
            initial_capital: Starting capital.
            
        Returns:
            PerformanceMetrics with all calculations.
        """
        metrics = PerformanceMetrics()
        
        if not equity_curve:
            return metrics
        
        # Return metrics
        final_equity = equity_curve[-1]
        metrics.total_return_usd = final_equity - initial_capital
        metrics.total_return_pct = (metrics.total_return_usd / initial_capital) * 100
        
        # Calculate returns series
        returns = self._calculate_returns(equity_curve)
        
        # Risk metrics
        if len(returns) > 1:
            metrics.volatility = float(np.std(returns) * np.sqrt(252))  # Annualized
            metrics.sharpe_ratio = self._calculate_sharpe(returns)
            metrics.sortino_ratio = self._calculate_sortino(returns)
        
        # Drawdown
        dd_pct, dd_usd = self._calculate_max_drawdown(equity_curve)
        metrics.max_drawdown_pct = dd_pct
        metrics.max_drawdown_usd = dd_usd
        
        # Calmar ratio
        if metrics.max_drawdown_pct > 0 and len(equity_curve) > 1:
            metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown_pct
        
        # Trade statistics
        if trades:
            self._compute_trade_stats(trades, metrics)
        
        return metrics
    
    def compute_from_trades(self, trades: List) -> PerformanceMetrics:
        """Compute metrics from trades only (no equity curve).
        
        Useful for per-category or per-expert breakdown.
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        self._compute_trade_stats(trades, metrics)
        
        # Estimate total return from trades
        metrics.total_return_pct = sum(t.profit_pct for t in trades)
        metrics.total_return_usd = sum(t.profit_usd for t in trades)
        
        return metrics
    
    def _calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """Calculate period returns from equity curve."""
        if len(equity_curve) < 2:
            return np.array([])
        
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        return returns
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        # Daily risk-free rate
        rf_daily = self.risk_free_rate / 252
        
        excess_returns = returns - rf_daily
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return float(sharpe)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        if len(returns) < 2:
            return 0.0
        
        rf_daily = self.risk_free_rate / 252
        excess_returns = returns - rf_daily
        
        # Downside returns only
        downside = excess_returns[excess_returns < 0]
        
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0 if np.mean(excess_returns) <= 0 else float('inf')
        
        sortino = np.mean(excess_returns) / np.std(downside) * np.sqrt(252)
        
        return float(sortino)
    
    def _calculate_max_drawdown(
        self, equity_curve: List[float]
    ) -> tuple[float, float]:
        """Calculate maximum drawdown."""
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        
        max_dd_pct = float(np.max(drawdown)) * 100
        
        # Find the actual USD drawdown at max point
        max_dd_idx = np.argmax(drawdown)
        max_dd_usd = float(peak[max_dd_idx] - equity[max_dd_idx])
        
        return max_dd_pct, max_dd_usd
    
    def _compute_trade_stats(self, trades: List, metrics: PerformanceMetrics) -> None:
        """Compute trade-level statistics."""
        closed_trades = [t for t in trades if t.is_closed]
        
        if not closed_trades:
            return
        
        metrics.total_trades = len(closed_trades)
        
        # Win/loss breakdown
        winners = [t for t in closed_trades if t.profit_pct > 0]
        losers = [t for t in closed_trades if t.profit_pct <= 0]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        metrics.win_rate = len(winners) / len(closed_trades) if closed_trades else 0
        
        # P&L statistics
        profits = [t.profit_pct for t in closed_trades]
        metrics.avg_trade_pct = float(np.mean(profits))
        metrics.best_trade_pct = max(profits) if profits else 0
        metrics.worst_trade_pct = min(profits) if profits else 0
        
        if winners:
            metrics.avg_win_pct = float(np.mean([t.profit_pct for t in winners]))
        if losers:
            metrics.avg_loss_pct = float(np.mean([t.profit_pct for t in losers]))
        
        # Profit factor
        gross_profit = sum(t.profit_usd for t in winners)
        gross_loss = abs(sum(t.profit_usd for t in losers))
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics.profit_factor = float('inf')
        
        # Hold times
        hold_times = []
        for t in closed_trades:
            if t.entry_time and t.exit_time:
                hold_hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                hold_times.append(hold_hours)
        
        if hold_times:
            metrics.avg_hold_time_hours = float(np.mean(hold_times))
        
        win_hold = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in winners if t.entry_time and t.exit_time
        ]
        loss_hold = [
            (t.exit_time - t.entry_time).total_seconds() / 3600
            for t in losers if t.entry_time and t.exit_time
        ]
        
        if win_hold:
            metrics.avg_win_hold_time = float(np.mean(win_hold))
        if loss_hold:
            metrics.avg_loss_hold_time = float(np.mean(loss_hold))
        
        # Streaks
        self._calculate_streaks(closed_trades, metrics)
    
    def _calculate_streaks(self, trades: List, metrics: PerformanceMetrics) -> None:
        """Calculate win/loss streaks."""
        if not trades:
            return
        
        max_win = 0
        max_loss = 0
        current = 0
        current_type = None
        
        for trade in sorted(trades, key=lambda t: t.entry_time or datetime.min):
            is_win = trade.profit_pct > 0
            
            if current_type is None:
                current_type = is_win
                current = 1
            elif is_win == current_type:
                current += 1
            else:
                if current_type:
                    max_win = max(max_win, current)
                else:
                    max_loss = max(max_loss, current)
                current_type = is_win
                current = 1
        
        # Final streak
        if current_type is not None:
            if current_type:
                max_win = max(max_win, current)
            else:
                max_loss = max(max_loss, current)
        
        metrics.max_win_streak = max_win
        metrics.max_loss_streak = max_loss
        metrics.current_streak = current if current_type else -current
    
    def generate_monthly_returns(
        self,
        trades: List,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:
        """Generate monthly return breakdown."""
        monthly = {}
        
        for trade in trades:
            if not trade.exit_time:
                continue
            
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = 0.0
            monthly[month_key] += trade.profit_pct
        
        return monthly
