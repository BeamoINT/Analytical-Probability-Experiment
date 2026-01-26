"""Backtesting framework for strategy evaluation.

This module provides comprehensive backtesting capabilities:
- Historical data replay
- Realistic execution simulation with slippage
- Orderbook-aware fill estimation
- Performance analytics (Sharpe, Sortino, drawdown, etc.)
- Multi-strategy comparison
"""

from polyb0t.backtest.engine import BacktestEngine, BacktestResult
from polyb0t.backtest.analytics import PerformanceAnalytics
from polyb0t.backtest.slippage import SlippageModel
from polyb0t.backtest.strategies import BacktestStrategy, MoEBacktestStrategy

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PerformanceAnalytics",
    "SlippageModel",
    "BacktestStrategy",
    "MoEBacktestStrategy",
]
