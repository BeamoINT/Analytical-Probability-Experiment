"""Tests for paper trading simulator."""

from datetime import datetime

import pytest

from polyb0t.execution.orders import OrderSide
from polyb0t.execution.portfolio import Portfolio
from polyb0t.execution.simulator import PaperTradingSimulator
from polyb0t.models.strategy_baseline import TradingSignal


def test_place_order(db_session):
    """Test order placement."""
    portfolio = Portfolio(10000.0)
    simulator = PaperTradingSimulator(portfolio, db_session)

    signal = TradingSignal(
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.6,
        edge=0.1,
        confidence=0.8,
        features={},
    )

    order = simulator.place_order(signal, size=100.0, cycle_id="test_cycle")

    assert order.token_id == "token_test"
    assert order.side == OrderSide.BUY
    assert order.size == 100.0
    assert order.order_id in simulator.open_orders


def test_order_expiration(db_session):
    """Test order expiration handling."""
    portfolio = Portfolio(10000.0)
    simulator = PaperTradingSimulator(portfolio, db_session)

    signal = TradingSignal(
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.6,
        edge=0.1,
        confidence=0.8,
        features={},
    )

    order = simulator.place_order(signal, size=100.0, cycle_id="test_cycle")

    # Manually expire the order
    order.expire()

    assert not order.is_active
    assert order.status.value == "EXPIRED"


def test_trade_crosses_order(db_session, sample_orderbook, sample_trades):
    """Test trade crossing logic."""
    portfolio = Portfolio(10000.0)
    simulator = PaperTradingSimulator(portfolio, db_session)

    signal = TradingSignal(
        token_id="token_yes",
        market_id="market_test",
        side="BUY",
        p_market=0.59,
        p_model=0.65,
        edge=0.06,
        confidence=0.8,
        features={},
    )

    # Place buy order at 0.59
    order = simulator.place_order(signal, size=100.0, cycle_id="test_cycle")

    # Check if trade at 0.58 crosses (should for BUY order at 0.59)
    trade = sample_trades[2]  # Trade at 0.58
    crosses = simulator._trade_crosses_order(order, trade)

    assert crosses  # Trade price 0.58 <= order price 0.59


def test_portfolio_update_on_fill(db_session, sample_orderbook, sample_trades):
    """Test portfolio updates when order fills."""
    portfolio = Portfolio(10000.0)
    simulator = PaperTradingSimulator(portfolio, db_session)

    signal = TradingSignal(
        token_id="token_yes",
        market_id="market_test",
        side="BUY",
        p_market=0.58,
        p_model=0.65,
        edge=0.07,
        confidence=0.8,
        features={},
    )

    initial_cash = portfolio.cash_balance

    order = simulator.place_order(signal, size=100.0, cycle_id="test_cycle")

    # Simulate a fill
    simulator._execute_fill(order, fill_price=0.58, fill_size=100.0, cycle_id="test_cycle")

    # Check portfolio was updated
    assert portfolio.cash_balance < initial_cash  # Cash decreased
    assert len(portfolio.positions) == 1  # Position created
    assert "token_yes" in portfolio.positions

