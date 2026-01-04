"""Tests for portfolio management."""

import uuid

from polyb0t.execution.orders import Fill, Order, OrderSide, OrderType
from polyb0t.execution.portfolio import Portfolio


def test_portfolio_initialization():
    """Test portfolio initialization."""
    portfolio = Portfolio(10000.0)

    assert portfolio.initial_cash == 10000.0
    assert portfolio.cash_balance == 10000.0
    assert portfolio.total_equity == 10000.0
    assert portfolio.num_positions == 0


def test_create_position():
    """Test position creation from fill."""
    portfolio = Portfolio(10000.0)

    order = Order(
        token_id="token_test",
        market_id="market_test",
        side=OrderSide.BUY,
        price=0.6,
        size=100.0,
    )

    fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        token_id="token_test",
        price=0.6,
        size=100.0,
        fee=0.2,
    )

    initial_cash = portfolio.cash_balance
    portfolio.process_fill(order, fill)

    # Check position created
    assert "token_test" in portfolio.positions
    position = portfolio.positions["token_test"]
    assert position.side == "LONG"
    assert position.quantity == 100.0
    assert position.avg_entry_price == 0.6

    # Check cash deducted
    assert portfolio.cash_balance == initial_cash - 100.0 - 0.2


def test_portfolio_exposure():
    """Test exposure calculation."""
    portfolio = Portfolio(10000.0)

    # Create two positions
    order1 = Order(
        token_id="token_1",
        market_id="market_1",
        side=OrderSide.BUY,
        price=0.5,
        size=100.0,
    )
    fill1 = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order1.order_id,
        token_id="token_1",
        price=0.5,
        size=100.0,
        fee=0.2,
    )

    order2 = Order(
        token_id="token_2",
        market_id="market_2",
        side=OrderSide.BUY,
        price=0.7,
        size=150.0,
    )
    fill2 = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order2.order_id,
        token_id="token_2",
        price=0.7,
        size=150.0,
        fee=0.3,
    )

    portfolio.process_fill(order1, fill1)
    portfolio.process_fill(order2, fill2)

    # Total exposure should be sum of position quantities
    assert portfolio.total_exposure == 250.0
    assert portfolio.num_positions == 2


def test_unrealized_pnl():
    """Test unrealized PnL calculation."""
    portfolio = Portfolio(10000.0)

    order = Order(
        token_id="token_test",
        market_id="market_test",
        side=OrderSide.BUY,
        price=0.6,
        size=100.0,
    )
    fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        token_id="token_test",
        price=0.6,
        size=100.0,
        fee=0.2,
    )

    portfolio.process_fill(order, fill)

    # Update market price
    portfolio.update_market_prices({"token_test": 0.7})

    # Should have unrealized PnL
    assert portfolio.unrealized_pnl > 0


def test_portfolio_summary():
    """Test portfolio summary generation."""
    portfolio = Portfolio(10000.0)

    summary = portfolio.get_summary()

    assert "cash_balance" in summary
    assert "total_equity" in summary
    assert "unrealized_pnl" in summary
    assert "realized_pnl" in summary
    assert "num_positions" in summary
    assert "return_pct" in summary

    assert summary["return_pct"] == 0.0  # No change yet

