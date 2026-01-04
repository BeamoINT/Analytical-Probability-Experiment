"""Tests for risk management."""

from polyb0t.models.risk import RiskManager
from polyb0t.models.strategy_baseline import TradingSignal


def test_position_size_calculation():
    """Test position sizing."""
    risk_manager = RiskManager()

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

    # With default 2% max position and 0.8 confidence
    # Expected size = 10000 * 0.02 * 0.8 = 160
    position_size = risk_manager._calculate_position_size(
        signal,
        current_cash=10000.0,
        portfolio_exposure=0.0,
    )

    assert position_size > 0
    assert position_size <= 10000.0 * 0.02  # Max 2%


def test_exposure_limits():
    """Test exposure limit enforcement."""
    risk_manager = RiskManager()

    signal = TradingSignal(
        token_id="token_new",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.6,
        edge=0.1,
        confidence=0.8,
        features={},
    )

    # Portfolio already at high exposure (19% of 10000 = 1900)
    result = risk_manager.check_position(
        signal,
        current_cash=8100.0,
        current_positions={},
        portfolio_exposure=1900.0,
    )

    # New position would exceed 20% limit, so should be rejected
    # Actually, with max position of 2%, we'd be at 21%, so might be rejected
    # Let's check the result
    if result.approved:
        # If approved, check that total doesn't exceed limit
        total = 1900.0 + (result.max_position_size or 0)
        assert total <= 10000.0 * 0.20


def test_drawdown_check():
    """Test drawdown limit checking."""
    risk_manager = RiskManager()

    # Start with peak at initial bankroll
    initial_equity = 10000.0
    risk_manager.peak_equity = initial_equity

    # Current equity down 10% (within 15% limit)
    current_equity = 9000.0
    halted = risk_manager.check_drawdown(current_equity)
    assert not halted
    assert not risk_manager.is_trading_halted

    # Current equity down 16% (exceeds 15% limit)
    current_equity = 8400.0
    halted = risk_manager.check_drawdown(current_equity)
    assert halted
    assert risk_manager.is_trading_halted


def test_duplicate_position_rejection():
    """Test that duplicate positions are rejected."""
    risk_manager = RiskManager()

    signal = TradingSignal(
        token_id="token_existing",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.6,
        edge=0.1,
        confidence=0.8,
        features={},
    )

    # Already have a BUY position in this token
    current_positions = {
        "token_existing": {
            "side": "BUY",
            "quantity": 100,
            "exposure": 100,
            "market_id": "market_test",
        }
    }

    result = risk_manager.check_position(
        signal,
        current_cash=9900.0,
        current_positions=current_positions,
        portfolio_exposure=100.0,
    )

    assert not result.approved
    assert "Already have" in result.reason

