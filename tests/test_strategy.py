"""Tests for baseline trading strategy."""

from polyb0t.models.strategy_baseline import BaselineStrategy


def test_generate_signals(sample_market, sample_orderbook, sample_trades):
    """Test signal generation."""
    strategy = BaselineStrategy()

    orderbooks = {"token_yes": sample_orderbook}
    trades = {"token_yes": sample_trades}

    signals = strategy.generate_signals([sample_market], orderbooks, trades)

    # Should generate signals if edge threshold is met
    # With shrinkage and adjustments, we expect at least some signal
    assert isinstance(signals, list)


def test_model_probability_calculation(sample_market, sample_orderbook, sample_trades):
    """Test model probability computation."""
    strategy = BaselineStrategy()

    features = strategy.feature_engine.compute_features(
        sample_market, 0, sample_orderbook, sample_trades
    )

    p_market = 0.6
    p_model = strategy._compute_model_probability(p_market, features)

    # Model probability should be different from market due to adjustments
    # And should be in valid range
    assert 0 < p_model < 1
    assert p_model != p_market  # Should have some adjustment


def test_confidence_calculation():
    """Test confidence scoring."""
    strategy = BaselineStrategy()

    # High quality features (many trades, tight spread, good depth)
    high_quality_features = {
        "num_trades": 20,
        "spread_pct": 0.01,
        "bid_depth": 2000,
        "ask_depth": 1800,
    }

    # Low quality features
    low_quality_features = {
        "num_trades": 2,
        "spread_pct": 0.08,
        "bid_depth": 100,
        "ask_depth": 100,
    }

    high_confidence = strategy._compute_confidence(high_quality_features)
    low_confidence = strategy._compute_confidence(low_quality_features)

    assert high_confidence > low_confidence
    assert 0 < high_confidence <= 1
    assert 0 < low_confidence <= 1

