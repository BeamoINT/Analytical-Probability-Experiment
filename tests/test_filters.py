"""Tests for market filtering."""

from datetime import datetime, timedelta

import pytest

from polyb0t.data.models import Market, MarketOutcome
from polyb0t.models.filters import MarketFilter


def test_filter_by_resolution_time():
    """Test filtering by resolution time window."""
    market_filter = MarketFilter()

    # Market resolving in 45 days (within 30-60 window)
    market_valid = Market(
        condition_id="valid_market",
        question="Valid market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token1", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    # Market resolving in 10 days (too soon)
    market_too_soon = Market(
        condition_id="too_soon",
        question="Too soon",
        end_date=datetime.utcnow() + timedelta(days=10),
        outcomes=[MarketOutcome(token_id="token2", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    # Market resolving in 90 days (too far)
    market_too_far = Market(
        condition_id="too_far",
        question="Too far",
        end_date=datetime.utcnow() + timedelta(days=90),
        outcomes=[MarketOutcome(token_id="token3", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    markets = [market_valid, market_too_soon, market_too_far]
    filtered = market_filter.filter_markets(markets)

    assert len(filtered) == 1
    assert filtered[0].condition_id == "valid_market"


def test_filter_by_liquidity():
    """Test filtering by minimum liquidity."""
    market_filter = MarketFilter()

    # Market with sufficient liquidity
    market_liquid = Market(
        condition_id="liquid",
        question="Liquid market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token1", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    # Market with insufficient liquidity
    market_illiquid = Market(
        condition_id="illiquid",
        question="Illiquid market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token2", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=500.0,  # Below default minimum of 1000
    )

    markets = [market_liquid, market_illiquid]
    filtered = market_filter.filter_markets(markets)

    assert len(filtered) == 1
    assert filtered[0].condition_id == "liquid"


def test_filter_blacklist():
    """Test manual blacklist filtering."""
    market_filter = MarketFilter()
    market_filter.load_blacklist(["blacklisted_market"])

    market_normal = Market(
        condition_id="normal",
        question="Normal market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token1", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    market_blacklisted = Market(
        condition_id="blacklisted_market",
        question="Blacklisted",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token2", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    markets = [market_normal, market_blacklisted]
    filtered = market_filter.filter_markets(markets)

    assert len(filtered) == 1
    assert filtered[0].condition_id == "normal"


def test_filter_inactive_markets():
    """Test filtering of inactive or closed markets."""
    market_filter = MarketFilter()

    market_active = Market(
        condition_id="active",
        question="Active market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token1", outcome="Yes")],
        active=True,
        closed=False,
        liquidity=5000.0,
    )

    market_inactive = Market(
        condition_id="inactive",
        question="Inactive market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token2", outcome="Yes")],
        active=False,
        closed=False,
        liquidity=5000.0,
    )

    market_closed = Market(
        condition_id="closed",
        question="Closed market",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[MarketOutcome(token_id="token3", outcome="Yes")],
        active=True,
        closed=True,
        liquidity=5000.0,
    )

    markets = [market_active, market_inactive, market_closed]
    filtered = market_filter.filter_markets(markets)

    assert len(filtered) == 1
    assert filtered[0].condition_id == "active"

