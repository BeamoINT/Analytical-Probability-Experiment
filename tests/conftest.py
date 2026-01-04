"""Pytest fixtures and configuration."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from polyb0t.data.models import Market, MarketOutcome, OrderBook, OrderBookLevel, Trade
from polyb0t.data.storage import Base


@pytest.fixture(autouse=True)
def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure required env vars exist for Settings in tests."""
    monkeypatch.setenv("POLYBOT_MODE", "paper")
    monkeypatch.setenv("POLYBOT_DRY_RUN", "true")
    monkeypatch.setenv("POLYBOT_LOOP_INTERVAL_SECONDS", "10")
    monkeypatch.setenv(
        "POLYBOT_USER_ADDRESS", "0x5cbb1a163f426097578eb4de9e3ecd987fc1c0d4"
    )

    # Clear cached settings between tests
    from polyb0t.config.settings import get_settings

    get_settings.cache_clear()


@pytest.fixture
def db_session() -> Session:
    """Create in-memory test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_market() -> Market:
    """Create sample market for testing."""
    return Market(
        condition_id="test_market_123",
        question="Will it rain tomorrow?",
        description="Test market description",
        end_date=datetime.utcnow() + timedelta(days=45),
        outcomes=[
            MarketOutcome(token_id="token_yes", outcome="Yes", price=0.6),
            MarketOutcome(token_id="token_no", outcome="No", price=0.4),
        ],
        category="Weather",
        volume=50000.0,
        liquidity=10000.0,
        active=True,
        closed=False,
    )


@pytest.fixture
def sample_orderbook() -> OrderBook:
    """Create sample orderbook for testing."""
    return OrderBook(
        token_id="token_yes",
        timestamp=datetime.utcnow(),
        bids=[
            OrderBookLevel(price=0.58, size=1000),
            OrderBookLevel(price=0.57, size=1500),
            OrderBookLevel(price=0.56, size=2000),
        ],
        asks=[
            OrderBookLevel(price=0.60, size=800),
            OrderBookLevel(price=0.61, size=1200),
            OrderBookLevel(price=0.62, size=1500),
        ],
    )


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades for testing."""
    base_time = datetime.utcnow()
    return [
        Trade(
            token_id="token_yes",
            timestamp=base_time - timedelta(minutes=5),
            price=0.59,
            size=500,
            side="BUY",
            trade_id="trade_1",
        ),
        Trade(
            token_id="token_yes",
            timestamp=base_time - timedelta(minutes=3),
            price=0.60,
            size=300,
            side="SELL",
            trade_id="trade_2",
        ),
        Trade(
            token_id="token_yes",
            timestamp=base_time - timedelta(minutes=1),
            price=0.58,
            size=400,
            side="BUY",
            trade_id="trade_3",
        ),
    ]

