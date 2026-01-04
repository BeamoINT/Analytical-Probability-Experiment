"""Tests for trade intent system."""

from datetime import datetime, timedelta

import pytest

from polyb0t.execution.intents import IntentManager, IntentStatus, IntentType, TradeIntent
from polyb0t.models.strategy_baseline import TradingSignal


def test_create_intent_from_signal(db_session):
    """Test creating intent from trading signal."""
    intent_manager = IntentManager(db_session)

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

    risk_checks = {"approved": True, "position_size": 150.0}

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=150.0,
        risk_checks=risk_checks,
        cycle_id="test_cycle",
    )

    assert intent.intent_id is not None
    assert intent.intent_type == IntentType.OPEN_POSITION
    assert intent.token_id == "token_test"
    assert intent.side == "BUY"
    assert intent.size_usd == 150.0
    assert intent.status == IntentStatus.PENDING


def test_intent_expiry(db_session):
    """Test intent expiry logic."""
    intent_manager = IntentManager(db_session)

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

    # Create intent with very short expiry
    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Set expiry to past
    intent.expires_at = datetime.utcnow() - timedelta(seconds=10)

    assert intent.is_expired()
    assert not intent.is_executable()


def test_approve_intent(db_session):
    """Test intent approval."""
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Approve intent
    approved_intent = intent_manager.approve_intent(intent.intent_id, approved_by="test_user")

    assert approved_intent is not None
    assert approved_intent.status == IntentStatus.APPROVED
    assert approved_intent.approved_by == "test_user"
    assert approved_intent.approved_at is not None
    assert approved_intent.is_executable()


def test_reject_intent(db_session):
    """Test intent rejection."""
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Reject intent
    rejected = intent_manager.reject_intent(intent.intent_id)

    assert rejected
    
    # Reload from DB
    reloaded = intent_manager.get_intent(intent.intent_id)
    assert reloaded.status == IntentStatus.REJECTED


def test_cannot_approve_expired_intent(db_session):
    """Test that expired intents cannot be approved."""
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Manually set expiry to past (simulate expired intent)
    from polyb0t.data.storage import TradeIntentDB

    db_intent = db_session.query(TradeIntentDB).filter_by(intent_id=intent.intent_id).first()
    db_intent.expires_at = datetime.utcnow() - timedelta(seconds=10)
    db_session.commit()

    # Try to approve
    approved = intent_manager.approve_intent(intent.intent_id)

    assert approved is None  # Should fail to approve expired intent


def test_get_pending_intents(db_session):
    """Test retrieving pending intents."""
    intent_manager = IntentManager(db_session)

    # Create multiple intents
    for i in range(3):
        signal = TradingSignal(
            token_id=f"token_{i}",
            market_id="market_test",
            side="BUY",
            p_market=0.5,
            p_model=0.6,
            edge=0.1,
            confidence=0.8,
            features={},
        )

        intent_manager.create_intent_from_signal(
            signal=signal,
            size=100.0,
            risk_checks={},
            cycle_id="test_cycle",
        )

    pending = intent_manager.get_pending_intents()

    assert len(pending) == 3
    assert all(intent.status == IntentStatus.PENDING for intent in pending)


def test_mark_executed(db_session):
    """Test marking intent as executed."""
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Approve first
    intent_manager.approve_intent(intent.intent_id)

    # Mark as executed
    execution_result = {"order_id": "order_123", "success": True}
    marked = intent_manager.mark_executed(intent.intent_id, execution_result)

    assert marked

    # Reload and check
    reloaded = intent_manager.get_intent(intent.intent_id)
    assert reloaded.status == IntentStatus.EXECUTED
    assert reloaded.executed_at is not None
    assert reloaded.execution_result == execution_result


def test_create_exit_intent(db_session):
    """Test creating exit intent."""
    intent_manager = IntentManager(db_session)

    exit_intent = intent_manager.create_exit_intent(
        token_id="token_test",
        market_id="market_test",
        side="SELL",
        price=0.65,
        size=100.0,
        reason="Take-profit: PnL 12% >= target 10%",
        cycle_id="test_cycle",
    )

    assert exit_intent is not None
    assert exit_intent.intent_type == IntentType.CLOSE_POSITION
    assert exit_intent.side == "SELL"
    assert exit_intent.price == 0.65
    assert exit_intent.status == IntentStatus.PENDING


def test_expire_old_intents(db_session):
    """Test automatic expiry of old intents."""
    intent_manager = IntentManager(db_session)

    # Create intent
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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )

    # Manually set expiry to past
    from polyb0t.data.storage import TradeIntentDB

    db_intent = db_session.query(TradeIntentDB).filter_by(intent_id=intent.intent_id).first()
    db_intent.expires_at = datetime.utcnow() - timedelta(seconds=10)
    db_session.commit()

    # Run expiry
    count = intent_manager.expire_old_intents()

    assert count == 1

    # Check status changed
    reloaded = intent_manager.get_intent(intent.intent_id)
    assert reloaded.status == IntentStatus.EXPIRED


def test_dedup_skips_identical_intent_within_cooldown(db_session):
    """Same fingerprint within cooldown should be suppressed."""
    intent_manager = IntentManager(db_session)

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

    i1 = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert i1 is not None

    # Second identical intent should be suppressed
    i2 = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert i2 is None


def test_dedup_allows_new_intent_if_edge_changes_enough(db_session):
    """Within cooldown, do NOT create a second pending intent; update existing if edge changes enough."""
    intent_manager = IntentManager(db_session)

    signal1 = TradingSignal(
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.6,
        edge=0.1,
        confidence=0.8,
        features={},
    )
    i1 = intent_manager.create_intent_from_signal(
        signal=signal1,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert i1 is not None

    # Keep same price so fingerprint remains same (rounded_price unchanged),
    # but bump edge by > dedup_edge_delta (default 0.01).
    signal2 = TradingSignal(
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        p_market=0.5,
        p_model=0.7,
        edge=0.2,
        confidence=0.8,
        features={},
    )
    i2 = intent_manager.create_intent_from_signal(
        signal=signal2,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert i2 is None

    # The existing pending intent should reflect the newer edge if it moved materially.
    reloaded = intent_manager.get_intent(i1.intent_id)
    assert reloaded is not None
    assert reloaded.edge == pytest.approx(0.2)


def test_approve_is_idempotent(db_session):
    """Approving twice should not be allowed."""
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert intent is not None

    approved1 = intent_manager.approve_intent(intent.intent_id, approved_by="test_user")
    assert approved1 is not None

    approved2 = intent_manager.approve_intent(intent.intent_id, approved_by="test_user")
    assert approved2 is None


def test_approve_in_live_dry_run_finalizes_to_executed_dryrun(db_session, monkeypatch):
    """In live dry-run, approve should transition to EXECUTED_DRYRUN so it doesn't linger."""
    monkeypatch.setenv("POLYBOT_MODE", "live")
    monkeypatch.setenv("POLYBOT_DRY_RUN", "true")
    from polyb0t.config.settings import get_settings

    get_settings.cache_clear()
    intent_manager = IntentManager(db_session)

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

    intent = intent_manager.create_intent_from_signal(
        signal=signal,
        size=100.0,
        risk_checks={},
        cycle_id="test_cycle",
    )
    assert intent is not None

    approved = intent_manager.approve_intent(intent.intent_id, approved_by="test_user")
    assert approved is not None
    assert approved.status == IntentStatus.EXECUTED_DRYRUN
    assert approved.executed_at is not None
    assert approved.submitted_order_id is not None


def test_backfill_and_cleanup_enforces_one_pending_per_fingerprint(db_session):
    """Legacy rows missing fingerprint should be backfilled and deduped down to one pending."""
    from polyb0t.data.storage import TradeIntentDB

    intent_manager = IntentManager(db_session)

    # Insert two duplicate pending intents with NULL fingerprint (legacy behavior)
    r1 = TradeIntentDB(
        intent_id="legacy-1",
        cycle_id="c1",
        intent_type="OPEN_POSITION",
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        price=0.5,
        size=100.0,
        edge=0.1,
        reason="legacy",
        risk_checks={},
        signal_data={},
        status="PENDING",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(seconds=300),
    )
    r2 = TradeIntentDB(
        intent_id="legacy-2",
        cycle_id="c2",
        intent_type="OPEN_POSITION",
        token_id="token_test",
        market_id="market_test",
        side="BUY",
        price=0.5,
        size=100.0,
        edge=0.1,
        reason="legacy",
        risk_checks={},
        signal_data={},
        status="PENDING",
        created_at=datetime.utcnow() + timedelta(seconds=1),
        expires_at=datetime.utcnow() + timedelta(seconds=300),
    )
    db_session.add(r1)
    db_session.add(r2)
    db_session.commit()

    backfilled = intent_manager.backfill_missing_fingerprints(statuses=[IntentStatus.PENDING])
    assert backfilled >= 2

    summary = intent_manager.cleanup_duplicate_pending_intents(mode="supersede")
    assert summary["kept"] == 1

    # Exactly one pending remains for that fingerprint
    pending = intent_manager.get_pending_intents()
    fps = [p.fingerprint for p in pending if p.market_id == "market_test" and p.token_id == "token_test"]
    assert len(fps) == 1

