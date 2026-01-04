"""Tests for kill switch system."""

from datetime import datetime, timedelta

import pytest

from polyb0t.models.kill_switches import KillSwitchManager, KillSwitchType


def test_drawdown_kill_switch(db_session):
    """Test drawdown kill switch triggering."""
    kill_switch_mgr = KillSwitchManager(db_session)

    peak_equity = 10000.0
    current_equity = 8400.0  # 16% drawdown (exceeds 15% default)

    triggered = kill_switch_mgr._check_drawdown(
        current_equity=current_equity,
        peak_equity=peak_equity,
        cycle_id="test_cycle",
    )

    assert triggered
    assert kill_switch_mgr.is_any_active()
    assert KillSwitchType.DRAWDOWN in kill_switch_mgr.active_switches


def test_drawdown_not_triggered_within_limit(db_session):
    """Test drawdown kill switch does not trigger within limit."""
    kill_switch_mgr = KillSwitchManager(db_session)

    peak_equity = 10000.0
    current_equity = 9000.0  # 10% drawdown (within 15% limit)

    triggered = kill_switch_mgr._check_drawdown(
        current_equity=current_equity,
        peak_equity=peak_equity,
        cycle_id="test_cycle",
    )

    assert not triggered
    assert not kill_switch_mgr.is_any_active()


def test_api_error_rate_kill_switch(db_session):
    """Test API error rate kill switch."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Record API calls with high error rate
    for i in range(20):
        success = i % 2 == 0  # 50% error rate
        kill_switch_mgr.record_api_call(success)

    triggered = kill_switch_mgr._check_api_error_rate(cycle_id="test_cycle")

    # Should trigger at 50% (default threshold)
    assert triggered
    assert KillSwitchType.API_ERROR_RATE in kill_switch_mgr.active_switches


def test_api_error_rate_not_triggered_low_errors(db_session):
    """Test API error rate does not trigger with low errors."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Record mostly successful calls
    for i in range(20):
        success = i < 18  # Only 2 failures (10% error rate)
        kill_switch_mgr.record_api_call(success)

    triggered = kill_switch_mgr._check_api_error_rate(cycle_id="test_cycle")

    assert not triggered


def test_stale_data_kill_switch(db_session):
    """Test stale data kill switch."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Set last data timestamp to old time
    kill_switch_mgr.last_data_timestamp = datetime.utcnow() - timedelta(seconds=120)

    triggered = kill_switch_mgr._check_stale_data(cycle_id="test_cycle")

    # Should trigger (120s > 60s default threshold)
    assert triggered
    assert KillSwitchType.STALE_DATA in kill_switch_mgr.active_switches


def test_stale_data_not_triggered_recent(db_session):
    """Test stale data does not trigger with recent data."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Set last data timestamp to recent time
    kill_switch_mgr.last_data_timestamp = datetime.utcnow() - timedelta(seconds=30)

    triggered = kill_switch_mgr._check_stale_data(cycle_id="test_cycle")

    assert not triggered


def test_spread_anomaly_kill_switch(db_session):
    """Test spread anomaly kill switch."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Normal max spread is 0.05, multiplier is 3x = 0.15
    # Provide spreads exceeding this
    current_spreads = {
        "token_1": 0.20,  # Anomaly
        "token_2": 0.03,  # Normal
    }

    triggered = kill_switch_mgr._check_spread_anomaly(
        current_spreads=current_spreads,
        cycle_id="test_cycle",
    )

    assert triggered
    assert KillSwitchType.SPREAD_ANOMALY in kill_switch_mgr.active_switches


def test_spread_anomaly_not_triggered_normal_spreads(db_session):
    """Test spread anomaly does not trigger with normal spreads."""
    kill_switch_mgr = KillSwitchManager(db_session)

    current_spreads = {
        "token_1": 0.03,
        "token_2": 0.04,
    }

    triggered = kill_switch_mgr._check_spread_anomaly(
        current_spreads=current_spreads,
        cycle_id="test_cycle",
    )

    assert not triggered


def test_daily_loss_kill_switch(db_session):
    """Test daily loss kill switch."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Set daily start equity
    kill_switch_mgr.daily_start_equity = 10000.0
    kill_switch_mgr.daily_start_date = datetime.utcnow()

    # Current equity shows 11% loss (exceeds 10% default)
    current_equity = 8900.0

    triggered = kill_switch_mgr._check_daily_loss(
        current_equity=current_equity,
        cycle_id="test_cycle",
    )

    assert triggered
    assert KillSwitchType.DAILY_LOSS in kill_switch_mgr.active_switches


def test_daily_loss_resets_new_day(db_session):
    """Test daily loss tracking resets on new day."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Set daily start to yesterday
    kill_switch_mgr.daily_start_equity = 10000.0
    kill_switch_mgr.daily_start_date = datetime.utcnow() - timedelta(days=1)

    # Check with lower equity
    current_equity = 8900.0

    triggered = kill_switch_mgr._check_daily_loss(
        current_equity=current_equity,
        cycle_id="test_cycle",
    )

    # Should not trigger because it resets daily start equity to current
    assert not triggered
    assert kill_switch_mgr.daily_start_equity == current_equity


def test_clear_kill_switch(db_session):
    """Test clearing a kill switch."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Trigger a switch
    kill_switch_mgr._trigger_switch(
        switch_type=KillSwitchType.DRAWDOWN,
        trigger_value=16.0,
        threshold_value=15.0,
        description="Test trigger",
        cycle_id="test_cycle",
    )

    assert kill_switch_mgr.is_any_active()

    # Clear it
    cleared = kill_switch_mgr.clear_switch(KillSwitchType.DRAWDOWN)

    assert cleared
    assert not kill_switch_mgr.is_any_active()


def test_clear_all_switches(db_session):
    """Test clearing all kill switches."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Trigger multiple switches
    kill_switch_mgr._trigger_switch(
        switch_type=KillSwitchType.DRAWDOWN,
        trigger_value=16.0,
        threshold_value=15.0,
        description="Test 1",
        cycle_id="test_cycle",
    )

    kill_switch_mgr._trigger_switch(
        switch_type=KillSwitchType.DAILY_LOSS,
        trigger_value=11.0,
        threshold_value=10.0,
        description="Test 2",
        cycle_id="test_cycle",
    )

    assert len(kill_switch_mgr.active_switches) == 2

    # Clear all
    count = kill_switch_mgr.clear_all_switches()

    assert count == 2
    assert not kill_switch_mgr.is_any_active()


def test_check_all_switches(db_session):
    """Test checking all kill switches at once."""
    kill_switch_mgr = KillSwitchManager(db_session)

    # Set up conditions for multiple triggers
    kill_switch_mgr.daily_start_equity = 10000.0
    kill_switch_mgr.daily_start_date = datetime.utcnow()

    # Record high error rate
    for i in range(20):
        kill_switch_mgr.record_api_call(i % 2 == 0)

    peak_equity = 10000.0
    current_equity = 8400.0  # 16% drawdown
    current_spreads = {"token_1": 0.20}  # High spread

    triggered = kill_switch_mgr.check_all_switches(
        current_equity=current_equity,
        peak_equity=peak_equity,
        current_spreads=current_spreads,
        cycle_id="test_cycle",
    )

    # Should trigger multiple switches
    assert len(triggered) > 1
    assert KillSwitchType.DRAWDOWN in triggered
    assert KillSwitchType.SPREAD_ANOMALY in triggered

