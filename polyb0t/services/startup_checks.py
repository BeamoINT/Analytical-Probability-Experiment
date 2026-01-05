"""Startup self-checks for reliability on clean machines."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from sqlalchemy import text

from polyb0t.config import Settings
from polyb0t.data.storage import get_session, init_db


def redact_db_url(db_url: str) -> str:
    """Redact credentials in DB URL for safe printing."""
    # redact user:pass@
    return re.sub(r"://([^:/]+):([^@]+)@", r"://\\1:***@", db_url)


def startup_banner(settings: Settings) -> str:
    """Create a safe startup banner (no secrets)."""
    return (
        "\n"
        "PolyB0T Startup\n"
        "============================================================\n"
        f"Mode:           {settings.mode}\n"
        f"Dry run:        {settings.dry_run}\n"
        f"Loop interval:  {settings.loop_interval_seconds}s\n"
        f"DB URL:         {redact_db_url(settings.db_url)}\n"
        "Clock:          UTC\n"
        "============================================================\n"
    )


def validate_clock_utc(max_skew_seconds: float = 5.0) -> None:
    """Validate system clock looks sane and UTC is available."""
    now_utc = datetime.now(timezone.utc).timestamp()
    now_epoch = time.time()
    skew = abs(now_utc - now_epoch)
    if skew > max_skew_seconds:
        raise RuntimeError(
            f"System clock skew too high ({skew:.1f}s). Please sync your system clock."
        )


def validate_db_connectivity(db_url: str) -> None:
    """Validate DB connectivity and schema availability."""
    init_db(db_url)
    session = get_session()
    try:
        session.execute(text("SELECT 1"))
    finally:
        session.close()


def validate_live_signing_key(settings: Settings) -> None:
    """Validate live execution signing key matches configured address.

    When `mode=live` and `dry_run=false`, orders are signed. If the configured
    private key does not correspond to the configured address, the CLOB will
    reject orders with an "invalid signature" style error.
    """
    if settings.mode != "live" or bool(settings.dry_run):
        return

    if not settings.polygon_private_key:
        raise RuntimeError(
            "Missing POLYBOT_POLYGON_PRIVATE_KEY (required when POLYBOT_DRY_RUN=false)."
        )

    try:
        from eth_account import Account

        derived = Account.from_key(settings.polygon_private_key).address
    except Exception:
        raise RuntimeError("Invalid POLYBOT_POLYGON_PRIVATE_KEY format.")

    expected = settings.funder_address or settings.user_address
    if not expected:
        raise RuntimeError("Missing POLYBOT_USER_ADDRESS (and/or POLYBOT_FUNDER_ADDRESS).")

    if derived.lower() != expected.lower():
        raise RuntimeError(
            "POLYBOT_POLYGON_PRIVATE_KEY does not match your configured address. "
            "Fix by either: (1) setting POLYBOT_USER_ADDRESS/POLYBOT_FUNDER_ADDRESS to the "
            "address for that private key, or (2) setting POLYBOT_POLYGON_PRIVATE_KEY for the "
            "configured address."
        )


