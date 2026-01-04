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


