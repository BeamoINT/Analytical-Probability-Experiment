"""Trade intent system for human-in-the-loop approval."""

import logging
import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.storage import TradeIntentDB
from polyb0t.models.strategy_baseline import TradingSignal

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Trade intent types."""

    OPEN_POSITION = "OPEN_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    CLAIM_SETTLEMENT = "CLAIM_SETTLEMENT"
    CANCEL_ORDER = "CANCEL_ORDER"


class IntentStatus(str, Enum):
    """Intent status values."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXECUTED = "EXECUTED"
    EXECUTED_DRYRUN = "EXECUTED_DRYRUN"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class TradeIntent:
    """Represents a proposed trade awaiting approval."""

    def __init__(
        self,
        intent_id: str,
        intent_type: IntentType,
        token_id: str,
        market_id: str | None,
        side: str | None,
        price: float | None,
        size_usd: float | None,
        edge: float | None,
        p_market: float | None,
        p_model: float | None,
        reason: str,
        risk_checks: dict[str, Any],
        signal_data: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        fingerprint: str | None = None,
    ) -> None:
        """Initialize trade intent.

        Args:
            intent_id: Unique intent identifier.
            intent_type: Type of intent.
            token_id: Token identifier.
            market_id: Market condition ID.
            side: BUY or SELL.
            price: Proposed price.
            size: Proposed size.
            edge: Expected edge.
            reason: Human-readable reason for trade.
            risk_checks: Risk check results.
            signal_data: Full signal data.
            expires_at: Expiration timestamp.
        """
        settings = get_settings()

        self.intent_id = intent_id
        self.intent_type = intent_type
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.price = price
        self.size_usd = size_usd
        self.edge = edge
        self.p_market = p_market
        self.p_model = p_model
        self.reason = reason
        self.risk_checks = risk_checks
        self.signal_data = signal_data or {}
        self.status = IntentStatus.PENDING
        self.created_at = datetime.utcnow()
        self.expires_at = expires_at or (
            self.created_at + timedelta(seconds=settings.intent_expiry_seconds)
        )
        self.fingerprint = fingerprint
        self.approved_at: datetime | None = None
        self.approved_by: str | None = None
        self.executed_at: datetime | None = None
        self.execution_result: dict[str, Any] | None = None
        self.submitted_order_id: str | None = None
        self.error_message: str | None = None
        self.superseded_by_intent_id: str | None = None
        self.superseded_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if intent has expired."""
        return datetime.utcnow() > self.expires_at

    def is_executable(self) -> bool:
        """Check if intent can be executed."""
        return (
            self.status == IntentStatus.APPROVED
            and not self.is_expired()
            and self.executed_at is None
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_id": self.intent_id,
            "intent_type": self.intent_type.value,
            "token_id": self.token_id,
            "market_id": self.market_id,
            "side": self.side,
            "price": self.price,
            "size_usd": self.size_usd,
            "edge": self.edge,
            "p_market": self.p_market,
            "p_model": self.p_model,
            "reason": self.reason,
            "risk_checks": self.risk_checks,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "fingerprint": self.fingerprint,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "submitted_order_id": self.submitted_order_id,
            "error_message": self.error_message,
            "seconds_until_expiry": max(
                0, (self.expires_at - datetime.utcnow()).total_seconds()
            ),
        }


class IntentManager:
    """Manages trade intents and approval workflow."""

    def __init__(self, db_session: Session) -> None:
        """Initialize intent manager.

        Args:
            db_session: Database session.
        """
        self.db_session = db_session
        self.settings = get_settings()

    def create_intent_from_signal(
        self,
        signal: TradingSignal,
        size: float,
        risk_checks: dict[str, Any],
        cycle_id: str,
    ) -> TradeIntent | None:
        """Create trade intent from trading signal.

        Args:
            signal: Trading signal.
            size: Position size.
            risk_checks: Risk check results.
            cycle_id: Current cycle ID.

        Returns:
            Created TradeIntent, or None if skipped due to dedup/cooldown.
        """
        intent_id = str(uuid.uuid4())

        reason = (
            f"{signal.side} {signal.token_id[:12]}... @ {signal.p_market:.3f} "
            f"(model: {signal.p_model:.3f}, edge: {signal.edge:+.3f})"
        )

        fingerprint = self._fingerprint_intent(
            intent_type=IntentType.OPEN_POSITION,
            market_id=signal.market_id,
            token_id=signal.token_id,
            side=signal.side,
            price=signal.p_market,
            size_usd=size,
        )
        skip_reason = self._should_skip_due_to_dedup(
            fingerprint=fingerprint,
            price=signal.p_market,
            edge=signal.edge,
        )
        if skip_reason is not None:
            if skip_reason == "existing_pending":
                self._maybe_update_existing_pending(
                    fingerprint=fingerprint,
                    new_price=signal.p_market,
                    new_edge=signal.edge,
                    new_p_model=signal.p_model,
                    new_size_usd=size,
                )
            logger.debug(
                "Skipped intent due to dedup/cooldown",
                extra={
                    "fingerprint": fingerprint,
                    "reason": skip_reason,
                    "token_id": signal.token_id,
                    "market_id": signal.market_id,
                },
            )
            return None

        intent = TradeIntent(
            intent_id=intent_id,
            intent_type=IntentType.OPEN_POSITION,
            token_id=signal.token_id,
            market_id=signal.market_id,
            side=signal.side,
            price=signal.p_market,
            size_usd=size,
            edge=signal.edge,
            p_market=signal.p_market,
            p_model=signal.p_model,
            reason=reason,
            risk_checks=risk_checks,
            signal_data={
                "p_market": signal.p_market,
                "p_model": signal.p_model,
                "edge": signal.edge,
                "confidence": signal.confidence,
                "features": signal.features,
            },
            fingerprint=fingerprint,
        )

        self._persist_intent(intent, cycle_id)

        logger.info(
            f"Created trade intent: {intent.intent_id[:8]}",
            extra={
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type.value,
                "side": signal.side,
                "edge": signal.edge,
                "fingerprint": intent.fingerprint,
            },
        )

        return intent

    def create_exit_intent(
        self,
        token_id: str,
        market_id: str,
        side: str,
        price: float,
        size: float,
        reason: str,
        cycle_id: str,
    ) -> TradeIntent | None:
        """Create exit intent.

        Args:
            token_id: Token identifier.
            market_id: Market ID.
            side: SELL (to close long) or BUY (to close short).
            price: Exit price.
            size: Position size to close.
            reason: Exit reason (take-profit, stop-loss, time-based).
            cycle_id: Current cycle ID.

        Returns:
            Created TradeIntent, or None if skipped due to dedup/cooldown.
        """
        intent_id = str(uuid.uuid4())

        fingerprint = self._fingerprint_intent(
            intent_type=IntentType.CLOSE_POSITION,
            market_id=market_id,
            token_id=token_id,
            side=side,
            price=price,
            size_usd=size,
        )
        skip_reason = self._should_skip_due_to_dedup(
            fingerprint=fingerprint,
            price=price,
            edge=None,
        )
        if skip_reason is not None:
            if skip_reason == "existing_pending":
                self._maybe_update_existing_pending(
                    fingerprint=fingerprint,
                    new_price=price,
                    new_edge=None,
                    new_p_model=None,
                    new_size_usd=size,
                )
            return None

        intent = TradeIntent(
            intent_id=intent_id,
            intent_type=IntentType.CLOSE_POSITION,
            token_id=token_id,
            market_id=market_id,
            side=side,
            price=price,
            size_usd=size,
            edge=None,
            p_market=price,
            p_model=None,
            reason=reason,
            risk_checks={"exit_intent": True},
            fingerprint=fingerprint,
        )

        self._persist_intent(intent, cycle_id)

        logger.info(
            f"Created exit intent: {intent.intent_id[:8]}",
            extra={"intent_id": intent.intent_id, "reason": reason},
        )

        return intent

    def create_claim_intent(
        self,
        token_id: str,
        market_id: str,
        reason: str,
        cycle_id: str,
    ) -> TradeIntent:
        """Create settlement claim intent.

        Args:
            token_id: Token identifier.
            market_id: Market ID.
            reason: Claim reason.
            cycle_id: Current cycle ID.

        Returns:
            Created TradeIntent.
        """
        intent_id = str(uuid.uuid4())

        intent = TradeIntent(
            intent_id=intent_id,
            intent_type=IntentType.CLAIM_SETTLEMENT,
            token_id=token_id,
            market_id=market_id,
            side=None,
            price=None,
            size_usd=None,
            edge=None,
            p_market=None,
            p_model=None,
            reason=reason,
            risk_checks={"claim_intent": True},
        )

        self._persist_intent(intent, cycle_id)

        logger.info(
            f"Created claim intent: {intent.intent_id[:8]}",
            extra={"intent_id": intent.intent_id, "market_id": market_id},
        )

        return intent

    def create_cancel_order_intent(
        self,
        token_id: str,
        market_id: str | None,
        order_id: str,
        reason: str,
        cycle_id: str,
    ) -> TradeIntent:
        """Create a cancel order intent (approval-gated)."""
        intent_id = str(uuid.uuid4())
        fingerprint = self._fingerprint_intent(
            intent_type=IntentType.CANCEL_ORDER,
            market_id=market_id,
            token_id=token_id,
            side="CANCEL",
            price=None,
            size_usd=None,
        )
        intent = TradeIntent(
            intent_id=intent_id,
            intent_type=IntentType.CANCEL_ORDER,
            token_id=token_id,
            market_id=market_id,
            side="CANCEL",
            price=None,
            size_usd=None,
            edge=None,
            p_market=None,
            p_model=None,
            reason=reason,
            risk_checks={"cancel_intent": True},
            signal_data={"order_id": order_id},
            fingerprint=fingerprint,
        )
        self._persist_intent(intent, cycle_id)
        logger.info(
            f"Created cancel intent: {intent.intent_id[:8]}",
            extra={"intent_id": intent.intent_id, "order_id": order_id},
        )
        return intent

    def _persist_intent(self, intent: TradeIntent, cycle_id: str) -> None:
        """Persist intent to database.

        Args:
            intent: Trade intent to persist.
            cycle_id: Current cycle ID.
        """
        db_intent = TradeIntentDB(
            intent_uuid=intent.intent_id if len(intent.intent_id) <= 36 else None,
            intent_id=intent.intent_id,
            cycle_id=cycle_id,
            intent_type=intent.intent_type.value,
            fingerprint=intent.fingerprint,
            token_id=intent.token_id,
            market_id=intent.market_id,
            side=intent.side,
            price=intent.price,
            size=intent.size_usd,
            size_usd=intent.size_usd,
            edge=intent.edge,
            p_market=intent.p_market,
            p_model=intent.p_model,
            reason=intent.reason,
            risk_checks=intent.risk_checks,
            signal_data=intent.signal_data,
            status=intent.status.value,
            created_at=intent.created_at,
            expires_at=intent.expires_at,
        )
        self.db_session.add(db_intent)
        self.db_session.commit()

    def approve_intent(
        self, intent_id: str, approved_by: str = "user"
    ) -> TradeIntent | None:
        """Approve a trade intent.

        Args:
            intent_id: Intent ID to approve.
            approved_by: User identifier.

        Returns:
            Approved intent or None if not found/expired.
        """
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )

        if not db_intent:
            logger.warning(f"Intent not found: {intent_id}")
            return None

        intent = self._load_intent_from_db(db_intent)

        if intent.is_expired():
            intent.status = IntentStatus.EXPIRED
            db_intent.status = IntentStatus.EXPIRED.value
            self.db_session.commit()
            logger.warning(f"Intent expired: {intent_id}")
            return None

        if intent.status != IntentStatus.PENDING:
            logger.warning(
                f"Intent {intent_id} cannot be approved (status: {intent.status.value})"
            )
            return None

        intent.status = IntentStatus.APPROVED
        intent.approved_at = datetime.utcnow()
        intent.approved_by = approved_by

        db_intent.status = IntentStatus.APPROVED.value
        db_intent.approved_at = intent.approved_at
        db_intent.approved_by = approved_by
        self.db_session.commit()

        # In dry-run, approvals should never place real orders; we finalize to a terminal state
        # so the intent doesn't keep reappearing in "approved" queues.
        if self.settings.mode == "live" and self.settings.dry_run:
            now = datetime.utcnow()
            db_intent.status = IntentStatus.EXECUTED_DRYRUN.value
            db_intent.executed_at = now
            db_intent.execution_result = {
                "dry_run": True,
                "success": True,
                "message": "Approved in dry-run; no order was submitted",
            }
            db_intent.submitted_order_id = f"dryrun:{intent_id[:8]}"
            self.db_session.commit()
            intent.status = IntentStatus.EXECUTED_DRYRUN
            intent.executed_at = now
            intent.execution_result = db_intent.execution_result
            intent.submitted_order_id = db_intent.submitted_order_id

        logger.info(
            f"Approved intent: {intent_id[:8]}",
            extra={"intent_id": intent_id, "approved_by": approved_by},
        )

        return intent

    def reject_intent(self, intent_id: str) -> bool:
        """Reject a trade intent.

        Args:
            intent_id: Intent ID to reject.

        Returns:
            True if rejected successfully.
        """
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )

        if not db_intent:
            logger.warning(f"Intent not found: {intent_id}")
            return False

        if db_intent.status != IntentStatus.PENDING.value:
            logger.warning(f"Intent {intent_id} cannot be rejected (status: {db_intent.status})")
            return False

        db_intent.status = IntentStatus.REJECTED.value
        self.db_session.commit()

        logger.info(f"Rejected intent: {intent_id[:8]}", extra={"intent_id": intent_id})

        return True

    def mark_executed(
        self, intent_id: str, execution_result: dict[str, Any]
    ) -> bool:
        """Mark intent as executed.

        Args:
            intent_id: Intent ID.
            execution_result: Execution result data.

        Returns:
            True if marked successfully.
        """
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )

        if not db_intent:
            return False

        if db_intent.status in (
            IntentStatus.EXECUTED.value,
            IntentStatus.EXECUTED_DRYRUN.value,
            IntentStatus.FAILED.value,
            IntentStatus.REJECTED.value,
            IntentStatus.EXPIRED.value,
            IntentStatus.SUPERSEDED.value,
        ):
            return False

        db_intent.status = IntentStatus.EXECUTED.value
        db_intent.executed_at = datetime.utcnow()
        db_intent.execution_result = execution_result
        if isinstance(execution_result, dict):
            oid = execution_result.get("order_id") or execution_result.get("submitted_order_id")
            if oid:
                db_intent.submitted_order_id = str(oid)
        self.db_session.commit()

        logger.info(
            f"Marked intent as executed: {intent_id[:8]}",
            extra={"intent_id": intent_id, "result": execution_result},
        )

        return True

    def mark_executed_dryrun(self, intent_id: str, note: str = "Dry-run: no order submitted") -> bool:
        """Mark intent as executed in dry-run (terminal, but no funds moved)."""
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )
        if not db_intent:
            return False

        if db_intent.status in (
            IntentStatus.EXECUTED.value,
            IntentStatus.EXECUTED_DRYRUN.value,
            IntentStatus.FAILED.value,
            IntentStatus.REJECTED.value,
            IntentStatus.EXPIRED.value,
            IntentStatus.SUPERSEDED.value,
        ):
            return False

        db_intent.status = IntentStatus.EXECUTED_DRYRUN.value
        db_intent.executed_at = datetime.utcnow()
        db_intent.execution_result = {"dry_run": True, "success": True, "message": note}
        db_intent.submitted_order_id = f"dryrun:{intent_id[:8]}"
        self.db_session.commit()
        return True

    def mark_failed(self, intent_id: str, error: str) -> bool:
        """Mark intent as failed.

        Args:
            intent_id: Intent ID.
            error: Error message.

        Returns:
            True if marked successfully.
        """
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )

        if not db_intent:
            return False

        if db_intent.status in (
            IntentStatus.EXECUTED.value,
            IntentStatus.EXECUTED_DRYRUN.value,
            IntentStatus.FAILED.value,
            IntentStatus.REJECTED.value,
            IntentStatus.EXPIRED.value,
            IntentStatus.SUPERSEDED.value,
        ):
            return False

        db_intent.status = IntentStatus.FAILED.value
        db_intent.execution_result = {"error": error}
        db_intent.error_message = error
        self.db_session.commit()

        logger.error(
            f"Marked intent as failed: {intent_id[:8]}",
            extra={"intent_id": intent_id, "error": error},
        )

        return True

    def expire_old_intents(self) -> int:
        """Expire old pending intents.

        Returns:
            Number of intents expired.
        """
        now = datetime.utcnow()
        expired = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.expires_at < now)
            .all()
        )

        count = 0
        for db_intent in expired:
            db_intent.status = IntentStatus.EXPIRED.value
            count += 1

        if count > 0:
            self.db_session.commit()
            logger.info(f"Expired {count} old intents")

        return count

    def expire_stale_open_intents(self, active_fingerprints: set[str]) -> int:
        """Expire pending OPEN_POSITION intents no longer backed by current signals.

        Args:
            active_fingerprints: Fingerprints for current-cycle signals.

        Returns:
            Number of intents expired.
        """
        now = datetime.utcnow()
        pending = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.intent_type == IntentType.OPEN_POSITION.value)
            .filter(TradeIntentDB.expires_at >= now)
            .all()
        )

        count = 0
        for db_intent in pending:
            fp = db_intent.fingerprint
            if not fp or fp not in active_fingerprints:
                db_intent.status = IntentStatus.EXPIRED.value
                count += 1

        if count:
            self.db_session.commit()
            logger.info(f"Expired {count} stale open intents")

        return count

    def backfill_missing_fingerprints(
        self,
        statuses: list[IntentStatus] | None = None,
        limit: int = 5000,
    ) -> int:
        """Backfill deterministic fingerprints for legacy intents missing them.

        This is required because older DB rows may have NULL `fingerprint`, which breaks
        DB-backed deduplication. This method is additive and safe to run repeatedly.
        """
        q = self.db_session.query(TradeIntentDB).filter(TradeIntentDB.fingerprint.is_(None))
        if statuses:
            q = q.filter(TradeIntentDB.status.in_([s.value for s in statuses]))
        rows = q.order_by(TradeIntentDB.created_at.desc()).limit(limit).all()

        updated = 0
        for r in rows:
            try:
                size_usd = r.size_usd if getattr(r, "size_usd", None) is not None else r.size
                fp = self._fingerprint_intent(
                    intent_type=IntentType(r.intent_type),
                    market_id=r.market_id,
                    token_id=r.token_id,
                    side=r.side,
                    price=r.price,
                    size_usd=size_usd,
                )
                r.fingerprint = fp
                if getattr(r, "size_usd", None) is None and size_usd is not None:
                    r.size_usd = float(size_usd)
                if getattr(r, "p_market", None) is None and r.price is not None:
                    r.p_market = float(r.price)
                updated += 1
            except Exception:
                continue

        if updated:
            self.db_session.commit()
        return updated

    def cleanup_duplicate_pending_intents(self, mode: str = "supersede") -> dict[str, int]:
        """Ensure at most ONE PENDING intent exists per fingerprint.

        Args:
            mode: 'supersede' (default) marks duplicates SUPERSEDED, or 'expire' marks EXPIRED.

        Returns:
            Dict with counts: scanned, deduped, kept, expired_null_fp.
        """
        now = datetime.utcnow()
        # Backfill fingerprints for PENDING so we can group deterministically.
        self.backfill_missing_fingerprints(statuses=[IntentStatus.PENDING])

        pending = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.expires_at >= now)
            .order_by(TradeIntentDB.created_at.desc())
            .all()
        )

        by_fp: dict[str, list[TradeIntentDB]] = {}
        null_fp = 0
        for r in pending:
            fp = r.fingerprint
            if not fp:
                null_fp += 1
                continue
            by_fp.setdefault(fp, []).append(r)

        deduped = 0
        kept = 0
        for fp, rows in by_fp.items():
            # rows already in created_at desc order from query ordering
            if not rows:
                continue
            keep = rows[0]
            kept += 1
            for dup in rows[1:]:
                if mode == "expire":
                    dup.status = IntentStatus.EXPIRED.value
                else:
                    dup.status = IntentStatus.SUPERSEDED.value
                    dup.superseded_by_intent_id = keep.intent_id
                    dup.superseded_at = now
                deduped += 1

        # If we still have null fingerprints (should be rare), expire them to stop spam.
        expired_null_fp = 0
        if null_fp:
            rows = (
                self.db_session.query(TradeIntentDB)
                .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
                .filter(TradeIntentDB.fingerprint.is_(None))
                .all()
            )
            for r in rows:
                r.status = IntentStatus.EXPIRED.value
                expired_null_fp += 1

        if deduped or expired_null_fp:
            self.db_session.commit()

        return {
            "scanned": len(pending),
            "kept": kept,
            "deduped": deduped,
            "expired_null_fp": expired_null_fp,
        }

    def get_pending_intents(self) -> list[TradeIntent]:
        """Get all pending intents.

        Returns:
            List of pending intents.
        """
        # Keep CLI/views clean and enforce lifecycle before returning data.
        self.expire_old_intents()
        self.backfill_missing_fingerprints(statuses=[IntentStatus.PENDING])
        self.cleanup_duplicate_pending_intents(mode="supersede")

        db_intents = (
            self.db_session.query(TradeIntentDB)
            .filter_by(status=IntentStatus.PENDING.value)
            .order_by(TradeIntentDB.created_at.desc())
            .all()
        )

        intents = []
        for db_intent in db_intents:
            intent = self._load_intent_from_db(db_intent)
            if not intent.is_expired():
                intents.append(intent)

        return intents

    def get_approved_intents(self) -> list[TradeIntent]:
        """Get all approved intents ready for execution.

        Returns:
            List of approved, non-expired intents.
        """
        db_intents = (
            self.db_session.query(TradeIntentDB)
            .filter_by(status=IntentStatus.APPROVED.value)
            .filter(TradeIntentDB.executed_at.is_(None))
            .order_by(TradeIntentDB.approved_at)
            .all()
        )

        intents = []
        for db_intent in db_intents:
            intent = self._load_intent_from_db(db_intent)
            if intent.is_executable():
                intents.append(intent)

        return intents

    def get_intent(self, intent_id: str) -> TradeIntent | None:
        """Get intent by ID.

        Args:
            intent_id: Intent ID.

        Returns:
            TradeIntent or None.
        """
        db_intent = (
            self.db_session.query(TradeIntentDB).filter_by(intent_id=intent_id).first()
        )

        if not db_intent:
            return None

        return self._load_intent_from_db(db_intent)

    def _load_intent_from_db(self, db_intent: TradeIntentDB) -> TradeIntent:
        """Load TradeIntent from database record.

        Args:
            db_intent: Database record.

        Returns:
            TradeIntent object.
        """
        intent = TradeIntent(
            intent_id=db_intent.intent_id,
            intent_type=IntentType(db_intent.intent_type),
            token_id=db_intent.token_id,
            market_id=db_intent.market_id,
            side=db_intent.side,
            price=db_intent.price,
            size_usd=db_intent.size_usd if db_intent.size_usd is not None else db_intent.size,
            edge=db_intent.edge,
            p_market=db_intent.p_market if db_intent.p_market is not None else db_intent.price,
            p_model=db_intent.p_model,
            reason=db_intent.reason or "",
            risk_checks=db_intent.risk_checks or {},
            signal_data=db_intent.signal_data,
            expires_at=db_intent.expires_at,
            fingerprint=db_intent.fingerprint,
        )

        intent.status = IntentStatus(db_intent.status)
        intent.created_at = db_intent.created_at
        intent.approved_at = db_intent.approved_at
        intent.approved_by = db_intent.approved_by
        intent.executed_at = db_intent.executed_at
        intent.execution_result = db_intent.execution_result
        intent.submitted_order_id = getattr(db_intent, "submitted_order_id", None)
        intent.error_message = getattr(db_intent, "error_message", None)
        intent.superseded_by_intent_id = getattr(db_intent, "superseded_by_intent_id", None)
        intent.superseded_at = getattr(db_intent, "superseded_at", None)

        return intent

    def _fingerprint_intent(
        self,
        intent_type: IntentType,
        market_id: str | None,
        token_id: str,
        side: str | None,
        price: float | None,
        size_usd: float | None,
    ) -> str:
        """Compute deterministic fingerprint for intent deduplication."""
        pr = self.settings.intent_price_round
        sb = self.settings.intent_size_bucket_usd
        rounded_price: float | None = None
        if price is not None and pr > 0:
            rounded_price = round(price / pr) * pr
        size_bucket: float | None = None
        if size_usd is not None and sb > 0:
            size_bucket = round(size_usd / sb) * sb
        base = "|".join(
            [
                intent_type.value,
                str(market_id or ""),
                str(token_id or ""),
                str(side or ""),
                "" if rounded_price is None else f"{rounded_price:.6f}",
                "" if size_bucket is None else f"{size_bucket:.2f}",
            ]
        )
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _should_skip_due_to_dedup(
        self, fingerprint: str, price: float | None, edge: float | None
    ) -> str | None:
        """Return skip reason if this intent should be suppressed; otherwise None."""
        now = datetime.utcnow()

        # Strong dedup: never create duplicates when there's already a live pending intent.
        pending = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.fingerprint == fingerprint)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.expires_at >= now)
            .first()
        )
        if pending is not None:
            return "existing_pending"

        cooldown = timedelta(seconds=self.settings.intent_cooldown_seconds)
        # Look for the most recent intent (any status) with the same fingerprint and enforce cooldown.
        recent = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.fingerprint == fingerprint)
            .filter(TradeIntentDB.created_at >= (now - cooldown))
            .order_by(TradeIntentDB.created_at.desc())
            .first()
        )
        if not recent:
            return None

        # If the underlying values materially changed, allow a new intent.
        if price is not None and recent.price is not None:
            if abs(price - float(recent.price)) >= self.settings.dedup_price_delta:
                return None
        if edge is not None and recent.edge is not None:
            if abs(edge - float(recent.edge)) >= self.settings.dedup_edge_delta:
                return None

        return "cooldown"

    def _maybe_update_existing_pending(
        self,
        fingerprint: str,
        new_price: float | None,
        new_edge: float | None,
        new_p_model: float | None,
        new_size_usd: float | None,
    ) -> bool:
        """Optionally update the existing pending intent's price/edge if materially changed.

        This keeps the invariant: <= 1 PENDING per fingerprint, while still reflecting latest signal quality.
        """
        now = datetime.utcnow()
        existing = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.fingerprint == fingerprint)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.expires_at >= now)
            .order_by(TradeIntentDB.created_at.desc())
            .first()
        )
        if not existing:
            return False

        updated = False
        if new_price is not None and existing.price is not None:
            if abs(float(existing.price) - float(new_price)) >= self.settings.dedup_price_delta:
                existing.price = float(new_price)
                existing.p_market = float(new_price)
                updated = True
        if new_edge is not None and existing.edge is not None:
            if abs(float(existing.edge) - float(new_edge)) >= self.settings.dedup_edge_delta:
                existing.edge = float(new_edge)
                updated = True
        if new_p_model is not None and getattr(existing, "p_model", None) is not None:
            # Update p_model whenever we update edge, or if it differs meaningfully.
            if updated or abs(float(existing.p_model) - float(new_p_model)) >= self.settings.dedup_edge_delta:
                existing.p_model = float(new_p_model)
                updated = True
        if new_size_usd is not None:
            existing_size = (
                float(existing.size_usd)
                if getattr(existing, "size_usd", None) is not None
                else float(existing.size)
                if getattr(existing, "size", None) is not None
                else None
            )
            if existing_size is None or abs(existing_size - float(new_size_usd)) >= 0.01:
                existing.size = float(new_size_usd)
                existing.size_usd = float(new_size_usd)
                updated = True

        if updated:
            self.db_session.commit()
        return updated
