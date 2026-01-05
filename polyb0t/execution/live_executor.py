"""Live trade executor with approval gate (human-in-the-loop)."""

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.storage import SimulatedFillDB, SimulatedOrderDB
from polyb0t.execution.intents import IntentManager, IntentType, TradeIntent
from polyb0t.execution.orders import OrderSide, OrderType

logger = logging.getLogger(__name__)


class LiveExecutor:
    """Executes approved trade intents in live mode.

    SAFETY: This executor ONLY executes intents that have been explicitly
    approved by the user. No autonomous trading occurs.
    """

    def __init__(self, db_session: Session, intent_manager: IntentManager) -> None:
        """Initialize live executor.

        Args:
            db_session: Database session.
            intent_manager: Intent manager.
        """
        self.db_session = db_session
        self.intent_manager = intent_manager
        self.settings = get_settings()

        # Validate we're in live mode
        if self.settings.mode != "live":
            raise ValueError(
                "LiveExecutor can only be used in live mode. "
                "Set POLYBOT_MODE=live"
            )

        # Check if dry-run mode
        self.dry_run = self.settings.dry_run

        if self.dry_run:
            logger.warning(
                "LiveExecutor in DRY-RUN mode: intents will be logged but NOT executed"
            )
        else:
            logger.warning(
                "LiveExecutor in LIVE mode: approved intents WILL be executed with real orders"
            )

    def process_approved_intents(self, cycle_id: str) -> dict[str, Any]:
        """Process all approved intents awaiting execution.

        Args:
            cycle_id: Current cycle ID.

        Returns:
            Execution summary.
        """
        approved_intents = self.intent_manager.get_approved_intents()

        if not approved_intents:
            return {
                "processed": 0,
                "executed": 0,
                "failed": 0,
                "dry_run": self.dry_run,
            }

        executed = 0
        failed = 0

        for intent in approved_intents:
            try:
                if self.dry_run:
                    # Dry-run: log but don't execute
                    self._log_dry_run_execution(intent)
                    self.intent_manager.mark_executed_dryrun(
                        intent.intent_id,
                        note="Logged but not executed (dry-run mode)",
                    )
                    executed += 1
                else:
                    # Live execution
                    result = self._execute_intent(intent, cycle_id)
                    if result["success"]:
                        self.intent_manager.mark_executed(intent.intent_id, result)
                        executed += 1
                    else:
                        self.intent_manager.mark_failed(intent.intent_id, result.get("error", "Unknown error"))
                        failed += 1

            except Exception as e:
                logger.error(
                    f"Error executing intent {intent.intent_id[:8]}: {e}",
                    exc_info=True,
                )
                self.intent_manager.mark_failed(intent.intent_id, str(e))
                failed += 1

        summary = {
            "processed": len(approved_intents),
            "executed": executed,
            "failed": failed,
            "dry_run": self.dry_run,
        }

        logger.info(
            f"Processed {summary['processed']} approved intents: "
            f"{executed} executed, {failed} failed",
            extra=summary,
        )

        return summary

    def execute_intent(self, intent: TradeIntent, cycle_id: str) -> dict[str, Any]:
        """Execute a single intent (must already be APPROVED).

        This is the only entry point intended to be used by schedulers/agents.
        """
        return self._execute_intent(intent, cycle_id)

    def _execute_intent(self, intent: TradeIntent, cycle_id: str) -> dict[str, Any]:
        """Execute a single trade intent.

        Args:
            intent: Trade intent to execute.
            cycle_id: Current cycle ID.

        Returns:
            Execution result.
        """
        logger.info(
            f"Executing intent {intent.intent_id[:8]}",
            extra={
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type.value,
                "token_id": intent.token_id,
            },
        )

        if intent.intent_type == IntentType.OPEN_POSITION:
            return self._execute_open_position(intent, cycle_id)
        elif intent.intent_type == IntentType.CLOSE_POSITION:
            return self._execute_close_position(intent, cycle_id)
        elif intent.intent_type == IntentType.CLAIM_SETTLEMENT:
            return self._execute_claim_settlement(intent, cycle_id)
        elif intent.intent_type == IntentType.CANCEL_ORDER:
            return self._execute_cancel_order(intent, cycle_id)
        else:
            return {
                "success": False,
                "error": f"Unknown intent type: {intent.intent_type}",
            }

    def _execute_open_position(
        self, intent: TradeIntent, cycle_id: str
    ) -> dict[str, Any]:
        """Execute position opening order.

        Args:
            intent: Trade intent.
            cycle_id: Current cycle ID.

        Returns:
            Execution result.

        Note:
            This is a PLACEHOLDER implementation. In production, this would:
            1. Generate properly signed order for Polymarket CLOB
            2. Submit order via authenticated CLOB API
            3. Monitor order status
            4. Handle fills and partial fills
        """
        # SAFETY CHECK: Ensure this intent was approved
        if intent.status.value != "APPROVED":
            return {
                "success": False,
                "error": "Intent not approved",
            }

        # SAFETY CHECK: Refuse live execution if required credentials are missing.
        if not self._has_live_credentials():
            return {
                "success": False,
                "error": (
                    "Missing CLOB credentials for live execution. "
                    "Set POLYBOT_CLOB_API_KEY/POLYBOT_CLOB_API_SECRET/POLYBOT_CLOB_PASSPHRASE "
                    "and POLYBOT_POLYGON_PRIVATE_KEY. (Dry-run does not require these.)"
                ),
            }

        # For MVP, we log the order details.
        # When dry_run=false and creds exist, we attempt a best-effort authenticated submit.
        order_details = {
            "token_id": intent.token_id,
            "market_id": intent.market_id,
            "side": intent.side,
            "price": intent.price,
            "size_usd": intent.size_usd,
            "order_type": "LIMIT",
        }

        logger.warning(
            "LIVE ORDER EXECUTION PLACEHOLDER: "
            "In production, this would submit signed order to CLOB API",
            extra=order_details,
        )

        # Attempt best-effort submit (may fail depending on Polymarket auth requirements).
        try:
            from polyb0t.services.clob_trading import CLOBTradingClient

            client = CLOBTradingClient(
                base_url=self.settings.clob_base_url,
                api_key=self.settings.clob_api_key or "",
                api_secret=self.settings.clob_api_secret or "",
                passphrase=self.settings.clob_passphrase or "",
                polygon_private_key=self.settings.polygon_private_key or "",
                chain_id=int(self.settings.chain_id),
                signature_type=int(self.settings.signature_type),
                funder=(self.settings.funder_address or self.settings.user_address),
            )
            res = client.submit_limit_order(
                token_id=intent.token_id,
                side=intent.side or "BUY",
                price=float(intent.price or 0.0),
                size_usd=float(intent.size_usd or 0.0),
                fee_rate_bps=int(self.settings.fee_bps),
            )
            client.close()
            if not res.success:
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            # If it succeeded, return the external order id
            return {
                "success": True,
                "order_id": res.order_id,
                "message": "Order submitted",
                "details": order_details,
            }
        except Exception as e:
            return {"success": False, "error": f"Order submit error: {e}"}

        # For now, simulate successful order placement
        order_id = str(uuid.uuid4())

        # Record in database (as simulated for now)
        db_order = SimulatedOrderDB(
            order_id=order_id,
            cycle_id=cycle_id,
            token_id=intent.token_id,
            market_id=intent.market_id or "",
            side=intent.side or "BUY",
            order_type="LIMIT",
            price=intent.price or 0.0,
            size=intent.size_usd or 0.0,
            status="OPEN",
            filled_size=0.0,
        )
        self.db_session.add(db_order)
        self.db_session.commit()

        return {
            "success": True,
            "order_id": order_id,
            "message": "Order placed (SIMULATED - implement CLOB API for production)",
            "details": order_details,
        }

    def _execute_close_position(
        self, intent: TradeIntent, cycle_id: str
    ) -> dict[str, Any]:
        """Execute position closing order.

        Args:
            intent: Trade intent.
            cycle_id: Current cycle ID.

        Returns:
            Execution result.
        """
        # Similar to open_position but for closing
        # Would use same CLOB API but with opposite side

        order_details = {
            "token_id": intent.token_id,
            "market_id": intent.market_id,
            "side": intent.side,
            "price": intent.price,
            "size_usd": intent.size_usd,
            "order_type": "LIMIT",
            "purpose": "CLOSE_POSITION",
        }

        if not self._has_live_credentials():
            return {
                "success": False,
                "error": (
                    "Missing CLOB credentials for live execution. "
                    "Set POLYBOT_CLOB_API_KEY/POLYBOT_CLOB_API_SECRET/POLYBOT_CLOB_PASSPHRASE "
                    "and POLYBOT_POLYGON_PRIVATE_KEY."
                ),
            }

        logger.warning(
            "LIVE CLOSE ORDER: attempting best-effort submit",
            extra=order_details,
        )

        try:
            from polyb0t.services.clob_trading import CLOBTradingClient

            client = CLOBTradingClient(
                base_url=self.settings.clob_base_url,
                api_key=self.settings.clob_api_key or "",
                api_secret=self.settings.clob_api_secret or "",
                passphrase=self.settings.clob_passphrase or "",
            )
            res = client.submit_limit_order(
                token_id=intent.token_id,
                side=intent.side or "SELL",
                price=float(intent.price or 0.0),
                size_usd=float(intent.size_usd or 0.0),
            )
            client.close()
            if not res.success:
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            return {"success": True, "order_id": res.order_id, "message": "Close order submitted", "details": order_details}
        except Exception as e:
            return {"success": False, "error": f"Close submit error: {e}"}

    def _execute_cancel_order(self, intent: TradeIntent, cycle_id: str) -> dict[str, Any]:
        """Cancel a previously submitted order (requires explicit approval).

        This is a placeholder; real implementation requires authenticated CLOB endpoints.
        """
        if intent.status.value != "APPROVED":
            return {"success": False, "error": "Intent not approved"}

        if not self._has_live_credentials():
            return {
                "success": False,
                "error": "Missing CLOB credentials; cannot cancel order in live execution mode.",
            }

        target = None
        if intent.signal_data and isinstance(intent.signal_data, dict):
            target = intent.signal_data.get("order_id") or intent.signal_data.get("submitted_order_id")
        if not target:
            return {"success": False, "error": "Cancel intent missing target order id in signal_data"}

        try:
            from polyb0t.services.clob_trading import CLOBTradingClient

            client = CLOBTradingClient(
                base_url=self.settings.clob_base_url,
                api_key=self.settings.clob_api_key or "",
                api_secret=self.settings.clob_api_secret or "",
                passphrase=self.settings.clob_passphrase or "",
            )
            res = client.cancel_order(str(target))
            client.close()
            if not res.success:
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            return {"success": True, "order_id": str(target), "message": "Order cancel submitted"}
        except Exception as e:
            return {"success": False, "error": f"Cancel error: {e}"}

    def _has_live_credentials(self) -> bool:
        """Check whether live execution credentials are configured."""
        s = self.settings
        return bool(
            s.polygon_private_key
            and s.clob_api_key
            and s.clob_api_secret
            and s.clob_passphrase
        )

    def _execute_claim_settlement(
        self, intent: TradeIntent, cycle_id: str
    ) -> dict[str, Any]:
        """Execute settlement claim.

        Args:
            intent: Trade intent.
            cycle_id: Current cycle ID.

        Returns:
            Execution result.

        Note:
            Settlement claiming requires on-chain transaction.
            This is a PLACEHOLDER - implement only if:
            1. Polymarket has a supported claim mechanism
            2. It's explicitly permitted by ToS
            3. User has explicitly approved
        """
        claim_details = {
            "token_id": intent.token_id,
            "market_id": intent.market_id,
            "reason": intent.reason,
        }

        logger.warning(
            "CLAIM SETTLEMENT PLACEHOLDER: "
            "In production, this would initiate on-chain settlement claim",
            extra=claim_details,
        )

        # TODO: Implement actual settlement claim
        # This would likely require:
        # 1. Call smart contract claim function
        # 2. Sign transaction with private key
        # 3. Submit to blockchain
        # 4. Monitor transaction status

        return {
            "success": True,
            "message": "Claim submitted (PLACEHOLDER - implement only if permitted)",
            "details": claim_details,
        }

    def _log_dry_run_execution(self, intent: TradeIntent) -> None:
        """Log dry-run execution details.

        Args:
            intent: Trade intent.
        """
        logger.info(
            f"DRY-RUN: Would execute intent {intent.intent_id[:8]}",
            extra={
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type.value,
                "token_id": intent.token_id,
                "side": intent.side,
                "price": intent.price,
                "size_usd": intent.size_usd,
                "reason": intent.reason,
                "dry_run": True,
            },
        )

