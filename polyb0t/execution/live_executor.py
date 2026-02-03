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

        # Pre-flight balance check for BUY orders
        # Verify we have enough USDC before submitting to avoid API rejections
        if intent.side == "BUY" and self.settings.polygon_rpc_url:
            try:
                from polyb0t.services.balance import BalanceService
                from sqlalchemy.orm import Session
                from polyb0t.data.storage import get_session

                with get_session() as bal_session:
                    bal_service = BalanceService(db_session=bal_session)
                    snap = bal_service.fetch_usdc_balance()
                    size_usd = float(intent.size_usd or 0.0)

                    # Check if we have enough available (with 5% safety margin for fees/slippage)
                    required_with_margin = size_usd * 1.05
                    if snap.available_usdc < required_with_margin:
                        logger.warning(
                            f"Pre-flight balance check failed: need ${required_with_margin:.2f}, "
                            f"have ${snap.available_usdc:.2f} available",
                            extra={
                                "size_usd": size_usd,
                                "required_with_margin": required_with_margin,
                                "available_usdc": snap.available_usdc,
                                "total_usdc": snap.total_usdc,
                                "reserved_usdc": snap.reserved_usdc,
                            },
                        )
                        return {
                            "success": False,
                            "error": f"Insufficient balance: need ${required_with_margin:.2f}, have ${snap.available_usdc:.2f}",
                            "skip_retry": True,
                        }
                    logger.debug(
                        f"Pre-flight balance check passed: ${snap.available_usdc:.2f} available >= ${required_with_margin:.2f} required"
                    )
            except Exception as e:
                logger.warning(f"Pre-flight balance check failed (non-blocking): {e}")
                # Continue anyway - the order will fail at the API level if balance is insufficient

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
            "LIVE ORDER EXECUTION",
            extra=order_details,
        )

        logger.info(f"About to submit order via CLOBTradingClient: {order_details}")
        
        # Attempt best-effort submit (may fail depending on Polymarket auth requirements).
        client = None
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
            logger.info(f"CLOBTradingClient returned: success={res.success}, order_id={res.order_id}")
            if not res.success:
                logger.error(f"Order rejected: {res.message}")
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            # If it succeeded, return the external order id
            logger.info(f"✅ Order succeeded! orderID={res.order_id}")
            return {
                "success": True,
                "order_id": res.order_id,
                "message": "Order submitted",
                "details": order_details,
            }
        except Exception as e:
            logger.error(f"Exception in order submission: {type(e).__name__}: {e}", exc_info=True)
            return {"success": False, "error": f"Order submit error: {e}"}
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

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

        Uses "panic sell" (market sell at best bid) ONLY when:
        - CRASH EXIT: Emergency flag is set (massive drop detected)
        - Price has dropped significantly from entry (configurable threshold)
        - Stop-loss triggered (reason contains "stop")
        
        Otherwise uses normal limit orders for better fill prices.
        
        CONSERVATIVE APPROACH:
        - Normal exits use limit orders (wait for good price)
        - Only crash/panic exits use market orders

        Args:
            intent: Trade intent.
            cycle_id: Current cycle ID.

        Returns:
            Execution result.
        """
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

        # Initialize variables
        actual_shares = 0.0
        actual_size_usd = float(intent.size_usd or 0.0)
        best_bid = float(intent.price or 0.0)
        use_market_sell = True  # ALWAYS use market sells for immediate fill at best bid
        entry_price: float | None = None
        price_drop_pct: float = 0.0
        
        # FRESH ORDERBOOK: Fetch current best bid right before selling
        try:
            import httpx
            ob_url = f"{self.settings.clob_base_url}/book?token_id={intent.token_id}"
            resp = httpx.get(ob_url, timeout=5.0)
            if resp.status_code == 200:
                ob_data = resp.json()
                bids = ob_data.get("bids", [])
                if bids and len(bids) > 0:
                    fresh_best_bid = float(bids[0].get("price", 0))
                    if fresh_best_bid > 0:
                        logger.info(
                            f"Fresh orderbook: best_bid={fresh_best_bid:.4f} (was {best_bid:.4f})",
                            extra={"token_id": intent.token_id[:20]}
                        )
                        best_bid = fresh_best_bid  # Use fresh price
        except Exception as e:
            logger.warning(f"Could not fetch fresh orderbook: {e}")
        
        # Check if this is an EMERGENCY (crash) exit - always use market sell
        is_emergency = False
        if intent.signal_data and isinstance(intent.signal_data, dict):
            is_emergency = intent.signal_data.get("is_emergency", False)
        
        # Also check reason for CRASH_EXIT indicator
        reason_lower = (intent.reason or "").lower()
        if "crash" in reason_lower or "🚨" in (intent.reason or ""):
            is_emergency = True
        
        if is_emergency:
            use_market_sell = True
            logger.warning(
                f"🚨 EMERGENCY EXIT: Using immediate market sell",
                extra={"token_id": intent.token_id[:20], "reason": intent.reason}
            )
        
        # Check if we actually have tokens to sell before attempting
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

            funder_addr = self.settings.funder_address or self.settings.user_address
            check_client = ClobClient(
                host=self.settings.clob_base_url,
                chain_id=int(self.settings.chain_id),
                key=self.settings.polygon_private_key or "",
                creds=ApiCreds(
                    api_key=self.settings.clob_api_key or "",
                    api_secret=self.settings.clob_api_secret or "",
                    api_passphrase=self.settings.clob_passphrase or "",
                ),
                signature_type=int(self.settings.signature_type),
                funder=funder_addr,
            )
            
            # Get token balance
            balance_result = check_client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=intent.token_id)
            )
            token_balance = int(balance_result.get("balance", 0))
            token_allowance = int(balance_result.get("allowance", 0))
            
            # Get best bid from orderbook
            try:
                orderbook = check_client.get_order_book(intent.token_id)
                bids = orderbook.get("bids", [])
                if bids:
                    best_bid = float(bids[0].get("price", intent.price or 0.01))
                else:
                    best_bid = max(0.01, float(intent.price or 0.01) * 0.5)
                    logger.warning(f"No bids found, using floor price: {best_bid:.4f}")
            except Exception as e:
                logger.warning(f"Could not fetch orderbook: {e}")
                best_bid = max(0.01, float(intent.price or 0.01) * 0.95)
            
            # Get entry price from intent signal_data (if available)
            if intent.signal_data and isinstance(intent.signal_data, dict):
                entry_price = intent.signal_data.get("entry_price") or intent.signal_data.get("avg_price")
            
            # Determine if we should panic sell based on price drop
            if self.settings.enable_panic_sell and entry_price and entry_price > 0:
                price_drop_pct = ((entry_price - best_bid) / entry_price) * 100
                
                # Only panic sell if price dropped by more than threshold
                if price_drop_pct >= self.settings.panic_sell_price_drop_pct:
                    use_market_sell = True
                    logger.warning(
                        f"PANIC SELL triggered: price dropped {price_drop_pct:.1f}% (threshold: {self.settings.panic_sell_price_drop_pct}%)",
                        extra={
                            "entry_price": entry_price,
                            "current_bid": best_bid,
                            "drop_pct": price_drop_pct,
                        },
                    )
            
            # Also trigger panic sell if this is a stop-loss
            reason = (intent.reason or "").lower()
            if "stop" in reason or "loss" in reason:
                use_market_sell = True
                logger.info("Market sell triggered: stop-loss exit")
            
            # Log the balance check result
            logger.info(
                f"Balance check for CLOSE_POSITION: balance={token_balance / 1e6:.2f} shares, allowance={token_allowance / 1e6:.2f}",
                extra={
                    "token_id": intent.token_id[:20],
                    "balance_raw": token_balance,
                    "allowance_raw": token_allowance,
                    "funder_address": funder_addr,
                    "best_bid": best_bid,
                    "entry_price": entry_price,
                    "price_drop_pct": price_drop_pct,
                    "use_market_sell": use_market_sell,
                },
            )
            
            # Minimum order size is 5 shares, balance is in raw units (1e6 = 1 share)
            min_sellable = 5 * 1_000_000  # 5 shares in raw units
            if token_balance < min_sellable:
                logger.info(
                    f"Skipping CLOSE_POSITION: token balance too low ({token_balance / 1e6:.2f} shares, need ≥5)",
                    extra={"token_id": intent.token_id[:20], "balance": token_balance},
                )
                return {
                    "success": False,
                    "error": f"Token balance too low to sell ({token_balance / 1e6:.2f} shares, minimum is 5)",
                    "skip_retry": True,  # Don't keep retrying this
                }
            
            # Also check allowance - if not enough, the sell will fail
            if token_allowance < token_balance:
                logger.warning(
                    f"Token allowance ({token_allowance / 1e6:.2f}) less than balance ({token_balance / 1e6:.2f}) - sell may fail",
                    extra={"token_id": intent.token_id[:20]},
                )
            
            # CRITICAL: Use actual balance for sell size
            actual_shares = token_balance / 1e6
            
            # ALWAYS use best bid for immediate fill - no slippage reduction
            # The best_bid IS the market price - sell at that price
            sell_price = best_bid
            order_details["order_type"] = "MARKET_SELL"
            order_details["slippage_applied"] = 0.0  # No additional slippage
            
            actual_size_usd = actual_shares * sell_price
            
            # Update order_details to reflect actual values
            order_details["actual_shares"] = actual_shares
            order_details["actual_size_usd"] = actual_size_usd
            order_details["intent_size_usd"] = float(intent.size_usd or 0.0)
            order_details["sell_price"] = sell_price
            order_details["best_bid"] = best_bid
            order_details["entry_price"] = entry_price
            order_details["price_drop_pct"] = price_drop_pct
            order_details["use_market_sell"] = use_market_sell
            
        except Exception as e:
            logger.warning(f"Could not check token balance before sell: {e}", exc_info=True)
            # Fall back to intent values if balance check fails
            actual_size_usd = float(intent.size_usd or 0.0)
            sell_price = float(intent.price or 0.0)

        logger.warning(
            f"LIVE CLOSE ORDER: {'PANIC SELL' if use_market_sell else 'LIMIT SELL'} @ {sell_price:.4f}",
            extra=order_details,
        )

        client = None
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
                side=intent.side or "SELL",
                price=sell_price,
                size_usd=actual_size_usd,
                fee_rate_bps=int(self.settings.fee_bps),
            )
            if not res.success:
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            return {"success": True, "order_id": res.order_id, "message": f"Close order submitted ({'panic' if use_market_sell else 'limit'})", "details": order_details}
        except Exception as e:
            logger.error(f"Close submit error: {e}")
            return {"success": False, "error": f"Close submit error: {e}"}
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

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

        client = None
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
            res = client.cancel_order(str(target))
            if not res.success:
                return {"success": False, "error": res.message, "status_code": res.status_code, "raw": res.raw}
            return {"success": True, "order_id": str(target), "message": "Order cancel submitted"}
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return {"success": False, "error": f"Cancel error: {e}"}
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

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

