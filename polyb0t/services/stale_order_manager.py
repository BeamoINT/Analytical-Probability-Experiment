"""Manages stale open orders - cancels and resubmits at market price."""

import logging
from datetime import datetime, timezone
from typing import Any

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


class StaleOrderManager:
    """Detects and handles stale open orders.
    
    When limit sell orders aren't filling, this manager:
    1. Detects orders older than the configured threshold
    2. Cancels them
    3. Resubmits at the current best bid (market sell)
    
    This prevents holding losing positions while waiting for limit orders to fill.
    """
    
    def __init__(self) -> None:
        """Initialize stale order manager."""
        self.settings = get_settings()
        self._order_first_seen: dict[str, datetime] = {}  # Track when we first saw each order
    
    def process_stale_orders(self) -> dict[str, Any]:
        """Check for and handle stale sell orders.
        
        Returns:
            Summary of actions taken.
        """
        if not self.settings.enable_panic_sell:
            return {"skipped": True, "reason": "panic_sell disabled"}
        
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds, OpenOrderParams
            
            client = ClobClient(
                host=self.settings.clob_base_url,
                chain_id=int(self.settings.chain_id),
                key=self.settings.polygon_private_key or "",
                creds=ApiCreds(
                    api_key=self.settings.clob_api_key or "",
                    api_secret=self.settings.clob_api_secret or "",
                    api_passphrase=self.settings.clob_passphrase or "",
                ),
                signature_type=int(self.settings.signature_type),
                funder=self.settings.funder_address or self.settings.user_address,
            )
            
            # Get all open orders
            orders = client.get_orders(OpenOrderParams())
            
            if not orders:
                self._order_first_seen.clear()
                return {"open_orders": 0, "stale_orders": 0, "resubmitted": 0}
            
            now = datetime.now(timezone.utc)
            stale_threshold = self.settings.panic_sell_order_age_seconds
            stale_orders = []
            
            # Track current order IDs
            current_order_ids = set()
            
            for order in orders:
                order_id = order.get("id") or order.get("order_id", "")
                side = order.get("side", "").upper()
                current_order_ids.add(order_id)
                
                # Only process SELL orders
                if side != "SELL":
                    continue
                
                # Track when we first saw this order
                if order_id not in self._order_first_seen:
                    self._order_first_seen[order_id] = now
                    continue  # Give new orders a chance
                
                # Check if order is stale
                first_seen = self._order_first_seen[order_id]
                age_seconds = (now - first_seen).total_seconds()
                
                if age_seconds >= stale_threshold:
                    stale_orders.append({
                        "order_id": order_id,
                        "token_id": order.get("asset_id", ""),
                        "side": side,
                        "price": float(order.get("price", 0)),
                        "size": float(order.get("original_size", 0)) or float(order.get("size", 0)),
                        "age_seconds": age_seconds,
                    })
            
            # Clean up tracking for orders that no longer exist
            self._order_first_seen = {
                oid: ts for oid, ts in self._order_first_seen.items()
                if oid in current_order_ids
            }
            
            if not stale_orders:
                return {
                    "open_orders": len(orders),
                    "sell_orders": sum(1 for o in orders if o.get("side", "").upper() == "SELL"),
                    "stale_orders": 0,
                    "resubmitted": 0,
                }
            
            logger.warning(
                f"Found {len(stale_orders)} stale SELL orders (>{stale_threshold}s old)",
                extra={"stale_orders": stale_orders},
            )
            
            # Cancel and resubmit stale orders at market price
            resubmitted = 0
            failed = 0
            
            for stale in stale_orders:
                try:
                    result = self._cancel_and_resubmit_at_market(client, stale)
                    if result["success"]:
                        resubmitted += 1
                        # Remove from tracking so we start fresh
                        self._order_first_seen.pop(stale["order_id"], None)
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error handling stale order {stale['order_id']}: {e}")
                    failed += 1
            
            return {
                "open_orders": len(orders),
                "stale_orders": len(stale_orders),
                "resubmitted": resubmitted,
                "failed": failed,
            }
            
        except Exception as e:
            logger.error(f"Error in stale order processing: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _cancel_and_resubmit_at_market(
        self,
        client: Any,
        stale_order: dict[str, Any],
    ) -> dict[str, Any]:
        """Cancel a stale order and resubmit at best bid.
        
        Args:
            client: CLOB client.
            stale_order: Details of the stale order.
            
        Returns:
            Result of the operation.
        """
        from py_clob_client.clob_types import OrderArgs
        
        order_id = stale_order["order_id"]
        token_id = stale_order["token_id"]
        original_price = stale_order["price"]
        size = stale_order["size"]
        
        # Get current best bid
        try:
            orderbook = client.get_order_book(token_id)
            bids = orderbook.get("bids", [])
            if bids:
                best_bid = float(bids[0].get("price", original_price))
            else:
                # No bids - use aggressive price (50% of original)
                best_bid = max(0.01, original_price * 0.5)
        except Exception as e:
            logger.warning(f"Could not get orderbook for {token_id[:20]}: {e}")
            best_bid = max(0.01, original_price * 0.9)
        
        # Apply slippage for guaranteed fill
        slippage_pct = self.settings.panic_sell_min_fill_slippage_pct
        market_price = max(0.005, best_bid * (1 - slippage_pct / 100))
        
        logger.info(
            f"Resubmitting stale order: was ${original_price:.3f}, now ${market_price:.3f} (best bid: ${best_bid:.3f})",
            extra={"order_id": order_id, "token_id": token_id[:20]},
        )
        
        # Cancel the old order
        try:
            client.cancel(order_id)
            logger.info(f"Cancelled stale order {order_id}")
        except Exception as e:
            # Order might already be filled or cancelled
            logger.warning(f"Could not cancel order {order_id}: {e}")
        
        # Resubmit at market price
        try:
            result = client.create_and_post_order(OrderArgs(
                token_id=token_id,
                price=market_price,
                size=size,
                side="SELL",
            ))
            
            new_order_id = result.get("orderID") or result.get("order_id", "unknown")
            logger.info(
                f"Resubmitted at market price: {new_order_id}",
                extra={
                    "old_order_id": order_id,
                    "new_order_id": new_order_id,
                    "old_price": original_price,
                    "new_price": market_price,
                },
            )
            
            return {"success": True, "new_order_id": new_order_id}
            
        except Exception as e:
            logger.error(f"Failed to resubmit order: {e}")
            return {"success": False, "error": str(e)}

