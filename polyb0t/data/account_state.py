"""Account state tracking for live mode (read-only)."""

import logging
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.storage import AccountStateDB

logger = logging.getLogger(__name__)


class AccountPosition:
    """Represents a live account position."""

    def __init__(
        self,
        token_id: str,
        market_id: str | None,
        side: str,
        quantity: float,
        avg_price: float,
        current_price: float | None = None,
    ) -> None:
        """Initialize account position.

        Args:
            token_id: Token identifier.
            market_id: Market ID.
            side: LONG or SHORT.
            quantity: Position size.
            avg_price: Average entry price.
            current_price: Current market price.
        """
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.avg_price = avg_price
        self.current_price = current_price or avg_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "market_id": self.market_id,
            "side": self.side,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
        }


class AccountOrder:
    """Represents a live open order."""

    def __init__(
        self,
        order_id: str,
        token_id: str,
        market_id: str | None,
        side: str,
        price: float,
        size: float,
        filled_size: float = 0.0,
    ) -> None:
        """Initialize account order.

        Args:
            order_id: Order identifier.
            token_id: Token identifier.
            market_id: Market ID.
            side: BUY or SELL.
            price: Order price.
            size: Order size.
            filled_size: Filled size.
        """
        self.order_id = order_id
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.price = price
        self.size = size
        self.filled_size = filled_size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "token_id": self.token_id,
            "market_id": self.market_id,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "filled_size": self.filled_size,
        }


class AccountState:
    """Represents current account state."""

    def __init__(
        self,
        wallet_address: str | None,
        cash_balance: float | None,
        total_equity: float | None,
        positions: list[AccountPosition],
        open_orders: list[AccountOrder],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize account state.

        Args:
            wallet_address: Wallet address.
            cash_balance: Available cash.
            total_equity: Total account equity.
            positions: List of positions.
            open_orders: List of open orders.
            metadata: Additional metadata.
        """
        self.wallet_address = wallet_address
        self.cash_balance = cash_balance
        self.total_equity = total_equity
        self.positions = positions
        self.open_orders = open_orders
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wallet_address": self.wallet_address,
            "cash_balance": self.cash_balance,
            "total_equity": self.total_equity,
            "positions_count": len(self.positions),
            "open_orders_count": len(self.open_orders),
            "positions": [p.to_dict() for p in self.positions],
            "open_orders": [o.to_dict() for o in self.open_orders],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class AccountStateProvider:
    """Fetches account state from Polymarket (read-only).

    IMPORTANT: This implementation makes assumptions about endpoint availability.
    Adjust based on actual Polymarket API documentation.
    """

    def __init__(self, clob_base_url: str | None = None) -> None:
        """Initialize account state provider.

        Args:
            clob_base_url: CLOB API base URL.
        """
        settings = get_settings()
        self.clob_base_url = clob_base_url or settings.clob_base_url
        self.wallet_address = settings.user_address
        self.api_key = settings.clob_api_key
        self.api_secret = settings.clob_api_secret
        self.api_passphrase = settings.clob_passphrase

        # Note: Authentication may be required for account endpoints
        headers = {"Accept": "application/json"}
        if self.api_key:
            # Adjust header names per official Polymarket CLOB docs.
            headers["X-API-Key"] = self.api_key

        self.client = httpx.AsyncClient(
            base_url=self.clob_base_url,
            timeout=30.0,
            headers=headers,
        )

    async def __aenter__(self) -> "AccountStateProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    async def fetch_account_state(self) -> AccountState:
        """Fetch current account state.

        Returns:
            AccountState object.

        Note:
            This is a READ-ONLY operation. Endpoint paths are assumed
            and should be verified against Polymarket documentation.
        """
        if not self.wallet_address:
            logger.warning("No wallet address configured, returning empty state")
            return AccountState(
                wallet_address=None,
                cash_balance=None,
                total_equity=None,
                positions=[],
                open_orders=[],
            )

        try:
            # Fetch positions (endpoint assumed - verify with API docs)
            positions = await self._fetch_positions()

            # Fetch open orders (endpoint assumed - verify with API docs)
            open_orders = await self._fetch_open_orders()

            # Fetch balance (endpoint assumed - verify with API docs)
            cash_balance, total_equity = await self._fetch_balances()

            account_state = AccountState(
                wallet_address=self.wallet_address,
                cash_balance=cash_balance,
                total_equity=total_equity,
                positions=positions,
                open_orders=open_orders,
            )

            logger.info(
                f"Fetched account state: {len(positions)} positions, "
                f"{len(open_orders)} open orders"
            )

            return account_state

        except Exception as e:
            logger.error(f"Error fetching account state: {e}", exc_info=True)
            # Return empty state on error - fail gracefully
            return AccountState(
                wallet_address=self.wallet_address,
                cash_balance=None,
                total_equity=None,
                positions=[],
                open_orders=[],
                metadata={"error": str(e)},
            )

    async def _fetch_positions(self) -> list[AccountPosition]:
        """Fetch account positions by reconstructing from trades.

        Returns:
            List of AccountPosition objects.

        Note:
            Uses py-clob-client to fetch trades and reconstruct positions.
        """
        logger.info("🔍 _fetch_positions called - starting position reconstruction with orderbook filtering")
        try:
            import asyncio
            from collections import defaultdict
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            
            settings = get_settings()
            logger.debug(f"Got settings: chain_id={settings.chain_id}, signature_type={settings.signature_type}")
            
            # Create CLOB client (synchronous)
            clob_client = ClobClient(
                host=settings.clob_base_url,
                chain_id=int(settings.chain_id),
                key=settings.polygon_private_key,
                creds=ApiCreds(
                    api_key=settings.clob_api_key,
                    api_secret=settings.clob_api_secret,
                    api_passphrase=settings.clob_passphrase,
                ),
                signature_type=int(settings.signature_type),
                funder=settings.funder_address,
            )
            
            # Fetch all trades (run sync code in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            trades = await loop.run_in_executor(None, clob_client.get_trades)
            
            # Reconstruct positions from trades
            position_data = defaultdict(lambda: {
                "quantity": 0.0,
                "cost": 0.0,
                "market_id": None,
            })
            
            for trade in trades:
                asset_id = trade.get("asset_id")
                side = trade.get("side")  # BUY or SELL
                size = float(trade.get("size", 0))
                price = float(trade.get("price", 0))
                market_id = trade.get("market")
                
                if side == "BUY":
                    position_data[asset_id]["quantity"] += size
                    position_data[asset_id]["cost"] += size * price
                elif side == "SELL":
                    position_data[asset_id]["quantity"] -= size
                    position_data[asset_id]["cost"] -= size * price
                
                # Store market_id
                if market_id and not position_data[asset_id]["market_id"]:
                    position_data[asset_id]["market_id"] = market_id
            
            # Convert to AccountPosition objects (only positions with quantity > 0)
            # Also filter out resolved markets by checking if orderbook exists
            positions = []
            tokens_to_check = [
                (token_id, data) 
                for token_id, data in position_data.items()
                if data["quantity"] > 0.001 and data["cost"] >= 0.10
            ]
            
            logger.debug(f"Found {len(tokens_to_check)} positions with $0.10+ value, checking orderbooks...")
            
            # Check orderbook for each position to filter out resolved markets
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                for token_id, data in tokens_to_check:
                    try:
                        # Check if token has an active orderbook (not resolved)
                        response = await http_client.get(
                            f"{settings.clob_base_url}/book?token_id={token_id}"
                        )
                        
                        if response.status_code == 200:
                            book = response.json()
                            # Only include if orderbook has bids or asks
                            has_bids = book.get("bids") and len(book["bids"]) > 0
                            has_asks = book.get("asks") and len(book["asks"]) > 0
                            
                            if has_bids or has_asks:
                                avg_price = data["cost"] / data["quantity"] if data["quantity"] > 0 else 0
                                position = AccountPosition(
                                    token_id=token_id,
                                    market_id=data["market_id"],
                                    side="LONG",
                                    quantity=data["quantity"],
                                    avg_price=avg_price,
                                    current_price=None,
                                )
                                positions.append(position)
                            else:
                                logger.debug(f"Skipping token {token_id[:16]}... - empty orderbook (resolved market)")
                        else:
                            logger.debug(f"Skipping token {token_id[:16]}... - no orderbook (resolved market)")
                    except Exception as e:
                        logger.debug(f"Error checking orderbook for {token_id[:16]}...: {e}")
            
            logger.info(f"Reconstructed {len(positions)} ACTIVE positions from {len(trades)} trades (filtered out {len(tokens_to_check) - len(positions)} resolved markets)")
            return positions

        except Exception as e:
            logger.error(f"CRITICAL: Error in _fetch_positions: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}, args: {e.args}")
            logger.error("Falling back to empty position list")
            return []

    async def _fetch_open_orders(self) -> list[AccountOrder]:
        """Fetch open orders.

        Returns:
            List of AccountOrder objects.

        Note:
            Endpoint path is assumed. Adjust based on actual API.
        """
        try:
            # Assumed endpoint - verify with API documentation
            response = await self.client.get(f"/orders/{self.wallet_address}")
            response.raise_for_status()
            data = response.json()

            orders = []
            order_list = data if isinstance(data, list) else data.get("orders", [])

            for item in order_list:
                try:
                    order = AccountOrder(
                        order_id=item.get("order_id", ""),
                        token_id=item.get("token_id", ""),
                        market_id=item.get("market_id"),
                        side=item.get("side", "BUY"),
                        price=float(item.get("price", 0)),
                        size=float(item.get("size", 0)),
                        filled_size=float(item.get("filled_size", 0)),
                    )
                    orders.append(order)
                except Exception as e:
                    logger.warning(f"Failed to parse order: {e}", extra={"item": item})

            return orders

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("No orders endpoint available or no orders found")
                return []
            logger.warning(f"HTTP error fetching orders: {e.response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"Error fetching orders: {e}")
            return []

    async def _fetch_balances(self) -> tuple[float | None, float | None]:
        """Fetch account balances.

        Returns:
            Tuple of (cash_balance, total_equity).

        Note:
            TEMPORARY: Hardcoded balance until py-clob-client proxy balance bug is fixed.
            The bot successfully places orders, so credentials work. Balance API is the issue.
        """
        try:
            # TEMPORARY WORKAROUND: Return a fixed balance so trading can proceed
            # The real balance should be fetched via py_clob_client.get_balance_allowance()
            # but that method has a bug with SIGNATURE_TYPE=1 (proxy wallets)
            # 
            # UPDATE: Set to actual current balance from Polymarket dashboard.
            # When you deposit more USDC, update this value!
            cash_balance = 0.76  # User's actual USDC balance - UPDATE THIS WHEN YOU DEPOSIT MORE
            total_equity = cash_balance
            
            logger.debug(f"Using hardcoded balance: ${cash_balance:.2f} USDC")
            return cash_balance, total_equity

        except Exception as e:
            logger.warning(f"Error fetching balances: {e}")
            return None, None

    def persist_state(self, state: AccountState, cycle_id: str, db_session: Session) -> None:
        """Persist account state to database.

        Args:
            state: Account state to persist.
            cycle_id: Current cycle ID.
            db_session: Database session.
        """
        db_state = AccountStateDB(
            cycle_id=cycle_id,
            timestamp=state.timestamp,
            wallet_address=state.wallet_address,
            cash_balance=state.cash_balance,
            total_equity=state.total_equity,
            open_orders_count=len(state.open_orders),
            positions_count=len(state.positions),
            positions=[p.to_dict() for p in state.positions],
            open_orders=[o.to_dict() for o in state.open_orders],
            meta_json=state.metadata,
        )

        db_session.add(db_state)
        db_session.commit()

        logger.debug(f"Persisted account state for cycle {cycle_id[:8]}")

