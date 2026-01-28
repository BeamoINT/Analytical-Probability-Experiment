"""CLOB API client for Polymarket order books and trades (read-only MVP)."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from polyb0t.config import get_settings
from polyb0t.data.models import OrderBook, OrderBookLevel, Trade

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price point from historical timeseries data."""
    timestamp: datetime
    price: float


@dataclass
class PriceHistory:
    """Historical price data for a token."""
    token_id: str
    interval: str  # e.g., "1m", "1h", "1d"
    points: list[PricePoint]
    points_count: int
    fetched_at: datetime


class CLOBClient:
    """Client for Polymarket CLOB API (read-only for MVP).

    Note:
        This implementation focuses on read-only operations for MVP.
        Live trading functionality would require authentication and
        additional methods for order placement, which are NOT included
        to ensure safety-by-default.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        """Initialize CLOB client.

        Args:
            base_url: Base URL for CLOB API. Defaults to config value.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.clob_base_url
        self.timeout = timeout
        headers = {
            "Accept": "application/json",
            "User-Agent": "polyb0t/0.1.0",
        }

        # Optional API key for authenticated READ-ONLY requests
        if settings.clob_api_key:
            headers["X-API-Key"] = settings.clob_api_key

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    async def __aenter__(self) -> "CLOBClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make GET request with retry logic.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            Response JSON data.

        Raises:
            httpx.HTTPStatusError: On HTTP error.
        """
        logger.debug(f"GET {endpoint}", extra={"params": params})
        response = await self.client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    async def _get_with_status(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> tuple[int | None, Any | None]:
        """GET that returns (status_code, json_or_none) without raising on HTTPStatusError."""
        try:
            response = await self.client.get(endpoint, params=params)
            status = response.status_code
            response.raise_for_status()
            return status, response.json()
        except httpx.HTTPStatusError as e:
            return e.response.status_code, None
        except Exception as e:
            logger.debug(f"CLOB API request failed: {e}")
            return None, None

    @staticmethod
    def debug_endpoints() -> dict[str, list[tuple[str, str]]]:
        """Return candidate endpoints for public reads (best-effort).

        We keep these guesses isolated here; they may vary by CLOB API version.
        """
        return {
            "orderbook": [
                ("GET", "/book?token_id={token_id}"),
                ("GET", "/book/{token_id}"),
                ("GET", "/orderbook/{token_id}"),
            ],
            "trades": [
                ("GET", "/trades?token_id={token_id}&limit={limit}"),
                ("GET", "/trades/{token_id}?limit={limit}"),
            ],
        }

    async def get_orderbook_debug(
        self, token_id: str
    ) -> tuple[OrderBook | None, int | None, str | None]:
        """Try multiple endpoints to fetch an orderbook, returning (orderbook, status, endpoint_used)."""
        candidates = self.debug_endpoints()["orderbook"]
        for _method, template in candidates:
            formatted = template.format(token_id=token_id)
            if "?" in formatted:
                path, qs = formatted.split("?", 1)
                # For our templates we only use token_id query param
                params = {"token_id": token_id}
            else:
                path = formatted
                params = None
            status, data = await self._get_with_status(path, params=params)
            if data:
                return self._parse_orderbook(token_id, data), status, template
            # If clearly not found, continue to try alternatives
            if status in (404, 400):
                continue
        return None, status, None

    async def get_trades_debug(
        self, token_id: str, limit: int = 100
    ) -> tuple[list[Trade], int | None, str | None]:
        """Try multiple endpoints to fetch trades, returning (trades, status, endpoint_used)."""
        candidates = self.debug_endpoints()["trades"]
        last_status: int | None = None
        for _method, template in candidates:
            formatted = template.format(token_id=token_id, limit=limit)
            if "?" in formatted:
                path, qs = formatted.split("?", 1)
                params = {"token_id": token_id, "limit": limit}
            else:
                path = formatted
                params = None
            status, data = await self._get_with_status(path, params=params)
            last_status = status
            if data is None:
                if status in (404, 400):
                    continue
                continue
            trade_list = data if isinstance(data, list) else data.get("trades", [])
            trades: list[Trade] = []
            for item in trade_list:
                if isinstance(item, dict):
                    try:
                        trades.append(self._parse_trade(token_id, item))
                    except Exception as e:
                        logger.debug(f"Failed to parse trade: {e}")
                        continue
            return trades, status, template
        return [], last_status, None

    async def get_orderbook(self, token_id: str) -> OrderBook | None:
        """Get order book snapshot for a token.

        Args:
            token_id: Token identifier.

        Returns:
            OrderBook object or None if not available.

        Note:
            Endpoint structure assumes /orderbook/{token_id} or similar.
            Adjust based on actual CLOB API documentation.
        """
        try:
            data = await self._get(f"/book", params={"token_id": token_id})
            return self._parse_orderbook(token_id, data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Order book not found for token: {token_id}")
                return None
            logger.error(
                f"HTTP error fetching orderbook for {token_id}: {e.response.status_code}"
            )
            # Don't raise - allow system to continue
            return None
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return None

    async def get_trades(
        self,
        token_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Get recent trades for a token.

        Args:
            token_id: Token identifier.
            since: Only return trades after this timestamp.
            limit: Maximum number of trades to return.

        Returns:
            List of Trade objects.

        Note:
            Endpoint structure is assumed. Adjust based on actual API.
        """
        params: dict[str, Any] = {
            "token_id": token_id,
            "limit": limit,
        }

        if since:
            # API might accept timestamp in various formats
            params["since"] = int(since.timestamp())

        try:
            data = await self._get("/trades", params=params)
            trades = []

            trade_list = data if isinstance(data, list) else data.get("trades", [])

            for item in trade_list:
                try:
                    trade = self._parse_trade(token_id, item)
                    trades.append(trade)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse trade: {e}",
                        extra={"trade_data": item},
                    )
                    continue

            logger.debug(f"Fetched {len(trades)} trades for token {token_id}")
            return trades

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching trades for {token_id}: {e.response.status_code}"
            )
            return []
        except Exception as e:
            logger.error(f"Error fetching trades for {token_id}: {e}")
            return []

    def _parse_orderbook(self, token_id: str, data: dict[str, Any]) -> OrderBook:
        """Parse order book data from API response.

        Args:
            token_id: Token identifier.
            data: Raw orderbook data from API.

        Returns:
            OrderBook object.
        """
        bids = []
        asks = []

        # Parse bids
        for bid in data.get("bids", []):
            if isinstance(bid, dict):
                bids.append(
                    OrderBookLevel(
                        price=float(bid.get("price", 0)),
                        size=float(bid.get("size", bid.get("quantity", 0))),
                    )
                )
            elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                # Sometimes [price, size] format
                bids.append(OrderBookLevel(price=float(bid[0]), size=float(bid[1])))

        # Parse asks
        for ask in data.get("asks", []):
            if isinstance(ask, dict):
                asks.append(
                    OrderBookLevel(
                        price=float(ask.get("price", 0)),
                        size=float(ask.get("size", ask.get("quantity", 0))),
                    )
                )
            elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                asks.append(OrderBookLevel(price=float(ask[0]), size=float(ask[1])))

        # Normalize ordering: bids descending (best first), asks ascending (best first)
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(
            token_id=token_id,
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            market_id=data.get("market_id"),
        )

    def _parse_trade(self, token_id: str, data: dict[str, Any]) -> Trade:
        """Parse trade data from API response.

        Args:
            token_id: Token identifier.
            data: Raw trade data from API.

        Returns:
            Trade object.
        """
        # Parse timestamp
        timestamp = datetime.utcnow()
        ts_value = data.get("timestamp") or data.get("time") or data.get("created_at")
        if ts_value:
            try:
                if isinstance(ts_value, str):
                    timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                elif isinstance(ts_value, (int, float)):
                    timestamp = datetime.fromtimestamp(ts_value)
            except Exception as e:
                logger.warning(f"Failed to parse trade timestamp: {ts_value}, error: {e}")

        return Trade(
            token_id=token_id,
            timestamp=timestamp,
            price=float(data.get("price", 0)),
            size=float(data.get("size", data.get("quantity", 0))),
            side=data.get("side", "UNKNOWN").upper(),
            trade_id=data.get("id") or data.get("trade_id"),
        )

    async def get_price_history(
        self,
        token_id: str,
        interval: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int | None = None,
    ) -> PriceHistory | None:
        """Get historical price timeseries data for a token.

        This endpoint provides detailed historical price data for ML training.

        Args:
            token_id: Token identifier (CLOB token ID).
            interval: Time interval string - one of "1m", "1h", "6h", "1d", "1w", "max".
                     Mutually exclusive with start_ts/end_ts.
            start_ts: Start Unix timestamp (UTC). Mutually exclusive with interval.
            end_ts: End Unix timestamp (UTC). Mutually exclusive with interval.
            fidelity: Data resolution in minutes (optional).

        Returns:
            PriceHistory object with list of (timestamp, price) points, or None on error.

        Note:
            - Use interval OR (start_ts + end_ts), not both.
            - fidelity controls data granularity (e.g., 1 = 1 min resolution, 60 = hourly).
            - Rate limit: 1500 req/10s for this endpoint.
        """
        params: dict[str, Any] = {"market": token_id}

        # Set time range parameters
        if interval:
            params["interval"] = interval
        elif start_ts is not None and end_ts is not None:
            params["startTs"] = start_ts
            params["endTs"] = end_ts

        if fidelity is not None:
            params["fidelity"] = fidelity

        try:
            data = await self._get("/prices-history", params=params)
            return self._parse_price_history(token_id, interval or "custom", data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Price history not found for token: {token_id}")
                return None
            logger.warning(
                f"HTTP error fetching price history for {token_id}: {e.response.status_code}"
            )
            return None
        except Exception as e:
            logger.warning(f"Error fetching price history for {token_id}: {e}")
            return None

    def _parse_price_history(
        self,
        token_id: str,
        interval: str,
        data: dict[str, Any],
    ) -> PriceHistory:
        """Parse price history response from CLOB API.

        Args:
            token_id: Token identifier.
            interval: Interval used for the request.
            data: Raw API response.

        Returns:
            PriceHistory object.
        """
        points = []
        history_data = data.get("history", [])

        for point in history_data:
            try:
                # API returns {"t": timestamp, "p": price}
                ts = point.get("t")
                price = point.get("p")

                if ts is None or price is None:
                    continue

                # Convert timestamp (Unix seconds) to datetime
                if isinstance(ts, (int, float)):
                    timestamp = datetime.utcfromtimestamp(ts)
                elif isinstance(ts, str):
                    # Try parsing ISO format
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    continue

                points.append(PricePoint(
                    timestamp=timestamp,
                    price=float(price),
                ))
            except Exception as e:
                logger.debug(f"Failed to parse price point: {point}, error: {e}")
                continue

        # Sort by timestamp ascending
        points.sort(key=lambda p: p.timestamp)

        return PriceHistory(
            token_id=token_id,
            interval=interval,
            points=points,
            points_count=len(points),
            fetched_at=datetime.utcnow(),
        )

    async def get_price_history_max(self, token_id: str, fidelity: int = 60) -> PriceHistory | None:
        """Get maximum available historical price data for a token.

        Convenience method to fetch all available price history.

        Args:
            token_id: Token identifier.
            fidelity: Data resolution in minutes (default: 60 = hourly).

        Returns:
            PriceHistory with all available data points.
        """
        return await self.get_price_history(token_id, interval="max", fidelity=fidelity)

