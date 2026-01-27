"""Gamma Markets API client for Polymarket market data."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from polyb0t.config import get_settings
from polyb0t.data.models import Market, MarketOutcome

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for Polymarket Gamma Markets API (read-only)."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        """Initialize Gamma client.

        Args:
            base_url: Base URL for Gamma API. Defaults to config value.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.gamma_base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "polyb0t/0.1.0",
            },
        )

    async def __aenter__(self) -> "GammaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

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
            logger.debug(f"Gamma API request failed: {e}")
            return None, None

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

    async def list_markets_debug(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Market], dict[str, Any]]:
        """List markets, returning (markets, diagnostics) including HTTP status counts."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        status, data = await self._get_with_status("/markets", params=params)
        diag: dict[str, Any] = {
            "endpoint": "/markets",
            "status": status,
            "parsed": 0,
            "failed_parse": 0,
        }
        if data is None:
            return [], diag

        markets: list[Market] = []
        market_list = data if isinstance(data, list) else data.get("markets", [])
        for item in market_list:
            try:
                market = self._parse_market(item)
                markets.append(market)
                diag["parsed"] += 1
            except Exception as e:
                logger.debug(f"Failed to parse market: {e}")
                diag["failed_parse"] += 1
                continue
        return markets, diag

    async def get_market_debug(self, market_id: str) -> tuple[Market | None, dict[str, Any]]:
        """Get single market, returning (Market|None, diagnostics including HTTP status)."""
        endpoint = f"/markets/{market_id}"
        status, data = await self._get_with_status(endpoint)
        diag: dict[str, Any] = {"endpoint": endpoint, "status": status}
        if data is None:
            return None, diag
        try:
            return self._parse_market(data), diag
        except Exception as e:
            diag["parse_error"] = str(e)
            return None, diag

    async def list_markets(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """List markets from Gamma API.

        Args:
            active: Filter by active status.
            closed: Filter by closed status.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of Market objects.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()

        try:
            data = await self._get("/markets", params=params)
            markets = []

            # Parse response - structure may vary, adapt as needed
            market_list = data if isinstance(data, list) else data.get("markets", [])

            for item in market_list:
                try:
                    market = self._parse_market(item)
                    markets.append(market)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse market: {e}",
                        extra={"market_data": item},
                    )
                    continue

            logger.info(f"Fetched {len(markets)} markets")
            return markets

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching markets: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error fetching markets: {e}")
            raise

    async def list_all_markets(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        batch_size: int = 500,
        max_markets: int = 5000,
    ) -> tuple[list[Market], dict[str, Any]]:
        """Fetch all available markets using pagination.

        Args:
            active: Filter by active status.
            closed: Filter by closed status.
            batch_size: Number of markets to fetch per request.
            max_markets: Maximum total markets to fetch.

        Returns:
            Tuple of (list of Market objects, diagnostics dict).
        """
        all_markets: list[Market] = []
        seen_ids: set[str] = set()  # Track condition_ids to avoid duplicates
        offset = 0
        total_parsed = 0
        total_failed = 0
        pages_fetched = 0
        duplicates_skipped = 0

        while True:
            markets, diag = await self.list_markets_debug(
                active=active, closed=closed,
                limit=batch_size, offset=offset
            )
            pages_fetched += 1
            total_parsed += diag.get("parsed", 0)
            total_failed += diag.get("failed_parse", 0)

            if not markets:
                break

            # Deduplicate by condition_id
            for market in markets:
                if market.condition_id not in seen_ids:
                    seen_ids.add(market.condition_id)
                    all_markets.append(market)
                else:
                    duplicates_skipped += 1

            logger.debug(f"Pagination: fetched {len(all_markets)} unique markets (page {pages_fetched})")

            if len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                break

            if len(markets) < batch_size:
                # Last page reached
                break

            offset += batch_size

        diagnostics = {
            "total_markets": len(all_markets),
            "pages_fetched": pages_fetched,
            "total_parsed": total_parsed,
            "total_failed": total_failed,
            "duplicates_skipped": duplicates_skipped,
        }
        logger.info(f"Fetched {len(all_markets)} markets via pagination ({pages_fetched} pages, {duplicates_skipped} duplicates skipped)")
        return all_markets, diagnostics

    async def get_market(self, condition_id: str) -> Market | None:
        """Get single market by condition ID.

        Args:
            condition_id: Market condition ID.

        Returns:
            Market object or None if not found.
        """
        try:
            data = await self._get(f"/markets/{condition_id}")
            return self._parse_market(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Market not found: {condition_id}")
                return None
            logger.error(f"HTTP error fetching market {condition_id}: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Error fetching market {condition_id}: {e}")
            raise

    def _parse_market(self, data: dict[str, Any]) -> Market:
        """Parse market data from API response.

        Args:
            data: Raw market data from API.

        Returns:
            Market object.

        Note:
            This parser makes assumptions about Gamma API structure.
            Adjust field names based on actual API response.
        """
        # Parse outcomes
        outcomes: list[MarketOutcome] = []

        # Gamma payloads vary. Try a few common structures:
        # - outcomes: [{ outcome/name, token_id/id/tokenId, price/probability/last }]
        # - tokens:   [{ token_id/id/tokenId, outcome/name, price }]
        # - outcomes/clobTokenIds/outcomePrices as parallel arrays (often JSON-encoded strings)
        raw_outcomes = data.get("outcomes")
        raw_tokens = data.get("tokens")

        # Some Gamma fields come back as JSON-encoded strings. Normalize.
        def _maybe_json_list(v: Any) -> Any:
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        return json.loads(s)
                    except (json.JSONDecodeError, TypeError):
                        return v
            return v

        raw_outcomes = _maybe_json_list(raw_outcomes)
        raw_tokens = _maybe_json_list(raw_tokens)

        candidate_list: list[dict[str, Any]] = []
        if isinstance(raw_outcomes, list):
            candidate_list = [o for o in raw_outcomes if isinstance(o, dict)]
        elif isinstance(raw_tokens, list):
            candidate_list = [t for t in raw_tokens if isinstance(t, dict)]

        def _get_token_id(obj: dict[str, Any]) -> str:
            tok = obj.get("token_id") or obj.get("tokenId") or obj.get("id") or obj.get("token")
            return str(tok) if tok is not None else ""

        def _get_outcome_name(obj: dict[str, Any]) -> str:
            return str(obj.get("outcome") or obj.get("name") or obj.get("label") or "Unknown")

        def _get_price(obj: dict[str, Any]) -> float | None:
            for k in ("price", "probability", "mid", "last", "last_price", "lastPrice"):
                if obj.get(k) is not None:
                    try:
                        return float(obj.get(k))
                    except (ValueError, TypeError):
                        continue
            return None

        for obj in candidate_list:
            outcomes.append(
                MarketOutcome(
                    token_id=_get_token_id(obj),
                    outcome=_get_outcome_name(obj),
                    price=_get_price(obj),
                )
            )

        # Fallback: try parallel arrays (best effort). These are commonly JSON-encoded strings.
        if not outcomes:
            names = _maybe_json_list(
                data.get("outcomeNames")
                or data.get("outcome_names")
                or data.get("outcomeLabels")
                or data.get("outcomes")
            )
            token_ids = _maybe_json_list(
                data.get("tokenIds")
                or data.get("token_ids")
                or data.get("outcomeTokenIds")
                or data.get("clobTokenIds")
            )
            prices = _maybe_json_list(
                data.get("outcomePrices")
                or data.get("outcome_prices")
                or data.get("prices")
            )
            if isinstance(names, list) and isinstance(token_ids, list):
                for i, name in enumerate(names):
                    tok = token_ids[i] if i < len(token_ids) else ""
                    pr = prices[i] if isinstance(prices, list) and i < len(prices) else None
                    try:
                        pr_f = float(pr) if pr is not None else None
                    except (ValueError, TypeError):
                        pr_f = None
                    outcomes.append(
                        MarketOutcome(
                            token_id=str(tok) if tok is not None else "",
                            outcome=str(name),
                            price=pr_f,
                        )
                    )

        # Parse end date
        end_date = None
        end_date_str = data.get("end_date") or data.get("endDate") or data.get("end_time")
        if end_date_str:
            try:
                if isinstance(end_date_str, str):
                    end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                elif isinstance(end_date_str, (int, float)):
                    end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to parse end_date: {end_date_str}, error: {e}")

        # Normalize to timezone-aware UTC to avoid naive/aware arithmetic bugs
        if end_date is not None and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        return Market(
            # Gamma uses both `id` (market id) and `conditionId` (on-chain). For our system,
            # we keep using the market id for fetching `/markets/{id}` and store raw payload in metadata.
            condition_id=str(data.get("id") or data.get("condition_id") or data.get("conditionId") or ""),
            question=data.get("question", data.get("title", "")),
            description=data.get("description"),
            end_date=end_date,
            outcomes=outcomes,
            category=data.get("category"),
            volume=data.get("volume"),
            liquidity=data.get("liquidity"),
            active=data.get("active", True),
            closed=data.get("closed", False),
            metadata=data,
        )

