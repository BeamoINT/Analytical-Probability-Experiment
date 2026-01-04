"""Market filtering pipeline."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from polyb0t.config import get_settings
from polyb0t.data.models import Market, OrderBook

logger = logging.getLogger(__name__)


class MarketFilter:
    """Filter markets based on resolution time, liquidity, and spread criteria."""

    def __init__(self) -> None:
        """Initialize market filter with settings."""
        self.settings = get_settings()
        self.blacklist: set[str] = set()  # Manual blacklist for ambiguous markets

    def load_blacklist(self, blacklist: list[str] | None = None) -> None:
        """Load manual blacklist of market IDs to exclude.

        Args:
            blacklist: List of condition IDs to exclude.
        """
        if blacklist:
            self.blacklist = set(blacklist)
            logger.info(f"Loaded {len(self.blacklist)} markets into blacklist")

    def filter_markets(
        self,
        markets: list[Market],
        orderbooks: dict[str, OrderBook] | None = None,
    ) -> list[Market]:
        """Apply all filters to market list.

        Args:
            markets: List of markets to filter.
            orderbooks: Optional dict of token_id -> OrderBook for spread filtering.

        Returns:
            Filtered list of tradable markets.
        """
        tradable = []

        for market in markets:
            if not self._passes_filters(market, orderbooks):
                continue
            tradable.append(market)

        logger.info(
            f"Filtered {len(markets)} markets down to {len(tradable)} tradable markets"
        )
        return tradable

    def _passes_filters(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook] | None = None,
    ) -> bool:
        """Check if market passes all filters.

        Args:
            market: Market to check.
            orderbooks: Optional orderbook data.

        Returns:
            True if market passes all filters.
        """
        # Blacklist check
        if market.condition_id in self.blacklist:
            logger.debug(f"Market {market.condition_id} is blacklisted")
            return False

        # Active/closed check
        if not market.active or market.closed:
            logger.debug(f"Market {market.condition_id} is not active or is closed")
            return False

        # Resolution time check
        if not self._check_resolution_time(market):
            return False

        # Liquidity check
        if not self._check_liquidity(market):
            return False

        # Spread check (if orderbook available)
        if orderbooks and not self._check_spread(market, orderbooks):
            return False

        return True

    def _check_resolution_time(self, market: Market) -> bool:
        """Check if market resolves within configured time window.

        Args:
            market: Market to check.

        Returns:
            True if resolution time is within bounds.
        """
        if market.end_date is None:
            logger.debug(f"Market {market.condition_id} has no end_date")
            return False

        end_date = market.end_date
        now = datetime.now(timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        days_until_resolution = (end_date - now).total_seconds() / 86400

        min_days = self.settings.resolve_min_days
        max_days = self.settings.resolve_max_days

        if not (min_days <= days_until_resolution <= max_days):
            logger.debug(
                f"Market {market.condition_id} resolves in {days_until_resolution:.1f} days "
                f"(outside {min_days}-{max_days} day window)"
            )
            return False

        return True

    def _check_liquidity(self, market: Market) -> bool:
        """Check if market has sufficient liquidity.

        Args:
            market: Market to check.

        Returns:
            True if liquidity meets minimum threshold.
        """
        # Use liquidity field if available, otherwise volume as proxy
        market_liquidity = market.liquidity or market.volume or 0

        if market_liquidity < self.settings.min_liquidity:
            logger.debug(
                f"Market {market.condition_id} liquidity {market_liquidity:.2f} "
                f"< minimum {self.settings.min_liquidity}"
            )
            return False

        return True

    def _check_spread(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook],
    ) -> bool:
        """Check if market has acceptable bid-ask spread.

        Args:
            market: Market to check.
            orderbooks: Dict of token_id -> OrderBook.

        Returns:
            True if all outcome spreads are acceptable.
        """
        for outcome in market.outcomes:
            orderbook = orderbooks.get(outcome.token_id)
            if not orderbook:
                continue

            spread = self._calculate_spread(orderbook)
            if spread is None:
                continue

            if spread > self.settings.max_spread:
                logger.debug(
                    f"Market {market.condition_id} outcome {outcome.outcome} "
                    f"spread {spread:.4f} > max {self.settings.max_spread}"
                )
                return False

        return True

    def _calculate_spread(self, orderbook: OrderBook) -> float | None:
        """Calculate relative bid-ask spread.

        Args:
            orderbook: OrderBook snapshot.

        Returns:
            Relative spread or None if cannot calculate.
        """
        if not orderbook.bids or not orderbook.asks:
            return None

        best_bid = orderbook.bids[0].price
        best_ask = orderbook.asks[0].price

        if best_bid <= 0 or best_ask <= 0:
            return None

        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return None

        spread = (best_ask - best_bid) / mid
        return spread

