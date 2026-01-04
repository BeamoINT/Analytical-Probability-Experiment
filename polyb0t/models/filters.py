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
        
        # Minimum orderbook depth thresholds (USD)
        self.min_bid_depth_usd = 50.0  # Minimum bid side depth
        self.min_ask_depth_usd = 50.0  # Minimum ask side depth
        self.min_total_depth_usd = 100.0  # Minimum combined depth
        
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
    ) -> tuple[list[Market], dict[str, int]]:
        """Apply all filters to market list with detailed rejection tracking.

        Args:
            markets: List of markets to filter.
            orderbooks: Optional dict of token_id -> OrderBook for spread filtering.

        Returns:
            Tuple of (filtered markets, rejection_reasons dict).
        """
        tradable = []
        rejection_reasons: dict[str, int] = {}

        for market in markets:
            passes, reason = self._passes_filters_with_reason(market, orderbooks)
            if passes:
                tradable.append(market)
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        logger.info(
            f"Filtered {len(markets)} markets down to {len(tradable)} tradable markets",
            extra={"rejection_reasons": rejection_reasons},
        )
        return tradable, rejection_reasons

    def _passes_filters_with_reason(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook] | None = None,
    ) -> tuple[bool, str]:
        """Check if market passes all filters with detailed reason.

        Args:
            market: Market to check.
            orderbooks: Optional orderbook data.

        Returns:
            Tuple of (passes, reason).
        """
        # Blacklist check
        if market.condition_id in self.blacklist:
            return False, "blacklisted"

        # Active/closed check
        if not market.active or market.closed:
            return False, "inactive_or_closed"

        # Resolution time check
        if not self._check_resolution_time(market):
            return False, "resolution_time_out_of_range"

        # Liquidity check (volume/liquidity from Gamma)
        if not self._check_liquidity(market):
            return False, "insufficient_gamma_liquidity"

        # Orderbook checks (if available)
        if orderbooks:
            # Missing orderbook check
            has_orderbook = any(ob is not None for ob in [orderbooks.get(o.token_id) for o in market.outcomes])
            if not has_orderbook:
                return False, "missing_orderbook"
            
            # Spread check
            if not self._check_spread(market, orderbooks):
                return False, "spread_too_wide"
            
            # Depth check
            if not self._check_orderbook_depth(market, orderbooks):
                return False, "insufficient_orderbook_depth"
            
            # Staleness check
            if not self._check_orderbook_freshness(market, orderbooks):
                return False, "stale_orderbook"

        return True, "passed"

    def _passes_filters(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook] | None = None,
    ) -> bool:
        """Check if market passes all filters (legacy method).

        Args:
            market: Market to check.
            orderbooks: Optional orderbook data.

        Returns:
            True if market passes all filters.
        """
        passes, _ = self._passes_filters_with_reason(market, orderbooks)
        return passes

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
    
    def _check_orderbook_depth(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook],
    ) -> bool:
        """Check if market has sufficient orderbook depth.

        Args:
            market: Market to check.
            orderbooks: Dict of token_id -> OrderBook.

        Returns:
            True if all outcomes have sufficient depth.
        """
        for outcome in market.outcomes:
            orderbook = orderbooks.get(outcome.token_id)
            if not orderbook:
                continue

            # Calculate depth at top levels (top 3 levels)
            bid_depth = self._calculate_depth_usd(orderbook.bids[:3])
            ask_depth = self._calculate_depth_usd(orderbook.asks[:3])
            total_depth = bid_depth + ask_depth

            # Check minimums
            if bid_depth < self.min_bid_depth_usd:
                logger.debug(
                    f"Market {market.condition_id} outcome {outcome.outcome} "
                    f"bid depth ${bid_depth:.2f} < minimum ${self.min_bid_depth_usd:.2f}"
                )
                return False
            
            if ask_depth < self.min_ask_depth_usd:
                logger.debug(
                    f"Market {market.condition_id} outcome {outcome.outcome} "
                    f"ask depth ${ask_depth:.2f} < minimum ${self.min_ask_depth_usd:.2f}"
                )
                return False
            
            if total_depth < self.min_total_depth_usd:
                logger.debug(
                    f"Market {market.condition_id} outcome {outcome.outcome} "
                    f"total depth ${total_depth:.2f} < minimum ${self.min_total_depth_usd:.2f}"
                )
                return False

        return True
    
    def _calculate_depth_usd(self, levels: list[Any]) -> float:
        """Calculate total depth in USD for given levels.

        Args:
            levels: List of orderbook levels (bids or asks).

        Returns:
            Total depth in USD.
        """
        total_depth = 0.0
        for level in levels:
            # Depth = size * price (shares * price = USD)
            depth_usd = level.size * level.price
            total_depth += depth_usd
        return total_depth
    
    def _check_orderbook_freshness(
        self,
        market: Market,
        orderbooks: dict[str, OrderBook],
    ) -> bool:
        """Check if orderbook data is recent (not stale).

        Args:
            market: Market to check.
            orderbooks: Dict of token_id -> OrderBook.

        Returns:
            True if orderbook is fresh enough.
        """
        max_age_seconds = self.settings.max_stale_data_seconds
        now = datetime.now(timezone.utc)

        for outcome in market.outcomes:
            orderbook = orderbooks.get(outcome.token_id)
            if not orderbook or not orderbook.timestamp:
                continue

            timestamp = orderbook.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            age_seconds = (now - timestamp).total_seconds()

            if age_seconds > max_age_seconds:
                logger.debug(
                    f"Market {market.condition_id} outcome {outcome.outcome} "
                    f"orderbook is {age_seconds:.0f}s old (max {max_age_seconds}s)"
                )
                return False

        return True

