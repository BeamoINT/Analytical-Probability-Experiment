"""Feature engineering for trading signals."""

import logging
from typing import Any

import numpy as np

from polyb0t.data.models import Market, OrderBook, Trade

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute trading features from market data."""

    def compute_features(
        self,
        market: Market,
        outcome_idx: int,
        orderbook: OrderBook | None = None,
        recent_trades: list[Trade] | None = None,
    ) -> dict[str, Any]:
        """Compute features for a market outcome.

        Args:
            market: Market data.
            outcome_idx: Index of outcome to compute features for.
            orderbook: Optional orderbook data.
            recent_trades: Optional recent trades.

        Returns:
            Dictionary of computed features.
        """
        features: dict[str, Any] = {
            "market_id": market.condition_id,
            "outcome_idx": outcome_idx,
        }

        if outcome_idx >= len(market.outcomes):
            logger.warning(f"Invalid outcome_idx {outcome_idx} for market {market.condition_id}")
            return features

        outcome = market.outcomes[outcome_idx]
        features["token_id"] = outcome.token_id
        features["outcome"] = outcome.outcome

        # Market features
        features["volume"] = market.volume or 0
        features["liquidity"] = market.liquidity or 0
        features["category"] = market.category

        # Price features from orderbook
        if orderbook:
            price_features = self._compute_price_features(orderbook)
            features.update(price_features)

        # Trade features
        if recent_trades:
            trade_features = self._compute_trade_features(recent_trades)
            features.update(trade_features)

        return features

    def _compute_price_features(self, orderbook: OrderBook) -> dict[str, Any]:
        """Compute price-based features from orderbook.

        Args:
            orderbook: OrderBook snapshot.

        Returns:
            Dictionary of price features.
        """
        features: dict[str, Any] = {}

        if orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price
            best_ask = orderbook.asks[0].price
            mid_price = (best_bid + best_ask) / 2

            features["best_bid"] = best_bid
            features["best_ask"] = best_ask
            features["mid_price"] = mid_price
            features["spread"] = best_ask - best_bid
            features["spread_pct"] = (best_ask - best_bid) / mid_price if mid_price > 0 else 0

            # Depth features
            bid_depth = sum(level.size for level in orderbook.bids[:5])
            ask_depth = sum(level.size for level in orderbook.asks[:5])

            features["bid_depth"] = bid_depth
            features["ask_depth"] = ask_depth
            features["depth_imbalance"] = (
                (bid_depth - ask_depth) / (bid_depth + ask_depth)
                if (bid_depth + ask_depth) > 0
                else 0
            )

        return features

    def _compute_trade_features(self, trades: list[Trade]) -> dict[str, Any]:
        """Compute features from recent trades.

        Args:
            trades: List of recent trades.

        Returns:
            Dictionary of trade features.
        """
        features: dict[str, Any] = {"num_trades": len(trades)}

        if not trades:
            return features

        prices = [t.price for t in trades]
        sizes = [t.size for t in trades]

        features["last_price"] = prices[-1] if prices else None
        features["avg_trade_price"] = np.mean(prices)
        features["price_std"] = np.std(prices) if len(prices) > 1 else 0
        features["total_volume"] = sum(sizes)

        # Buy/sell pressure
        buy_volume = sum(t.size for t in trades if t.side == "BUY")
        sell_volume = sum(t.size for t in trades if t.side == "SELL")
        total_volume = buy_volume + sell_volume

        features["buy_volume"] = buy_volume
        features["sell_volume"] = sell_volume
        features["buy_pressure"] = buy_volume / total_volume if total_volume > 0 else 0.5

        # Momentum: compare recent vs earlier prices
        if len(prices) >= 4:
            recent_avg = np.mean(prices[-len(prices) // 2 :])
            earlier_avg = np.mean(prices[: len(prices) // 2])
            features["momentum"] = (
                (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            )
        else:
            features["momentum"] = 0

        return features

    def compute_implied_probability(
        self,
        orderbook: OrderBook | None = None,
        last_price: float | None = None,
        gamma_price: float | None = None,
    ) -> float | None:
        """Compute implied probability from market prices.

        Args:
            orderbook: Optional orderbook to use mid price.
            last_price: Optional last trade price.
            gamma_price: Optional price from Gamma market payload.

        Returns:
            Implied probability (0-1), or None if no price source available.
        """
        p, _src = self.compute_implied_probability_with_source(
            orderbook=orderbook, last_price=last_price, gamma_price=gamma_price
        )
        return p

    def compute_implied_probability_with_source(
        self,
        orderbook: OrderBook | None = None,
        last_price: float | None = None,
        gamma_price: float | None = None,
    ) -> tuple[float | None, str | None]:
        """Compute implied probability and report which source was used.

        Source priority:
          a) orderbook mid of best bid/ask
          b) last trade price
          c) Gamma price
          d) None
        """
        if orderbook and orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price
            best_ask = orderbook.asks[0].price
            mid_price = (best_bid + best_ask) / 2
            return self._clamp_probability(mid_price), "orderbook_mid"

        if last_price is not None:
            return self._clamp_probability(last_price), "last_trade"

        if gamma_price is not None:
            return self._clamp_probability(gamma_price), "gamma_price"

        return None, None

    def _clamp_probability(self, p: float) -> float:
        """Clamp probability to valid range with small margin.

        Args:
            p: Raw probability.

        Returns:
            Clamped probability in [0.01, 0.99].
        """
        return max(0.01, min(0.99, p))

