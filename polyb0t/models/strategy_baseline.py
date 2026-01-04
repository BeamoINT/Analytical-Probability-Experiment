"""Baseline trading strategy with explainable logic."""

import logging
from datetime import datetime
from typing import Any

import numpy as np

from polyb0t.config import get_settings
from polyb0t.data.models import Market, OrderBook, Trade
from polyb0t.models.features import FeatureEngine

logger = logging.getLogger(__name__)


class TradingSignal:
    """Trading signal with edge calculation."""

    def __init__(
        self,
        token_id: str,
        market_id: str,
        side: str,  # "BUY" or "SELL"
        p_market: float,
        p_model: float,
        edge: float,
        confidence: float,
        features: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize trading signal.

        Args:
            token_id: Token identifier.
            market_id: Market condition ID.
            side: Trade direction (BUY or SELL).
            p_market: Market implied probability.
            p_model: Model probability estimate.
            edge: Expected edge (p_model - p_market for BUY).
            confidence: Signal confidence (0-1).
            features: Feature dictionary.
            timestamp: Signal generation time.
        """
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.p_market = p_market
        self.p_model = p_model
        self.edge = edge
        self.confidence = confidence
        self.features = features
        self.timestamp = timestamp or datetime.utcnow()


class BaselineStrategy:
    """Baseline trading strategy using value-based approach.

    Strategy:
        1. Estimate market probability from orderbook/trades (p_market).
        2. Compute model probability using shrinkage + features (p_model).
        3. Calculate edge = p_model - p_market (for BUY side).
        4. Generate signal if abs(edge) >= threshold.

    Model Approach:
        - Shrinkage toward 0.5 (reduces overconfidence).
        - Momentum adjustment (recent price trends).
        - Mean-reversion component (extremes tend to revert).
        - Volume-weighted confidence.
    """

    def __init__(self) -> None:
        """Initialize baseline strategy."""
        self.settings = get_settings()
        self.feature_engine = FeatureEngine()
        self.shrinkage_factor = 0.3  # Weight on prior (0.5)
        self.momentum_weight = 0.2
        self.mean_reversion_weight = 0.1

    def generate_signals(
        self,
        markets: list[Market],
        orderbooks: dict[str, OrderBook],
        trades: dict[str, list[Trade]],
    ) -> list[TradingSignal]:
        """Generate trading signals for all tradable markets.

        Args:
            markets: List of filtered markets.
            orderbooks: Dict of token_id -> OrderBook.
            trades: Dict of token_id -> list of recent trades.

        Returns:
            List of TradingSignal objects meeting edge threshold.
        """
        signals: list[TradingSignal] = []
        price_source_counts: dict[str, int] = {"orderbook_mid": 0, "last_trade": 0, "gamma_price": 0, "none": 0}

        for market in markets:
            for idx, outcome in enumerate(market.outcomes):
                token_id = outcome.token_id
                orderbook = orderbooks.get(token_id)
                token_trades = trades.get(token_id, [])

                signal = self._generate_signal(market, idx, orderbook, token_trades)
                if signal:
                    signals.append(signal)
                # Count price sources for observability (even if no signal)
                src = None
                try:
                    src = signal.features.get("p_market_source") if signal else None
                except Exception:
                    src = None
                # If no signal, still try to infer the price source used/available
                if src is None:
                    gamma_price = market.outcomes[idx].price if idx < len(market.outcomes) else None
                    p, src2 = self.feature_engine.compute_implied_probability_with_source(
                        orderbook=orderbook,
                        last_price=(token_trades[-1].price if token_trades else None),
                        gamma_price=gamma_price,
                    )
                    src = src2
                price_source_counts[src or "none"] = price_source_counts.get(src or "none", 0) + 1

        logger.info(
            f"Generated {len(signals)} signals meeting threshold",
            extra={"price_sources": price_source_counts},
        )
        return signals

    def _generate_signal(
        self,
        market: Market,
        outcome_idx: int,
        orderbook: OrderBook | None,
        recent_trades: list[Trade],
    ) -> TradingSignal | None:
        """Generate signal for a single outcome.

        Args:
            market: Market data.
            outcome_idx: Outcome index.
            orderbook: OrderBook data.
            recent_trades: Recent trades.

        Returns:
            TradingSignal if edge threshold met, else None.
        """
        # Compute features
        features = self.feature_engine.compute_features(
            market, outcome_idx, orderbook, recent_trades
        )

        # Market probability with fallback:
        #  a) orderbook mid
        #  b) last trade
        #  c) Gamma outcome price
        #  d) None => skip
        gamma_price = None
        if outcome_idx < len(market.outcomes):
            gamma_price = market.outcomes[outcome_idx].price

        p_market, p_market_source = self.feature_engine.compute_implied_probability_with_source(
            orderbook=orderbook,
            last_price=features.get("last_price"),
            gamma_price=gamma_price,
        )

        if p_market is None:
            return None

        features["p_market_source"] = p_market_source

        # Model probability
        p_model = self._compute_model_probability(p_market, features)

        # Edge calculation
        edge = p_model - p_market

        # Confidence based on data quality
        confidence = self._compute_confidence(features)

        # Check threshold
        if abs(edge) < self.settings.edge_threshold:
            return None

        # Determine side
        side = "BUY" if edge > 0 else "SELL"

        outcome = market.outcomes[outcome_idx]
        return TradingSignal(
            token_id=outcome.token_id,
            market_id=market.condition_id,
            side=side,
            p_market=p_market,
            p_model=p_model,
            edge=edge,
            confidence=confidence,
            features=features,
        )

    def _compute_model_probability(
        self,
        p_market: float,
        features: dict[str, Any],
    ) -> float:
        """Compute model probability estimate.

        Args:
            p_market: Market implied probability.
            features: Feature dictionary.

        Returns:
            Model probability estimate (0-1).
        """
        # Start with shrinkage toward 0.5 (reduces overconfidence)
        prior = 0.5
        p_base = (1 - self.shrinkage_factor) * p_market + self.shrinkage_factor * prior

        # Momentum adjustment
        momentum = features.get("momentum", 0)
        momentum_adj = self.momentum_weight * momentum

        # Mean reversion: extreme prices tend to revert
        distance_from_center = abs(p_market - 0.5)
        mean_reversion_adj = -self.mean_reversion_weight * distance_from_center * np.sign(
            p_market - 0.5
        )

        # Combine adjustments
        p_model = p_base + momentum_adj + mean_reversion_adj

        # Clamp to valid range
        p_model = max(0.01, min(0.99, p_model))

        return p_model

    def _compute_confidence(self, features: dict[str, Any]) -> float:
        """Compute signal confidence based on data quality.

        Args:
            features: Feature dictionary.

        Returns:
            Confidence score (0-1).
        """
        confidence = 0.5  # Base confidence

        # Higher confidence with more trades
        num_trades = features.get("num_trades", 0)
        if num_trades > 10:
            confidence += 0.2
        elif num_trades > 5:
            confidence += 0.1

        # Higher confidence with tighter spreads
        spread_pct = features.get("spread_pct", 1.0)
        if spread_pct < 0.02:
            confidence += 0.2
        elif spread_pct < 0.05:
            confidence += 0.1

        # Higher confidence with good depth
        bid_depth = features.get("bid_depth", 0)
        ask_depth = features.get("ask_depth", 0)
        total_depth = bid_depth + ask_depth
        if total_depth > 1000:
            confidence += 0.1

        return min(1.0, confidence)

