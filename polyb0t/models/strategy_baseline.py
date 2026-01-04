"""Baseline trading strategy with explainable logic."""

import logging
from datetime import datetime
from typing import Any

import numpy as np

from polyb0t.config import get_settings
from polyb0t.data.models import Market, OrderBook, Trade
from polyb0t.models.features import FeatureEngine
from polyb0t.models.fill_estimation import FillPriceEstimator, FillEstimate
from polyb0t.models.position_sizing import PositionSizer, SizingResult

logger = logging.getLogger(__name__)


class TradingSignal:
    """Trading signal with edge calculation and fill estimation."""

    def __init__(
        self,
        token_id: str,
        market_id: str,
        side: str,  # "BUY" or "SELL"
        p_market: float,
        p_model: float,
        edge: float,
        edge_raw: float,  # Raw mid-price edge
        edge_net: float,  # Net edge after fills/fees
        confidence: float,
        features: dict[str, Any],
        fill_estimate: FillEstimate | None = None,
        sizing_result: SizingResult | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize trading signal.

        Args:
            token_id: Token identifier.
            market_id: Market condition ID.
            side: Trade direction (BUY or SELL).
            p_market: Market implied probability (mid-price).
            p_model: Model probability estimate.
            edge: Primary edge metric (net edge).
            edge_raw: Raw edge based on mid-price.
            edge_net: Net edge based on expected fill.
            confidence: Signal confidence (0-1).
            features: Feature dictionary.
            fill_estimate: Expected fill pricing details.
            sizing_result: Position sizing calculation.
            timestamp: Signal generation time.
        """
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.p_market = p_market
        self.p_model = p_model
        self.edge = edge  # Use net edge as primary metric
        self.edge_raw = edge_raw
        self.edge_net = edge_net
        self.confidence = confidence
        self.features = features
        self.fill_estimate = fill_estimate
        self.sizing_result = sizing_result
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
        self.fill_estimator = FillPriceEstimator()
        self.position_sizer = PositionSizer()
        self.shrinkage_factor = 0.3  # Weight on prior (0.5)
        self.momentum_weight = 0.2
        self.mean_reversion_weight = 0.1
        
        # Minimum net edge threshold (after fees/slippage)
        self.min_net_edge = 0.02  # Require at least 2% net edge
        
        # ML components (optional)
        self.ml_feature_engine = None
        self.ml_model_manager = None
        self.ml_data_collector = None
        self.ml_model_updater = None
        
        # Initialize ML if enabled
        if self.settings.enable_ml:
            self._initialize_ml_components()

    def generate_signals(
        self,
        markets: list[Market],
        orderbooks: dict[str, OrderBook],
        trades: dict[str, list[Trade]],
        available_usdc: float = 0.0,
        reserved_usdc: float = 0.0,
    ) -> tuple[list[TradingSignal], dict[str, int]]:
        """Generate trading signals for all tradable markets.

        Args:
            markets: List of filtered markets.
            orderbooks: Dict of token_id -> OrderBook.
            trades: Dict of token_id -> list of recent trades.
            available_usdc: Available USDC balance.
            reserved_usdc: Reserved USDC (for position sizing).

        Returns:
            Tuple of (signals meeting threshold, rejection_reasons dict).
        """
        signals: list[TradingSignal] = []
        price_source_counts: dict[str, int] = {"orderbook_mid": 0, "last_trade": 0, "gamma_price": 0, "none": 0}
        rejection_reasons: dict[str, int] = {}

        for market in markets:
            for idx, outcome in enumerate(market.outcomes):
                token_id = outcome.token_id
                orderbook = orderbooks.get(token_id)
                token_trades = trades.get(token_id, [])

                signal, reason = self._generate_signal_with_reason(
                    market, idx, orderbook, token_trades, available_usdc, reserved_usdc
                )
                if signal:
                    signals.append(signal)
                else:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    
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
            extra={"price_sources": price_source_counts, "rejection_reasons": rejection_reasons},
        )
        return signals, rejection_reasons

    def _generate_signal_with_reason(
        self,
        market: Market,
        outcome_idx: int,
        orderbook: OrderBook | None,
        recent_trades: list[Trade],
        available_usdc: float,
        reserved_usdc: float,
    ) -> tuple[TradingSignal | None, str]:
        """Generate signal for a single outcome with rejection reason.

        Args:
            market: Market data.
            outcome_idx: Outcome index.
            orderbook: OrderBook data.
            recent_trades: Recent trades.
            available_usdc: Available balance.
            reserved_usdc: Reserved balance.

        Returns:
            Tuple of (TradingSignal if threshold met, rejection_reason).
        """
        # Compute features
        features = self.feature_engine.compute_features(
            market, outcome_idx, orderbook, recent_trades
        )

        # Market probability with fallback
        gamma_price = None
        if outcome_idx < len(market.outcomes):
            gamma_price = market.outcomes[outcome_idx].price

        p_market, p_market_source = self.feature_engine.compute_implied_probability_with_source(
            orderbook=orderbook,
            last_price=features.get("last_price"),
            gamma_price=gamma_price,
        )

        if p_market is None:
            return None, "no_market_price"

        features["p_market_source"] = p_market_source

        # Model probability
        p_model = self._compute_model_probability(p_market, features)

        # Raw edge (mid-price based)
        edge_raw = p_model - p_market
        
        # Check raw edge threshold
        if abs(edge_raw) < self.settings.edge_threshold:
            return None, "raw_edge_below_threshold"

        # Determine side
        side = "BUY" if edge_raw > 0 else "SELL"
        
        # Confidence based on data quality
        confidence = self._compute_confidence(features)

        # Estimate initial position size for fill estimation
        sizing = self.position_sizer.compute_size(
            edge_net=abs(edge_raw),  # Use raw edge for initial sizing
            confidence=confidence,
            available_usdc=available_usdc,
            reserved_usdc=reserved_usdc,
        )
        
        # If size is below minimum, reject
        if sizing.size_usd_final < self.settings.min_order_usd:
            return None, f"size_below_minimum_{sizing.sizing_reason}"

        # Compute expected fill price and net edge
        if not orderbook:
            return None, "no_orderbook_for_fill_estimation"
            
        edge_raw_calc, edge_net, fill_est = self.fill_estimator.compute_net_edge(
            p_model=p_model,
            p_market_mid=p_market,
            orderbook=orderbook,
            side=side,
            size_usd=sizing.size_usd_final,
        )

        # Check if fill is feasible
        if not fill_est or not fill_est.is_feasible:
            return None, "fill_not_feasible"
        
        # Check net edge threshold (after fees/slippage)
        if abs(edge_net) < self.min_net_edge:
            return None, "net_edge_below_threshold"
        
        # Recompute sizing with net edge
        sizing_final = self.position_sizer.compute_size(
            edge_net=abs(edge_net),
            confidence=confidence,
            available_usdc=available_usdc,
            reserved_usdc=reserved_usdc,
        )
        
        # Final size check
        if sizing_final.size_usd_final < self.settings.min_order_usd:
            return None, f"final_size_below_minimum_{sizing_final.sizing_reason}"

        # Store fill and sizing info in features
        features["fill_expected_price"] = fill_est.expected_price
        features["fill_price_impact_pct"] = fill_est.price_impact_pct
        features["fill_slippage_bps"] = fill_est.slippage_bps
        features["fill_levels_consumed"] = fill_est.levels_consumed
        features["sizing_reason"] = sizing_final.sizing_reason
        features["kelly_fraction"] = sizing_final.kelly_fraction

        outcome = market.outcomes[outcome_idx]
        return TradingSignal(
            token_id=outcome.token_id,
            market_id=market.condition_id,
            side=side,
            p_market=p_market,
            p_model=p_model,
            edge=edge_net,  # Use net edge as primary
            edge_raw=edge_raw,
            edge_net=edge_net,
            confidence=confidence,
            features=features,
            fill_estimate=fill_est,
            sizing_result=sizing_final,
        ), "passed"
    
    def _generate_signal(
        self,
        market: Market,
        outcome_idx: int,
        orderbook: OrderBook | None,
        recent_trades: list[Trade],
    ) -> TradingSignal | None:
        """Generate signal for a single outcome (legacy method).

        Args:
            market: Market data.
            outcome_idx: Outcome index.
            orderbook: OrderBook data.
            recent_trades: Recent trades.

        Returns:
            TradingSignal if edge threshold met, else None.
        """
        signal, _ = self._generate_signal_with_reason(
            market, outcome_idx, orderbook, recent_trades, 0.0, 0.0
        )
        return signal

    def _initialize_ml_components(self) -> None:
        """Initialize ML components (model manager, data collector, updater)."""
        try:
            from pathlib import Path
            from polyb0t.ml.features import AdvancedFeatureEngine
            from polyb0t.ml.manager import ModelManager
            from polyb0t.ml.data import DataCollector
            from polyb0t.ml.updater import ModelUpdater
            
            # Advanced feature engine
            self.ml_feature_engine = AdvancedFeatureEngine()
            
            # Model manager (hot-swappable inference)
            model_dir = Path(self.settings.ml_model_dir)
            self.ml_model_manager = ModelManager(
                model_dir=model_dir,
                use_ensemble=self.settings.ml_use_ensemble,
                fallback_enabled=True,
            )
            
            # Data collector
            self.ml_data_collector = DataCollector(self.settings.ml_data_db)
            
            # Model updater (background learning)
            self.ml_model_updater = ModelUpdater(
                data_collector=self.ml_data_collector,
                model_dir=model_dir,
                retrain_interval_hours=self.settings.ml_retrain_interval_hours,
                min_total_examples=self.settings.ml_min_training_examples,
                validation_threshold_r2=self.settings.ml_validation_threshold_r2,
            )
            
            # Start background learning
            self.ml_model_updater.start()
            
            logger.info("ML components initialized and learning started")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            # Disable ML on failure
            self.settings.enable_ml = False
            self.ml_feature_engine = None
            self.ml_model_manager = None
            self.ml_data_collector = None
            self.ml_model_updater = None
    
    def _compute_model_probability(
        self,
        p_market: float,
        features: dict[str, Any],
    ) -> float:
        """Compute model probability estimate (ML-enhanced if enabled).

        Args:
            p_market: Market implied probability.
            features: Feature dictionary.

        Returns:
            Model probability estimate (0-1).
        """
        # Baseline prediction
        baseline_prob = self._compute_baseline_probability(p_market, features)
        
        # If ML disabled or not available, return baseline
        if not self.settings.enable_ml or self.ml_model_manager is None:
            return baseline_prob
        
        # Try ML prediction
        try:
            import pandas as pd
            
            # Convert features to DataFrame for ML model
            ml_features_df = pd.DataFrame([features])
            
            # Get ML prediction (predicted future return)
            predicted_return = self.ml_model_manager.predict(ml_features_df)
            
            # Convert return prediction to probability
            # Predicted return is change from current price
            p_ml = p_market + predicted_return
            p_ml = max(0.01, min(0.99, p_ml))
            
            # Blend ML with baseline for robustness
            blend_weight = self.settings.ml_prediction_blend_weight
            p_model = blend_weight * p_ml + (1 - blend_weight) * baseline_prob
            
            # Clamp final prediction
            p_model = max(0.01, min(0.99, p_model))
            
            return p_model
            
        except Exception as e:
            logger.warning(f"ML prediction failed, using baseline: {e}")
            return baseline_prob
    
    def _compute_baseline_probability(
        self,
        p_market: float,
        features: dict[str, Any],
    ) -> float:
        """Compute baseline probability (no ML).

        Args:
            p_market: Market implied probability.
            features: Feature dictionary.

        Returns:
            Baseline probability estimate (0-1).
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
    
    def collect_training_data(
        self,
        markets: list[Market],
        orderbooks: dict[str, OrderBook],
        trades: dict[str, list[Trade]],
        cycle_id: str,
    ) -> int:
        """Collect training data from current cycle (for ML learning).
        
        Args:
            markets: Current markets.
            orderbooks: Current orderbooks.
            trades: Recent trades.
            cycle_id: Current cycle ID.
            
        Returns:
            Number of examples collected.
        """
        if not self.settings.enable_ml or self.ml_data_collector is None or self.ml_feature_engine is None:
            return 0
        
        try:
            features_dict = {}
            prices = {}
            market_ids = {}
            
            for market in markets:
                for idx, outcome in enumerate(market.outcomes):
                    token_id = outcome.token_id
                    orderbook = orderbooks.get(token_id)
                    token_trades = trades.get(token_id, [])
                    
                    # Get current price
                    if orderbook and orderbook.bids and orderbook.asks:
                        current_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
                    elif token_trades:
                        current_price = token_trades[-1].price
                    elif outcome.price:
                        current_price = outcome.price
                    else:
                        continue
                    
                    # Compute ML features
                    ml_features = self.ml_feature_engine.compute_features(
                        market=market,
                        outcome_idx=idx,
                        orderbook=orderbook,
                        recent_trades=token_trades,
                        current_price=current_price,
                    )
                    
                    features_dict[token_id] = ml_features
                    prices[token_id] = current_price
                    market_ids[token_id] = market.condition_id
            
            # Store for future labeling
            collected = self.ml_data_collector.collect_cycle_data(
                features_dict=features_dict,
                prices=prices,
                cycle_id=cycle_id,
                market_ids=market_ids,
            )
            
            return collected
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            return 0
    
    def get_ml_status(self) -> dict:
        """Get ML system status.
        
        Returns:
            Dictionary with ML status information.
        """
        if not self.settings.enable_ml:
            return {'enabled': False}
        
        status = {'enabled': True}
        
        # Model status
        if self.ml_model_manager:
            status['model'] = self.ml_model_manager.get_model_info()
        
        # Data collection status
        if self.ml_data_collector:
            status['data'] = self.ml_data_collector.get_statistics()
        
        # Updater status
        if self.ml_model_updater:
            status['updater'] = self.ml_model_updater.get_status()
        
        return status
    
    def shutdown_ml(self) -> None:
        """Gracefully shutdown ML components."""
        if self.ml_model_updater:
            logger.info("Stopping ML updater...")
            self.ml_model_updater.stop()
            logger.info("ML updater stopped")

