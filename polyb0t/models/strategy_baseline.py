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
from polyb0t.models.market_edge import get_market_edge_engine, MarketEdgeEngine

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
        
        # Market Edge Intelligence (smarter rules-based edge)
        self.market_edge_engine: MarketEdgeEngine | None = None
        if self.settings.enable_market_edge_intelligence:
            try:
                self.market_edge_engine = get_market_edge_engine()
                logger.info("Market Edge Intelligence engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Market Edge engine: {e}")
        
        # Microstructure analysis (advanced signals)
        self.microstructure_analyzer = None
        self.kelly_sizer = None
        if self.settings.enable_microstructure_analysis:
            from polyb0t.models.market_microstructure import (
                MicrostructureAnalyzer,
                KellyCriterionSizer,
            )
            self.microstructure_analyzer = MicrostructureAnalyzer()
            if self.settings.enable_kelly_sizing:
                self.kelly_sizer = KellyCriterionSizer(
                    kelly_fraction=self.settings.kelly_fraction
                )
        
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

        # === RUN MICROSTRUCTURE ANALYSIS FIRST ===
        # This populates order_book_imbalance, entry_score, etc. BEFORE model probability
        if self.microstructure_analyzer and orderbook:
            try:
                ob_dict = {
                    "bids": [{"price": l.price, "size": l.size} for l in (orderbook.bids or [])],
                    "asks": [{"price": l.price, "size": l.size} for l in (orderbook.asks or [])],
                }
                price_history = features.get("price_history", [])
                if not price_history and features.get("last_price"):
                    price_history = [{"price": p_market}, {"price": features.get("last_price", p_market)}]
                
                early_microstructure = self.microstructure_analyzer.analyze(
                    token_id=market.outcomes[outcome_idx].token_id,
                    orderbook=ob_dict,
                    current_price=p_market,
                    price_history=price_history,
                )
                # Store in features for baseline probability calculation
                features["order_book_imbalance"] = early_microstructure.order_book_imbalance
                features["entry_score"] = early_microstructure.entry_score
                features["momentum"] = early_microstructure.price_momentum_24h
                features["volume_ratio"] = early_microstructure.volume_ratio
                features["is_falling_knife"] = early_microstructure.is_falling_knife
                features["is_pump"] = early_microstructure.is_pump
            except Exception as e:
                logger.debug(f"Early microstructure analysis failed: {e}")

        # Model probability (now has microstructure features available)
        p_model = self._compute_model_probability(p_market, features)

        # Raw edge (mid-price based)
        edge_raw = p_model - p_market
        
        # === MARKET EDGE INTELLIGENCE ===
        # Apply smarter edge calculation using multi-factor analysis
        edge_score = None
        if self.market_edge_engine:
            try:
                # Convert orderbook for edge engine
                ob_dict = None
                if orderbook:
                    ob_dict = {
                        "bids": [{"price": l.price, "size": l.size} for l in (orderbook.bids or [])],
                        "asks": [{"price": l.price, "size": l.size} for l in (orderbook.asks or [])],
                    }
                
                # Convert trades for edge engine
                trades_dict = None
                if recent_trades:
                    trades_dict = [
                        {"price": t.price, "size": t.size, "side": t.side, "timestamp": t.timestamp}
                        for t in recent_trades
                    ]
                
                # Get days to resolution
                days_to_resolution = None
                if market.end_date:
                    days_to_resolution = (market.end_date - datetime.utcnow()).total_seconds() / 86400
                
                # Compute edge score with market intelligence
                edge_score = self.market_edge_engine.compute_edge(
                    token_id=market.outcomes[outcome_idx].token_id,
                    current_price=p_market,
                    p_model=p_model,
                    orderbook=ob_dict,
                    recent_trades=trades_dict,
                    market_category=getattr(market, "category", None),
                    days_to_resolution=days_to_resolution,
                    volume_24h=float(features.get("volume_24h", 0) or market.volume or 0),
                )
                
                # Store edge intelligence features
                features.update(edge_score.to_dict())
                
                # Use composite edge if it provides stronger signal
                if abs(edge_score.composite_edge) > abs(edge_raw) * 0.8:
                    # Blend raw edge with composite edge
                    edge_raw = 0.6 * edge_raw + 0.4 * edge_score.composite_edge
                    features["edge_source"] = "market_intelligence_blend"
                else:
                    features["edge_source"] = "raw_model"
                
                # Boost confidence based on edge score confidence
                if edge_score.confidence > 0.7:
                    features["intelligence_confidence_boost"] = True
                    
            except Exception as e:
                logger.debug(f"Market edge intelligence failed: {e}")
                features["edge_source"] = "raw_model_fallback"
        
        # Check raw edge threshold
        if abs(edge_raw) < self.settings.edge_threshold:
            # Log rejection details for tuning
            token_short = market.outcomes[outcome_idx].token_id[:12] if outcome_idx < len(market.outcomes) else "unknown"
            logger.debug(
                f"Edge below threshold: {token_short} edge={edge_raw:+.4f} vs threshold={self.settings.edge_threshold:.4f} "
                f"(p_model={p_model:.3f}, p_market={p_market:.3f})"
            )
            return None, "raw_edge_below_threshold"

        # Determine side
        side = "BUY" if edge_raw > 0 else "SELL"
        
        # === MICROSTRUCTURE ANALYSIS ===
        # Check order book imbalance, momentum, and entry timing
        microstructure_signal = None
        if self.microstructure_analyzer and orderbook:
            try:
                # Convert orderbook to dict format
                ob_dict = {
                    "bids": [{"price": l.price, "size": l.size} for l in (orderbook.bids or [])],
                    "asks": [{"price": l.price, "size": l.size} for l in (orderbook.asks or [])],
                }
                
                # Get price history from features if available
                price_history = features.get("price_history", [])
                if not price_history and features.get("last_price"):
                    # Create minimal price history
                    price_history = [{"price": p_market}, {"price": features.get("last_price", p_market)}]
                
                microstructure_signal = self.microstructure_analyzer.analyze(
                    token_id=market.outcomes[outcome_idx].token_id,
                    orderbook=ob_dict,
                    current_price=p_market,
                    price_history=price_history,
                    volume_24h=float(features.get("volume_24h", 0) or market.volume or 0),
                    avg_volume=float(features.get("avg_volume", 0) or market.volume or 1),
                )
                
                # Store microstructure data in features
                features["microstructure"] = microstructure_signal.to_dict()
                features["order_book_imbalance"] = microstructure_signal.order_book_imbalance
                features["is_falling_knife"] = microstructure_signal.is_falling_knife
                features["is_pump"] = microstructure_signal.is_pump
                features["entry_score"] = microstructure_signal.entry_score
                
                # === REJECT FALLING KNIVES ===
                if side == "BUY" and self.settings.avoid_falling_knives:
                    if microstructure_signal.is_falling_knife:
                        logger.info(
                            f"Rejecting signal: falling knife detected (24h momentum: {microstructure_signal.price_momentum_24h:.1%})",
                            extra={"token_id": market.outcomes[outcome_idx].token_id[:20]},
                        )
                        return None, "falling_knife_detected"
                
                # === REJECT CHASING PUMPS ===
                if side == "BUY" and self.settings.avoid_chasing_pumps:
                    if microstructure_signal.is_pump:
                        logger.info(
                            f"Rejecting signal: pump detected, don't chase (24h momentum: {microstructure_signal.price_momentum_24h:.1%})",
                            extra={"token_id": market.outcomes[outcome_idx].token_id[:20]},
                        )
                        return None, "pump_chasing_rejected"
                
                # === CHECK ORDER BOOK IMBALANCE ===
                if side == "BUY" and self.settings.min_orderbook_imbalance > 0:
                    if microstructure_signal.order_book_imbalance < self.settings.min_orderbook_imbalance:
                        logger.debug(
                            f"Rejecting: orderbook imbalance {microstructure_signal.order_book_imbalance:.2f} "
                            f"< threshold {self.settings.min_orderbook_imbalance:.2f}"
                        )
                        return None, "orderbook_imbalance_unfavorable"
                
                # === CHECK SPREAD ===
                if microstructure_signal.spread_pct > self.settings.max_spread_for_entry:
                    logger.debug(
                        f"Rejecting: spread {microstructure_signal.spread_pct:.2%} too wide "
                        f"(max: {self.settings.max_spread_for_entry:.2%})"
                    )
                    return None, "spread_too_wide"
                
                # === CHECK LIQUIDITY DEPTH ===
                if microstructure_signal.bid_depth_usd < self.settings.min_liquidity_depth_usd:
                    logger.debug(
                        f"Rejecting: bid depth ${microstructure_signal.bid_depth_usd:.0f} "
                        f"< minimum ${self.settings.min_liquidity_depth_usd:.0f}"
                    )
                    return None, "insufficient_liquidity"
                
                # === SHOULD WAIT FOR BETTER ENTRY ===
                if microstructure_signal.should_wait:
                    logger.info(
                        f"Rejecting signal: {microstructure_signal.wait_reason}",
                        extra={"token_id": market.outcomes[outcome_idx].token_id[:20]},
                    )
                    return None, "bad_entry_timing"
                    
            except Exception as e:
                logger.warning(f"Microstructure analysis failed: {e}")
                # Continue without microstructure analysis
        
        # Confidence based on data quality
        confidence = self._compute_confidence(features)
        
        # Boost confidence if microstructure signals align
        if microstructure_signal:
            # If entry_score is positive and aligns with our edge, boost confidence
            if (edge_raw > 0 and microstructure_signal.entry_score > 0.3) or \
               (edge_raw < 0 and microstructure_signal.entry_score < -0.3):
                confidence = min(1.0, confidence * 1.2)
                features["confidence_boosted"] = "microstructure_alignment"

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
        
        # === SLIPPAGE ABORT CHECK ===
        # Abort if slippage would eat too much of our edge
        if self.settings.enable_slippage_abort:
            slippage_pct_of_edge = 0.0
            if abs(edge_raw) > 0:
                slippage_pct_of_edge = fill_est.slippage_bps / (abs(edge_raw) * 10000)
            
            max_slippage_ratio = self.settings.max_slippage_of_edge_pct / 100.0
            
            if slippage_pct_of_edge > max_slippage_ratio:
                logger.debug(
                    f"SLIPPAGE ABORT: slippage {fill_est.slippage_bps}bps would eat "
                    f"{slippage_pct_of_edge*100:.0f}% of edge (max: {max_slippage_ratio*100:.0f}%)"
                )
                return None, f"slippage_exceeds_edge_ratio_{slippage_pct_of_edge*100:.0f}pct"
            
            if fill_est.slippage_bps > self.settings.absolute_max_slippage_bps:
                logger.debug(
                    f"SLIPPAGE ABORT: slippage {fill_est.slippage_bps}bps > max {self.settings.absolute_max_slippage_bps}bps"
                )
                return None, f"slippage_exceeds_absolute_max_{fill_est.slippage_bps}bps"
        
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
        
        # === KELLY CRITERION SIZING ===
        # Use Kelly for optimal position sizing if enabled
        kelly_size_usd = sizing_final.size_usd_final
        if self.kelly_sizer and side == "BUY":
            try:
                kelly_result = self.kelly_sizer.calculate_position_size(
                    bankroll=available_usdc + reserved_usdc,
                    market_price=p_market,
                    estimated_probability=p_model,
                    max_position_pct=self.settings.max_position_pct / 100.0,
                    min_edge=self.settings.min_net_edge,
                )
                
                if kelly_result["should_bet"]:
                    kelly_size_usd = kelly_result["recommended_size_usd"]
                    features["kelly_recommended_pct"] = kelly_result["recommended_pct"]
                    features["kelly_edge_pct"] = kelly_result["edge_pct"]
                    features["kelly_full_fraction"] = kelly_result["full_kelly_fraction"]
                    
                    # Use the smaller of position sizer and Kelly (conservative)
                    if kelly_size_usd < sizing_final.size_usd_final:
                        sizing_final.size_usd_final = kelly_size_usd
                        sizing_final.sizing_reason = f"kelly_optimal_{kelly_result['recommended_pct']:.1f}pct"
                        features["sizing_method"] = "kelly_criterion"
                    else:
                        features["sizing_method"] = "position_sizer"
                else:
                    # Kelly says don't bet - but we passed edge threshold, so use small size
                    features["kelly_no_bet_reason"] = kelly_result.get("reason", "unknown")
                    features["sizing_method"] = "position_sizer_override_kelly"
                    
            except Exception as e:
                logger.warning(f"Kelly sizing failed: {e}")
                features["sizing_method"] = "position_sizer_kelly_error"
        
        # === VOLUME SPIKE BOOST ===
        # Increase size when unusual volume detected (early accumulation signal)
        if microstructure_signal and self.settings.enable_volume_spike_boost:
            if microstructure_signal.is_volume_spike and side == "BUY":
                original_size = sizing_final.size_usd_final
                boosted_size = original_size * self.settings.volume_spike_size_multiplier
                
                # Cap at maximum
                max_size = (available_usdc + reserved_usdc) * (self.settings.max_position_pct / 100.0)
                sizing_final.size_usd_final = min(boosted_size, max_size)
                
                if sizing_final.size_usd_final > original_size:
                    features["volume_spike_boost_applied"] = True
                    features["volume_spike_ratio"] = microstructure_signal.volume_ratio
                    logger.info(
                        f"Volume spike boost: ${original_size:.2f} -> ${sizing_final.size_usd_final:.2f} "
                        f"(volume {microstructure_signal.volume_ratio:.1f}x normal)"
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
        features["final_size_usd"] = sizing_final.size_usd_final

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
            
            # Determine blend weight - check if ML should fully take over
            blend_weight = self.settings.ml_prediction_blend_weight
            
            # Check if ML model is performing well enough for full takeover
            model_info = self.ml_model_manager.get_model_info()
            model_r2 = model_info.get("r2_score", 0) if model_info else 0
            training_examples = model_info.get("training_examples", 0) if model_info else 0
            
            if (model_r2 >= self.settings.ml_full_takeover_r2_threshold and 
                training_examples >= self.settings.ml_full_takeover_min_examples):
                # ML model is performing well - full takeover
                blend_weight = 1.0
                logger.debug(
                    f"ML full takeover active: R²={model_r2:.3f}, examples={training_examples}"
                )
            
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
        """Compute baseline probability with market microstructure signals.

        APPROACH: Start with market price, adjust based on orderbook and momentum.
        - Lower thresholds to actually generate trades
        - Small but consistent edges from market inefficiencies

        Args:
            p_market: Market implied probability.
            features: Feature dictionary.

        Returns:
            Baseline probability estimate (0-1).
        """
        # BASELINE: Start with market price
        adjustment = 0.0
        adjustment_reasons = []
        signal_strength = 0.0  # Track how much evidence we have
        
        # === SIGNAL 1: ORDER BOOK IMBALANCE ===
        # Any imbalance > 10% is meaningful
        orderbook_imbalance = features.get("order_book_imbalance", 0.0)
        if abs(orderbook_imbalance) > 0.10:  # 10%+ imbalance (lowered from 25%)
            ob_adj = orderbook_imbalance * 0.05  # Up to 5% adjustment
            adjustment += ob_adj
            signal_strength += abs(orderbook_imbalance)
            adjustment_reasons.append(f"ob_imbalance_{orderbook_imbalance:.2f}")
        
        # === SIGNAL 2: VOLUME SPIKE (information signal) ===
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio > 1.2:  # 1.2x+ normal volume (lowered from 1.5)
            # High volume = something is happening
            direction = np.sign(orderbook_imbalance) if abs(orderbook_imbalance) > 0.05 else 0
            if direction == 0:
                momentum = features.get("momentum", 0)
                direction = np.sign(momentum) if abs(momentum) > 0.005 else 0
            
            vol_adj = direction * min((volume_ratio - 1.0) * 0.03, 0.04)
            adjustment += vol_adj
            if vol_adj != 0:
                signal_strength += 0.3
                adjustment_reasons.append(f"volume_{volume_ratio:.1f}x")
        
        # === SIGNAL 3: MOMENTUM ===
        momentum = features.get("momentum", 0)
        if abs(momentum) > 0.005:  # 0.5%+ momentum (lowered from 2%)
            # Follow momentum - trend continuation
            mom_adj = np.sign(momentum) * min(abs(momentum) * 1.0, 0.04)
            adjustment += mom_adj
            signal_strength += min(abs(momentum) * 5, 0.5)
            adjustment_reasons.append(f"momentum_{momentum:+.3f}")
        
        # === SIGNAL 4: MICROSTRUCTURE SIGNALS ===
        entry_score = features.get("entry_score", 0)
        if abs(entry_score) > 0.1:  # lowered from 0.3
            micro_adj = entry_score * 0.03
            adjustment += micro_adj
            signal_strength += abs(entry_score) * 0.4
            adjustment_reasons.append(f"entry_score_{entry_score:.2f}")
        
        # === SIGNAL 5: EDGE INTELLIGENCE (if available) ===
        composite_edge = features.get("edge_composite", 0)
        if abs(composite_edge) > 0.005:  # lowered from 0.02
            intel_adj = composite_edge * 0.8  # Blend in 80% of intelligence edge
            adjustment += intel_adj
            signal_strength += abs(composite_edge) * 3
            adjustment_reasons.append(f"intel_edge_{composite_edge:+.3f}")
        
        # === SIGNAL 6: CONTRARIAN AT EXTREMES ===
        if p_market < 0.15 and orderbook_imbalance < -0.2:  # lowered thresholds
            adjustment += 0.02
            signal_strength += 0.2
            adjustment_reasons.append("contrarian_oversold")
        elif p_market > 0.85 and orderbook_imbalance > 0.2:
            adjustment -= 0.02
            signal_strength += 0.2
            adjustment_reasons.append("contrarian_overbought")
        
        # === SIGNAL 7: BASE EDGE FROM SPREAD ===
        # If spread is tight, there's less uncertainty = small edge opportunity
        spread_pct = features.get("spread_pct", 0.05)
        if spread_pct < 0.03 and abs(orderbook_imbalance) > 0.05:
            # Tight spread + any imbalance = small edge
            base_edge = orderbook_imbalance * 0.02
            adjustment += base_edge
            signal_strength += 0.15
            adjustment_reasons.append("tight_spread_edge")
        
        # === APPLY DYNAMIC LIMITS ===
        if signal_strength < 0.15:
            max_deviation = 0.015  # Very weak: 1.5%
        elif signal_strength < 0.3:
            max_deviation = 0.03  # Weak: 3%
        elif signal_strength < 0.5:
            max_deviation = 0.05  # Moderate: 5%
        else:
            max_deviation = 0.08  # Strong: 8%
        
        adjustment = max(-max_deviation, min(max_deviation, adjustment))
        
        # === SPREAD-BASED DAMPENING ===
        if spread_pct > 0.08:  # Only dampen for very wide spreads
            dampening = max(0.4, 1.0 - spread_pct * 3)
            adjustment *= dampening
            adjustment_reasons.append(f"spread_dampen_{dampening:.2f}")
        
        p_model = p_market + adjustment
        
        # Clamp to valid range
        p_model = max(0.01, min(0.99, p_model))
        
        # Log adjustments for debugging
        if abs(adjustment) >= 0.01:
            logger.debug(
                f"Model: market={p_market:.3f} -> model={p_model:.3f} "
                f"(adj={adjustment:+.3f}, strength={signal_strength:.2f}, reasons={adjustment_reasons})"
            )
        
        # Store for debugging
        features["baseline_adjustment"] = adjustment
        features["baseline_signal_strength"] = signal_strength
        features["baseline_reasons"] = adjustment_reasons
        
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
        
        IMPORTANT: Collects from ALL markets, not just tradable ones.
        This enables learning from the entire market landscape.
        
        Args:
            markets: ALL markets (not just tradable subset).
            orderbooks: Current orderbooks (may be partial).
            trades: Recent trades (may be partial).
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
            
            # Process ALL markets for comprehensive learning
            for market in markets:
                for idx, outcome in enumerate(market.outcomes):
                    token_id = outcome.token_id
                    if not token_id:
                        continue
                    
                    orderbook = orderbooks.get(token_id)
                    token_trades = trades.get(token_id, [])
                    
                    # Get current price (prioritize orderbook, fallback to Gamma)
                    current_price = None
                    
                    if orderbook and orderbook.bids and orderbook.asks:
                        current_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
                    elif token_trades:
                        current_price = token_trades[-1].price
                    elif outcome.price is not None:
                        current_price = outcome.price
                    
                    # Skip if no price available at all
                    if current_price is None or current_price <= 0 or current_price >= 1:
                        continue
                    
                    # Compute ML features (handles missing orderbook gracefully)
                    try:
                        ml_features = self.ml_feature_engine.compute_features(
                            market=market,
                            outcome_idx=idx,
                            orderbook=orderbook,  # May be None
                            recent_trades=token_trades,  # May be empty
                            current_price=current_price,
                        )
                        
                        # Add market metadata for better learning
                        ml_features['market_volume'] = float(market.volume or 0)
                        ml_features['market_liquidity'] = float(market.liquidity or 0)
                        ml_features['market_active'] = float(market.active)
                        
                        # Add Market Edge Intelligence features for smarter learning
                        if self.market_edge_engine:
                            try:
                                # Convert orderbook for edge engine
                                ob_dict = None
                                if orderbook and orderbook.bids and orderbook.asks:
                                    ob_dict = {
                                        "bids": [{"price": l.price, "size": l.size} for l in orderbook.bids],
                                        "asks": [{"price": l.price, "size": l.size} for l in orderbook.asks],
                                    }
                                
                                # Convert trades
                                trades_dict = None
                                if token_trades:
                                    trades_dict = [
                                        {"price": t.price, "size": t.size, "side": t.side, "timestamp": t.timestamp}
                                        for t in token_trades
                                    ]
                                
                                # Get days to resolution
                                days_to_resolution = None
                                if market.end_date:
                                    days_to_resolution = (market.end_date - datetime.utcnow()).total_seconds() / 86400
                                
                                # Get all edge features for ML training
                                edge_features = self.market_edge_engine.get_all_features(
                                    token_id=token_id,
                                    current_price=current_price,
                                    p_model=current_price,  # Use market price as baseline
                                    orderbook=ob_dict,
                                    recent_trades=trades_dict,
                                    market_category=getattr(market, "category", None),
                                    days_to_resolution=days_to_resolution,
                                    volume_24h=float(market.volume or 0),
                                )
                                
                                # Merge edge features into ML features
                                ml_features.update(edge_features)
                                
                            except Exception as e:
                                logger.debug(f"Failed to compute edge features for {token_id}: {e}")
                        
                        features_dict[token_id] = ml_features
                        prices[token_id] = current_price
                        market_ids[token_id] = market.condition_id
                        
                    except Exception as e:
                        # Log but continue - don't let one bad market stop collection
                        logger.debug(f"Failed to compute features for {token_id}: {e}")
                        continue
            
            if not features_dict:
                logger.debug("No features collected this cycle (no valid market data)")
                return 0
            
            # Store for future labeling
            collected = self.ml_data_collector.collect_cycle_data(
                features_dict=features_dict,
                prices=prices,
                cycle_id=cycle_id,
                market_ids=market_ids,
            )
            
            return collected
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}", exc_info=True)
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

