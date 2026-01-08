"""Exit manager for proposing position exits (take-profit, stop-loss, time-based).

Conservative Loss Philosophy:
- Never sell at small losses - wait for recovery
- Only exit on sustained declines (multiple confirmation cycles)
- React QUICKLY to massive crashes (>25% drops or rapid velocity)
- Minimum hold time before any loss-taking allowed
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from polyb0t.config import get_settings
from polyb0t.data.models import Market, OrderBook
from polyb0t.execution.intents import IntentManager, TradeIntent
from polyb0t.execution.portfolio import Position

logger = logging.getLogger(__name__)


class ExitProposal:
    """Represents a proposed exit."""

    def __init__(
        self,
        token_id: str,
        market_id: str,
        exit_type: str,
        reason: str,
        side: str,
        price: float,
        size: float,
        pnl: float | None = None,
        is_emergency: bool = False,
    ) -> None:
        """Initialize exit proposal.

        Args:
            token_id: Token identifier.
            market_id: Market ID.
            exit_type: TAKE_PROFIT, STOP_LOSS, TIME_EXIT, LIQUIDITY_EXIT, CRASH_EXIT.
            reason: Human-readable reason.
            side: SELL (close long) or BUY (close short).
            price: Proposed exit price.
            size: Position size to close.
            pnl: Expected PnL.
            is_emergency: If True, execute immediately at market price (crash protection).
        """
        self.token_id = token_id
        self.market_id = market_id
        self.exit_type = exit_type
        self.reason = reason
        self.side = side
        self.price = price
        self.size = size
        self.pnl = pnl
        self.is_emergency = is_emergency


class ExitManager:
    """Manages exit proposals based on take-profit, stop-loss, and other criteria.
    
    Exit Types:
        - TAKE_PROFIT: Price increased by target %
        - STOP_LOSS: Price decreased by limit % (CONSERVATIVE - needs confirmation)
        - TIME_EXIT: Market resolving soon
        - LIQUIDITY_EXIT: Spread widened significantly
        - MOMENTUM_EXIT: Momentum turned strongly negative
        - TRAILING_STOP: Price dropped from recent high
        - CRASH_EXIT: EMERGENCY - massive rapid drop, immediate market sell
    
    Conservative Loss Philosophy:
        - Small losses (<15%): Wait and see, likely to recover
        - Medium losses (15-25%): Only sell after 5 consecutive cycles confirming loss
        - Crash (>25% or rapid): Immediate emergency exit to preserve capital
    """

    def __init__(self) -> None:
        """Initialize exit manager."""
        self.settings = get_settings()
        
        # Track highest price seen for trailing stops
        self._position_high_watermarks: dict[str, float] = {}
        
        # Track consecutive cycles below stop-loss for confirmation
        # token_id -> (cycles_below_threshold, first_seen_timestamp, price_history)
        self._stop_loss_tracking: dict[str, tuple[int, datetime, list[float]]] = {}
        
        # Track position entry times for min hold enforcement
        self._position_entry_times: dict[str, datetime] = {}
        
        # Microstructure analyzer for momentum-based exits
        self.microstructure_analyzer = None
        if self.settings.enable_microstructure_analysis:
            try:
                from polyb0t.models.market_microstructure import MicrostructureAnalyzer
                self.microstructure_analyzer = MicrostructureAnalyzer()
            except ImportError:
                pass

    def propose_exits(
        self,
        positions: dict[str, Position],
        markets: dict[str, Market],
        orderbooks: dict[str, OrderBook] | None = None,
        fallback_prices: dict[str, float] | None = None,
    ) -> list[ExitProposal]:
        """Propose exits for current positions.

        Args:
            positions: Current positions (token_id -> Position).
            markets: Market data (condition_id -> Market).
            orderbooks: Optional orderbook data (token_id -> OrderBook).
            fallback_prices: Optional fallback prices (token_id -> price/probability) when no orderbook is available.

        Returns:
            List of exit proposals.
        """
        proposals = []

        for token_id, position in positions.items():
            orderbook = orderbooks.get(token_id) if orderbooks else None
            current_price = None
            if orderbook and orderbook.bids and orderbook.asks:
                current_price = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
            elif fallback_prices and token_id in fallback_prices:
                current_price = fallback_prices[token_id]
            if current_price is None:
                continue

            # Update position price
            position.update_price(current_price)

            # Get market data
            market = markets.get(position.market_id)

            # Check take-profit
            if self.settings.enable_take_profit:
                tp_proposal = self._check_take_profit(position, orderbook)
                if tp_proposal:
                    proposals.append(tp_proposal)

            # Check CRASH first (takes priority over everything - emergency exit)
            if self.settings.enable_crash_detection:
                crash_proposal = self._check_crash_exit(position, orderbook)
                if crash_proposal:
                    # Crash overrides all other exits - return immediately
                    proposals.append(crash_proposal)
                    logger.warning(
                        f"CRASH exit takes priority - skipping other checks",
                        extra={"token_id": token_id[:12]}
                    )
                    continue  # Don't check anything else for this position
            
            # Check stop-loss (conservative - requires confirmation)
            if self.settings.enable_stop_loss:
                sl_proposal = self._check_stop_loss(position, orderbook)
                if sl_proposal:
                    proposals.append(sl_proposal)

            # Check time-based exit
            if self.settings.enable_time_exit and market:
                time_proposal = self._check_time_exit(position, market, orderbook)
                if time_proposal:
                    proposals.append(time_proposal)

            # Check liquidity-based exit
            if orderbook:
                liquidity_proposal = self._check_liquidity_exit(position, orderbook)
                if liquidity_proposal:
                    proposals.append(liquidity_proposal)
            
            # Check momentum-based exit (sell when momentum turns negative)
            if self.microstructure_analyzer and orderbook:
                momentum_proposal = self._check_momentum_exit(position, orderbook)
                if momentum_proposal:
                    proposals.append(momentum_proposal)
            
            # Check trailing stop
            trailing_proposal = self._check_trailing_stop(position, orderbook)
            if trailing_proposal:
                proposals.append(trailing_proposal)

        return proposals

    def _check_take_profit(
        self, position: Position, orderbook: OrderBook | None
    ) -> ExitProposal | None:
        """Check if take-profit threshold is met.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if take-profit triggered, else None.
        """
        # Calculate PnL percentage
        pnl_pct = (position.unrealized_pnl / position.quantity) * 100

        if pnl_pct >= self.settings.take_profit_pct:
            # Propose exit at current bid (for long) or ask (for short)
            if position.side == "LONG":
                exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else position.current_price
                exit_side = "SELL"
            else:
                exit_price = orderbook.asks[0].price if orderbook and orderbook.asks else position.current_price
                exit_side = "BUY"

            reason = (
                f"Take-profit: PnL {pnl_pct:.1f}% >= target {self.settings.take_profit_pct}%. "
                f"Current: ${position.current_price:.3f}, Entry: ${position.avg_entry_price:.3f}"
            )

            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="TAKE_PROFIT",
                reason=reason,
                side=exit_side,
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
            )

        return None

    def _check_stop_loss(
        self, position: Position, orderbook: OrderBook | None
    ) -> ExitProposal | None:
        """Check if stop-loss threshold is met - CONSERVATIVE approach.

        Conservative Loss Management:
        1. Never sell within min_hold_time of entry (avoid whipsaws)
        2. Require multiple consecutive cycles below threshold (confirmation)
        3. Give positions time to recover before selling
        4. Only trigger after sustained decline, not temporary dips

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if stop-loss triggered, else None.
        """
        token_id = position.token_id
        current_price = position.current_price
        now = datetime.utcnow()
        
        # Calculate loss percentage
        pnl_pct = (position.unrealized_pnl / position.quantity) * 100 if position.quantity > 0 else 0
        
        # Track position entry time if not already tracked
        if token_id not in self._position_entry_times:
            # Use current time as a fallback (position was opened before bot started)
            self._position_entry_times[token_id] = now
        
        entry_time = self._position_entry_times[token_id]
        time_held_minutes = (now - entry_time).total_seconds() / 60
        
        # RULE 1: Never sell at a loss within minimum hold time
        if time_held_minutes < self.settings.min_hold_time_minutes:
            logger.debug(
                f"Stop-loss check skipped: Min hold time not reached",
                extra={
                    "token_id": token_id[:12],
                    "time_held_min": time_held_minutes,
                    "min_hold_min": self.settings.min_hold_time_minutes,
                    "pnl_pct": pnl_pct,
                }
            )
            return None
        
        # Check if we're below the stop-loss threshold
        if pnl_pct <= -self.settings.stop_loss_pct:
            # Initialize or update tracking
            if token_id not in self._stop_loss_tracking:
                # First time seeing this position below threshold
                self._stop_loss_tracking[token_id] = (1, now, [current_price])
                logger.info(
                    f"Stop-loss monitoring started (need {self.settings.stop_loss_confirmation_cycles} confirmations)",
                    extra={
                        "token_id": token_id[:12],
                        "pnl_pct": pnl_pct,
                        "cycles_confirmed": 1,
                        "cycles_needed": self.settings.stop_loss_confirmation_cycles,
                    }
                )
                return None  # Wait for more confirmations
            
            cycles, first_seen, price_history = self._stop_loss_tracking[token_id]
            cycles += 1
            price_history.append(current_price)
            
            # Keep only recent prices (last 10)
            if len(price_history) > 10:
                price_history = price_history[-10:]
            
            self._stop_loss_tracking[token_id] = (cycles, first_seen, price_history)
            
            # RULE 2: Require confirmation cycles before selling
            if cycles < self.settings.stop_loss_confirmation_cycles:
                logger.info(
                    f"Stop-loss pending confirmation ({cycles}/{self.settings.stop_loss_confirmation_cycles})",
                    extra={
                        "token_id": token_id[:12],
                        "pnl_pct": pnl_pct,
                        "cycles_confirmed": cycles,
                        "cycles_needed": self.settings.stop_loss_confirmation_cycles,
                    }
                )
                return None  # Need more confirmations
            
            # RULE 3: Check if price is recovering (don't sell if bouncing back)
            if len(price_history) >= 3:
                recent_avg = sum(price_history[-3:]) / 3
                older_avg = sum(price_history[:3]) / 3 if len(price_history) >= 6 else price_history[0]
                
                # If recent prices are higher than older prices, it might be recovering
                if recent_avg > older_avg * 1.02:  # 2% recovery signal
                    logger.info(
                        f"Stop-loss deferred: Position showing recovery",
                        extra={
                            "token_id": token_id[:12],
                            "pnl_pct": pnl_pct,
                            "recent_avg": recent_avg,
                            "older_avg": older_avg,
                        }
                    )
                    # Reset cycle count but keep tracking
                    self._stop_loss_tracking[token_id] = (1, now, price_history[-3:])
                    return None
            
            # RULE 4: Check recovery window
            time_below_threshold = (now - first_seen).total_seconds() / 60
            if time_below_threshold < self.settings.loss_recovery_window_minutes:
                logger.info(
                    f"Stop-loss deferred: Still in recovery window ({time_below_threshold:.0f}/{self.settings.loss_recovery_window_minutes} min)",
                    extra={
                        "token_id": token_id[:12],
                        "pnl_pct": pnl_pct,
                        "time_below_min": time_below_threshold,
                    }
                )
                return None
            
            # All checks passed - this is a sustained loss, trigger stop-loss
            if position.side == "LONG":
                exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else position.current_price
                exit_side = "SELL"
            else:
                exit_price = orderbook.asks[0].price if orderbook and orderbook.asks else position.current_price
                exit_side = "BUY"
            
            # Clean up tracking
            del self._stop_loss_tracking[token_id]
            if token_id in self._position_entry_times:
                del self._position_entry_times[token_id]

            reason = (
                f"Stop-loss (CONFIRMED): Loss {abs(pnl_pct):.1f}% sustained for {time_below_threshold:.0f} min "
                f"across {cycles} cycles. Entry: ${position.avg_entry_price:.3f}, Current: ${position.current_price:.3f}"
            )

            logger.warning(
                f"Stop-loss triggered after confirmation",
                extra={
                    "token_id": token_id[:12],
                    "pnl_pct": pnl_pct,
                    "cycles_confirmed": cycles,
                    "time_below_min": time_below_threshold,
                }
            )

            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="STOP_LOSS",
                reason=reason,
                side=exit_side,
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
            )
        else:
            # Price recovered above threshold - reset tracking
            if token_id in self._stop_loss_tracking:
                logger.info(
                    f"Stop-loss tracking reset: Position recovered",
                    extra={"token_id": token_id[:12], "pnl_pct": pnl_pct}
                )
                del self._stop_loss_tracking[token_id]

        return None
    
    def _check_crash_exit(
        self, position: Position, orderbook: OrderBook | None
    ) -> ExitProposal | None:
        """Check for CRASH conditions - immediate emergency exit.

        This is separate from regular stop-loss. A crash is:
        1. Massive drop (>25% from entry) - catastrophic loss imminent
        2. Rapid velocity (>2% per minute) - falling too fast to wait

        For crashes, we DO NOT wait for confirmation - immediate action required
        to preserve remaining capital.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal with is_emergency=True if crash detected, else None.
        """
        if not self.settings.enable_crash_detection:
            return None
        
        token_id = position.token_id
        current_price = position.current_price
        entry_price = position.avg_entry_price
        
        # Calculate loss percentage
        loss_pct = ((entry_price - current_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # CRASH CONDITION 1: Massive absolute loss
        is_massive_loss = loss_pct >= self.settings.crash_threshold_pct
        
        # CRASH CONDITION 2: Rapid velocity
        # Check price history from stop-loss tracking if available
        is_rapid_drop = False
        if token_id in self._stop_loss_tracking:
            _, first_seen, price_history = self._stop_loss_tracking[token_id]
            
            if len(price_history) >= 2:
                time_span_minutes = (datetime.utcnow() - first_seen).total_seconds() / 60
                if time_span_minutes > 0:
                    price_drop = price_history[0] - current_price
                    drop_pct = (price_drop / price_history[0]) * 100 if price_history[0] > 0 else 0
                    velocity_per_min = drop_pct / time_span_minutes
                    
                    if velocity_per_min >= self.settings.crash_velocity_pct_per_minute:
                        is_rapid_drop = True
                        logger.warning(
                            f"CRASH velocity detected: {velocity_per_min:.2f}%/min",
                            extra={
                                "token_id": token_id[:12],
                                "velocity_per_min": velocity_per_min,
                                "threshold": self.settings.crash_velocity_pct_per_minute,
                            }
                        )
        
        if is_massive_loss or is_rapid_drop:
            # EMERGENCY EXIT - no waiting, no confirmation
            if position.side == "LONG":
                # Use best bid for immediate exit
                exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else position.current_price
                exit_side = "SELL"
            else:
                exit_price = orderbook.asks[0].price if orderbook and orderbook.asks else position.current_price
                exit_side = "BUY"
            
            # Clean up all tracking for this position
            if token_id in self._stop_loss_tracking:
                del self._stop_loss_tracking[token_id]
            if token_id in self._position_entry_times:
                del self._position_entry_times[token_id]
            if token_id in self._position_high_watermarks:
                del self._position_high_watermarks[token_id]
            
            crash_type = []
            if is_massive_loss:
                crash_type.append(f"massive drop {loss_pct:.1f}%")
            if is_rapid_drop:
                crash_type.append("rapid velocity")
            
            reason = (
                f"🚨 CRASH EXIT ({', '.join(crash_type)}): Emergency sell to preserve capital. "
                f"Entry: ${entry_price:.3f}, Current: ${current_price:.3f}, Loss: {loss_pct:.1f}%"
            )
            
            logger.error(
                f"CRASH DETECTED - Emergency exit triggered",
                extra={
                    "token_id": token_id[:12],
                    "loss_pct": loss_pct,
                    "is_massive_loss": is_massive_loss,
                    "is_rapid_drop": is_rapid_drop,
                }
            )
            
            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="CRASH_EXIT",
                reason=reason,
                side=exit_side,
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
                is_emergency=True,  # Flag for immediate market execution
            )

        return None

    def _check_time_exit(
        self, position: Position, market: Market, orderbook: OrderBook | None
    ) -> ExitProposal | None:
        """Check if time-based exit should be triggered.

        Args:
            position: Position to check.
            market: Market data.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if time-based exit triggered, else None.
        """
        if not market.end_date:
            return None

        days_until_resolution = (market.end_date - datetime.utcnow()).total_seconds() / 86400

        if days_until_resolution <= self.settings.time_exit_days_before:
            # Propose exit at current bid (for long) or ask (for short)
            if position.side == "LONG":
                exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else position.current_price
                exit_side = "SELL"
            else:
                exit_price = orderbook.asks[0].price if orderbook and orderbook.asks else position.current_price
                exit_side = "BUY"

            reason = (
                f"Time-based exit: Market resolves in {days_until_resolution:.1f} days "
                f"(<= {self.settings.time_exit_days_before} day threshold). "
                f"PnL: ${position.unrealized_pnl:.2f}"
            )

            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="TIME_EXIT",
                reason=reason,
                side=exit_side,
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
            )

        return None

    def _check_liquidity_exit(
        self, position: Position, orderbook: OrderBook
    ) -> ExitProposal | None:
        """Check if liquidity deterioration warrants exit.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if liquidity exit triggered, else None.
        """
        # Calculate spread
        if not orderbook.bids or not orderbook.asks:
            return None

        best_bid = orderbook.bids[0].price
        best_ask = orderbook.asks[0].price
        mid = (best_bid + best_ask) / 2

        if mid == 0:
            return None

        spread = (best_ask - best_bid) / mid

        # If spread exceeds threshold by large margin, propose exit
        spread_threshold = self.settings.max_spread * 2.0  # 2x normal

        if spread > spread_threshold:
            # Propose exit at current bid (for long) or ask (for short)
            if position.side == "LONG":
                exit_price = best_bid
                exit_side = "SELL"
            else:
                exit_price = best_ask
                exit_side = "BUY"

            reason = (
                f"Liquidity deterioration: Spread {spread:.3f} > threshold {spread_threshold:.3f}. "
                f"PnL: ${position.unrealized_pnl:.2f}"
            )

            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="LIQUIDITY_EXIT",
                reason=reason,
                side=exit_side,
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
            )

        return None

    def _check_momentum_exit(
        self, position: Position, orderbook: OrderBook
    ) -> ExitProposal | None:
        """Check if momentum has turned strongly negative, warranting early exit.
        
        This catches positions where price is dropping rapidly before stop-loss triggers.
        Better to exit early at a smaller loss than wait for full stop-loss.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if momentum exit triggered, else None.
        """
        if not self.microstructure_analyzer:
            return None
        
        try:
            # Analyze current momentum
            ob_dict = {
                "bids": [{"price": l.price, "size": l.size} for l in (orderbook.bids or [])],
                "asks": [{"price": l.price, "size": l.size} for l in (orderbook.asks or [])],
            }
            
            current_price = position.current_price
            
            signal = self.microstructure_analyzer.analyze(
                token_id=position.token_id,
                orderbook=ob_dict,
                current_price=current_price,
                price_history=[{"price": position.avg_entry_price}, {"price": current_price}],
            )
            
            # Exit if:
            # 1. It's a falling knife (momentum very negative)
            # 2. Order book imbalance is strongly negative (lots of sellers)
            # 3. We're in profit but momentum just turned negative (lock in gains)
            
            pnl_pct = (position.unrealized_pnl / position.quantity) * 100 if position.quantity > 0 else 0
            
            # CONSERVATIVE APPROACH: Only use momentum exits for PROFIT PROTECTION
            # Don't sell at a loss just because of momentum - let crash/stop-loss handle that
            
            # Case 1: Falling knife - ONLY exit if we're still in profit
            # If we're at a loss, let the conservative stop-loss handle it (needs confirmation)
            if signal.is_falling_knife and pnl_pct > 0:
                exit_price = orderbook.bids[0].price if orderbook.bids else position.current_price
                
                reason = (
                    f"Momentum exit: Falling knife detected, locking in profits. "
                    f"(24h momentum: {signal.price_momentum_24h:.1%}). Current PnL: {pnl_pct:.1f}%"
                )
                
                return ExitProposal(
                    token_id=position.token_id,
                    market_id=position.market_id,
                    exit_type="MOMENTUM_EXIT",
                    reason=reason,
                    side="SELL" if position.side == "LONG" else "BUY",
                    price=exit_price,
                    size=position.quantity,
                    pnl=position.unrealized_pnl,
                )
            
            # Case 2: Heavy sell pressure - ONLY exit if we have meaningful profit to protect
            # Don't panic sell at a loss due to order book imbalance
            HEAVY_SELL_THRESHOLD = -0.5  # 50% more asks than bids
            MIN_PROFIT_TO_PROTECT = 3.0  # Only protect if up at least 3%
            if signal.order_book_imbalance < HEAVY_SELL_THRESHOLD and pnl_pct >= MIN_PROFIT_TO_PROTECT:
                exit_price = orderbook.bids[0].price if orderbook.bids else position.current_price
                
                reason = (
                    f"Momentum exit: Heavy sell pressure, locking in {pnl_pct:.1f}% gain "
                    f"(imbalance: {signal.order_book_imbalance:.2f})."
                )
                
                return ExitProposal(
                    token_id=position.token_id,
                    market_id=position.market_id,
                    exit_type="MOMENTUM_EXIT",
                    reason=reason,
                    side="SELL" if position.side == "LONG" else "BUY",
                    price=exit_price,
                    size=position.quantity,
                    pnl=position.unrealized_pnl,
                )
            
            # Case 3: Lock in gains when momentum turns (profit > 5% and momentum negative)
            if pnl_pct >= 5.0 and signal.momentum_score < -0.3:
                exit_price = orderbook.bids[0].price if orderbook.bids else position.current_price
                
                reason = (
                    f"Momentum exit: Locking in gains as momentum turns negative "
                    f"(score: {signal.momentum_score:.2f}). Current PnL: {pnl_pct:.1f}%"
                )
                
                return ExitProposal(
                    token_id=position.token_id,
                    market_id=position.market_id,
                    exit_type="MOMENTUM_EXIT",
                    reason=reason,
                    side="SELL" if position.side == "LONG" else "BUY",
                    price=exit_price,
                    size=position.quantity,
                    pnl=position.unrealized_pnl,
                )
                
        except Exception as e:
            logger.warning(f"Momentum exit check failed: {e}")
        
        return None

    def _check_trailing_stop(
        self, position: Position, orderbook: OrderBook | None
    ) -> ExitProposal | None:
        """Check if trailing stop should be triggered.
        
        Trailing stop tracks the highest price seen and triggers exit
        if price drops a certain % from that high.
        
        This locks in profits when price rises but then starts falling.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if trailing stop triggered, else None.
        """
        # Trailing stop configuration (could be made into settings)
        TRAILING_STOP_ACTIVATION_PCT = 5.0  # Only activate after 5% gain
        TRAILING_STOP_DISTANCE_PCT = 3.0  # Trigger if drops 3% from high
        
        current_price = position.current_price
        entry_price = position.avg_entry_price
        token_id = position.token_id
        
        # Calculate current gain from entry
        gain_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        # Update high watermark
        if token_id not in self._position_high_watermarks:
            self._position_high_watermarks[token_id] = current_price
        else:
            if current_price > self._position_high_watermarks[token_id]:
                self._position_high_watermarks[token_id] = current_price
        
        high_watermark = self._position_high_watermarks[token_id]
        
        # Only check trailing stop if we've reached activation threshold
        high_gain_pct = ((high_watermark - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        
        if high_gain_pct < TRAILING_STOP_ACTIVATION_PCT:
            return None  # Not enough profit to activate trailing stop
        
        # Check if current price dropped enough from high
        drop_from_high_pct = ((high_watermark - current_price) / high_watermark) * 100 if high_watermark > 0 else 0
        
        if drop_from_high_pct >= TRAILING_STOP_DISTANCE_PCT:
            # Trailing stop triggered!
            exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else current_price
            
            # Clear the watermark
            del self._position_high_watermarks[token_id]
            
            reason = (
                f"Trailing stop: Price dropped {drop_from_high_pct:.1f}% from high of ${high_watermark:.3f}. "
                f"Locking in gains. High gain was {high_gain_pct:.1f}%, current gain: {gain_pct:.1f}%"
            )
            
            return ExitProposal(
                token_id=position.token_id,
                market_id=position.market_id,
                exit_type="TRAILING_STOP",
                reason=reason,
                side="SELL" if position.side == "LONG" else "BUY",
                price=exit_price,
                size=position.quantity,
                pnl=position.unrealized_pnl,
            )

        return None

    def create_exit_intents(
        self,
        proposals: list[ExitProposal],
        intent_manager: IntentManager,
        cycle_id: str,
        positions: dict[str, "Position"] | None = None,
    ) -> list[TradeIntent]:
        """Create trade intents from exit proposals.

        Args:
            proposals: Exit proposals.
            intent_manager: Intent manager.
            cycle_id: Current cycle ID.
            positions: Optional positions dict for entry price lookup.

        Returns:
            List of created intents.
        """
        intents = []

        for proposal in proposals:
            # Get entry price if positions provided
            entry_price = None
            if positions and proposal.token_id in positions:
                entry_price = positions[proposal.token_id].avg_entry_price
            
            intent = intent_manager.create_exit_intent(
                token_id=proposal.token_id,
                market_id=proposal.market_id,
                side=proposal.side,
                price=proposal.price,
                size=proposal.size,
                reason=proposal.reason,
                cycle_id=cycle_id,
                is_emergency=proposal.is_emergency,
                entry_price=entry_price,
            )
            
            if intent is None:
                continue  # Dedup skipped
                
            intents.append(intent)

            log_level = logging.WARNING if proposal.is_emergency else logging.INFO
            logger.log(
                log_level,
                f"Created exit intent: {proposal.exit_type}{' 🚨 EMERGENCY' if proposal.is_emergency else ''}",
                extra={
                    "intent_id": intent.intent_id,
                    "exit_type": proposal.exit_type,
                    "token_id": proposal.token_id[:12],
                    "pnl": proposal.pnl,
                    "is_emergency": proposal.is_emergency,
                },
            )

        return intents

