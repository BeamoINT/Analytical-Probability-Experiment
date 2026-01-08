"""Exit manager for proposing position exits (take-profit, stop-loss, time-based)."""

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
    ) -> None:
        """Initialize exit proposal.

        Args:
            token_id: Token identifier.
            market_id: Market ID.
            exit_type: TAKE_PROFIT, STOP_LOSS, TIME_EXIT, LIQUIDITY_EXIT.
            reason: Human-readable reason.
            side: SELL (close long) or BUY (close short).
            price: Proposed exit price.
            size: Position size to close.
            pnl: Expected PnL.
        """
        self.token_id = token_id
        self.market_id = market_id
        self.exit_type = exit_type
        self.reason = reason
        self.side = side
        self.price = price
        self.size = size
        self.pnl = pnl


class ExitManager:
    """Manages exit proposals based on take-profit, stop-loss, and other criteria.
    
    Exit Types:
        - TAKE_PROFIT: Price increased by target %
        - STOP_LOSS: Price decreased by limit %
        - TIME_EXIT: Market resolving soon
        - LIQUIDITY_EXIT: Spread widened significantly
        - MOMENTUM_EXIT: Momentum turned strongly negative (new)
        - TRAILING_STOP: Price dropped from recent high (new)
    """

    def __init__(self) -> None:
        """Initialize exit manager."""
        self.settings = get_settings()
        
        # Track highest price seen for trailing stops
        self._position_high_watermarks: dict[str, float] = {}
        
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

            # Check stop-loss
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
        """Check if stop-loss threshold is met.

        Args:
            position: Position to check.
            orderbook: Current orderbook.

        Returns:
            ExitProposal if stop-loss triggered, else None.
        """
        # Calculate loss percentage
        pnl_pct = (position.unrealized_pnl / position.quantity) * 100

        if pnl_pct <= -self.settings.stop_loss_pct:
            # Propose exit at current bid (for long) or ask (for short)
            if position.side == "LONG":
                exit_price = orderbook.bids[0].price if orderbook and orderbook.bids else position.current_price
                exit_side = "SELL"
            else:
                exit_price = orderbook.asks[0].price if orderbook and orderbook.asks else position.current_price
                exit_side = "BUY"

            reason = (
                f"Stop-loss: Loss {abs(pnl_pct):.1f}% >= limit {self.settings.stop_loss_pct}%. "
                f"Current: ${position.current_price:.3f}, Entry: ${position.avg_entry_price:.3f}"
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
            
            # Case 1: Falling knife - get out before it gets worse
            if signal.is_falling_knife:
                exit_price = orderbook.bids[0].price if orderbook.bids else position.current_price
                
                reason = (
                    f"Momentum exit: Falling knife detected (24h momentum: {signal.price_momentum_24h:.1%}). "
                    f"Exiting before further loss. Current PnL: {pnl_pct:.1f}%"
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
            
            # Case 2: Heavy sell pressure in order book
            HEAVY_SELL_THRESHOLD = -0.5  # 50% more asks than bids
            if signal.order_book_imbalance < HEAVY_SELL_THRESHOLD and pnl_pct < 3:
                exit_price = orderbook.bids[0].price if orderbook.bids else position.current_price
                
                reason = (
                    f"Momentum exit: Heavy sell pressure in orderbook (imbalance: {signal.order_book_imbalance:.2f}). "
                    f"Exiting to avoid being caught in selloff. Current PnL: {pnl_pct:.1f}%"
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
    ) -> list[TradeIntent]:
        """Create trade intents from exit proposals.

        Args:
            proposals: Exit proposals.
            intent_manager: Intent manager.
            cycle_id: Current cycle ID.

        Returns:
            List of created intents.
        """
        intents = []

        for proposal in proposals:
            intent = intent_manager.create_exit_intent(
                token_id=proposal.token_id,
                market_id=proposal.market_id,
                side=proposal.side,
                price=proposal.price,
                size=proposal.size,
                reason=proposal.reason,
                cycle_id=cycle_id,
            )
            intents.append(intent)

            logger.info(
                f"Created exit intent: {proposal.exit_type}",
                extra={
                    "intent_id": intent.intent_id,
                    "exit_type": proposal.exit_type,
                    "token_id": proposal.token_id[:12],
                    "pnl": proposal.pnl,
                },
            )

        return intents

