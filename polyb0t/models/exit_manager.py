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
    """Manages exit proposals based on take-profit, stop-loss, and other criteria."""

    def __init__(self) -> None:
        """Initialize exit manager."""
        self.settings = get_settings()

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

