"""Arbitrage Scanner - Finds and executes risk-free arbitrage opportunities.

Arbitrage Types on Polymarket:

1. BINARY ARBITRAGE (Same Market)
   - YES + NO should always sum to ~$1.00
   - If YES = $0.45 and NO = $0.45: Buy both for $0.90, guaranteed $1.00 payout
   - Profit: $0.10 per $0.90 invested = 11.1% risk-free return
   
2. COMPLEMENTARY MARKET ARBITRAGE
   - Markets that should sum to 100% probability
   - Example: "Winner of Super Bowl" - all teams should sum to ~100%
   - If they sum to <100%, buy the underpriced combination
   
3. RELATED MARKET ARBITRAGE
   - Markets that are logically linked
   - Example: "Trump wins" vs "Republican wins" (Trump is Republican nominee)
   - If Trump wins implies Republican wins, prices should reflect that
"""

import logging
from dataclasses import dataclass
from typing import Any

from polyb0t.config import get_settings
from polyb0t.data.models import Market

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""
    
    arb_type: str  # BINARY, COMPLEMENTARY, RELATED
    market_ids: list[str]
    token_ids: list[str]
    description: str
    
    # Costs
    total_cost: float  # Total cost to execute
    guaranteed_payout: float  # Guaranteed payout on resolution
    profit: float  # Guaranteed profit
    profit_pct: float  # Profit as percentage
    
    # Execution details
    trades: list[dict[str, Any]]  # List of trades to execute
    
    # Risk assessment
    is_profitable_after_fees: bool
    net_profit_after_fees: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "arb_type": self.arb_type,
            "market_ids": self.market_ids,
            "token_ids": self.token_ids,
            "description": self.description,
            "total_cost": self.total_cost,
            "guaranteed_payout": self.guaranteed_payout,
            "profit": self.profit,
            "profit_pct": self.profit_pct,
            "trades": self.trades,
            "is_profitable_after_fees": self.is_profitable_after_fees,
            "net_profit_after_fees": self.net_profit_after_fees,
        }


class ArbitrageScanner:
    """Scans markets for arbitrage opportunities.
    
    Arbitrage is the holy grail of trading: risk-free profit.
    On Polymarket, arbitrage exists when market prices are inconsistent.
    """
    
    def __init__(self) -> None:
        """Initialize arbitrage scanner."""
        self.settings = get_settings()
        
        # Minimum profit thresholds (after fees)
        self.min_profit_pct = 1.0  # At least 1% profit
        self.min_profit_usd = 0.50  # At least $0.50 profit
        
        # Fee estimation
        self.fee_per_trade_bps = self.settings.fee_bps  # Basis points per trade
        self.slippage_bps = self.settings.slippage_bps  # Expected slippage
    
    def scan_for_binary_arbitrage(
        self,
        markets: list[Market],
        orderbooks: dict[str, dict[str, Any]] | None = None,
    ) -> list[ArbitrageOpportunity]:
        """Scan binary markets for YES/NO arbitrage.
        
        Binary markets have exactly 2 outcomes: YES and NO.
        YES + NO should always = $1.00 (minus fees).
        If total < $1.00 (after fees), there's arbitrage.
        
        Args:
            markets: List of markets to scan.
            orderbooks: Optional orderbook data for better pricing.
            
        Returns:
            List of arbitrage opportunities found.
        """
        opportunities = []
        
        for market in markets:
            # Only binary markets
            if len(market.outcomes) != 2:
                continue
            
            try:
                opp = self._check_binary_arbitrage(market, orderbooks)
                if opp and opp.is_profitable_after_fees:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error checking binary arbitrage for {market.condition_id[:12]}: {e}")
        
        # Sort by profit percentage (best first)
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        if opportunities:
            logger.info(
                f"Found {len(opportunities)} binary arbitrage opportunities",
                extra={"best_profit_pct": opportunities[0].profit_pct if opportunities else 0},
            )
        
        return opportunities
    
    def _check_binary_arbitrage(
        self,
        market: Market,
        orderbooks: dict[str, dict[str, Any]] | None,
    ) -> ArbitrageOpportunity | None:
        """Check a single binary market for arbitrage.
        
        Args:
            market: Binary market to check.
            orderbooks: Optional orderbook data.
            
        Returns:
            ArbitrageOpportunity if found, else None.
        """
        if len(market.outcomes) != 2:
            return None
        
        outcome_yes = market.outcomes[0]
        outcome_no = market.outcomes[1]
        
        # Get best ask prices (what we'd pay to buy)
        yes_price = self._get_best_ask(outcome_yes.token_id, orderbooks) or outcome_yes.price
        no_price = self._get_best_ask(outcome_no.token_id, orderbooks) or outcome_no.price
        
        if yes_price is None or no_price is None:
            return None
        
        if yes_price <= 0 or no_price <= 0:
            return None
        
        # Total cost to buy both YES and NO
        total_cost = yes_price + no_price
        
        # Guaranteed payout: $1.00 (one of them always wins)
        guaranteed_payout = 1.0
        
        # Calculate fees (per trade)
        # For 2 trades (buy YES + buy NO), fees are:
        total_fee_rate = (self.fee_per_trade_bps + self.slippage_bps) * 2 / 10000
        total_fees = total_cost * total_fee_rate
        
        # Profit calculation
        gross_profit = guaranteed_payout - total_cost
        net_profit = gross_profit - total_fees
        net_profit_pct = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Check if profitable
        is_profitable = net_profit >= self.min_profit_usd or net_profit_pct >= self.min_profit_pct
        
        if not is_profitable:
            return None
        
        # Build trade instructions
        trades = [
            {
                "token_id": outcome_yes.token_id,
                "side": "BUY",
                "price": yes_price,
                "outcome": outcome_yes.name or "YES",
            },
            {
                "token_id": outcome_no.token_id,
                "side": "BUY",
                "price": no_price,
                "outcome": outcome_no.name or "NO",
            },
        ]
        
        description = (
            f"Binary arbitrage: Buy {outcome_yes.name or 'YES'} @ ${yes_price:.3f} + "
            f"{outcome_no.name or 'NO'} @ ${no_price:.3f} = ${total_cost:.3f}. "
            f"Guaranteed payout: $1.00. Net profit: ${net_profit:.4f} ({net_profit_pct:.2f}%)"
        )
        
        logger.info(
            f"ARBITRAGE FOUND: {market.question[:50]}... Net profit: {net_profit_pct:.2f}%",
            extra={
                "market_id": market.condition_id,
                "yes_price": yes_price,
                "no_price": no_price,
                "total_cost": total_cost,
                "net_profit_pct": net_profit_pct,
            },
        )
        
        return ArbitrageOpportunity(
            arb_type="BINARY",
            market_ids=[market.condition_id],
            token_ids=[outcome_yes.token_id, outcome_no.token_id],
            description=description,
            total_cost=total_cost,
            guaranteed_payout=guaranteed_payout,
            profit=gross_profit,
            profit_pct=(gross_profit / total_cost) * 100 if total_cost > 0 else 0,
            trades=trades,
            is_profitable_after_fees=is_profitable,
            net_profit_after_fees=net_profit,
        )
    
    def scan_for_complementary_arbitrage(
        self,
        markets: list[Market],
        orderbooks: dict[str, dict[str, Any]] | None = None,
    ) -> list[ArbitrageOpportunity]:
        """Scan for arbitrage in multi-outcome markets.
        
        In markets like "Who wins the Super Bowl?", all outcomes should sum to ~100%.
        If they sum to significantly less, there's arbitrage.
        
        Args:
            markets: Markets to scan.
            orderbooks: Optional orderbook data.
            
        Returns:
            List of arbitrage opportunities.
        """
        opportunities = []
        
        for market in markets:
            # Only multi-outcome markets (3+ outcomes)
            if len(market.outcomes) < 3:
                continue
            
            try:
                opp = self._check_complementary_arbitrage(market, orderbooks)
                if opp and opp.is_profitable_after_fees:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error checking complementary arb for {market.condition_id[:12]}: {e}")
        
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        return opportunities
    
    def _check_complementary_arbitrage(
        self,
        market: Market,
        orderbooks: dict[str, dict[str, Any]] | None,
    ) -> ArbitrageOpportunity | None:
        """Check multi-outcome market for complementary arbitrage.
        
        Args:
            market: Multi-outcome market.
            orderbooks: Optional orderbook data.
            
        Returns:
            ArbitrageOpportunity if found, else None.
        """
        if len(market.outcomes) < 3:
            return None
        
        # Get prices for all outcomes
        total_cost = 0.0
        trades = []
        
        for outcome in market.outcomes:
            price = self._get_best_ask(outcome.token_id, orderbooks) or outcome.price
            if price is None or price <= 0:
                return None  # Can't price all outcomes
            
            total_cost += price
            trades.append({
                "token_id": outcome.token_id,
                "side": "BUY",
                "price": price,
                "outcome": outcome.name,
            })
        
        # Guaranteed payout is $1.00 (one outcome wins)
        guaranteed_payout = 1.0
        
        # Calculate fees
        num_trades = len(trades)
        total_fee_rate = (self.fee_per_trade_bps + self.slippage_bps) * num_trades / 10000
        total_fees = total_cost * total_fee_rate
        
        # Profit
        gross_profit = guaranteed_payout - total_cost
        net_profit = gross_profit - total_fees
        net_profit_pct = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Must be profitable
        is_profitable = net_profit >= self.min_profit_usd or net_profit_pct >= self.min_profit_pct
        
        if not is_profitable:
            return None
        
        outcomes_str = ", ".join(f"{t['outcome']} @ ${t['price']:.3f}" for t in trades[:3])
        if len(trades) > 3:
            outcomes_str += f", ... ({len(trades)} total)"
        
        description = (
            f"Complementary arbitrage: Buy all {len(trades)} outcomes for ${total_cost:.3f}. "
            f"Guaranteed $1.00 payout. Net profit: ${net_profit:.4f} ({net_profit_pct:.2f}%)"
        )
        
        logger.info(
            f"COMPLEMENTARY ARB FOUND: {market.question[:50]}... Net profit: {net_profit_pct:.2f}%",
            extra={
                "market_id": market.condition_id,
                "num_outcomes": len(trades),
                "total_cost": total_cost,
                "net_profit_pct": net_profit_pct,
            },
        )
        
        return ArbitrageOpportunity(
            arb_type="COMPLEMENTARY",
            market_ids=[market.condition_id],
            token_ids=[t["token_id"] for t in trades],
            description=description,
            total_cost=total_cost,
            guaranteed_payout=guaranteed_payout,
            profit=gross_profit,
            profit_pct=(gross_profit / total_cost) * 100 if total_cost > 0 else 0,
            trades=trades,
            is_profitable_after_fees=is_profitable,
            net_profit_after_fees=net_profit,
        )
    
    def scan_all(
        self,
        markets: list[Market],
        orderbooks: dict[str, dict[str, Any]] | None = None,
    ) -> list[ArbitrageOpportunity]:
        """Scan all markets for any type of arbitrage.
        
        Args:
            markets: Markets to scan.
            orderbooks: Optional orderbook data.
            
        Returns:
            All arbitrage opportunities found, sorted by profit.
        """
        opportunities = []
        
        # Binary arbitrage
        binary_opps = self.scan_for_binary_arbitrage(markets, orderbooks)
        opportunities.extend(binary_opps)
        
        # Complementary arbitrage
        comp_opps = self.scan_for_complementary_arbitrage(markets, orderbooks)
        opportunities.extend(comp_opps)
        
        # Sort by net profit percentage
        opportunities.sort(key=lambda x: x.net_profit_after_fees, reverse=True)
        
        return opportunities
    
    def _get_best_ask(
        self,
        token_id: str,
        orderbooks: dict[str, dict[str, Any]] | None,
    ) -> float | None:
        """Get best ask price for a token.
        
        Args:
            token_id: Token to get price for.
            orderbooks: Orderbook data.
            
        Returns:
            Best ask price or None.
        """
        if not orderbooks or token_id not in orderbooks:
            return None
        
        ob = orderbooks[token_id]
        asks = ob.get("asks", [])
        
        if not asks:
            return None
        
        # Best ask is lowest price someone will sell at
        return float(asks[0].get("price", 0))
    
    def create_arbitrage_intents(
        self,
        opportunity: ArbitrageOpportunity,
        intent_manager: Any,
        cycle_id: str,
        max_usd_per_arb: float = 100.0,
    ) -> list[Any]:
        """Create trade intents to execute an arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage to execute.
            intent_manager: Intent manager for creating intents.
            cycle_id: Current cycle ID.
            max_usd_per_arb: Maximum USD to allocate to this arb.
            
        Returns:
            List of created intents.
        """
        from polyb0t.execution.intents import IntentType
        
        # Calculate how many "units" we can afford
        # One unit = buy all outcomes once
        units = max_usd_per_arb / opportunity.total_cost
        units = max(1.0, min(units, 10.0))  # 1-10 units
        
        intents = []
        
        for trade in opportunity.trades:
            # Size for this trade
            size_usd = trade["price"] * units
            
            intent = intent_manager.create_intent(
                intent_type=IntentType.OPEN_POSITION,
                token_id=trade["token_id"],
                market_id=opportunity.market_ids[0],
                side=trade["side"],
                price=trade["price"],
                size_usd=size_usd,
                reason=f"ARBITRAGE: {opportunity.arb_type} - {opportunity.description[:100]}",
                cycle_id=cycle_id,
                signal_data={
                    "arb_type": opportunity.arb_type,
                    "profit_pct": opportunity.profit_pct,
                    "net_profit": opportunity.net_profit_after_fees,
                    "is_arbitrage": True,
                },
            )
            intents.append(intent)
        
        logger.info(
            f"Created {len(intents)} arbitrage intents",
            extra={
                "arb_type": opportunity.arb_type,
                "profit_pct": opportunity.profit_pct,
                "total_size_usd": sum(t["price"] * units for t in opportunity.trades),
            },
        )
        
        return intents

