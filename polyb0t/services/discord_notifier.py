"""Discord Webhook Notification Service.

Sends notifications to Discord for important trading events:
- Trade executions (entries/exits with P&L)
- Whale activity detection
- Expert state changes (ACTIVE, DEPRECATED, etc.)
- Training completed
- Errors and warnings
- Hourly status summaries
- Daily performance reports
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# Discord embed colors (decimal values)
COLOR_SUCCESS = 0x00FF00  # Green
COLOR_ERROR = 0xFF0000    # Red
COLOR_WARNING = 0xFFA500  # Orange
COLOR_INFO = 0x0099FF     # Blue
COLOR_WHALE = 0x9B59B6    # Purple
COLOR_TRAINING = 0x3498DB  # Light blue


@dataclass
class DiscordEmbed:
    """Discord embed message structure."""
    
    title: str
    description: str = ""
    color: int = COLOR_INFO
    fields: List[Dict[str, Any]] = None
    footer: str = ""
    timestamp: Optional[str] = None
    thumbnail_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Discord webhook payload format."""
        embed = {
            "title": self.title,
            "color": self.color,
        }
        
        if self.description:
            embed["description"] = self.description
        
        if self.fields:
            embed["fields"] = self.fields
        
        if self.footer:
            embed["footer"] = {"text": self.footer}
        
        if self.timestamp:
            embed["timestamp"] = self.timestamp
        else:
            embed["timestamp"] = datetime.utcnow().isoformat()
        
        if self.thumbnail_url:
            embed["thumbnail"] = {"url": self.thumbnail_url}
        
        return embed


class DiscordNotifier:
    """Discord webhook notification service."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize the notifier.
        
        Args:
            webhook_url: Discord webhook URL. If not provided, uses settings.
        """
        settings = get_settings()
        
        self.webhook_url = webhook_url or getattr(settings, 'discord_webhook_url', '')
        self.enabled = bool(self.webhook_url) and getattr(settings, 'discord_notifications_enabled', True)
        
        # Feature flags
        self.notify_trades = getattr(settings, 'discord_notify_on_trade', True)
        self.notify_whales = getattr(settings, 'discord_notify_on_whale', True)
        self.notify_errors = getattr(settings, 'discord_notify_on_error', True)
        self.hourly_summary = getattr(settings, 'discord_hourly_summary', True)
        self.daily_report = getattr(settings, 'discord_daily_report', True)
        
        # Rate limiting
        self._last_send_time: Optional[datetime] = None
        self._min_interval_seconds = 1.0  # Minimum time between messages
        self._error_count = 0
        self._max_errors = 10  # Disable after too many errors
        
        if self.enabled:
            logger.info("Discord notifier initialized")
        else:
            logger.info("Discord notifier disabled (no webhook URL)")
    
    async def send(self, embed: DiscordEmbed, username: str = "Polybot") -> bool:
        """Send an embed message to Discord.
        
        Args:
            embed: The embed to send.
            username: Bot username to display.
            
        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False
        
        if self._error_count >= self._max_errors:
            logger.warning("Discord notifier disabled due to too many errors")
            return False
        
        # Rate limiting
        now = datetime.utcnow()
        if self._last_send_time:
            elapsed = (now - self._last_send_time).total_seconds()
            if elapsed < self._min_interval_seconds:
                await asyncio.sleep(self._min_interval_seconds - elapsed)
        
        payload = {
            "username": username,
            "embeds": [embed.to_dict()],
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.webhook_url, json=payload)
                
                if response.status_code == 204:
                    self._last_send_time = datetime.utcnow()
                    self._error_count = 0
                    return True
                elif response.status_code == 429:
                    # Rate limited
                    retry_after = response.json().get("retry_after", 5)
                    logger.warning(f"Discord rate limited, retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return False
                else:
                    logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                    self._error_count += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            self._error_count += 1
            return False
    
    async def notify_trade(self, outcome) -> bool:
        """Send trade execution notification.
        
        Args:
            outcome: TradeOutcome object.
        """
        if not self.notify_trades:
            return False
        
        # Color based on profit/loss
        color = COLOR_SUCCESS if outcome.profit_pct > 0 else COLOR_ERROR
        
        # Direction emoji
        direction = "📈" if outcome.profit_pct > 0 else "📉"
        
        fields = [
            {"name": "Market", "value": outcome.market_title[:50] or outcome.token_id[:16], "inline": False},
            {"name": "Side", "value": outcome.side, "inline": True},
            {"name": "Size", "value": f"${outcome.size:.2f}", "inline": True},
            {"name": "P&L", "value": f"{outcome.profit_pct:+.2f}% (${outcome.profit_usd:+.2f})", "inline": True},
            {"name": "Entry", "value": f"${outcome.entry_price:.3f}", "inline": True},
            {"name": "Exit", "value": f"${outcome.exit_price:.3f}", "inline": True},
            {"name": "Hold Time", "value": f"{outcome.hold_time_hours:.1f}h", "inline": True},
        ]
        
        if outcome.expert_predictions:
            top_expert = max(outcome.expert_predictions.keys(), 
                           key=lambda k: outcome.expert_weights.get(k, 0))
            fields.append({
                "name": "Top Expert", 
                "value": top_expert, 
                "inline": True
            })
        
        embed = DiscordEmbed(
            title=f"{direction} Trade Closed",
            color=color,
            fields=fields,
            footer=f"Confidence: {outcome.prediction_confidence:.1%}",
        )
        
        return await self.send(embed)
    
    async def notify_whale(self, whale_event) -> bool:
        """Send whale activity notification.
        
        Args:
            whale_event: WhaleEvent object.
        """
        if not self.notify_whales:
            return False
        
        direction = "🐋📈" if whale_event.is_buy else "🐋📉"
        
        fields = [
            {"name": "Asset", "value": whale_event.asset_id[:20] + "...", "inline": True},
            {"name": "Side", "value": whale_event.side, "inline": True},
            {"name": "Size", "value": f"${whale_event.value_usd:,.0f}", "inline": True},
            {"name": "Price", "value": f"${whale_event.price:.3f}", "inline": True},
            {"name": "Shares", "value": f"{whale_event.size:,.0f}", "inline": True},
        ]
        
        if whale_event.pct_of_volume:
            fields.append({
                "name": "% of Volume",
                "value": f"{whale_event.pct_of_volume*100:.1f}%",
                "inline": True,
            })
        
        embed = DiscordEmbed(
            title=f"{direction} Whale Detected",
            color=COLOR_WHALE,
            fields=fields,
        )
        
        return await self.send(embed)
    
    async def notify_expert_state_change(
        self,
        expert_id: str,
        old_state: str,
        new_state: str,
        profit_pct: float = 0.0,
    ) -> bool:
        """Send expert state change notification."""
        # Color based on state
        if new_state == "active":
            color = COLOR_SUCCESS
            emoji = "✅"
        elif new_state == "deprecated":
            color = COLOR_ERROR
            emoji = "❌"
        elif new_state == "suspended":
            color = COLOR_WARNING
            emoji = "⚠️"
        else:
            color = COLOR_INFO
            emoji = "ℹ️"
        
        embed = DiscordEmbed(
            title=f"{emoji} Expert State Change",
            description=f"**{expert_id}**: {old_state} → {new_state}",
            color=color,
            fields=[
                {"name": "Profit", "value": f"{profit_pct:+.1%}", "inline": True},
            ],
        )
        
        return await self.send(embed)
    
    async def notify_training_complete(
        self,
        num_experts: int,
        training_time: float,
        active_count: int,
    ) -> bool:
        """Send training completion notification."""
        embed = DiscordEmbed(
            title="🎓 Training Complete",
            color=COLOR_TRAINING,
            fields=[
                {"name": "Experts Trained", "value": str(num_experts), "inline": True},
                {"name": "Active Experts", "value": str(active_count), "inline": True},
                {"name": "Training Time", "value": f"{training_time:.1f}s", "inline": True},
            ],
        )
        
        return await self.send(embed)
    
    async def notify_error(
        self,
        error: str,
        traceback_str: str = "",
        severity: str = "error",
    ) -> bool:
        """Send error notification.
        
        Args:
            error: Error message.
            traceback_str: Optional traceback.
            severity: 'error' or 'warning'.
        """
        if not self.notify_errors:
            return False
        
        color = COLOR_ERROR if severity == "error" else COLOR_WARNING
        emoji = "🚨" if severity == "error" else "⚠️"
        
        description = error[:1000]  # Truncate long errors
        if traceback_str:
            description += f"\n```\n{traceback_str[:500]}\n```"
        
        embed = DiscordEmbed(
            title=f"{emoji} {severity.title()}",
            description=description,
            color=color,
        )
        
        return await self.send(embed)
    
    async def send_hourly_summary(
        self,
        portfolio_value: float,
        unrealized_pnl: float,
        trades_this_hour: int,
        active_positions: int,
        ai_status: Dict[str, Any],
    ) -> bool:
        """Send hourly status summary."""
        if not self.hourly_summary:
            return False
        
        pnl_emoji = "📈" if unrealized_pnl >= 0 else "📉"
        
        fields = [
            {"name": "Portfolio Value", "value": f"${portfolio_value:,.2f}", "inline": True},
            {"name": "Unrealized P&L", "value": f"{pnl_emoji} ${unrealized_pnl:+,.2f}", "inline": True},
            {"name": "Active Positions", "value": str(active_positions), "inline": True},
            {"name": "Trades (1h)", "value": str(trades_this_hour), "inline": True},
            {"name": "Active Experts", "value": str(ai_status.get("active_experts", 0)), "inline": True},
            {"name": "Training Examples", "value": f"{ai_status.get('training_examples', 0):,}", "inline": True},
        ]
        
        embed = DiscordEmbed(
            title="📊 Hourly Status",
            color=COLOR_INFO,
            fields=fields,
        )
        
        return await self.send(embed)
    
    async def send_daily_report(
        self,
        starting_balance: float,
        ending_balance: float,
        total_trades: int,
        win_rate: float,
        best_trade: float,
        worst_trade: float,
        expert_summary: Dict[str, Any],
    ) -> bool:
        """Send daily performance report."""
        if not self.daily_report:
            return False
        
        daily_pnl = ending_balance - starting_balance
        daily_pnl_pct = (daily_pnl / starting_balance * 100) if starting_balance > 0 else 0
        
        color = COLOR_SUCCESS if daily_pnl >= 0 else COLOR_ERROR
        emoji = "📈" if daily_pnl >= 0 else "📉"
        
        fields = [
            {"name": "Starting Balance", "value": f"${starting_balance:,.2f}", "inline": True},
            {"name": "Ending Balance", "value": f"${ending_balance:,.2f}", "inline": True},
            {"name": "Daily P&L", "value": f"{emoji} ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)", "inline": False},
            {"name": "Total Trades", "value": str(total_trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True},
            {"name": "Best Trade", "value": f"{best_trade:+.2f}%", "inline": True},
            {"name": "Worst Trade", "value": f"{worst_trade:+.2f}%", "inline": True},
        ]
        
        if expert_summary:
            top_expert = max(
                expert_summary.items(),
                key=lambda x: x[1].get("profit", 0),
                default=(None, {})
            )
            if top_expert[0]:
                fields.append({
                    "name": "Best Expert",
                    "value": f"{top_expert[0]} ({top_expert[1].get('profit', 0):+.1%})",
                    "inline": True,
                })
        
        embed = DiscordEmbed(
            title="📅 Daily Performance Report",
            color=color,
            fields=fields,
            footer=f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}",
        )
        
        return await self.send(embed)
    
    async def send_startup_notification(self, mode: str, balance: float) -> bool:
        """Send bot startup notification."""
        embed = DiscordEmbed(
            title="🚀 Polybot Started",
            color=COLOR_INFO,
            fields=[
                {"name": "Mode", "value": mode.upper(), "inline": True},
                {"name": "Balance", "value": f"${balance:,.2f}", "inline": True},
                {"name": "Time", "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "inline": True},
            ],
        )
        
        return await self.send(embed)
    
    async def send_shutdown_notification(self, reason: str = "Manual") -> bool:
        """Send bot shutdown notification."""
        embed = DiscordEmbed(
            title="🛑 Polybot Stopped",
            description=f"Reason: {reason}",
            color=COLOR_WARNING,
        )
        
        return await self.send(embed)


# Singleton instance
_discord_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier() -> DiscordNotifier:
    """Get or create the Discord notifier singleton."""
    global _discord_notifier
    if _discord_notifier is None:
        _discord_notifier = DiscordNotifier()
    return _discord_notifier
