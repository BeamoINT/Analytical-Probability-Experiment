"""GPT-Powered Notification Summarizer.

Uses GPT-5.2 to create human-readable summaries of complex bot events
before sending them to Discord. Makes technical trading updates easy to understand.
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# System prompt that gives GPT context about the trading bot
SYSTEM_PROMPT = """You are a trading bot assistant for Polymarket, a prediction market platform.
Your job is to summarize technical trading events into clear, concise updates (2-3 sentences max).
The user is monitoring their autonomous trading bot and wants to understand what's happening.

Key context:
- MoE (Mixture of Experts) = AI system with multiple specialized prediction models
- Expert states: ACTIVE (making predictions), PROBATION (new/testing), SUSPENDED (underperforming), DEPRECATED (removed permanently)
- Whale = Large market trade (>$25k or >5% of daily volume) that may signal smart money
- Training examples = Market data snapshots used to improve the AI's predictions
- Confidence multiplier = How much the system trusts each expert (0.3 to 1.0)
- Cross-expert awareness = Experts can see what other experts predict and adjust

Write in a friendly, informative tone. Focus on what the user cares about:
- Is the system healthy?
- Is it making money?
- Are there opportunities or risks?

Keep it brief and actionable. No technical jargon unless necessary."""


class NotificationSummarizer:
    """Uses GPT-5.2 to create human-readable summaries of bot events."""
    
    MODEL = "gpt-4o"  # Using gpt-4o as fallback if gpt-5.2 not available
    MAX_TOKENS = 150
    TEMPERATURE = 0.7  # Slightly creative for natural language
    
    def __init__(self):
        """Initialize the summarizer."""
        settings = get_settings()
        self.api_key = settings.openai_api_key
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            logger.info("Notification summarizer initialized with GPT")
        else:
            logger.info("Notification summarizer disabled (no OpenAI API key)")
    
    async def summarize(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        """Generate a human-readable summary of an event.
        
        Args:
            event_type: Type of event (training, expert_state, whale, hourly, daily, trade)
            data: Event data to summarize
            
        Returns:
            Plain English summary, or None if summarization fails
        """
        if not self.enabled:
            return None
        
        try:
            prompt = self._build_prompt(event_type, data)
            summary = self._call_gpt(prompt)
            return summary
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return None
    
    def summarize_sync(self, event_type: str, data: Dict[str, Any]) -> Optional[str]:
        """Synchronous version of summarize for non-async contexts."""
        if not self.enabled:
            return None
        
        try:
            prompt = self._build_prompt(event_type, data)
            summary = self._call_gpt(prompt)
            return summary
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return None
    
    def _build_prompt(self, event_type: str, data: Dict[str, Any]) -> str:
        """Build a context-aware prompt for the event type."""
        
        if event_type == "training":
            return self._build_training_prompt(data)
        elif event_type == "expert_state":
            return self._build_expert_state_prompt(data)
        elif event_type == "whale":
            return self._build_whale_prompt(data)
        elif event_type == "hourly":
            return self._build_hourly_prompt(data)
        elif event_type == "daily":
            return self._build_daily_prompt(data)
        elif event_type == "trade":
            return self._build_trade_prompt(data)
        elif event_type == "model_debrief":
            return self._build_model_debrief_prompt(data)
        else:
            return f"Summarize this trading bot event:\n{json.dumps(data, indent=2)}"
    
    def _build_training_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for training completion event."""
        return f"""The AI trading system just completed a training cycle. Summarize what happened:

Training Results:
- Training time: {data.get('training_time_seconds', 0):.1f} seconds
- Training samples used: {data.get('n_samples', 0):,}
- Experts trained: {data.get('n_experts_trained', 0)}
- Active experts (making predictions): {data.get('n_active', 0)}
- Suspended experts (temporarily paused): {data.get('n_suspended', 0)}
- Deprecated experts (permanently removed): {data.get('n_deprecated', 0)}
- New experts created: {data.get('n_new_experts', 0)}

Top performing experts: {data.get('top_experts', 'N/A')}

Explain what this means for the trading system's health and performance."""
    
    def _build_expert_state_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for expert state change event."""
        return f"""An AI expert model changed its state. Summarize what happened:

Expert: {data.get('expert_id', 'Unknown')}
Previous state: {data.get('old_state', 'Unknown')}
New state: {data.get('new_state', 'Unknown')}
Expert's profit performance: {data.get('profit_pct', 0):+.1%}

Explain what this state change means. Is this good or bad for the trading system?"""
    
    def _build_whale_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for whale activity event."""
        return f"""A large trade (whale activity) was detected on Polymarket. Summarize:

Market/Asset: {data.get('asset_id', 'Unknown')[:40]}
Trade side: {data.get('side', 'Unknown')} (buying or selling)
Trade size: ${data.get('value_usd', 0):,.0f}
Price: ${data.get('price', 0):.3f}
Percentage of daily volume: {data.get('pct_of_volume', 0)*100:.1f}%

Is this a bullish or bearish signal? What might this whale know that others don't?"""
    
    def _build_hourly_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for hourly summary."""
        return f"""Summarize the trading bot's hourly status in 2-3 sentences:

Portfolio value: ${data.get('portfolio_value', 0):,.2f}
Active AI experts: {data.get('ai_status', {}).get('active_experts', 0)}
Training examples collected: {data.get('ai_status', {}).get('training_examples', 0):,}
Time until next AI training: {data.get('time_until_training', 'Unknown')}

Give a brief, friendly status update. Mention how long until the next training cycle."""
    
    def _build_daily_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for daily report."""
        daily_pnl = data.get('ending_balance', 0) - data.get('starting_balance', 0)
        daily_pnl_pct = (daily_pnl / data.get('starting_balance', 1)) * 100 if data.get('starting_balance', 0) > 0 else 0
        
        return f"""Summarize the trading bot's daily performance:

Starting balance: ${data.get('starting_balance', 0):,.2f}
Ending balance: ${data.get('ending_balance', 0):,.2f}
Daily P&L: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)
Total trades: {data.get('total_trades', 0)}
Win rate: {data.get('win_rate', 0):.1%}
Best trade: {data.get('best_trade', 0):+.2f}%
Worst trade: {data.get('worst_trade', 0):+.2f}%

Was today a good day? What's the outlook?"""
    
    def _build_trade_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for trade closed event."""
        return f"""A trade just closed. Summarize the result:

Market: {data.get('market_title', 'Unknown')}
Side: {data.get('side', 'Unknown')}
Entry price: ${data.get('entry_price', 0):.3f}
Exit price: ${data.get('exit_price', 0):.3f}
Profit/Loss: {data.get('profit_pct', 0):+.2f}% (${data.get('profit_usd', 0):+.2f})
Hold time: {data.get('hold_time_hours', 0):.1f} hours
AI confidence: {data.get('prediction_confidence', 0):.1%}
Top expert used: {data.get('top_expert', 'N/A')}

Explain what happened and whether the AI made a good decision."""
    
    def _build_model_debrief_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for model performance debrief.
        
        This provides a detailed explanation of how the AI model is performing,
        shown once per training cycle.
        """
        total_experts = data.get('total_experts', 0)
        active = data.get('active_experts', 0)
        suspended = data.get('suspended_experts', 0)
        deprecated = data.get('deprecated_experts', 0)
        probation = data.get('probation_experts', 0)
        
        return f"""Provide a detailed debrief on the AI trading model's current performance. 
Explain what these metrics mean and whether the system is healthy. Be educational but concise (3-4 sentences).

Model Performance Metrics:
- Total experts in the pool: {total_experts}
- Active experts (trusted, making predictions): {active}
- Probation experts (new, being tested): {probation}
- Suspended experts (temporarily paused due to poor performance): {suspended}
- Deprecated experts (permanently removed): {deprecated}
- Total simulated trades by active experts: {data.get('total_simulated_trades', 0)}
- Best performing expert: {data.get('best_expert', 'N/A')} with {data.get('best_expert_profit', 0):+.1%} profit
- Training examples collected: {data.get('training_examples', 0):,}
- Labeled examples (with known outcomes): {data.get('labeled_examples', 0):,}

Key questions to address:
1. Is the ratio of active to suspended experts healthy? (More active = better)
2. Are there enough training examples for reliable learning?
3. Is the best expert showing positive profit?
4. What does this mean for the system's ability to make good trading decisions?

Keep the explanation accessible to someone who isn't a machine learning expert."""
    
    def _call_gpt(self, user_prompt: str) -> Optional[str]:
        """Call the OpenAI API to get a summary."""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": self.MAX_TOKENS,
                    "temperature": self.TEMPERATURE,
                },
                timeout=15,
            )
            
            if response.status_code == 429:
                logger.warning("OpenAI rate limit - skipping summary")
                return None
            
            if response.status_code != 200:
                logger.warning(f"OpenAI error: {response.status_code}")
                return None
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Clean up the response
            summary = content.strip()
            
            # Truncate if too long (Discord embed description limit)
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            return summary
            
        except requests.RequestException as e:
            logger.warning(f"GPT request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to parse GPT response: {e}")
            return None


# Singleton instance
_notification_summarizer: Optional[NotificationSummarizer] = None


def get_notification_summarizer() -> NotificationSummarizer:
    """Get or create the notification summarizer singleton."""
    global _notification_summarizer
    if _notification_summarizer is None:
        _notification_summarizer = NotificationSummarizer()
    return _notification_summarizer
