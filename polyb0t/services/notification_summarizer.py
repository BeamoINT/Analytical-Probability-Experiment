"""GPT-Powered Notification Summarizer.

Uses GPT to create human-readable summaries of trading bot events for Discord.
Optimized for minimal token usage while providing clear, actionable updates.
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# Minimal system prompt - context is provided per-message to save tokens
SYSTEM_PROMPT = """You summarize trading bot updates in 2-3 sentences. Be concise, friendly, and focus on: system health, profit potential, and actionable insights. No jargon."""


class NotificationSummarizer:
    """Uses GPT to create human-readable summaries of bot events."""
    
    MODEL = "gpt-5.2"  # Using GPT-5.2 for best quality
    MAX_TOKENS = 200  # Allow fuller responses
    TEMPERATURE = 0.5  # More focused responses
    
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
        return f"""Training done: {data.get('n_samples', 0)} samples, {data.get('n_active', 0)} active/{data.get('n_suspended', 0)} suspended experts. Summarize health."""
    
    def _build_expert_state_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for expert state change event."""
        return f"""Expert {data.get('old_state', '?')}->{data.get('new_state', '?')}, profit {data.get('profit_pct', 0):+.1%}. Good or bad?"""
    
    def _build_whale_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for whale activity event."""
        return f"""Whale: {data.get('side', '?')} ${data.get('value_usd', 0):,.0f} at {data.get('price', 0):.2f} ({data.get('pct_of_volume', 0)*100:.0f}% vol). Signal?"""
    
    def _build_hourly_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for hourly summary."""
        return f"""Bot status: ${data.get('portfolio_value', 0):,.0f}, {data.get('ai_status', {}).get('active_experts', 0)} experts, {data.get('ai_status', {}).get('training_examples', 0)} examples, next train {data.get('time_until_training', '?')}. Brief update."""
    
    def _build_daily_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for daily report."""
        daily_pnl = data.get('ending_balance', 0) - data.get('starting_balance', 0)
        return f"""Daily: ${data.get('starting_balance', 0):,.0f}->${data.get('ending_balance', 0):,.0f} ({daily_pnl:+.0f}), {data.get('total_trades', 0)} trades, {data.get('win_rate', 0):.0%} win. Summary."""
    
    def _build_trade_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for trade closed event."""
        return f"""Trade closed: {data.get('side', '?')} {data.get('profit_pct', 0):+.1%} (${data.get('profit_usd', 0):+.0f}), {data.get('hold_time_hours', 0):.0f}h hold. Analyze."""
    
    def _build_model_debrief_prompt(self, data: Dict[str, Any]) -> str:
        """Build prompt for model performance debrief."""
        active = data.get('active_experts', 0)
        suspended = data.get('suspended_experts', 0)
        total = data.get('total_experts', 0)
        examples = data.get('training_examples', 0)
        best_profit = data.get('best_expert_profit', 0)
        
        return f"""AI model: {active}/{total} experts active, {suspended} suspended, {examples} training examples, best expert {best_profit:+.1%}. Is this healthy? 2 sentences max."""
    
    def _call_gpt(self, user_prompt: str) -> Optional[str]:
        """Call the OpenAI API to get a summary with model fallback."""
        # Try primary model first, then fallback
        models_to_try = [self.MODEL, "gpt-4o", "gpt-4o-mini"]
        
        for model in models_to_try:
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
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
                    logger.warning(f"OpenAI rate limit on {model} - skipping")
                    return None
                
                if response.status_code != 200:
                    # Log detailed error for debugging
                    error_text = response.text[:300] if response.text else "No response body"
                    logger.warning(f"OpenAI error with {model}: {response.status_code} - {error_text}")
                    # Try next model
                    continue
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Clean up the response
                summary = content.strip()
                if len(summary) > 500:
                    summary = summary[:497] + "..."
                
                if model != self.MODEL:
                    logger.info(f"Used fallback model {model} for summary")
                
                return summary
                
            except requests.RequestException as e:
                logger.warning(f"GPT request failed with {model}: {e}")
                continue
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to parse GPT response from {model}: {e}")
                continue
        
        logger.warning("All GPT models failed - no summary available")
        return None


# Singleton instance
_notification_summarizer: Optional[NotificationSummarizer] = None


def get_notification_summarizer() -> NotificationSummarizer:
    """Get or create the notification summarizer singleton."""
    global _notification_summarizer
    if _notification_summarizer is None:
        _notification_summarizer = NotificationSummarizer()
    return _notification_summarizer
