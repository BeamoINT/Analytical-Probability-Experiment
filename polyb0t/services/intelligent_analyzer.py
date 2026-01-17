"""Intelligent News Analyzer using GPT-5.2 for true understanding.

Uses OpenAI's GPT-5.2 to deeply understand news content
and determine if it definitively confirms a market outcome.

Features:
- Chain-of-thought reasoning for accuracy
- Structured JSON output
- Aggressive caching to minimize API costs
- Cost tracking and efficiency metrics
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class IntelligentConfirmation:
    """Result of intelligent news analysis."""
    market_question: str
    headline: str
    confirmed_outcome: Optional[str]  # "YES", "NO", or None if uncertain
    confidence: float  # 0.0 to 1.0
    reasoning: str  # LLM's explanation
    source: str
    analyzed_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "market_question": self.market_question,
            "headline": self.headline,
            "confirmed_outcome": self.confirmed_outcome,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "source": self.source,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


# System prompt - crafted for precision and accuracy
SYSTEM_PROMPT = """You are an expert financial analyst specializing in prediction market arbitrage.

Your task is to determine if a news article DEFINITIVELY answers a prediction market question.

CRITICAL RULES:
1. You are looking for ARBITRAGE - this means near-certain outcomes only
2. Only answer YES if the news PROVES the event HAS happened or WILL definitely happen
3. Only answer NO if the news PROVES the event has NOT happened or WILL NOT happen
4. Answer UNCERTAIN for anything else - speculation, predictions, opinions, partial info

EXAMPLES OF DEFINITIVE NEWS:
- "President Biden pardons Hunter Biden" → YES for "Will Biden pardon Hunter?"
- "Supreme Court rules 6-3 against..." → Depending on question, YES or NO
- "Company X files for bankruptcy" → YES for "Will Company X go bankrupt?"
- "Bill fails to pass Senate vote" → NO for "Will the bill pass?"

EXAMPLES OF NON-DEFINITIVE NEWS (answer UNCERTAIN):
- "Sources say Biden is considering..." → UNCERTAIN (not confirmed)
- "Experts predict Company X will..." → UNCERTAIN (prediction, not fact)
- "Bill likely to pass according to..." → UNCERTAIN (speculation)
- Old news about a different event → UNCERTAIN (not relevant)

THINK STEP BY STEP:
1. What exactly is the market question asking?
2. What does the headline/article actually say happened?
3. Does the news DIRECTLY and DEFINITIVELY answer the question?
4. Is this actual news or just speculation/opinion?
5. Am I 100% certain about this? If not, say UNCERTAIN.

Your response must be JSON with exactly these fields:
- outcome: "YES", "NO", or "UNCERTAIN"
- confidence: number from 0.0 to 1.0 (only >0.85 for definitive answers)
- reasoning: one clear sentence explaining your logic"""


class IntelligentAnalyzer:
    """Uses GPT-5.2 to intelligently analyze news for market confirmation."""
    
    # Configuration
    MODEL = "gpt-5.2"  # Latest model with best reasoning
    MAX_TOKENS = 150  # Keep output small for efficiency
    TEMPERATURE = 0.0  # Zero temperature for deterministic output
    
    # Cache settings - aggressive caching to save costs
    _cache: dict = {}
    _cache_duration = timedelta(hours=4)  # Cache for 4 hours
    
    # Cost tracking (GPT-5.2 pricing: $1.75/1M input, $14.00/1M output)
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_requests = 0
    _cache_hits = 0
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from settings or env var.
        """
        self.api_key = api_key
        
        # Try to get from settings first, then env var
        if not self.api_key:
            try:
                from polyb0t.config.settings import get_settings
                settings = get_settings()
                self.api_key = settings.openai_api_key
            except:
                pass
        
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        
        if not self.api_key:
            logger.warning(
                "No OpenAI API key found. Set POLYBOT_OPENAI_API_KEY in .env. "
                "Intelligent analysis will be disabled."
            )
        else:
            logger.info(f"Intelligent analyzer initialized with {self.MODEL}")
    
    def is_available(self) -> bool:
        """Check if LLM analysis is available."""
        return bool(self.api_key)
    
    def analyze_headline(
        self,
        market_question: str,
        headline: str,
        article_content: str = "",
        source: str = "Unknown",
    ) -> Optional[IntelligentConfirmation]:
        """Analyze if a headline confirms a market outcome.
        
        Uses GPT-5.2 to deeply understand the content and determine if it
        definitively answers the market question.
        
        Args:
            market_question: The Polymarket question (e.g., "Will Biden pardon Hunter?")
            headline: The news headline
            article_content: Optional article body text (truncated for efficiency)
            source: News source name
            
        Returns:
            IntelligentConfirmation with YES/NO/None outcome
        """
        if not self.api_key:
            return None
        
        # Normalize inputs for better cache hits
        market_q_normalized = market_question.strip().lower()
        headline_normalized = headline.strip().lower()
        
        # Check cache first
        cache_key = f"{market_q_normalized}:{headline_normalized}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_duration:
                self._cache_hits += 1
                logger.debug(f"Cache hit for: {headline[:40]}...")
                return cached_result
        
        # Build the analysis prompt - keep it concise for efficiency
        user_prompt = self._build_prompt(market_question, headline, article_content)
        
        try:
            response = self._call_llm(user_prompt)
            if not response:
                return None
            
            # Parse the response
            result = self._parse_response(response, market_question, headline, source)
            
            # Cache the result (cache both positive and negative results)
            if result:
                self._cache[cache_key] = (datetime.utcnow(), result)
                
                if result.confirmed_outcome:
                    logger.info(
                        f"[GPT-5.2] {result.confirmed_outcome} for '{market_question[:35]}...' "
                        f"(conf={result.confidence:.0%}): {result.reasoning}"
                    )
                else:
                    logger.debug(f"[GPT-5.2] UNCERTAIN for '{market_question[:35]}...'")
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None
    
    def _build_prompt(
        self,
        market_question: str,
        headline: str,
        article_content: str,
    ) -> str:
        """Build an efficient, focused prompt."""
        # Truncate article content to save tokens
        content_snippet = ""
        if article_content:
            # Only include first 200 chars - headlines usually have the key info
            content_snippet = f"\nArticle snippet: {article_content[:200].strip()}..."
        
        return f"""MARKET QUESTION: {market_question}

NEWS HEADLINE: {headline}{content_snippet}

Does this news DEFINITIVELY answer the market question? Think carefully, then respond with JSON only."""
    
    def _call_llm(self, user_prompt: str) -> Optional[str]:
        """Call the OpenAI API with GPT-5.2."""
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
                    "response_format": {"type": "json_object"},  # Force JSON output
                },
                timeout=30,
            )
            
            self._total_requests += 1
            
            if response.status_code == 429:
                logger.warning("OpenAI rate limit reached - backing off")
                return None
            
            if response.status_code != 200:
                logger.warning(f"OpenAI error: {response.status_code} - {response.text[:200]}")
                return None
            
            data = response.json()
            
            # Track token usage for cost monitoring
            usage = data.get("usage", {})
            self._total_input_tokens += usage.get("prompt_tokens", 0)
            self._total_output_tokens += usage.get("completion_tokens", 0)
            
            content = data["choices"][0]["message"]["content"]
            return content
            
        except requests.RequestException as e:
            logger.warning(f"OpenAI request failed: {e}")
            return None
    
    def _parse_response(
        self,
        response: str,
        market_question: str,
        headline: str,
        source: str,
    ) -> Optional[IntelligentConfirmation]:
        """Parse the LLM response into a confirmation object."""
        try:
            # Parse JSON response
            data = json.loads(response)
            
            outcome = data.get("outcome", "UNCERTAIN").upper().strip()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Map outcome
            if outcome == "YES":
                confirmed = "YES"
            elif outcome == "NO":
                confirmed = "NO"
            else:
                confirmed = None  # UNCERTAIN or anything else
            
            # Require high confidence for definitive answers (arbitrage must be sure)
            if confirmed and confidence < 0.80:
                logger.debug(
                    f"Confidence too low ({confidence:.0%}) for definitive answer - "
                    f"treating as UNCERTAIN"
                )
                confirmed = None
            
            return IntelligentConfirmation(
                market_question=market_question,
                headline=headline,
                confirmed_outcome=confirmed,
                confidence=confidence,
                reasoning=reasoning,
                source=source,
                analyzed_at=datetime.utcnow(),
            )
            
        except json.JSONDecodeError:
            # Try to extract JSON from response if it's wrapped in markdown
            try:
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    return self._parse_response(json_match.group(), market_question, headline, source)
            except:
                pass
            logger.warning(f"Failed to parse JSON from LLM: {response[:100]}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response fields: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get detailed usage statistics."""
        # Calculate costs (GPT-5.2 pricing)
        input_cost = (self._total_input_tokens / 1_000_000) * 1.75
        output_cost = (self._total_output_tokens / 1_000_000) * 14.00
        total_cost = input_cost + output_cost
        
        # Calculate efficiency
        cache_rate = (
            self._cache_hits / (self._cache_hits + self._total_requests) * 100
            if (self._cache_hits + self._total_requests) > 0 else 0
        )
        
        return {
            "model": self.MODEL,
            "is_available": self.is_available(),
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "cache_size": len(self._cache),
            "tokens": {
                "input": self._total_input_tokens,
                "output": self._total_output_tokens,
                "total": self._total_input_tokens + self._total_output_tokens,
            },
            "cost_usd": {
                "input": round(input_cost, 4),
                "output": round(output_cost, 4),
                "total": round(total_cost, 4),
            },
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("Intelligent analyzer cache cleared")
    
    def prune_cache(self):
        """Remove expired cache entries."""
        now = datetime.utcnow()
        expired = [
            key for key, (cached_time, _) in self._cache.items()
            if now - cached_time > self._cache_duration
        ]
        for key in expired:
            del self._cache[key]
        if expired:
            logger.debug(f"Pruned {len(expired)} expired cache entries")


# Singleton instance
_analyzer_instance: Optional[IntelligentAnalyzer] = None


def get_intelligent_analyzer() -> IntelligentAnalyzer:
    """Get or create the intelligent analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = IntelligentAnalyzer()
    return _analyzer_instance
