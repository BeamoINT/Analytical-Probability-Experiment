"""Intelligent News Analyzer using LLM for true understanding.

Uses OpenAI/compatible LLM to actually understand news content
and determine if it confirms a market outcome.

Much more accurate than keyword matching.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

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


class IntelligentAnalyzer:
    """Uses LLM to intelligently analyze news for market confirmation."""
    
    # Configuration
    API_KEY_ENV = "OPENAI_API_KEY"
    MODEL = "gpt-3.5-turbo"  # Cheap and fast, good enough for this
    MAX_TOKENS = 300
    TEMPERATURE = 0.1  # Low temperature for consistent analysis
    
    # Cache to avoid duplicate API calls
    _cache: dict = {}
    _cache_duration = timedelta(hours=1)
    
    # Cost tracking
    _total_tokens_used = 0
    _total_requests = 0
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV)
        
        if not self.api_key:
            logger.warning(
                f"No OpenAI API key found. Set {self.API_KEY_ENV} environment variable. "
                "Intelligent analysis will be disabled."
            )
    
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
        
        Uses LLM to truly understand the content and determine if it
        answers the market question.
        
        Args:
            market_question: The Polymarket question (e.g., "Will Biden pardon Hunter?")
            headline: The news headline
            article_content: Optional article body text
            source: News source name
            
        Returns:
            IntelligentConfirmation with YES/NO/None outcome
        """
        if not self.api_key:
            return None
        
        # Check cache
        cache_key = f"{market_question}:{headline}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_duration:
                return cached_result
        
        # Build the analysis prompt
        prompt = self._build_prompt(market_question, headline, article_content)
        
        try:
            response = self._call_llm(prompt)
            if not response:
                return None
            
            # Parse the response
            result = self._parse_response(response, market_question, headline, source)
            
            # Cache the result
            if result:
                self._cache[cache_key] = (datetime.utcnow(), result)
                
                if result.confirmed_outcome:
                    logger.info(
                        f"LLM confirms {result.confirmed_outcome} for '{market_question[:40]}' "
                        f"(conf={result.confidence:.0%}): {result.reasoning[:50]}..."
                    )
            
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
        """Build the analysis prompt for the LLM."""
        content = headline
        if article_content:
            # Limit content length
            content += f"\n\nArticle excerpt: {article_content[:500]}"
        
        return f"""You are analyzing news to determine if it answers a prediction market question.

MARKET QUESTION: "{market_question}"

NEWS HEADLINE: "{headline}"
{f'ARTICLE CONTENT: {article_content[:500]}' if article_content else ''}

TASK: Determine if this news DEFINITIVELY answers the market question.

Rules:
1. Only say YES if the news CONFIRMS the event happened/will happen
2. Only say NO if the news CONFIRMS the event did NOT/will NOT happen
3. Say UNCERTAIN if the news doesn't definitively answer the question
4. Be VERY careful - only confirm if the news is clear and definitive

Respond in this exact JSON format:
{{
    "outcome": "YES" or "NO" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "reasoning": "One sentence explanation"
}}

JSON Response:"""
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the OpenAI API."""
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
                        {"role": "system", "content": "You are a precise news analyst for prediction markets. Respond only in JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.MAX_TOKENS,
                    "temperature": self.TEMPERATURE,
                },
                timeout=30,
            )
            
            self._total_requests += 1
            
            if response.status_code == 429:
                logger.warning("OpenAI rate limit reached")
                return None
            
            if response.status_code != 200:
                logger.warning(f"OpenAI error: {response.status_code} - {response.text[:200]}")
                return None
            
            data = response.json()
            self._total_tokens_used += data.get("usage", {}).get("total_tokens", 0)
            
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
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if not json_match:
                logger.warning(f"Could not find JSON in LLM response: {response[:100]}")
                return None
            
            data = json.loads(json_match.group())
            
            outcome = data.get("outcome", "UNCERTAIN").upper()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Map outcome
            if outcome == "YES":
                confirmed = "YES"
            elif outcome == "NO":
                confirmed = "NO"
            else:
                confirmed = None  # UNCERTAIN
            
            # Only return if confident enough
            if confirmed and confidence < 0.7:
                logger.debug(f"LLM confidence too low ({confidence:.0%}) for {market_question[:30]}")
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
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "is_available": self.is_available(),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens_used,
            "cache_size": len(self._cache),
            "estimated_cost_usd": self._total_tokens_used * 0.000002,  # GPT-3.5 pricing
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()


# Singleton
_analyzer_instance: Optional[IntelligentAnalyzer] = None


def get_intelligent_analyzer() -> IntelligentAnalyzer:
    """Get or create the intelligent analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = IntelligentAnalyzer()
    return _analyzer_instance
