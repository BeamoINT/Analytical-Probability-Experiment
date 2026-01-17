"""Headline Analyzer for correlating news with Polymarket outcomes.

Analyzes news headlines to determine if they confirm market outcomes.
Uses keyword matching and sentiment detection to determine YES/NO.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from polyb0t.services.news_client import NewsArticle, get_news_client

logger = logging.getLogger(__name__)


@dataclass
class OutcomeConfirmation:
    """Represents a confirmed market outcome from news."""
    market_id: str
    market_title: str
    confirmed_outcome: str  # "YES" or "NO"
    confidence: float  # 0.0 to 1.0
    headline: str
    source: str
    published_at: datetime
    reasoning: str
    
    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "market_title": self.market_title,
            "confirmed_outcome": self.confirmed_outcome,
            "confidence": self.confidence,
            "headline": self.headline,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "reasoning": self.reasoning,
        }


class HeadlineAnalyzer:
    """Analyzes headlines to confirm market outcomes."""
    
    # Words that indicate positive/affirmative outcomes
    POSITIVE_INDICATORS = [
        "confirmed", "announces", "announced", "wins", "won", "passes", "passed",
        "approves", "approved", "signs", "signed", "accepts", "accepted",
        "succeeds", "succeeded", "achieves", "achieved", "completes", "completed",
        "launches", "launched", "starts", "started", "begins", "began",
        "reaches", "reached", "breaks", "broke", "surpasses", "surpassed",
        "elected", "elects", "chooses", "chosen", "selects", "selected",
        "defeats", "defeated", "beats", "beat", "overcomes", "overcame",
        "pardons", "pardoned", "grants", "granted", "issues", "issued",
        "declares", "declared", "officially", "finally", "successfully",
        "will", "going to", "set to", "expected to", "is going",
    ]
    
    # Words that indicate negative outcomes
    NEGATIVE_INDICATORS = [
        "fails", "failed", "loses", "lost", "rejects", "rejected",
        "denies", "denied", "refuses", "refused", "blocks", "blocked",
        "cancels", "cancelled", "canceled", "stops", "stopped", "ends", "ended",
        "withdraws", "withdrew", "drops", "dropped", "abandons", "abandoned",
        "postpones", "postponed", "delays", "delayed", "suspends", "suspended",
        "defeats", "defeated",  # Note: context matters - who defeated whom
        "vetoes", "vetoed", "kills", "killed", "dies", "dead",
        "won't", "will not", "not going to", "unlikely", "rules out",
        "no longer", "not expected", "fails to", "unable to",
    ]
    
    # Stop words to ignore when extracting keywords
    STOP_WORDS = {
        "will", "the", "a", "an", "in", "on", "at", "to", "for", "of", "by",
        "with", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "this", "that", "these", "those",
        "before", "after", "during", "between", "above", "below", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "2024", "2025", "2026", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december",
    }
    
    def __init__(self):
        self.news_client = get_news_client()
        self._confirmation_cache: dict[str, OutcomeConfirmation] = {}
        self._stats = {
            "markets_analyzed": 0,
            "confirmations_found": 0,
            "headlines_checked": 0,
        }
    
    def extract_keywords(self, market_title: str) -> list[str]:
        """Extract searchable keywords from a market question.
        
        Args:
            market_title: The market question (e.g., "Will Biden pardon Hunter?")
            
        Returns:
            List of keywords for news search.
        """
        # Remove common question words
        title = market_title.lower()
        title = re.sub(r'^(will|does|is|are|has|have|can|could|would|should)\s+', '', title)
        title = re.sub(r'\?$', '', title)
        
        # Extract words
        words = re.findall(r'[a-zA-Z]+', title)
        
        # Filter stop words and short words
        keywords = [
            w for w in words 
            if w.lower() not in self.STOP_WORDS and len(w) > 2
        ]
        
        # Return unique keywords, prioritizing proper nouns (capitalized in original)
        original_words = re.findall(r'[A-Z][a-z]+', market_title)
        proper_nouns = [w.lower() for w in original_words]
        
        # Sort: proper nouns first, then by length (longer = more specific)
        keywords.sort(key=lambda w: (w not in proper_nouns, -len(w)))
        
        return keywords[:5]  # Top 5 keywords
    
    def analyze_headline(
        self,
        headline: str,
        market_title: str,
        keywords: list[str],
    ) -> Tuple[Optional[str], float, str]:
        """Analyze if a headline confirms YES or NO for a market.
        
        Args:
            headline: The news headline text.
            market_title: The market question.
            keywords: Keywords extracted from market title.
            
        Returns:
            Tuple of (outcome, confidence, reasoning) or (None, 0, "") if no match.
        """
        headline_lower = headline.lower()
        market_lower = market_title.lower()
        
        # Check if headline contains relevant keywords
        keyword_matches = sum(1 for kw in keywords if kw in headline_lower)
        if keyword_matches < 2:
            return None, 0, ""
        
        # Count positive and negative indicators
        positive_count = sum(1 for ind in self.POSITIVE_INDICATORS if ind in headline_lower)
        negative_count = sum(1 for ind in self.NEGATIVE_INDICATORS if ind in headline_lower)
        
        # Determine outcome
        if positive_count == 0 and negative_count == 0:
            return None, 0, "No outcome indicators found"
        
        # Calculate confidence based on indicator strength
        total_indicators = positive_count + negative_count
        
        if positive_count > negative_count:
            outcome = "YES"
            confidence = min(0.95, 0.6 + (positive_count - negative_count) * 0.1)
            reasoning = f"Positive indicators ({positive_count}) > negative ({negative_count})"
        elif negative_count > positive_count:
            outcome = "NO"
            confidence = min(0.95, 0.6 + (negative_count - positive_count) * 0.1)
            reasoning = f"Negative indicators ({negative_count}) > positive ({positive_count})"
        else:
            return None, 0, "Conflicting indicators"
        
        # Boost confidence if headline strongly matches market question
        keyword_ratio = keyword_matches / max(len(keywords), 1)
        if keyword_ratio >= 0.8:
            confidence = min(0.98, confidence + 0.1)
            reasoning += f"; Strong keyword match ({keyword_ratio:.0%})"
        
        return outcome, confidence, reasoning
    
    def check_market_outcome(
        self,
        market_id: str,
        market_title: str,
        max_articles: int = 10,
    ) -> Optional[OutcomeConfirmation]:
        """Check news to see if a market's outcome can be confirmed.
        
        Args:
            market_id: The market/condition ID.
            market_title: The market question.
            max_articles: Maximum articles to check.
            
        Returns:
            OutcomeConfirmation if found, None otherwise.
        """
        # Check cache first
        if market_id in self._confirmation_cache:
            cached = self._confirmation_cache[market_id]
            # Cache for 30 minutes
            if (datetime.utcnow() - cached.published_at).total_seconds() < 1800:
                return cached
        
        self._stats["markets_analyzed"] += 1
        
        if not self.news_client.is_available():
            logger.debug("News API not available for outcome confirmation")
            return None
        
        # Extract keywords from market question
        keywords = self.extract_keywords(market_title)
        if len(keywords) < 2:
            logger.debug(f"Not enough keywords from: {market_title}")
            return None
        
        # Search for relevant news
        search_query = " ".join(keywords[:3])  # Use top 3 keywords
        articles = self.news_client.search_headlines(search_query, page_size=max_articles)
        
        self._stats["headlines_checked"] += len(articles)
        
        if not articles:
            return None
        
        # Analyze each article for outcome confirmation
        best_confirmation = None
        best_confidence = 0.0
        
        for article in articles:
            # Skip old articles (more than 7 days)
            article_age = datetime.utcnow() - article.published_at.replace(tzinfo=None)
            if article_age > timedelta(days=7):
                continue
            
            # Check title and description
            for text in [article.title, article.description]:
                if not text:
                    continue
                
                outcome, confidence, reasoning = self.analyze_headline(
                    text, market_title, keywords
                )
                
                if outcome and confidence > best_confidence:
                    # Boost confidence for recent articles
                    if article_age < timedelta(hours=24):
                        confidence = min(0.99, confidence + 0.05)
                        reasoning += "; Recent article (< 24h)"
                    
                    best_confirmation = OutcomeConfirmation(
                        market_id=market_id,
                        market_title=market_title,
                        confirmed_outcome=outcome,
                        confidence=confidence,
                        headline=article.title,
                        source=article.source,
                        published_at=article.published_at,
                        reasoning=reasoning,
                    )
                    best_confidence = confidence
        
        if best_confirmation and best_confidence >= 0.7:
            self._stats["confirmations_found"] += 1
            self._confirmation_cache[market_id] = best_confirmation
            
            logger.info(
                f"News confirms {best_confirmation.confirmed_outcome} for '{market_title[:50]}' "
                f"(conf={best_confidence:.0%}, source={best_confirmation.source})"
            )
            return best_confirmation
        
        return None
    
    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        return {
            **self._stats,
            "cache_size": len(self._confirmation_cache),
            "news_api_available": self.news_client.is_available(),
            "api_requests": self.news_client.get_request_count(),
        }
    
    def clear_cache(self):
        """Clear the confirmation cache."""
        self._confirmation_cache.clear()


# Singleton
_analyzer_instance: Optional[HeadlineAnalyzer] = None


def get_headline_analyzer() -> HeadlineAnalyzer:
    """Get or create the headline analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HeadlineAnalyzer()
    return _analyzer_instance
