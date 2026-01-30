"""Sentiment feature extraction for ML training.

This module connects the existing NewsClient and HeadlineAnalyzer services
to the ML training pipeline, extracting sentiment-based features from
news articles relevant to each market.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SentimentFeatures:
    """Container for sentiment-based ML features."""
    news_article_count: int = 0
    news_recency_hours: float = 999.0
    news_sentiment_score: float = 0.0
    news_sentiment_confidence: float = 0.0
    keyword_positive_count: int = 0
    keyword_negative_count: int = 0
    headline_confirmation: float = 0.0
    headline_conf_confidence: float = 0.0
    intelligent_confirmation: float = 0.0
    intelligent_conf_confidence: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML features."""
        return {
            "news_article_count": float(self.news_article_count),
            "news_recency_hours": self.news_recency_hours,
            "news_sentiment_score": self.news_sentiment_score,
            "news_sentiment_confidence": self.news_sentiment_confidence,
            "keyword_positive_count": float(self.keyword_positive_count),
            "keyword_negative_count": float(self.keyword_negative_count),
            "headline_confirmation": self.headline_confirmation,
            "headline_conf_confidence": self.headline_conf_confidence,
            "intelligent_confirmation": self.intelligent_confirmation,
            "intelligent_conf_confidence": self.intelligent_conf_confidence,
        }


class SentimentFeatureEngine:
    """Extracts ML features from news and sentiment analysis.

    Connects to existing services:
    - NewsClient: Fetches headlines from NewsAPI
    - HeadlineAnalyzer: Keyword-based sentiment analysis
    - IntelligentAnalyzer: GPT-based deep analysis (optional, costly)

    Features are cached per market to minimize API calls.
    """

    def __init__(self, cache_ttl_minutes: int = 30):
        """Initialize the sentiment feature engine.

        Args:
            cache_ttl_minutes: Cache time-to-live for sentiment features
        """
        self._cache: Dict[str, Tuple[datetime, SentimentFeatures]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Lazy-load services
        self._news_client = None
        self._headline_analyzer = None
        self._intelligent_analyzer = None

        # Track API usage
        self._api_calls = 0
        self._cache_hits = 0

    @property
    def news_client(self):
        """Lazy-load news client."""
        if self._news_client is None:
            try:
                from polyb0t.services.news_client import get_news_client
                self._news_client = get_news_client()
            except Exception as e:
                logger.debug(f"NewsClient not available: {e}")
        return self._news_client

    @property
    def headline_analyzer(self):
        """Lazy-load headline analyzer."""
        if self._headline_analyzer is None:
            try:
                from polyb0t.services.headline_analyzer import get_headline_analyzer
                self._headline_analyzer = get_headline_analyzer()
            except Exception as e:
                logger.debug(f"HeadlineAnalyzer not available: {e}")
        return self._headline_analyzer

    @property
    def intelligent_analyzer(self):
        """Lazy-load intelligent analyzer."""
        if self._intelligent_analyzer is None:
            try:
                from polyb0t.services.intelligent_analyzer import get_intelligent_analyzer
                self._intelligent_analyzer = get_intelligent_analyzer()
            except Exception as e:
                logger.debug(f"IntelligentAnalyzer not available: {e}")
        return self._intelligent_analyzer

    def is_available(self) -> bool:
        """Check if sentiment analysis is available."""
        return self.news_client is not None and self.news_client.api_key

    def get_sentiment_features(
        self,
        market_id: str,
        market_title: str,
        category: str = "",
        use_intelligent: bool = False
    ) -> SentimentFeatures:
        """Get sentiment features for a market.

        Args:
            market_id: Unique market identifier
            market_title: Market title/question for keyword extraction
            category: Market category for filtering
            use_intelligent: Whether to use GPT-based analysis (costs money)

        Returns:
            SentimentFeatures with all computed values
        """
        # Check cache first
        if market_id in self._cache:
            cached_time, cached_features = self._cache[market_id]
            if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                self._cache_hits += 1
                return cached_features

        # Initialize with defaults
        features = SentimentFeatures()

        if not self.is_available():
            return features

        try:
            features = self._compute_features(
                market_id, market_title, category, use_intelligent
            )
            self._api_calls += 1
        except Exception as e:
            logger.warning(f"Error computing sentiment features for {market_id}: {e}")

        # Cache the result
        self._cache[market_id] = (datetime.now(timezone.utc), features)

        return features

    def _compute_features(
        self,
        market_id: str,
        market_title: str,
        category: str,
        use_intelligent: bool
    ) -> SentimentFeatures:
        """Internal method to compute sentiment features."""
        features = SentimentFeatures()

        # Extract keywords from market title
        if not self.headline_analyzer:
            return features

        keywords = self.headline_analyzer.extract_keywords(market_title)
        if not keywords:
            return features

        # Build search query from top keywords
        search_query = " ".join(keywords[:3])

        # Fetch relevant news articles
        if not self.news_client:
            return features

        articles = self.news_client.search_headlines(search_query, page_size=10)

        if not articles:
            return features

        features.news_article_count = len(articles)

        # Compute news recency
        article_times = []
        for article in articles:
            if hasattr(article, 'published_at') and article.published_at:
                pub_time = article.published_at
                # Normalize to timezone-aware UTC
                if pub_time.tzinfo is None:
                    pub_time = pub_time.replace(tzinfo=timezone.utc)
                article_times.append(pub_time)

        if article_times:
            most_recent = max(article_times)
            now_utc = datetime.now(timezone.utc)
            hours_ago = (now_utc - most_recent).total_seconds() / 3600
            features.news_recency_hours = max(0.0, min(999.0, hours_ago))

        # Analyze headlines for sentiment
        sentiment_sum = 0.0
        confidence_sum = 0.0
        positive_count = 0
        negative_count = 0

        for article in articles[:5]:  # Top 5 articles
            try:
                result = self.headline_analyzer.analyze_headline(
                    headline=article.title,
                    market_title=market_title,
                    keywords=keywords
                )

                if result:
                    outcome, confidence, _ = result
                    if outcome == "YES":
                        positive_count += 1
                        sentiment_sum += confidence * 0.2
                        confidence_sum += confidence
                    elif outcome == "NO":
                        negative_count += 1
                        sentiment_sum -= confidence * 0.2
                        confidence_sum += confidence
            except Exception as e:
                logger.debug(f"Error analyzing headline: {e}")

        features.keyword_positive_count = positive_count
        features.keyword_negative_count = negative_count
        features.news_sentiment_score = max(-1.0, min(1.0, sentiment_sum))

        if positive_count + negative_count > 0:
            features.news_sentiment_confidence = confidence_sum / (positive_count + negative_count)

        # Check headline-based confirmation
        try:
            confirmation = self.headline_analyzer.check_market_outcome(market_id, market_title)
            if confirmation:
                if confirmation.confirmed_outcome == "YES":
                    features.headline_confirmation = 1.0
                elif confirmation.confirmed_outcome == "NO":
                    features.headline_confirmation = -1.0
                features.headline_conf_confidence = confirmation.confidence
        except Exception as e:
            logger.debug(f"Error checking headline confirmation: {e}")

        # Use intelligent analyzer if requested and available
        if use_intelligent and self.intelligent_analyzer and articles:
            try:
                if self.intelligent_analyzer.is_available():
                    top_article = articles[0]
                    result = self.intelligent_analyzer.analyze_headline(
                        market_question=market_title,
                        headline=top_article.title,
                        article_content=getattr(top_article, 'description', '') or '',
                        source=getattr(top_article, 'source', '') or ''
                    )
                    if result and result.confirmed_outcome:
                        if result.confirmed_outcome == "YES":
                            features.intelligent_confirmation = 1.0
                        elif result.confirmed_outcome == "NO":
                            features.intelligent_confirmation = -1.0
                        features.intelligent_conf_confidence = result.confidence
            except Exception as e:
                logger.debug(f"Error in intelligent analysis: {e}")

        return features

    def get_features_dict(
        self,
        market_id: str,
        market_title: str,
        category: str = "",
        use_intelligent: bool = False
    ) -> Dict[str, float]:
        """Get sentiment features as a dictionary for ML training.

        This is the main entry point for feature extraction.

        Args:
            market_id: Unique market identifier
            market_title: Market title/question
            category: Market category
            use_intelligent: Whether to use GPT analysis

        Returns:
            Dictionary of feature name -> value
        """
        features = self.get_sentiment_features(
            market_id, market_title, category, use_intelligent
        )
        return features.to_dict()

    def clear_cache(self, market_id: Optional[str] = None):
        """Clear the feature cache.

        Args:
            market_id: Specific market to clear, or None for all
        """
        if market_id:
            self._cache.pop(market_id, None)
        else:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "api_calls": self._api_calls,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._cache_hits / max(1, self._api_calls + self._cache_hits),
            "news_client_available": self.news_client is not None,
            "headline_analyzer_available": self.headline_analyzer is not None,
            "intelligent_analyzer_available": self.intelligent_analyzer is not None,
        }


# Singleton instance
_sentiment_engine: Optional[SentimentFeatureEngine] = None


def get_sentiment_feature_engine() -> SentimentFeatureEngine:
    """Get the singleton SentimentFeatureEngine instance."""
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = SentimentFeatureEngine()
    return _sentiment_engine


def get_sentiment_features_for_market(
    market_id: str,
    market_title: str,
    category: str = ""
) -> Dict[str, float]:
    """Convenience function to get sentiment features.

    Args:
        market_id: Market identifier
        market_title: Market title/question
        category: Market category

    Returns:
        Dictionary of sentiment features
    """
    engine = get_sentiment_feature_engine()
    return engine.get_features_dict(market_id, market_title, category)
