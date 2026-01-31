"""News API client for fetching headlines.

Integrates with NewsAPI.org to fetch current headlines
for correlation with Polymarket events.

Plan: 1000 requests/day, 1 month old articles max.
"""

import json
import logging
import os
import re
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """A news article from the API."""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    content: str = ""
    
    def get_full_text(self) -> str:
        """Get all text for analysis."""
        return f"{self.title} {self.description} {self.content}"


class NewsClient:
    """Client for fetching news from NewsAPI.org.
    
    Features:
    - 1000 requests/day limit tracking
    - Aggressive caching (30 min) to maximize efficiency
    - Daily usage reset at midnight UTC
    """
    
    # Configuration
    API_KEY_ENV = "NEWSAPI_KEY"
    BASE_URL = "https://newsapi.org/v2"
    DAILY_LIMIT = 1000  # Requests per day
    DAILY_BUDGET = 500  # Conservative budget to leave headroom for retries

    # Cache settings - longer cache = fewer API calls
    _cache: dict = {}
    _cache_duration = timedelta(minutes=60)  # 60 min cache (was 30)
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the news client.

        Args:
            api_key: NewsAPI.org API key. If not provided, reads from settings or env var.
        """
        self.api_key = api_key
        self._last_request_time = None
        self._request_count = 0
        self._daily_count = 0
        self._daily_reset_date = datetime.now(timezone.utc).date()
        self._cache_hits = 0

        # Rate limit tracking
        self._is_rate_limited = False
        self._rate_limit_hit_time: Optional[datetime] = None
        self._consecutive_429s = 0
        
        # Try to get from settings first, then env var
        if not self.api_key:
            try:
                from polyb0t.config.settings import get_settings
                settings = get_settings()
                self.api_key = settings.newsapi_key
            except:
                pass
        
        if not self.api_key:
            self.api_key = os.environ.get("NEWSAPI_KEY", "")
        
        if not self.api_key:
            logger.warning(
                "No NewsAPI key found. Set POLYBOT_NEWSAPI_KEY in .env. "
                "Get free key at https://newsapi.org/register"
            )
        else:
            logger.info(f"NewsClient initialized (limit: {self.DAILY_LIMIT}/day)")
    
    def _check_daily_reset(self):
        """Reset daily counter at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today > self._daily_reset_date:
            old_count = self._daily_count
            was_limited = self._is_rate_limited
            self._daily_count = 0
            self._daily_reset_date = today
            self._is_rate_limited = False
            self._rate_limit_hit_time = None
            self._consecutive_429s = 0
            logger.info(
                f"NewsAPI daily quota reset. Yesterday: {old_count} requests, "
                f"was_rate_limited: {was_limited}"
            )

    def get_time_until_reset(self) -> Tuple[int, int, int]:
        """Get time until NewsAPI quota resets (midnight UTC).

        Returns:
            Tuple of (hours, minutes, seconds) until reset
        """
        now = datetime.now(timezone.utc)
        # Next midnight UTC
        tomorrow = now.date() + timedelta(days=1)
        reset_time = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)
        delta = reset_time - now
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return (hours, minutes, seconds)

    def get_reset_time_str(self) -> str:
        """Get human-readable time until reset."""
        hours, minutes, seconds = self.get_time_until_reset()
        return f"{hours}h {minutes}m until reset (midnight UTC)"

    def is_rate_limited(self) -> bool:
        """Check if we're currently rate-limited by NewsAPI."""
        self._check_daily_reset()  # May clear the flag
        return self._is_rate_limited
    
    def _can_make_request(self) -> bool:
        """Check if we can make another request today."""
        self._check_daily_reset()
        return self._daily_count < self.DAILY_LIMIT
    
    def is_available(self) -> bool:
        """Check if news API is available."""
        return bool(self.api_key)
    
    def search_headlines(
        self,
        query: str,
        language: str = "en",
        page_size: int = 10,
    ) -> list[NewsArticle]:
        """Search for headlines matching a query.
        
        Args:
            query: Search query (keywords from market question).
            language: Language code.
            page_size: Number of results.
            
        Returns:
            List of matching articles.
        """
        if not self.api_key:
            return []
        
        # Check cache first
        cache_key = hashlib.md5(f"{query}:{language}:{page_size}".encode()).hexdigest()
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < self._cache_duration:
                self._cache_hits += 1
                return cached_data
        
        # Check daily limit
        if not self._can_make_request():
            logger.warning(f"NewsAPI daily limit reached ({self.DAILY_LIMIT})")
            return []
        
        try:
            # Use "everything" endpoint for broader search
            url = f"{self.BASE_URL}/everything"
            params = {
                "q": query,
                "language": language,
                "pageSize": page_size,
                "sortBy": "publishedAt",
                "apiKey": self.api_key,
            }
            
            response = requests.get(url, params=params, timeout=10)
            self._request_count += 1
            self._daily_count += 1
            
            if response.status_code == 429:
                self._consecutive_429s += 1
                if not self._is_rate_limited:
                    self._is_rate_limited = True
                    self._rate_limit_hit_time = datetime.now(timezone.utc)
                    hours, minutes, _ = self.get_time_until_reset()
                    logger.warning(
                        f"NewsAPI rate limit reached (429). "
                        f"Quota resets in {hours}h {minutes}m (midnight UTC). "
                        f"Daily requests: {self._daily_count}"
                    )
                elif self._consecutive_429s % 50 == 0:  # Log occasionally
                    hours, minutes, _ = self.get_time_until_reset()
                    logger.info(f"NewsAPI still rate-limited. Resets in {hours}h {minutes}m")
                return []
            
            if response.status_code != 200:
                logger.warning(f"NewsAPI error: {response.status_code}")
                return []
            
            data = response.json()
            articles = []
            
            for item in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        item.get("publishedAt", "").replace("Z", "+00:00")
                    )
                except:
                    published = datetime.now(timezone.utc)
                
                article = NewsArticle(
                    title=item.get("title", "") or "",
                    description=item.get("description", "") or "",
                    source=item.get("source", {}).get("name", "Unknown"),
                    url=item.get("url", ""),
                    published_at=published,
                    content=item.get("content", "") or "",
                )
                articles.append(article)
            
            # Cache results
            self._cache[cache_key] = (datetime.now(timezone.utc), articles)
            
            logger.debug(f"NewsAPI: Found {len(articles)} articles for '{query}'")
            return articles
            
        except requests.RequestException as e:
            logger.warning(f"NewsAPI request failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            return []
    
    def get_top_headlines(
        self,
        category: str = "general",
        country: str = "us",
        page_size: int = 20,
    ) -> list[NewsArticle]:
        """Get top headlines.
        
        Args:
            category: business, entertainment, general, health, science, sports, technology
            country: Country code (us, gb, etc.)
            page_size: Number of results.
            
        Returns:
            List of top headlines.
        """
        if not self.api_key:
            return []
        
        cache_key = f"top:{category}:{country}:{page_size}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < self._cache_duration:
                return cached_data
        
        try:
            url = f"{self.BASE_URL}/top-headlines"
            params = {
                "category": category,
                "country": country,
                "pageSize": page_size,
                "apiKey": self.api_key,
            }
            
            response = requests.get(url, params=params, timeout=10)
            self._request_count += 1
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            articles = []
            
            for item in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        item.get("publishedAt", "").replace("Z", "+00:00")
                    )
                except:
                    published = datetime.now(timezone.utc)
                
                article = NewsArticle(
                    title=item.get("title", "") or "",
                    description=item.get("description", "") or "",
                    source=item.get("source", {}).get("name", "Unknown"),
                    url=item.get("url", ""),
                    published_at=published,
                    content=item.get("content", "") or "",
                )
                articles.append(article)
            
            self._cache[cache_key] = (datetime.now(timezone.utc), articles)
            return articles
            
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            return []
    
    def get_request_count(self) -> int:
        """Get number of API requests made."""
        return self._request_count
    
    def get_stats(self) -> dict:
        """Get detailed usage statistics."""
        self._check_daily_reset()
        remaining = self.DAILY_LIMIT - self._daily_count

        cache_rate = (
            self._cache_hits / (self._cache_hits + self._request_count) * 100
            if (self._cache_hits + self._request_count) > 0 else 0
        )

        hours, minutes, _ = self.get_time_until_reset()

        stats = {
            "is_available": self.is_available(),
            "daily_limit": self.DAILY_LIMIT,
            "daily_budget": self.DAILY_BUDGET,
            "daily_used": self._daily_count,
            "daily_remaining": remaining,
            "is_rate_limited": self._is_rate_limited,
            "hours_until_reset": hours,
            "minutes_until_reset": minutes,
            "reset_time_str": f"{hours}h {minutes}m until midnight UTC",
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "cache_size": len(self._cache),
            "consecutive_429s": self._consecutive_429s,
        }

        if self._rate_limit_hit_time:
            stats["rate_limit_hit_at"] = self._rate_limit_hit_time.isoformat()

        return stats


# Singleton
_client_instance: Optional[NewsClient] = None


def get_news_client() -> NewsClient:
    """Get or create the news client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = NewsClient()
    return _client_instance
