"""WebSocket client for real-time Polymarket CLOB data.

Replaces HTTP polling with event-driven WebSocket streaming for:
- Real-time orderbook updates
- Trade notifications
- Price changes
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = None

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# WebSocket endpoints
WS_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_ENDPOINT_USER = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

# Connection constants
MAX_RECONNECT_ATTEMPTS = 10
INITIAL_RECONNECT_DELAY = 1.0  # seconds
MAX_RECONNECT_DELAY = 60.0  # seconds
HEALTH_CHECK_INTERVAL = 30.0  # seconds
MAX_SUBSCRIPTIONS_PER_CONNECTION = 100  # Polymarket limit


@dataclass
class OrderBookLevel:
    """Single orderbook level."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Full orderbook for an asset."""
    asset_id: str
    market: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.utcnow)
    hash: str = ""
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if self.bids:
            return max(b.price for b in self.bids)
        return None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if self.asks:
            return min(a.price for a in self.asks)
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Trade:
    """Trade event from WebSocket."""
    asset_id: str
    market: str
    price: float
    size: float
    side: str  # BUY or SELL
    timestamp: datetime
    fee_rate_bps: int = 0
    
    @property
    def value_usd(self) -> float:
        """Trade value in USD."""
        return self.price * self.size


class PolymarketWebSocket:
    """WebSocket client for Polymarket CLOB real-time data.
    
    Features:
    - Automatic reconnection with exponential backoff
    - Subscription management for multiple markets
    - Local orderbook state maintenance
    - Trade event streaming
    - Health check heartbeats
    """
    
    def __init__(self):
        """Initialize WebSocket client."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets package not installed. WebSocket features disabled.")
            
        self.settings = get_settings()
        
        # Connection state
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._last_message_time: Optional[datetime] = None
        
        # Subscription management
        self._subscribed_assets: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        
        # Local state
        self._orderbooks: Dict[str, OrderBook] = {}
        self._recent_trades: List[Trade] = []
        self._max_recent_trades = 1000
        
        # Event callbacks
        self._on_trade_callbacks: List[Callable[[Trade], None]] = []
        self._on_book_update_callbacks: List[Callable[[OrderBook], None]] = []
        self._on_connect_callbacks: List[Callable[[], None]] = []
        self._on_disconnect_callbacks: List[Callable[[], None]] = []
        
        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Lock to prevent multiple concurrent recv() calls
        self._recv_lock = asyncio.Lock()
        
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._ws is not None
    
    @property
    def subscribed_count(self) -> int:
        """Get number of subscribed assets."""
        return len(self._subscribed_assets)
    
    async def connect(self) -> bool:
        """Connect to WebSocket server.
        
        Returns:
            True if connection successful.
        """
        if not HAS_WEBSOCKETS:
            logger.error("Cannot connect: websockets package not installed")
            return False
            
        if self._connected:
            return True
        
        # Cancel any existing background tasks before creating new ones
        # This prevents multiple receive loops during reconnection
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            
        try:
            logger.info(f"Connecting to Polymarket WebSocket: {WS_ENDPOINT}")
            
            self._ws = await websockets.connect(
                WS_ENDPOINT,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            
            self._connected = True
            self._reconnect_attempts = 0
            self._last_message_time = datetime.utcnow()
            
            # Start background tasks (only one of each)
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("WebSocket connected successfully")
            
            # Notify callbacks
            for callback in self._on_connect_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Connect callback error: {e}")
            
            # Resubscribe to previously subscribed assets
            if self._subscribed_assets:
                await self.subscribe(list(self._subscribed_assets))
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        self._connected = False
        logger.info("WebSocket disconnected")
        
        # Notify callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")
    
    async def subscribe(self, asset_ids: List[str]) -> bool:
        """Subscribe to market data for assets.
        
        Args:
            asset_ids: List of token IDs to subscribe to.
            
        Returns:
            True if subscription sent successfully.
        """
        if not self._connected or not self._ws:
            # Queue for later
            self._pending_subscriptions.update(asset_ids)
            return False
        
        # Limit subscriptions
        new_assets = [a for a in asset_ids if a not in self._subscribed_assets]
        if not new_assets:
            return True
        
        # Batch if too many
        for i in range(0, len(new_assets), MAX_SUBSCRIPTIONS_PER_CONNECTION):
            batch = new_assets[i:i + MAX_SUBSCRIPTIONS_PER_CONNECTION]
            
            try:
                message = {
                    "type": "MARKET",
                    "assets_ids": batch,
                    "custom_feature_enabled": True,  # Enable best_bid_ask etc
                }
                
                await self._ws.send(json.dumps(message))
                self._subscribed_assets.update(batch)
                logger.debug(f"Subscribed to {len(batch)} assets")
                
            except Exception as e:
                logger.error(f"Subscribe error: {e}")
                return False
        
        return True
    
    async def unsubscribe(self, asset_ids: List[str]) -> bool:
        """Unsubscribe from market data.
        
        Args:
            asset_ids: List of token IDs to unsubscribe from.
            
        Returns:
            True if unsubscription sent successfully.
        """
        if not self._connected or not self._ws:
            self._subscribed_assets.difference_update(asset_ids)
            return False
        
        try:
            message = {
                "assets_ids": asset_ids,
                "operation": "unsubscribe",
            }
            
            await self._ws.send(json.dumps(message))
            self._subscribed_assets.difference_update(asset_ids)
            
            # Clean up orderbooks
            for asset_id in asset_ids:
                self._orderbooks.pop(asset_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False
    
    def get_orderbook(self, asset_id: str) -> Optional[OrderBook]:
        """Get current orderbook for asset.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            OrderBook or None if not subscribed.
        """
        return self._orderbooks.get(asset_id)
    
    def get_recent_trades(
        self,
        asset_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Trade]:
        """Get recent trades.
        
        Args:
            asset_id: Optional filter by asset.
            limit: Maximum trades to return.
            
        Returns:
            List of recent trades.
        """
        trades = self._recent_trades
        
        if asset_id:
            trades = [t for t in trades if t.asset_id == asset_id]
        
        return trades[-limit:]
    
    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register callback for trade events."""
        self._on_trade_callbacks.append(callback)
    
    def on_book_update(self, callback: Callable[[OrderBook], None]) -> None:
        """Register callback for orderbook updates."""
        self._on_book_update_callbacks.append(callback)
    
    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register callback for connection events."""
        self._on_connect_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register callback for disconnection events."""
        self._on_disconnect_callbacks.append(callback)
    
    async def _receive_loop(self) -> None:
        """Background task to receive messages.
        
        Only one receive loop should run at a time.
        """
        while self._running and self._ws:
            try:
                # Use lock to prevent concurrent recv() calls
                async with self._recv_lock:
                    if not self._ws or not self._connected:
                        break
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=HEALTH_CHECK_INTERVAL + 10,
                    )
                
                self._last_message_time = datetime.utcnow()
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self._connected = False
                # Don't reconnect from here - let health check handle it
                break
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                # Only log once, don't spam
                if "another coroutine" not in str(e):
                    logger.error(f"WebSocket receive error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, raw_message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            
            # Handle array of messages (Polymarket sometimes sends batches)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        await self._process_single_message(item)
            elif isinstance(data, dict):
                await self._process_single_message(data)
            else:
                logger.debug(f"Unexpected message type: {type(data)}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _process_single_message(self, data: Dict[str, Any]) -> None:
        """Process a single message dict."""
        event_type = data.get("event_type", "")
        
        if event_type == "book":
            await self._handle_book_message(data)
            
        elif event_type == "price_change":
            await self._handle_price_change(data)
            
        elif event_type == "last_trade_price":
            await self._handle_trade(data)
            
        elif event_type == "best_bid_ask":
            await self._handle_best_bid_ask(data)
            
        elif event_type == "tick_size_change":
            logger.debug(f"Tick size change: {data.get('asset_id')}")
            
        elif event_type == "new_market":
            logger.info(f"New market: {data.get('question')}")
            
        elif event_type == "market_resolved":
            logger.info(f"Market resolved: {data.get('question')} -> {data.get('winning_outcome')}")
            
        elif event_type:
            logger.debug(f"Unknown event type: {event_type}")
    
    async def _handle_book_message(self, data: Dict[str, Any]) -> None:
        """Handle full orderbook snapshot."""
        asset_id = data.get("asset_id", "")
        market = data.get("market", "")
        
        bids = []
        for level in data.get("bids", data.get("buys", [])):
            try:
                bids.append(OrderBookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                ))
            except (ValueError, TypeError):
                pass
        
        asks = []
        for level in data.get("asks", data.get("sells", [])):
            try:
                asks.append(OrderBookLevel(
                    price=float(level.get("price", 0)),
                    size=float(level.get("size", 0)),
                ))
            except (ValueError, TypeError):
                pass
        
        orderbook = OrderBook(
            asset_id=asset_id,
            market=market,
            bids=bids,
            asks=asks,
            hash=data.get("hash", ""),
            last_update=datetime.utcnow(),
        )
        
        self._orderbooks[asset_id] = orderbook
        
        # Notify callbacks
        for callback in self._on_book_update_callbacks:
            try:
                callback(orderbook)
            except Exception as e:
                logger.error(f"Book update callback error: {e}")
    
    async def _handle_price_change(self, data: Dict[str, Any]) -> None:
        """Handle orderbook price level change."""
        market = data.get("market", "")
        
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id", "")
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            side = change.get("side", "")
            
            orderbook = self._orderbooks.get(asset_id)
            if not orderbook:
                orderbook = OrderBook(asset_id=asset_id, market=market)
                self._orderbooks[asset_id] = orderbook
            
            # Update the orderbook
            if side == "BUY":
                # Update or add bid level
                orderbook.bids = [b for b in orderbook.bids if b.price != price]
                if size > 0:
                    orderbook.bids.append(OrderBookLevel(price=price, size=size))
            else:
                # Update or add ask level
                orderbook.asks = [a for a in orderbook.asks if a.price != price]
                if size > 0:
                    orderbook.asks.append(OrderBookLevel(price=price, size=size))
            
            orderbook.last_update = datetime.utcnow()
            
            # Notify callbacks
            for callback in self._on_book_update_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    logger.error(f"Book update callback error: {e}")
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade event."""
        try:
            trade = Trade(
                asset_id=data.get("asset_id", ""),
                market=data.get("market", ""),
                price=float(data.get("price", 0)),
                size=float(data.get("size", 0)),
                side=data.get("side", ""),
                timestamp=datetime.utcnow(),
                fee_rate_bps=int(data.get("fee_rate_bps", 0)),
            )
            
            # Store trade
            self._recent_trades.append(trade)
            if len(self._recent_trades) > self._max_recent_trades:
                self._recent_trades = self._recent_trades[-self._max_recent_trades:]
            
            # Notify callbacks
            for callback in self._on_trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid trade data: {e}")
    
    async def _handle_best_bid_ask(self, data: Dict[str, Any]) -> None:
        """Handle best bid/ask update."""
        asset_id = data.get("asset_id", "")
        market = data.get("market", "")
        
        orderbook = self._orderbooks.get(asset_id)
        if not orderbook:
            orderbook = OrderBook(asset_id=asset_id, market=market)
            self._orderbooks[asset_id] = orderbook
        
        # Update with best bid/ask (simplified orderbook)
        try:
            best_bid = float(data.get("best_bid", 0))
            best_ask = float(data.get("best_ask", 0))
            
            if best_bid > 0:
                orderbook.bids = [OrderBookLevel(price=best_bid, size=0)]
            if best_ask > 0:
                orderbook.asks = [OrderBookLevel(price=best_ask, size=0)]
            
            orderbook.last_update = datetime.utcnow()
            
        except (ValueError, TypeError):
            pass
    
    async def _health_check_loop(self) -> None:
        """Background task for connection health checks."""
        while self._running:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
                # Check if receive task died and we need to reconnect
                if self._receive_task and self._receive_task.done() and self._running:
                    logger.warning("Receive task died, reconnecting...")
                    await self._reconnect()
                    continue
                
                if not self._connected:
                    # Try to reconnect if not connected
                    if self._running:
                        await self._reconnect()
                    continue
                
                # Check last message time
                if self._last_message_time:
                    seconds_since = (datetime.utcnow() - self._last_message_time).total_seconds()
                    if seconds_since > HEALTH_CHECK_INTERVAL * 2:
                        logger.warning(f"No messages for {seconds_since:.0f}s, reconnecting...")
                        await self._reconnect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self._connected = False
        
        # Notify callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")
        
        while self._running and self._reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            
            # Exponential backoff
            delay = min(
                INITIAL_RECONNECT_DELAY * (2 ** (self._reconnect_attempts - 1)),
                MAX_RECONNECT_DELAY,
            )
            
            logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})")
            await asyncio.sleep(delay)
            
            if await self.connect():
                return
        
        logger.error("Max reconnection attempts reached")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket client statistics."""
        return {
            "connected": self._connected,
            "subscribed_assets": len(self._subscribed_assets),
            "orderbooks_cached": len(self._orderbooks),
            "recent_trades": len(self._recent_trades),
            "reconnect_attempts": self._reconnect_attempts,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
        }


# Singleton instance
_ws_client: Optional[PolymarketWebSocket] = None


def get_ws_client() -> PolymarketWebSocket:
    """Get or create the WebSocket client singleton."""
    global _ws_client
    if _ws_client is None:
        _ws_client = PolymarketWebSocket()
    return _ws_client
