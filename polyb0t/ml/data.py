"""Data collection and management for online learning.

Collects features at time T, labels them with outcomes at T+horizon,
and provides training datasets for model updates.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single labeled training example."""
    
    timestamp: datetime
    market_id: str
    token_id: str
    features: Dict[str, float]
    target: Optional[float]  # None until labeled
    cycle_id: str
    was_traded: bool = False
    actual_pnl: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'timestamp': self.timestamp.timestamp(),
            'market_id': self.market_id,
            'token_id': self.token_id,
            'features_json': json.dumps(self.features),
            'target': self.target,
            'cycle_id': self.cycle_id,
            'was_traded': int(self.was_traded),
            'actual_pnl': self.actual_pnl,
        }


class DataCollector:
    """Collects and manages training data for online learning."""
    
    def __init__(self, db_path: str = "data/training_data.db"):
        """Initialize data collector.
        
        Args:
            db_path: Path to SQLite database for storage.
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        
        # In-memory price cache for labeling
        self.price_cache: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Main training data table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                features_json TEXT NOT NULL,
                target REAL,
                cycle_id TEXT,
                was_traded INTEGER DEFAULT 0,
                actual_pnl REAL,
                labeled INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        
        # Index for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_timestamp 
            ON training_data(token_id, timestamp)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_labeled 
            ON training_data(labeled, timestamp)
        """)
        
        # Price history table for labeling
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                token_id TEXT NOT NULL,
                price REAL NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_token_time 
            ON price_history(token_id, timestamp)
        """)
        
        # Model performance tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata_json TEXT
            )
        """)
        
        # Multi-horizon predictions table (1h, 6h, 24h, 7d)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_horizon_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_data_id INTEGER NOT NULL,
                horizon_hours INTEGER NOT NULL,
                target_return REAL,
                target_direction INTEGER,  -- 1=up, 0=flat, -1=down
                target_volatility REAL,
                labeled INTEGER DEFAULT 0,
                FOREIGN KEY (training_data_id) REFERENCES training_data(id)
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_multi_horizon 
            ON multi_horizon_labels(training_data_id, horizon_hours)
        """)
        
        # Advanced features table (stored separately for efficiency)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS advanced_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_data_id INTEGER NOT NULL,
                smart_money_features TEXT,
                technical_features TEXT,
                pattern_features TEXT,
                composite_signals TEXT,
                FOREIGN KEY (training_data_id) REFERENCES training_data(id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Initialize advanced intelligence engine
        self.advanced_intelligence = None
        try:
            from polyb0t.ml.advanced_intelligence import get_advanced_intelligence
            self.advanced_intelligence = get_advanced_intelligence()
            logger.info("Advanced intelligence engine initialized for data collection")
        except Exception as e:
            logger.warning(f"Could not initialize advanced intelligence: {e}")
        
        logger.info(f"Data collector initialized: {self.db_path}")
    
    def collect_cycle_data(
        self,
        features_dict: Dict[str, Dict[str, float]],
        prices: Dict[str, float],
        cycle_id: str,
        market_ids: Optional[Dict[str, str]] = None,
        orderbooks: Optional[Dict[str, any]] = None,
        trades: Optional[Dict[str, List]] = None,
        market_end_dates: Optional[Dict[str, datetime]] = None,
        market_categories: Optional[Dict[str, str]] = None,
        market_volumes: Optional[Dict[str, float]] = None,
    ) -> int:
        """Store features from current cycle for future labeling.
        
        Enhanced to include advanced intelligence features for smarter predictions.
        
        Args:
            features_dict: Dict of token_id -> features.
            prices: Dict of token_id -> current price.
            cycle_id: Current cycle identifier.
            market_ids: Optional dict of token_id -> market_id.
            orderbooks: Optional dict of token_id -> orderbook data.
            trades: Optional dict of token_id -> trade list.
            market_end_dates: Optional dict of token_id -> end date.
            market_categories: Optional dict of token_id -> category.
            market_volumes: Optional dict of token_id -> 24h volume.
            
        Returns:
            Number of examples stored.
        """
        if not features_dict:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        now = datetime.utcnow()
        
        stored_count = 0
        
        for token_id, features in features_dict.items():
            market_id = market_ids.get(token_id, "unknown") if market_ids else "unknown"
            current_price = prices.get(token_id)
            
            if current_price is None:
                continue
            
            # === COMPUTE ADVANCED INTELLIGENCE FEATURES ===
            advanced_features = {}
            if self.advanced_intelligence:
                try:
                    # Get orderbook in dict format
                    ob_data = None
                    if orderbooks and token_id in orderbooks:
                        ob = orderbooks[token_id]
                        if hasattr(ob, 'bids') and hasattr(ob, 'asks'):
                            ob_data = {
                                "bids": [{"price": l.price, "size": l.size} for l in (ob.bids or [])],
                                "asks": [{"price": l.price, "size": l.size} for l in (ob.asks or [])],
                            }
                    
                    # Get trades as dicts
                    trade_data = None
                    if trades and token_id in trades:
                        trade_data = [
                            {
                                "price": t.price if hasattr(t, 'price') else t.get('price', 0),
                                "size": t.size if hasattr(t, 'size') else t.get('size', 0),
                                "side": t.side if hasattr(t, 'side') else t.get('side', ''),
                                "timestamp": t.timestamp if hasattr(t, 'timestamp') else t.get('timestamp'),
                            }
                            for t in trades[token_id]
                        ]
                    
                    advanced_features = self.advanced_intelligence.compute_advanced_features(
                        token_id=token_id,
                        current_price=current_price,
                        orderbook=ob_data,
                        recent_trades=trade_data,
                        market_end_date=market_end_dates.get(token_id) if market_end_dates else None,
                        market_category=market_categories.get(token_id) if market_categories else None,
                        market_volume_24h=market_volumes.get(token_id, 0) if market_volumes else 0,
                    )
                    
                    # Merge advanced features into main features
                    features = {**features, **advanced_features}
                    
                except Exception as e:
                    logger.debug(f"Advanced features failed for {token_id[:20]}: {e}")
            
            # Store features (target=NULL, will label later)
            cursor = conn.execute("""
                INSERT INTO training_data 
                (timestamp, market_id, token_id, features_json, 
                 target, cycle_id, was_traded, labeled)
                VALUES (?, ?, ?, ?, NULL, ?, 0, 0)
            """, (
                now.timestamp(),
                market_id,
                token_id,
                json.dumps(features),
                cycle_id,
            ))
            
            training_data_id = cursor.lastrowid
            
            # === STORE ADVANCED FEATURES SEPARATELY ===
            if advanced_features:
                # Group features by category
                smart_money = {k: v for k, v in advanced_features.items() 
                              if 'smart_money' in k or 'whale' in k or 'accumulation' in k or 'distribution' in k}
                technical = {k: v for k, v in advanced_features.items() 
                            if 'rsi' in k or 'macd' in k or 'bollinger' in k or 'trend' in k or 'sma' in k}
                patterns = {k: v for k, v in advanced_features.items() 
                           if 'pattern' in k or 'divergence' in k or 'higher' in k or 'lower' in k}
                composite = {k: v for k, v in advanced_features.items() 
                            if 'composite' in k or 'signal' in k or 'quality' in k}
                
                conn.execute("""
                    INSERT INTO advanced_features 
                    (training_data_id, smart_money_features, technical_features, 
                     pattern_features, composite_signals)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    training_data_id,
                    json.dumps(smart_money),
                    json.dumps(technical),
                    json.dumps(patterns),
                    json.dumps(composite),
                ))
            
            # === CREATE MULTI-HORIZON LABEL PLACEHOLDERS ===
            for horizon_hours in [1, 6, 24, 168]:  # 1h, 6h, 24h, 1 week
                conn.execute("""
                    INSERT INTO multi_horizon_labels 
                    (training_data_id, horizon_hours, labeled)
                    VALUES (?, ?, 0)
                """, (training_data_id, horizon_hours))
            
            # Store price for labeling
            conn.execute("""
                INSERT INTO price_history (timestamp, token_id, price)
                VALUES (?, ?, ?)
            """, (now.timestamp(), token_id, current_price))
            
            # Update in-memory cache
            if token_id not in self.price_cache:
                self.price_cache[token_id] = []
            self.price_cache[token_id].append((now, current_price))
            
            stored_count += 1
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Collected {stored_count} examples from cycle {cycle_id} (with advanced features)")
        
        return stored_count
    
    def label_historical_data(
        self,
        horizon_hours: int = 1,
        max_examples: int = 10000,
    ) -> int:
        """Label unlabeled examples with actual outcomes.
        
        Args:
            horizon_hours: Prediction horizon (hours into future).
            max_examples: Maximum examples to label per run.
            
        Returns:
            Number of examples labeled.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Find unlabeled examples old enough to have outcomes
        cutoff_timestamp = (datetime.utcnow() - timedelta(hours=horizon_hours)).timestamp()
        
        unlabeled = conn.execute("""
            SELECT id, token_id, timestamp, features_json
            FROM training_data
            WHERE labeled = 0 AND timestamp < ?
            ORDER BY timestamp ASC
            LIMIT ?
        """, (cutoff_timestamp, max_examples)).fetchall()
        
        if not unlabeled:
            conn.close()
            return 0
        
        labeled_count = 0
        
        for row_id, token_id, timestamp, features_json in unlabeled:
            # Get price at time T
            price_t = self._get_price_at(conn, token_id, timestamp)
            
            if price_t is None:
                continue
            
            # Get price at time T+horizon
            target_timestamp = timestamp + (horizon_hours * 3600)
            price_t_plus = self._get_price_at(conn, token_id, target_timestamp, tolerance=1800)
            
            if price_t_plus is None:
                # Market might have resolved or no data available
                # Mark as labeled but with NULL target (can't use for training)
                conn.execute("""
                    UPDATE training_data 
                    SET labeled = 1 
                    WHERE id = ?
                """, (row_id,))
                continue
            
            # Compute target (future return)
            if price_t > 0:
                target = (price_t_plus - price_t) / price_t
            else:
                target = 0.0
            
            # Update with label
            conn.execute("""
                UPDATE training_data 
                SET target = ?, labeled = 1
                WHERE id = ?
            """, (target, row_id))
            
            labeled_count += 1
        
        conn.commit()
        conn.close()
        
        if labeled_count > 0:
            logger.info(f"Labeled {labeled_count} examples with {horizon_hours}h horizon")
        
        return labeled_count
    
    def label_multi_horizon_data(
        self,
        max_examples: int = 10000,
    ) -> Dict[int, int]:
        """Label examples with multiple prediction horizons.
        
        Creates labels for 1h, 6h, 24h, and 1-week predictions.
        This allows the model to make predictions at different timeframes.
        
        Args:
            max_examples: Maximum examples to label per run.
            
        Returns:
            Dict of horizon_hours -> labeled_count.
        """
        conn = sqlite3.connect(self.db_path)
        
        horizons = [1, 6, 24, 168]  # 1h, 6h, 24h, 1 week
        results = {}
        
        for horizon_hours in horizons:
            # Find unlabeled multi-horizon entries old enough
            cutoff_timestamp = (datetime.utcnow() - timedelta(hours=horizon_hours)).timestamp()
            
            unlabeled = conn.execute("""
                SELECT mh.id, mh.training_data_id, td.token_id, td.timestamp
                FROM multi_horizon_labels mh
                JOIN training_data td ON td.id = mh.training_data_id
                WHERE mh.horizon_hours = ? AND mh.labeled = 0 AND td.timestamp < ?
                ORDER BY td.timestamp ASC
                LIMIT ?
            """, (horizon_hours, cutoff_timestamp, max_examples)).fetchall()
            
            if not unlabeled:
                results[horizon_hours] = 0
                continue
            
            labeled_count = 0
            
            for mh_id, td_id, token_id, timestamp in unlabeled:
                # Get price at time T
                price_t = self._get_price_at(conn, token_id, timestamp)
                
                if price_t is None or price_t <= 0:
                    conn.execute("UPDATE multi_horizon_labels SET labeled = 1 WHERE id = ?", (mh_id,))
                    continue
                
                # Get price at time T+horizon
                target_timestamp = timestamp + (horizon_hours * 3600)
                price_t_plus = self._get_price_at(conn, token_id, target_timestamp, tolerance=1800)
                
                if price_t_plus is None:
                    conn.execute("UPDATE multi_horizon_labels SET labeled = 1 WHERE id = ?", (mh_id,))
                    continue
                
                # Compute target return
                target_return = (price_t_plus - price_t) / price_t
                
                # Compute target direction (-1, 0, +1)
                if target_return > 0.02:  # Up more than 2%
                    target_direction = 1
                elif target_return < -0.02:  # Down more than 2%
                    target_direction = -1
                else:
                    target_direction = 0  # Flat
                
                # Compute volatility in the window (if we have enough data)
                prices_in_window = self._get_prices_in_range(
                    conn, token_id, timestamp, target_timestamp
                )
                if len(prices_in_window) >= 3:
                    import numpy as np
                    returns = np.diff(prices_in_window) / prices_in_window[:-1]
                    target_volatility = float(np.std(returns)) if len(returns) > 0 else 0.0
                else:
                    target_volatility = 0.0
                
                # Update label
                conn.execute("""
                    UPDATE multi_horizon_labels 
                    SET target_return = ?, target_direction = ?, target_volatility = ?, labeled = 1
                    WHERE id = ?
                """, (target_return, target_direction, target_volatility, mh_id))
                
                labeled_count += 1
            
            results[horizon_hours] = labeled_count
        
        conn.commit()
        conn.close()
        
        total_labeled = sum(results.values())
        if total_labeled > 0:
            logger.info(f"Multi-horizon labeling: {results}")
        
        return results
    
    def _get_prices_in_range(
        self,
        conn: sqlite3.Connection,
        token_id: str,
        start_timestamp: float,
        end_timestamp: float,
    ) -> List[float]:
        """Get all prices in a time range."""
        rows = conn.execute("""
            SELECT price FROM price_history
            WHERE token_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (token_id, start_timestamp, end_timestamp)).fetchall()
        
        return [row[0] for row in rows]
    
    def get_training_set_multi_horizon(
        self,
        horizon_hours: int = 1,
        min_examples: int = 1000,
        max_examples: int = 50000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve training data with multi-horizon labels.
        
        Args:
            horizon_hours: Which horizon to get labels for.
            min_examples: Minimum examples required.
            max_examples: Maximum examples to retrieve.
            
        Returns:
            Tuple of (features DataFrame, labels DataFrame with return/direction/volatility).
        """
        conn = sqlite3.connect(self.db_path)
        
        rows = conn.execute("""
            SELECT td.features_json, mh.target_return, mh.target_direction, mh.target_volatility
            FROM training_data td
            JOIN multi_horizon_labels mh ON mh.training_data_id = td.id
            WHERE mh.horizon_hours = ? AND mh.labeled = 1 AND mh.target_return IS NOT NULL
            ORDER BY td.timestamp DESC
            LIMIT ?
        """, (horizon_hours, max_examples)).fetchall()
        
        conn.close()
        
        if len(rows) < min_examples:
            logger.warning(f"Only {len(rows)} examples for {horizon_hours}h horizon, need {min_examples}")
        
        if not rows:
            return pd.DataFrame(), pd.DataFrame()
        
        features_list = []
        labels_list = []
        
        for features_json, target_return, target_direction, target_volatility in rows:
            try:
                features = json.loads(features_json)
                features_list.append(features)
                labels_list.append({
                    'target_return': target_return,
                    'target_direction': target_direction,
                    'target_volatility': target_volatility,
                })
            except Exception:
                continue
        
        features_df = pd.DataFrame(features_list).fillna(0)
        labels_df = pd.DataFrame(labels_list)
        
        return features_df, labels_df
    
    def get_training_set(
        self,
        min_examples: int = 1000,
        max_examples: int = 50000,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve labeled training data.
        
        Args:
            min_examples: Minimum examples required.
            max_examples: Maximum examples to retrieve.
            
        Returns:
            Tuple of (features DataFrame, targets Series).
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get labeled examples with non-NULL targets
        rows = conn.execute("""
            SELECT features_json, target
            FROM training_data
            WHERE labeled = 1 AND target IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (max_examples,)).fetchall()
        
        conn.close()
        
        if len(rows) < min_examples:
            logger.warning(
                f"Only {len(rows)} examples available, need {min_examples}"
            )
        
        if not rows:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Parse features and targets
        features_list = []
        targets = []
        
        for features_json, target in rows:
            try:
                features = json.loads(features_json)
                features_list.append(features)
                targets.append(target)
            except json.JSONDecodeError:
                continue
        
        # Convert to pandas
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        
        logger.info(f"Retrieved {len(X)} training examples")
        
        return X, y
    
    def record_trade_outcome(
        self,
        token_id: str,
        cycle_id: str,
        pnl: float,
    ) -> None:
        """Record actual PnL from executed trade.
        
        Args:
            token_id: Token that was traded.
            cycle_id: Cycle when trade was created.
            pnl: Realized profit/loss.
        """
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            UPDATE training_data
            SET was_traded = 1, actual_pnl = ?
            WHERE token_id = ? AND cycle_id = ?
        """, (pnl, token_id, cycle_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade outcome: {token_id}, PnL={pnl:.4f}")
    
    def record_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record model performance metrics.
        
        Args:
            model_name: Model identifier.
            metrics: Dictionary of metric_name -> value.
            metadata: Optional additional metadata.
        """
        conn = sqlite3.connect(self.db_path)
        now = datetime.utcnow().timestamp()
        
        for metric_name, metric_value in metrics.items():
            conn.execute("""
                INSERT INTO model_performance
                (model_name, timestamp, metric_name, metric_value, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_name,
                now,
                metric_name,
                metric_value,
                json.dumps(metadata) if metadata else None,
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded performance for {model_name}: {metrics}")
    
    def get_statistics(self) -> Dict:
        """Get collection statistics.
        
        Returns:
            Dictionary of statistics.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Total examples
        total = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        
        # Labeled examples
        labeled = conn.execute(
            "SELECT COUNT(*) FROM training_data WHERE labeled = 1"
        ).fetchone()[0]
        
        # Examples with non-NULL targets
        with_targets = conn.execute(
            "SELECT COUNT(*) FROM training_data WHERE target IS NOT NULL"
        ).fetchone()[0]
        
        # Traded examples
        traded = conn.execute(
            "SELECT COUNT(*) FROM training_data WHERE was_traded = 1"
        ).fetchone()[0]
        
        # Price history size
        prices = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        
        conn.close()
        
        return {
            'total_examples': total,
            'labeled_examples': labeled,
            'examples_with_targets': with_targets,
            'traded_examples': traded,
            'price_history_size': prices,
            'labeling_rate': labeled / total if total > 0 else 0,
            'training_ready': with_targets >= 1000,
        }
    
    def cleanup_old_data(self, days: int = 730) -> int:
        """Remove old training data to save space.
        
        Args:
            days: Remove data older than N days (default: 730 = 2 years).
            
        Returns:
            Number of rows deleted.
        """
        conn = sqlite3.connect(self.db_path)
        
        cutoff = (datetime.utcnow() - timedelta(days=days)).timestamp()
        
        # Delete old training data
        cursor = conn.execute("""
            DELETE FROM training_data
            WHERE timestamp < ?
        """, (cutoff,))
        
        deleted_training = cursor.rowcount
        
        # Delete old price history
        cursor = conn.execute("""
            DELETE FROM price_history
            WHERE timestamp < ?
        """, (cutoff,))
        
        deleted_prices = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(
            f"Cleanup: deleted {deleted_training} training examples "
            f"and {deleted_prices} price records older than {days} days"
        )
        
        return deleted_training + deleted_prices
    
    def _get_price_at(
        self,
        conn: sqlite3.Connection,
        token_id: str,
        timestamp: float,
        tolerance: int = 600,  # 10 minutes
    ) -> Optional[float]:
        """Get price at specific timestamp (with tolerance).
        
        Args:
            conn: Database connection.
            token_id: Token identifier.
            timestamp: Target timestamp.
            tolerance: Acceptable time difference in seconds.
            
        Returns:
            Price or None if not found.
        """
        # Try to find price within tolerance
        result = conn.execute("""
            SELECT price
            FROM price_history
            WHERE token_id = ?
            AND ABS(timestamp - ?) <= ?
            ORDER BY ABS(timestamp - ?) ASC
            LIMIT 1
        """, (token_id, timestamp, tolerance, timestamp)).fetchone()
        
        if result:
            return result[0]
        
        return None
    
    def export_to_csv(self, output_path: str, labeled_only: bool = True) -> int:
        """Export training data to CSV for analysis.
        
        Args:
            output_path: Path to write CSV file.
            labeled_only: Only export labeled examples.
            
        Returns:
            Number of rows exported.
        """
        X, y = self.get_training_set(min_examples=1, max_examples=100000)
        
        if len(X) == 0:
            logger.warning("No data to export")
            return 0
        
        # Combine features and target
        X['target'] = y
        
        X.to_csv(output_path, index=False)
        logger.info(f"Exported {len(X)} examples to {output_path}")
        
        return len(X)

