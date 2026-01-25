"""Unified Status Aggregator Service.

Aggregates all status data from across the system into a single,
comprehensive status report for CLI and API consumption.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class SystemStatus:
    """System resource status."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    uptime_hours: float = 0.0
    python_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "uptime_hours": self.uptime_hours,
            "python_version": self.python_version,
        }


@dataclass
class TradingStatus:
    """Trading activity status."""
    is_live: bool = False
    placing_orders: bool = False
    balance_usd: float = 0.0
    available_balance: float = 0.0
    position_count: int = 0
    total_exposure: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    trades_today: int = 0
    signals_today: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_live": self.is_live,
            "placing_orders": self.placing_orders,
            "balance_usd": self.balance_usd,
            "available_balance": self.available_balance,
            "position_count": self.position_count,
            "total_exposure": self.total_exposure,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl_today": self.realized_pnl_today,
            "trades_today": self.trades_today,
            "signals_today": self.signals_today,
        }


@dataclass
class AIStatus:
    """AI/ML system status."""
    is_ready: bool = False
    model_version: Optional[str] = None
    training_examples: int = 0
    labeled_examples: int = 0
    last_training: Optional[str] = None
    next_training: Optional[str] = None
    training_interval_hours: int = 6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "model_version": self.model_version,
            "training_examples": self.training_examples,
            "labeled_examples": self.labeled_examples,
            "last_training": self.last_training,
            "next_training": self.next_training,
            "training_interval_hours": self.training_interval_hours,
        }


@dataclass
class MoEStatus:
    """Mixture of Experts status."""
    total_experts: int = 0
    active_experts: int = 0
    suspended_experts: int = 0
    probation_experts: int = 0
    deprecated_experts: int = 0
    untrained_experts: int = 0
    state_counts: Dict[str, int] = field(default_factory=dict)
    top_experts: List[Dict[str, Any]] = field(default_factory=list)
    gating_accuracy: float = 0.0
    last_training: Optional[str] = None
    training_cycles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_experts": self.total_experts,
            "active_experts": self.active_experts,
            "suspended_experts": self.suspended_experts,
            "probation_experts": self.probation_experts,
            "deprecated_experts": self.deprecated_experts,
            "untrained_experts": self.untrained_experts,
            "state_counts": self.state_counts,
            "top_experts": self.top_experts,
            "gating_accuracy": self.gating_accuracy,
            "last_training": self.last_training,
            "training_cycles": self.training_cycles,
        }


@dataclass
class MetaControllerStatus:
    """Meta-Controller status."""
    total_mixtures: int = 0
    resolved_mixtures: int = 0
    total_profit_pct: float = 0.0
    win_rate: float = 0.0
    top_mixtures: List[tuple] = field(default_factory=list)
    synergy_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mixtures": self.total_mixtures,
            "resolved_mixtures": self.resolved_mixtures,
            "total_profit_pct": self.total_profit_pct,
            "win_rate": self.win_rate,
            "top_mixtures": self.top_mixtures,
            "synergy_insights": self.synergy_insights,
        }


@dataclass
class ArbitrageStatus:
    """Arbitrage scanner status."""
    is_enabled: bool = True
    total_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_enabled": self.is_enabled,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_profit": self.total_profit,
        }


class StatusAggregator:
    """Aggregates status from all system components."""
    
    def __init__(self):
        self._cached_status: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 5  # Cache for 5 seconds
    
    def get_full_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Args:
            use_cache: Use cached status if available
        
        Returns:
            Dictionary with all status sections
        """
        # Check cache
        if use_cache and self._cached_status and self._cache_time:
            age = (datetime.utcnow() - self._cache_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._cached_status
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": self.get_system_status().to_dict(),
            "trading": self.get_trading_status().to_dict(),
            "ai": self.get_ai_status().to_dict(),
            "moe": self.get_moe_status().to_dict(),
            "meta_controller": self.get_meta_controller_status().to_dict(),
            "arbitrage": self.get_arbitrage_status().to_dict(),
            "recent_activity": self.get_recent_activity(),
            "errors": self.get_recent_errors(),
        }
        
        # Cache it
        self._cached_status = status
        self._cache_time = datetime.utcnow()
        
        return status
    
    def get_system_status(self) -> SystemStatus:
        """Get system resource status."""
        status = SystemStatus()
        
        try:
            import psutil
            import sys
            
            status.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            status.memory_percent = mem.percent
            status.memory_used_mb = mem.used / (1024 * 1024)
            
            disk = psutil.disk_usage(str(PROJECT_ROOT))
            status.disk_percent = disk.percent
            status.disk_used_gb = disk.used / (1024 * 1024 * 1024)
            
            # Uptime (of Python process)
            import time
            status.uptime_hours = (time.time() - psutil.Process().create_time()) / 3600
            
            status.python_version = sys.version.split()[0]
            
        except ImportError:
            logger.debug("psutil not available for system status")
        except Exception as e:
            logger.debug(f"Failed to get system status: {e}")
        
        return status
    
    def get_trading_status(self) -> TradingStatus:
        """Get trading activity status."""
        status = TradingStatus()
        
        try:
            from polyb0t.config.settings import get_settings
            settings = get_settings()
            
            status.placing_orders = settings.placing_orders
            status.is_live = settings.placing_orders and settings.mode == "live"
            
            # Try to get balance info
            balance_path = DATA_DIR / "account_state.json"
            if balance_path.exists():
                with open(balance_path, "r") as f:
                    account = json.load(f)
                status.balance_usd = account.get("balance_usd", 0)
                status.available_balance = account.get("available_balance", 0)
            
            # Get position info from database
            db_path = PROJECT_ROOT / "polybot.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                try:
                    # Count positions
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM positions WHERE exit_price IS NULL"
                    )
                    status.position_count = cur.fetchone()[0]
                    
                    # Get exposure
                    cur = conn.execute(
                        "SELECT SUM(size_usd) FROM positions WHERE exit_price IS NULL"
                    )
                    result = cur.fetchone()[0]
                    status.total_exposure = result if result else 0
                    
                    # Trades today
                    today = datetime.now(timezone.utc).date().isoformat()
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM fills WHERE date(created_at) = ?",
                        (today,)
                    )
                    status.trades_today = cur.fetchone()[0]
                    
                    # Signals today
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM signals WHERE date(created_at) = ?",
                        (today,)
                    )
                    status.signals_today = cur.fetchone()[0]
                    
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.debug(f"Failed to get trading status: {e}")
        
        return status
    
    def get_ai_status(self) -> AIStatus:
        """Get AI system status."""
        status = AIStatus()
        
        try:
            # Check model state
            model_dir = DATA_DIR / "ai_models"
            state_path = model_dir / "trainer_state.json"
            
            if state_path.exists():
                with open(state_path, "r") as f:
                    state = json.load(f)
                
                status.is_ready = state.get("model_deployed", False)
                status.model_version = state.get("current_version")
                status.training_examples = state.get("training_examples", 0)
                status.last_training = state.get("last_training")
            
            # Get training data count from database
            db_path = DATA_DIR / "ai_training.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                try:
                    cur = conn.execute("SELECT COUNT(*) FROM training_examples")
                    status.training_examples = cur.fetchone()[0]
                    
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM training_examples WHERE price_change_24h IS NOT NULL"
                    )
                    status.labeled_examples = cur.fetchone()[0]
                finally:
                    conn.close()
            
            # Calculate next training
            if status.last_training:
                try:
                    last = datetime.fromisoformat(status.last_training.replace("Z", "+00:00"))
                    next_train = last + timedelta(hours=status.training_interval_hours)
                    status.next_training = next_train.isoformat()
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Failed to get AI status: {e}")
        
        return status
    
    def get_moe_status(self) -> MoEStatus:
        """Get Mixture of Experts status."""
        status = MoEStatus()
        
        try:
            pool_state_path = DATA_DIR / "moe_models" / "pool_state.json"
            
            if pool_state_path.exists():
                with open(pool_state_path, "r") as f:
                    pool_data = json.load(f)
                
                state = pool_data.get("state", {})
                expert_ids = pool_data.get("expert_ids", [])
                
                status.total_experts = len(expert_ids)
                status.last_training = state.get("last_training")
                status.training_cycles = state.get("total_training_cycles", 0)
                
                # Load individual expert data
                experts_dir = DATA_DIR / "moe_models" / "experts"
                experts = []
                
                state_counts = {
                    "untrained": 0, "probation": 0, "active": 0,
                    "suspended": 0, "rollback": 0, "deprecated": 0
                }
                
                for expert_id in expert_ids:
                    meta_path = experts_dir / f"{expert_id}.meta.pkl"
                    version_path = experts_dir / f"{expert_id}.versions.json"
                    
                    if meta_path.exists():
                        import pickle
                        with open(meta_path, "rb") as f:
                            expert_data = pickle.load(f)
                        
                        metrics = expert_data.get("metrics", {})
                        expert_state = "untrained"
                        
                        if version_path.exists():
                            with open(version_path, "r") as f:
                                version_data = json.load(f)
                            expert_state = version_data.get("state", "untrained")
                        
                        if expert_state in state_counts:
                            state_counts[expert_state] += 1
                        
                        experts.append({
                            "expert_id": expert_id,
                            "domain": expert_data.get("domain", expert_id),
                            "state": expert_state,
                            "profit_pct": metrics.get("simulated_profit_pct", 0),
                            "trades": metrics.get("simulated_num_trades", 0),
                            "win_rate": metrics.get("simulated_win_rate", 0),
                        })
                
                status.state_counts = state_counts
                status.active_experts = state_counts.get("active", 0) + state_counts.get("rollback", 0)
                status.suspended_experts = state_counts.get("suspended", 0)
                status.probation_experts = state_counts.get("probation", 0)
                status.deprecated_experts = state_counts.get("deprecated", 0)
                status.untrained_experts = state_counts.get("untrained", 0)
                
                # Top experts by profit
                experts.sort(key=lambda x: x["profit_pct"], reverse=True)
                status.top_experts = experts[:5]
                
                # Gating accuracy
                gating_path = DATA_DIR / "moe_models" / "gating.meta.pkl"
                if gating_path.exists():
                    import pickle
                    with open(gating_path, "rb") as f:
                        gating_data = pickle.load(f)
                    status.gating_accuracy = gating_data.get("metrics", {}).get("routing_accuracy", 0)
                    
        except Exception as e:
            logger.debug(f"Failed to get MoE status: {e}")
        
        return status
    
    def get_meta_controller_status(self) -> MetaControllerStatus:
        """Get Meta-Controller status."""
        status = MetaControllerStatus()
        
        try:
            state_path = DATA_DIR / "meta_controller_state.json"
            history_path = DATA_DIR / "mixture_history.json"
            
            if history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
                
                outcomes = history.get("outcomes", [])
                status.total_mixtures = len(outcomes)
                
                resolved = [o for o in outcomes if o.get("profit_pct") is not None]
                status.resolved_mixtures = len(resolved)
                
                if resolved:
                    status.total_profit_pct = sum(o["profit_pct"] for o in resolved)
                    status.win_rate = sum(1 for o in resolved if o["profit_pct"] > 0) / len(resolved)
                
                # Count mixture usage
                mixture_counts = {}
                for o in outcomes[-100:]:
                    key = f"{o.get('primary_expert', 'unknown')}+{','.join(o.get('supporting_experts', []))}"
                    mixture_counts[key] = mixture_counts.get(key, 0) + 1
                
                status.top_mixtures = sorted(
                    mixture_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
            
            if state_path.exists():
                with open(state_path, "r") as f:
                    state = json.load(f)
                
                pair_perf = state.get("pair_performance", {})
                synergies = []
                for key, perf in pair_perf.items():
                    if perf.get("combined_trades", 0) >= 20:
                        synergies.append({
                            "pair": key,
                            "trades": perf["combined_trades"],
                            "synergy": perf.get("synergy_score", 0),
                        })
                
                synergies.sort(key=lambda x: x["synergy"], reverse=True)
                status.synergy_insights = synergies[:10]
                
        except Exception as e:
            logger.debug(f"Failed to get Meta-Controller status: {e}")
        
        return status
    
    def get_arbitrage_status(self) -> ArbitrageStatus:
        """Get arbitrage scanner status."""
        status = ArbitrageStatus()
        
        try:
            # First check settings to see if arbitrage is enabled
            from polyb0t.config.settings import get_settings
            settings = get_settings()
            status.is_enabled = settings.enable_arbitrage_scanner
            
            # Then check state file for stats
            arb_path = DATA_DIR / "arbitrage_state.json"
            if arb_path.exists():
                with open(arb_path, "r") as f:
                    arb = json.load(f)
                
                # If disabled in state file, override
                if arb.get("is_disabled", False):
                    status.is_enabled = False
                    
                stats = arb.get("stats", {})
                status.total_trades = stats.get("total_trades", 0)
                status.win_rate = stats.get("win_rate", 0)
                status.total_profit = stats.get("total_profit", 0)
                
        except Exception as e:
            logger.debug(f"Failed to get arbitrage status: {e}")
        
        return status
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading activity."""
        activity = []
        
        try:
            db_path = PROJECT_ROOT / "polybot.db"
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                try:
                    # Get recent fills
                    cur = conn.execute("""
                        SELECT * FROM fills 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                    
                    for row in cur.fetchall():
                        activity.append({
                            "type": "fill",
                            "timestamp": row["created_at"],
                            "market_id": row["market_id"],
                            "side": row["side"],
                            "size_usd": row["size_usd"],
                            "price": row["price"],
                        })
                finally:
                    conn.close()
                    
        except Exception as e:
            logger.debug(f"Failed to get recent activity: {e}")
        
        return activity
    
    def get_recent_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent errors from logs."""
        errors = []
        
        try:
            log_path = PROJECT_ROOT / "live_run.log"
            if log_path.exists():
                with open(log_path, "r") as f:
                    lines = f.readlines()[-1000:]  # Last 1000 lines
                
                for line in reversed(lines):
                    if "ERROR" in line or "CRITICAL" in line:
                        errors.append({
                            "message": line.strip()[:200],
                        })
                        if len(errors) >= limit:
                            break
                            
        except Exception as e:
            logger.debug(f"Failed to get recent errors: {e}")
        
        return errors
    
    def format_cli_output(self, status: Optional[Dict[str, Any]] = None) -> str:
        """Format status for CLI display."""
        if status is None:
            status = self.get_full_status()
        
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("                    POLYB0T STATUS DASHBOARD")
        lines.append(f"                    {status['timestamp'][:19]}")
        lines.append("=" * 70)
        
        # System Status
        sys = status["system"]
        lines.append(f"\n[SYSTEM]")
        lines.append(f"  CPU: {sys['cpu_percent']:.1f}%  |  Memory: {sys['memory_percent']:.1f}%  |  Disk: {sys['disk_percent']:.1f}%")
        lines.append(f"  Uptime: {sys['uptime_hours']:.1f} hours  |  Python: {sys['python_version']}")
        
        # Trading Status
        trading = status["trading"]
        mode = "LIVE" if trading["is_live"] else ("DRY RUN" if trading["placing_orders"] else "OFF")
        lines.append(f"\n[TRADING] {mode}")
        lines.append(f"  Balance: ${trading['balance_usd']:,.2f}  |  Available: ${trading['available_balance']:,.2f}")
        lines.append(f"  Positions: {trading['position_count']}  |  Exposure: ${trading['total_exposure']:,.2f}")
        lines.append(f"  Trades Today: {trading['trades_today']}  |  Signals Today: {trading['signals_today']}")
        
        # AI Status
        ai = status["ai"]
        ai_status = "READY" if ai["is_ready"] else "TRAINING"
        lines.append(f"\n[AI] {ai_status}")
        lines.append(f"  Training Examples: {ai['training_examples']:,}  |  Labeled: {ai['labeled_examples']:,}")
        if ai["last_training"]:
            lines.append(f"  Last Training: {ai['last_training'][:19]}")
        
        # MoE Status
        moe = status["moe"]
        lines.append(f"\n[MIXTURE OF EXPERTS] {moe['total_experts']} experts")
        lines.append(f"  Active: {moe['active_experts']}  |  Suspended: {moe['suspended_experts']}  |  Probation: {moe['probation_experts']}")
        lines.append(f"  Deprecated: {moe['deprecated_experts']}  |  Untrained: {moe['untrained_experts']}")
        if moe["gating_accuracy"]:
            lines.append(f"  Gating Accuracy: {moe['gating_accuracy']:.1%}")
        
        if moe["top_experts"]:
            lines.append("\n  Top Experts:")
            for expert in moe["top_experts"][:3]:
                profit = expert["profit_pct"]
                icon = "+" if profit > 0 else "-" if profit < 0 else " "
                lines.append(f"    [{icon}] {expert['domain']}: {profit:+.1%} ({expert['trades']} trades)")
        
        # Meta-Controller Status
        meta = status["meta_controller"]
        if meta["total_mixtures"] > 0:
            lines.append(f"\n[META-CONTROLLER]")
            lines.append(f"  Total Mixtures: {meta['total_mixtures']}  |  Resolved: {meta['resolved_mixtures']}")
            if meta["resolved_mixtures"] > 0:
                lines.append(f"  Total Profit: {meta['total_profit_pct']:+.1%}  |  Win Rate: {meta['win_rate']:.1%}")
            
            if meta["top_mixtures"]:
                lines.append("  Popular Mixtures:")
                for mix, count in meta["top_mixtures"][:3]:
                    lines.append(f"    {mix}: {count} uses")
        
        # Arbitrage Status
        arb = status["arbitrage"]
        arb_status = "ENABLED" if arb["is_enabled"] else "DISABLED"
        lines.append(f"\n[ARBITRAGE] {arb_status}")
        if arb["total_trades"] > 0:
            lines.append(f"  Trades: {arb['total_trades']}  |  Win Rate: {arb['win_rate']:.1%}  |  Profit: {arb['total_profit']:+.1%}")
        
        # Recent Errors
        if status["errors"]:
            lines.append(f"\n[RECENT ERRORS]")
            for err in status["errors"][:3]:
                lines.append(f"  ! {err['message'][:60]}...")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


# Singleton instance
_aggregator: Optional[StatusAggregator] = None


def get_status_aggregator() -> StatusAggregator:
    """Get or create the singleton StatusAggregator."""
    global _aggregator
    if _aggregator is None:
        _aggregator = StatusAggregator()
    return _aggregator
