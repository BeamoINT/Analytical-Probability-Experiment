"""System Resource Monitor for tracking CPU, memory, and disk usage.

Tracks min, max, and average usage over time to help determine
if the system needs to be upgraded.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque

logger = logging.getLogger(__name__)

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system monitoring disabled. Install with: pip install psutil")


@dataclass
class ResourceSnapshot:
    """A snapshot of system resources at a point in time."""
    timestamp: str
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    process_cpu_percent: float = 0.0  # This process only
    process_memory_mb: float = 0.0  # This process only
    context: str = "normal"  # "normal", "training", "collecting"


@dataclass
class ResourceStats:
    """Aggregated statistics for a resource."""
    current: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    avg: float = 0.0
    samples: int = 0
    
    def update(self, value: float) -> None:
        """Update stats with a new value."""
        self.current = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        # Running average
        self.samples += 1
        self.avg = self.avg + (value - self.avg) / self.samples
    
    def to_dict(self) -> dict:
        return {
            "current": round(self.current, 2),
            "min": round(self.min, 2) if self.min != float('inf') else 0,
            "max": round(self.max, 2),
            "avg": round(self.avg, 2),
            "samples": self.samples,
        }


@dataclass 
class SystemStats:
    """Complete system statistics."""
    start_time: str = ""
    uptime_hours: float = 0.0
    
    # CPU stats
    cpu_percent: ResourceStats = field(default_factory=ResourceStats)
    cpu_count: int = 0
    cpu_freq_mhz: float = 0.0
    load_avg: ResourceStats = field(default_factory=ResourceStats)
    
    # Memory stats
    memory_percent: ResourceStats = field(default_factory=ResourceStats)
    memory_total_gb: float = 0.0
    
    # Disk stats
    disk_percent: ResourceStats = field(default_factory=ResourceStats)
    disk_total_gb: float = 0.0
    
    # Process-specific stats
    process_cpu: ResourceStats = field(default_factory=ResourceStats)
    process_memory_mb: ResourceStats = field(default_factory=ResourceStats)
    
    # Training-specific stats
    training_cpu: ResourceStats = field(default_factory=ResourceStats)
    training_memory: ResourceStats = field(default_factory=ResourceStats)
    training_duration_seconds: List[float] = field(default_factory=list)
    
    # Peak tracking
    peak_cpu_time: str = ""
    peak_memory_time: str = ""
    
    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "uptime_hours": round(self.uptime_hours, 2),
            "cpu": {
                "percent": self.cpu_percent.to_dict(),
                "count": self.cpu_count,
                "freq_mhz": round(self.cpu_freq_mhz, 0),
                "load_avg": self.load_avg.to_dict(),
            },
            "memory": {
                "percent": self.memory_percent.to_dict(),
                "total_gb": round(self.memory_total_gb, 2),
            },
            "disk": {
                "percent": self.disk_percent.to_dict(),
                "total_gb": round(self.disk_total_gb, 2),
            },
            "process": {
                "cpu_percent": self.process_cpu.to_dict(),
                "memory_mb": self.process_memory_mb.to_dict(),
            },
            "training": {
                "cpu_percent": self.training_cpu.to_dict(),
                "memory_percent": self.training_memory.to_dict(),
                "avg_duration_seconds": round(sum(self.training_duration_seconds) / max(1, len(self.training_duration_seconds)), 1) if self.training_duration_seconds else 0,
                "training_count": len(self.training_duration_seconds),
            },
            "peaks": {
                "cpu_time": self.peak_cpu_time,
                "memory_time": self.peak_memory_time,
            },
        }


class SystemMonitor:
    """Monitors system resources and tracks statistics over time."""
    
    def __init__(
        self,
        sample_interval_seconds: float = 10.0,
        history_size: int = 1000,
        state_file: str = "data/system_stats.json",
    ):
        """Initialize the system monitor.
        
        Args:
            sample_interval_seconds: How often to sample resources.
            history_size: Number of recent snapshots to keep in memory.
            state_file: Path to persist stats.
        """
        self.sample_interval = sample_interval_seconds
        self.history_size = history_size
        self.state_file = state_file
        
        self._stats = SystemStats()
        self._history: deque = deque(maxlen=history_size)
        self._current_context = "normal"
        self._context_start_time: Optional[float] = None
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._process: Optional[psutil.Process] = None
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()
        
        self._load_state()
        
    def _load_state(self) -> None:
        """Load persisted state if available."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Restore basic stats
                self._stats.start_time = data.get("start_time", "")
                self._stats.cpu_count = data.get("cpu", {}).get("count", 0)
                self._stats.memory_total_gb = data.get("memory", {}).get("total_gb", 0)
                self._stats.disk_total_gb = data.get("disk", {}).get("total_gb", 0)
                
                # Restore resource stats
                self._restore_resource_stats(self._stats.cpu_percent, data.get("cpu", {}).get("percent", {}))
                self._restore_resource_stats(self._stats.memory_percent, data.get("memory", {}).get("percent", {}))
                self._restore_resource_stats(self._stats.disk_percent, data.get("disk", {}).get("percent", {}))
                self._restore_resource_stats(self._stats.process_cpu, data.get("process", {}).get("cpu_percent", {}))
                self._restore_resource_stats(self._stats.process_memory_mb, data.get("process", {}).get("memory_mb", {}))
                self._restore_resource_stats(self._stats.training_cpu, data.get("training", {}).get("cpu_percent", {}))
                self._restore_resource_stats(self._stats.training_memory, data.get("training", {}).get("memory_percent", {}))
                
                logger.info(f"Loaded system stats: {self._stats.cpu_percent.samples} samples")
            except Exception as e:
                logger.warning(f"Could not load system stats: {e}")
        
        # Set start time if not set
        if not self._stats.start_time:
            self._stats.start_time = datetime.utcnow().isoformat()
    
    def _restore_resource_stats(self, stats: ResourceStats, data: dict) -> None:
        """Restore ResourceStats from dict."""
        if data:
            stats.min = data.get("min", float('inf'))
            stats.max = data.get("max", 0)
            stats.avg = data.get("avg", 0)
            stats.samples = data.get("samples", 0)
    
    def _save_state(self) -> None:
        """Persist current stats to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(self._stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save system stats: {e}")
    
    def start(self) -> None:
        """Start background monitoring."""
        if not PSUTIL_AVAILABLE:
            logger.warning("Cannot start system monitor - psutil not installed")
            return
            
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitor started")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self._save_state()
        logger.info("System monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        save_counter = 0
        while self._running:
            try:
                snapshot = self._take_snapshot()
                self._update_stats(snapshot)
                
                # Save periodically (every 60 samples = ~10 minutes at 10s interval)
                save_counter += 1
                if save_counter >= 60:
                    self._save_state()
                    save_counter = 0
                    
            except Exception as e:
                logger.warning(f"System monitor error: {e}")
            
            time.sleep(self.sample_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current system resources."""
        now = datetime.utcnow().isoformat()
        
        snapshot = ResourceSnapshot(
            timestamp=now,
            context=self._current_context,
        )
        
        if not PSUTIL_AVAILABLE:
            return snapshot
        
        try:
            # CPU
            snapshot.cpu_percent = psutil.cpu_percent(interval=None)
            snapshot.cpu_count = psutil.cpu_count()
            
            freq = psutil.cpu_freq()
            if freq:
                snapshot.cpu_freq_mhz = freq.current
            
            # Load average (Unix only)
            try:
                load = os.getloadavg()
                snapshot.load_avg_1m = load[0]
                snapshot.load_avg_5m = load[1]
                snapshot.load_avg_15m = load[2]
            except (AttributeError, OSError):
                pass
            
            # Memory
            mem = psutil.virtual_memory()
            snapshot.memory_percent = mem.percent
            snapshot.memory_used_gb = mem.used / (1024 ** 3)
            snapshot.memory_total_gb = mem.total / (1024 ** 3)
            
            # Disk
            disk = psutil.disk_usage('/')
            snapshot.disk_percent = disk.percent
            snapshot.disk_used_gb = disk.used / (1024 ** 3)
            snapshot.disk_total_gb = disk.total / (1024 ** 3)
            
            # Process-specific
            if self._process:
                snapshot.process_cpu_percent = self._process.cpu_percent()
                snapshot.process_memory_mb = self._process.memory_info().rss / (1024 ** 2)
                
        except Exception as e:
            logger.debug(f"Error taking snapshot: {e}")
        
        return snapshot
    
    def _update_stats(self, snapshot: ResourceSnapshot) -> None:
        """Update statistics with new snapshot."""
        with self._lock:
            # Add to history
            self._history.append(snapshot)
            
            # Update uptime
            try:
                start = datetime.fromisoformat(self._stats.start_time)
                self._stats.uptime_hours = (datetime.utcnow() - start).total_seconds() / 3600
            except:
                pass
            
            # Update static info
            self._stats.cpu_count = snapshot.cpu_count
            self._stats.cpu_freq_mhz = snapshot.cpu_freq_mhz
            self._stats.memory_total_gb = snapshot.memory_total_gb
            self._stats.disk_total_gb = snapshot.disk_total_gb
            
            # Update resource stats
            self._stats.cpu_percent.update(snapshot.cpu_percent)
            self._stats.load_avg.update(snapshot.load_avg_1m)
            self._stats.memory_percent.update(snapshot.memory_percent)
            self._stats.disk_percent.update(snapshot.disk_percent)
            self._stats.process_cpu.update(snapshot.process_cpu_percent)
            self._stats.process_memory_mb.update(snapshot.process_memory_mb)
            
            # Track peaks
            if snapshot.cpu_percent >= self._stats.cpu_percent.max:
                self._stats.peak_cpu_time = snapshot.timestamp
            if snapshot.memory_percent >= self._stats.memory_percent.max:
                self._stats.peak_memory_time = snapshot.timestamp
            
            # Training-specific stats
            if snapshot.context == "training":
                self._stats.training_cpu.update(snapshot.cpu_percent)
                self._stats.training_memory.update(snapshot.memory_percent)
    
    def set_context(self, context: str) -> None:
        """Set the current activity context.
        
        Args:
            context: One of "normal", "training", "collecting"
        """
        if context == self._current_context:
            return
            
        # If leaving training context, record duration
        if self._current_context == "training" and self._context_start_time:
            duration = time.time() - self._context_start_time
            self._stats.training_duration_seconds.append(duration)
            logger.debug(f"Training completed in {duration:.1f}s")
        
        self._current_context = context
        self._context_start_time = time.time() if context != "normal" else None
        logger.debug(f"System monitor context: {context}")
    
    def get_stats(self) -> dict:
        """Get current statistics as dict."""
        with self._lock:
            return self._stats.to_dict()
    
    def get_recent_history(self, minutes: int = 10) -> List[dict]:
        """Get recent snapshots.
        
        Args:
            minutes: Number of minutes of history to return.
            
        Returns:
            List of snapshot dicts.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        cutoff_str = cutoff.isoformat()
        
        with self._lock:
            return [
                asdict(s) for s in self._history
                if s.timestamp >= cutoff_str
            ]
    
    def get_recommendation(self) -> dict:
        """Get upgrade recommendation based on stats.
        
        Returns:
            Dict with recommendation and details.
        """
        stats = self._stats
        recommendations = []
        status = "healthy"
        
        # Check CPU
        if stats.cpu_percent.avg > 80:
            recommendations.append("CPU consistently high (>80% avg). Consider upgrading CPU or adding more cores.")
            status = "needs_upgrade"
        elif stats.cpu_percent.max > 95 and stats.cpu_percent.samples > 100:
            recommendations.append("CPU occasionally maxes out. May cause slowdowns during training.")
            if status == "healthy":
                status = "monitor"
        
        # Check Memory
        if stats.memory_percent.avg > 85:
            recommendations.append(f"Memory usage high (>85% avg). Current: {stats.memory_total_gb:.1f}GB. Consider adding RAM.")
            status = "needs_upgrade"
        elif stats.memory_percent.max > 95:
            recommendations.append("Memory occasionally maxes out. May cause swapping.")
            if status == "healthy":
                status = "monitor"
        
        # Check Disk
        if stats.disk_percent.current > 90:
            recommendations.append(f"Disk usage critical (>90%). Free up space or add storage.")
            status = "needs_upgrade"
        elif stats.disk_percent.current > 80:
            recommendations.append("Disk usage high (>80%). Consider freeing space.")
            if status == "healthy":
                status = "monitor"
        
        # Check training performance
        if stats.training_cpu.samples > 0 and stats.training_cpu.avg > 90:
            recommendations.append("Training uses nearly all CPU. Consider more cores for faster training.")
        
        avg_training_time = (
            sum(stats.training_duration_seconds) / len(stats.training_duration_seconds)
            if stats.training_duration_seconds else 0
        )
        if avg_training_time > 300:  # 5 minutes
            recommendations.append(f"Training takes {avg_training_time/60:.1f} minutes on average. More CPU could help.")
        
        # Check load average vs cores
        if stats.load_avg.avg > stats.cpu_count and stats.cpu_count > 0:
            recommendations.append(f"Load average ({stats.load_avg.avg:.1f}) exceeds CPU cores ({stats.cpu_count}). System is overloaded.")
            status = "needs_upgrade"
        
        if not recommendations:
            recommendations.append("System resources look healthy. No upgrades needed.")
        
        return {
            "status": status,
            "status_icon": "🟢" if status == "healthy" else "🟡" if status == "monitor" else "🔴",
            "recommendations": recommendations,
            "summary": {
                "cpu_avg": f"{stats.cpu_percent.avg:.1f}%",
                "memory_avg": f"{stats.memory_percent.avg:.1f}%",
                "disk_current": f"{stats.disk_percent.current:.1f}%",
                "uptime": f"{stats.uptime_hours:.1f}h",
            }
        }
    
    def take_manual_snapshot(self) -> dict:
        """Take and return a manual snapshot (for immediate stats)."""
        snapshot = self._take_snapshot()
        return asdict(snapshot)


# Singleton instance
_monitor_instance: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get or create the singleton system monitor."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


def start_system_monitor() -> None:
    """Start the system monitor."""
    monitor = get_system_monitor()
    monitor.start()


def stop_system_monitor() -> None:
    """Stop the system monitor."""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop()
