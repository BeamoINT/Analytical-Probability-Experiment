#!/usr/bin/env python3
"""Comprehensive PolyB0T status checker.

Run from anywhere: poetry run python scripts/check_ai_status.py
Or: cd ~/Analytical-Probability-Experiment && poetry run python scripts/check_ai_status.py
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Find project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


def format_usd(amount):
    """Format as USD."""
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"


def format_pct(value):
    """Format as percentage."""
    if value is None:
        return "N/A"
    return f"{value*100:.1f}%"


def format_time_ago(timestamp_str):
    """Format timestamp as time ago."""
    if not timestamp_str:
        return "Never"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        delta = datetime.now(timezone.utc).replace(tzinfo=None) - dt
        
        if delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())}s ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        elif delta.total_seconds() < 86400:
            return f"{delta.total_seconds() / 3600:.1f}h ago"
        else:
            return f"{delta.days}d ago"
    except:
        return timestamp_str


def get_ai_status():
    """Get AI model and training status."""
    result = {
        "has_model": False,
        "model_version": None,
        "model_created": None,
        "training_examples": 0,
        "metrics": {},
        "total_examples": 0,
        "labeled_examples": 0,
        "partial_labels": 0,
        "snapshots": 0,
        "price_points": 0,
        "db_size_mb": 0,
    }
    
    # Check model state
    model_dir = "data/ai_models"
    state_path = os.path.join(model_dir, "trainer_state.json")
    
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            result["has_model"] = True
            result["model_version"] = state.get("version")
            result["model_created"] = state.get("created_at")
            result["training_examples"] = state.get("training_examples", 0)
            result["metrics"] = state.get("metrics", {})
        except:
            pass
    
    # Check training database
    db_path = "data/ai_training.db"
    if os.path.exists(db_path):
        try:
            result["db_size_mb"] = os.path.getsize(db_path) / (1024 * 1024)

            # Use timeout to wait for lock (bot may be writing)
            # Also use WAL mode for better concurrent access
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            cursor = conn.cursor()

            # Total examples
            cursor.execute("SELECT COUNT(*) FROM training_examples")
            result["total_examples"] = cursor.fetchone()[0]

            # Labeled = has 24h price change (usable for training)
            cursor.execute("SELECT COUNT(*) FROM training_examples WHERE price_change_24h IS NOT NULL")
            result["labeled_examples"] = cursor.fetchone()[0]
            
            # Fully resolved (market closed)
            cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 1")
            result["resolved_examples"] = cursor.fetchone()[0]

            # Partial labels (has some data but not 24h yet)
            cursor.execute("SELECT COUNT(*) FROM training_examples WHERE price_change_1h IS NOT NULL AND price_change_24h IS NULL")
            result["partial_labels"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM market_snapshots")
            result["snapshots"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM price_history")
            result["price_points"] = cursor.fetchone()[0]

            conn.close()
        except Exception as e:
            result["db_error"] = str(e)
    else:
        result["db_error"] = f"DB not found at {os.path.abspath(db_path)}"

    return result


def get_trading_status():
    """Get trading and balance status from main database."""
    result = {
        "total_balance": None,
        "available_balance": None,
        "drawdown_pct": None,
        "open_positions": 0,
        "pending_intents": 0,
        "recent_trades": 0,
        "last_cycle": None,
    }
    
    db_path = "polybot.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get latest balance snapshot
            cursor.execute("""
                SELECT total_usdc, available_usdc, drawdown_pct, timestamp
                FROM balance_snapshots
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                result["total_balance"] = row[0]
                result["available_balance"] = row[1]
                result["drawdown_pct"] = row[2]
                result["last_cycle"] = row[3]
            
            # Count pending intents
            cursor.execute("""
                SELECT COUNT(*) FROM trade_intents
                WHERE status IN ('PENDING', 'APPROVED')
            """)
            result["pending_intents"] = cursor.fetchone()[0]
            
            # Count recent executed intents (last 24h)
            cursor.execute("""
                SELECT COUNT(*) FROM trade_intents
                WHERE status = 'EXECUTED'
                AND created_at > datetime('now', '-1 day')
            """)
            result["recent_trades"] = cursor.fetchone()[0]
            
            conn.close()
        except Exception as e:
            result["db_error"] = str(e)
    
    return result


def get_log_status():
    """Get recent activity from log file."""
    result = {
        "last_log": None,
        "last_ai_collection": None,
        "errors_24h": 0,
        "signals_generated": 0,
        "bot_running": False,
        "recent_simulations": [],  # Last 5 simulation results
        "recent_errors": [],  # Last 5 error messages
        "recent_training": None,  # Last training completion
        "last_training_start": None,
        "last_training_end": None,
        "training_in_progress": False,
        "signals_today": 0,
        "markets_scanned": 0,
        "last_cycle_markets": 0,
        "arbitrage_scans": 0,
        "recent_activity": [],  # Last 10 notable events
    }
    
    log_path = "live_run.log"
    if os.path.exists(log_path):
        try:
            # Read more of the log for better history
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 200000))  # Last 200KB
                content = f.read().decode('utf-8', errors='ignore')
            
            lines = content.strip().split('\n')
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    ts = data.get("asctime")
                    msg = data.get("message", "")
                    level = data.get("levelname", "")
                    
                    if result["last_log"] is None:
                        result["last_log"] = ts
                        try:
                            log_time = datetime.fromisoformat(ts)
                            if (now - log_time).total_seconds() < 120:
                                result["bot_running"] = True
                        except:
                            pass
                    
                    # Parse timestamp for time-based filtering
                    try:
                        log_time = datetime.fromisoformat(ts)
                    except:
                        continue
                    
                    # AI data collection
                    if "AI data collection" in msg and result["last_ai_collection"] is None:
                        result["last_ai_collection"] = ts
                    
                    # Count errors (last 24h)
                    if level == "ERROR" and (now - log_time).total_seconds() < 86400:
                        result["errors_24h"] += 1
                        if len(result["recent_errors"]) < 5:
                            result["recent_errors"].append({
                                "time": ts,
                                "message": msg[:200]  # Truncate long messages
                            })
                    
                    # Training simulations (SIMULATION: lines)
                    if "SIMULATION:" in msg and len(result["recent_simulations"]) < 5:
                        # Parse: "SIMULATION: 2556 trades, profit=-3847.12%, ..."
                        result["recent_simulations"].append({
                            "time": ts,
                            "message": msg
                        })
                    
                    # Training start/end
                    if "Starting thorough classifier training" in msg:
                        if result["last_training_start"] is None:
                            result["last_training_start"] = ts
                            result["training_in_progress"] = True
                    
                    if "Model training completed" in msg or "Deployed model v" in msg:
                        if result["last_training_end"] is None:
                            result["last_training_end"] = ts
                            result["training_in_progress"] = False
                            result["recent_training"] = ts
                    
                    # Count today's signals
                    if "signal" in msg.lower() and log_time >= today_start:
                        result["signals_today"] += 1
                    
                    # Markets scanned
                    if "Scanned" in msg and "markets" in msg:
                        try:
                            # Extract number from "Scanned X markets"
                            import re
                            match = re.search(r'Scanned (\d+) markets', msg)
                            if match:
                                result["last_cycle_markets"] = int(match.group(1))
                        except:
                            pass
                    
                    # Arbitrage activity
                    if "arbitrage" in msg.lower():
                        result["arbitrage_scans"] += 1
                    
                    # Collect recent notable activity
                    notable_keywords = [
                        "Deployed model", "Training", "Signal", "Arbitrage", 
                        "Error", "Warning", "Collected", "Executing", "Trade"
                    ]
                    if len(result["recent_activity"]) < 15:
                        for kw in notable_keywords:
                            if kw.lower() in msg.lower():
                                result["recent_activity"].append({
                                    "time": ts,
                                    "level": level,
                                    "message": msg[:150]
                                })
                                break
                        
                except:
                    continue
            
            # Check if training is truly in progress (started but not ended)
            if result["last_training_start"] and result["last_training_end"]:
                try:
                    start = datetime.fromisoformat(result["last_training_start"])
                    end = datetime.fromisoformat(result["last_training_end"])
                    result["training_in_progress"] = start > end
                except:
                    result["training_in_progress"] = False
                    
        except Exception as e:
            result["log_error"] = str(e)
    
    return result


def get_system_stats():
    """Get system resource statistics from the monitor."""
    stats_path = "data/system_stats.json"
    
    if os.path.exists(stats_path):
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    # Try to get live stats if file doesn't exist yet
    try:
        import psutil
        # Return basic live stats
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "uptime_hours": 0,
            "cpu": {
                "percent": {"current": cpu, "avg": cpu, "min": cpu, "max": cpu, "samples": 1},
                "count": psutil.cpu_count(),
                "load_avg": {"current": 0, "avg": 0, "min": 0, "max": 0, "samples": 0},
            },
            "memory": {
                "percent": {"current": mem.percent, "avg": mem.percent, "min": mem.percent, "max": mem.percent, "samples": 1},
                "total_gb": mem.total / (1024**3),
            },
            "disk": {
                "percent": {"current": disk.percent, "avg": disk.percent, "min": disk.percent, "max": disk.percent, "samples": 1},
                "total_gb": disk.total / (1024**3),
            },
            "process": {"cpu_percent": {"samples": 0}, "memory_mb": {"samples": 0}},
            "training": {"cpu_percent": {"samples": 0}, "avg_duration_seconds": 0, "training_count": 0},
            "_live": True,
        }
    except ImportError:
        return None
    except Exception:
        return None


def get_live_performance():
    """Get actual live prediction performance from CURRENT MODEL ONLY."""
    result = {
        "total_predictions": 0,
        "evaluated_predictions": 0,
        "correct_direction": 0,
        "profitable_predictions": 0,
        "total_pnl": 0.0,
        "by_category": {},
        "model_created": None,
    }
    
    # First, get the current model's creation time
    model_created = None
    state_path = "data/ai_models/trainer_state.json"
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            model_created = state.get("created_at")
            result["model_created"] = model_created
        except:
            pass

    # Check category stats database for actual prediction outcomes
    db_path = "data/category_stats.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            cursor = conn.cursor()

            # Get overall stats from predictions table
            # Only count predictions from AFTER current model was created
            if model_created:
                cursor.execute("""
                    SELECT COUNT(*),
                           SUM(was_correct),
                           SUM(was_profitable),
                           SUM(pnl)
                    FROM predictions
                    WHERE recorded_at >= ?
                """, (model_created,))
            else:
                cursor.execute("""
                    SELECT COUNT(*),
                           SUM(was_correct),
                           SUM(was_profitable),
                           SUM(pnl)
                    FROM predictions
                """)
            row = cursor.fetchone()
            if row and row[0]:
                result["total_predictions"] = row[0]
                result["correct_direction"] = row[1] or 0
                result["profitable_predictions"] = row[2] or 0
                result["total_pnl"] = row[3] or 0.0

            conn.close()
        except Exception as e:
            result["error"] = str(e)
    
    # Also check training_examples for evaluated predictions
    ai_db_path = "data/ai_training.db"
    if os.path.exists(ai_db_path):
        try:
            conn = sqlite3.connect(ai_db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Only count predictions made AFTER current model was created
            # This filters out old predictions from previous (worse) models
            if model_created:
                # Count evaluated predictions from current model
                cursor.execute("""
                    SELECT COUNT(*) FROM training_examples 
                    WHERE prediction_evaluated = 1 
                      AND predicted_change IS NOT NULL
                      AND created_at >= ?
                """, (model_created,))
                row = cursor.fetchone()
                result["evaluated_predictions"] = row[0] if row else 0
                
                # Calculate actual accuracy from current model's predictions only
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN (predicted_change > 0 AND price_change_24h > 0) 
                                 OR (predicted_change < 0 AND price_change_24h < 0) 
                                 OR (predicted_change = 0 AND price_change_24h = 0)
                            THEN 1 ELSE 0 END) as correct_direction,
                        SUM(CASE WHEN (predicted_change > 0.01 AND price_change_24h > 0.02)
                                 OR (predicted_change < -0.01 AND price_change_24h < -0.02)
                            THEN 1 ELSE 0 END) as profitable
                    FROM training_examples
                    WHERE prediction_evaluated = 1 
                      AND predicted_change IS NOT NULL 
                      AND price_change_24h IS NOT NULL
                      AND created_at >= ?
                """, (model_created,))
            else:
                # Fallback: all predictions
                cursor.execute("""
                    SELECT COUNT(*) FROM training_examples 
                    WHERE prediction_evaluated = 1 AND predicted_change IS NOT NULL
                """)
                row = cursor.fetchone()
                result["evaluated_predictions"] = row[0] if row else 0
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN (predicted_change > 0 AND price_change_24h > 0) 
                                 OR (predicted_change < 0 AND price_change_24h < 0) 
                                 OR (predicted_change = 0 AND price_change_24h = 0)
                            THEN 1 ELSE 0 END) as correct_direction,
                        SUM(CASE WHEN (predicted_change > 0.01 AND price_change_24h > 0.02)
                                 OR (predicted_change < -0.01 AND price_change_24h < -0.02)
                            THEN 1 ELSE 0 END) as profitable
                    FROM training_examples
                    WHERE prediction_evaluated = 1 
                      AND predicted_change IS NOT NULL 
                      AND price_change_24h IS NOT NULL
                """)
            
            row = cursor.fetchone()
            if row and row[0] and row[0] > 0:
                result["simulated_total"] = row[0]
                result["simulated_correct"] = row[1] or 0
                result["simulated_profitable"] = row[2] or 0
                result["simulated_accuracy"] = (row[1] or 0) / row[0]
                result["simulated_profitable_pct"] = (row[2] or 0) / row[0]
            
            conn.close()
        except Exception as e:
            if "error" not in result:
                result["error"] = str(e)
    
    return result


def get_category_stats():
    """Get market category performance stats."""
    db_path = "data/category_stats.db"
    
    if not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        
        # Get all category stats
        cursor.execute("""
            SELECT category, total_predictions, correct_predictions, profitable_predictions,
                   total_pnl, is_avoided
            FROM category_stats
            ORDER BY total_predictions DESC
        """)
        
        categories = []
        for row in cursor.fetchall():
            total = row[1]
            if total > 0:
                accuracy = row[2] / total
                profitable_acc = row[3] / total
            else:
                accuracy = 0.5
                profitable_acc = 0.5
            
            categories.append({
                "category": row[0],
                "total": total,
                "accuracy": accuracy,
                "profitable_acc": profitable_acc,
                "avg_pnl": row[4] / total if total > 0 else 0,
                "is_avoided": bool(row[5]),
            })
        
        conn.close()
        return categories
        
    except Exception as e:
        return {"error": str(e)}


def get_system_recommendation(stats: dict) -> dict:
    """Get upgrade recommendation based on stats."""
    if not stats or "cpu" not in stats:
        return None
    
    recommendations = []
    status = "healthy"
    
    cpu = stats.get("cpu", {}).get("percent", {})
    mem = stats.get("memory", {}).get("percent", {})
    disk = stats.get("disk", {}).get("percent", {})
    load = stats.get("cpu", {}).get("load_avg", {})
    cpu_count = stats.get("cpu", {}).get("count", 1)
    training = stats.get("training", {})
    
    # Check CPU
    if cpu.get("avg", 0) > 80:
        recommendations.append("CPU avg >80%. Consider upgrading.")
        status = "needs_upgrade"
    elif cpu.get("max", 0) > 95 and cpu.get("samples", 0) > 100:
        recommendations.append("CPU occasionally maxes out.")
        if status == "healthy":
            status = "monitor"
    
    # Check Memory
    if mem.get("avg", 0) > 85:
        recommendations.append("Memory avg >85%. Add more RAM.")
        status = "needs_upgrade"
    elif mem.get("max", 0) > 95:
        recommendations.append("Memory occasionally maxes out.")
        if status == "healthy":
            status = "monitor"
    
    # Check Disk
    if disk.get("current", 0) > 90:
        recommendations.append("Disk >90%. Free space needed.")
        status = "needs_upgrade"
    elif disk.get("current", 0) > 80:
        recommendations.append("Disk >80%. Consider cleanup.")
        if status == "healthy":
            status = "monitor"
    
    # Check load vs cores
    if load.get("avg", 0) > cpu_count and cpu_count > 0:
        recommendations.append(f"Load ({load.get('avg', 0):.1f}) > cores ({cpu_count}). Overloaded.")
        status = "needs_upgrade"
    
    # Check training performance
    avg_train_time = training.get("avg_duration_seconds", 0)
    if avg_train_time > 300:
        recommendations.append(f"Training takes {avg_train_time/60:.1f}m. More CPU helps.")
    
    if not recommendations:
        recommendations.append("System resources healthy. No upgrade needed.")
    
    return {
        "status": status,
        "status_icon": "🟢" if status == "healthy" else "🟡" if status == "monitor" else "🔴",
        "recommendations": recommendations,
    }


def get_config_status():
    """Get current configuration."""
    result = {
        "strategy_mode": "unknown",
        "placing_orders": "unknown",
        "dry_run": "unknown",
    }
    
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip().upper()
                    value = value.strip().strip('"').strip("'")
                    
                    if key == "POLYBOT_STRATEGY_MODE":
                        result["strategy_mode"] = value
                    elif key == "POLYBOT_PLACING_ORDERS":
                        result["placing_orders"] = value.lower()
                    elif key == "POLYBOT_DRY_RUN":
                        result["dry_run"] = value.lower()
        except:
            pass
    
    return result


def get_moe_status():
    """Get Mixture of Experts status with full state machine support."""
    result = {
        "has_moe": False,
        "total_experts": 0,
        "active_experts": 0,
        "suspended_experts": 0,
        "probation_experts": 0,
        "deprecated_experts": 0,
        "untrained_experts": 0,
        "rollback_experts": 0,
        "state_counts": {},
        "total_profit_pct": 0.0,
        "total_trades": 0,
        "experts": [],
        "suspended": [],
        "probation": [],
        "deprecated": [],
        "gating": None,
        "best_expert": None,
        "candidates": [],
    }
    
    pool_state_path = "data/moe_models/pool_state.json"
    if os.path.exists(pool_state_path):
        result["has_moe"] = True
        try:
            with open(pool_state_path, "r") as f:
                pool_data = json.load(f)
            
            state = pool_data.get("state", {})
            expert_ids = pool_data.get("expert_ids", [])
            
            result["total_experts"] = len(expert_ids)
            result["total_profit_pct"] = state.get("total_profit_pct", 0)
            result["total_trades"] = state.get("total_trades", 0)
            result["best_expert_id"] = state.get("best_expert_id")
            result["last_training"] = state.get("last_training")
            result["training_cycles"] = state.get("total_training_cycles", 0)
            
            # State counts initialization
            state_counts = {
                "untrained": 0,
                "probation": 0,
                "active": 0,
                "suspended": 0,
                "rollback": 0,
                "deprecated": 0,
            }
            
            # Load individual expert data
            experts_dir = "data/moe_models/experts"
            for expert_id in expert_ids:
                meta_path = os.path.join(experts_dir, f"{expert_id}.meta.pkl")
                version_path = os.path.join(experts_dir, f"{expert_id}.versions.json")
                
                if os.path.exists(meta_path):
                    try:
                        import pickle
                        with open(meta_path, "rb") as f:
                            expert_data = pickle.load(f)
                        
                        metrics = expert_data.get("metrics", {})
                        
                        # Load versioning state if exists
                        expert_state = "untrained"  # Default
                        current_version = None
                        confidence_mult = expert_data.get("confidence_multiplier", 0.5)
                        trend = 0.0
                        
                        if os.path.exists(version_path):
                            try:
                                with open(version_path, "r") as f:
                                    version_data = json.load(f)
                                expert_state = version_data.get("state", "untrained")
                                current_version = version_data.get("current_version_id", None)
                            except:
                                pass
                        
                        # Calculate trend from training history
                        training_history = expert_data.get("training_history", [])
                        if len(training_history) >= 5:
                            recent = training_history[-5:]
                            profits = [r.get("profit_pct", 0) for r in recent]
                            trend = (profits[-1] - profits[0]) / 5
                        
                        expert_info = {
                            "expert_id": expert_id,
                            "domain": expert_data.get("domain", expert_id),
                            "expert_type": expert_data.get("expert_type", "unknown"),
                            "state": expert_state,
                            "is_active": expert_state in ["active", "rollback"],
                            "is_deprecated": expert_state == "deprecated",
                            "current_version": current_version,
                            "confidence_mult": confidence_mult,
                            "trend": trend,
                            "profit_pct": metrics.get("simulated_profit_pct", 0),
                            "num_trades": metrics.get("simulated_num_trades", 0),
                            "win_rate": metrics.get("simulated_win_rate", 0),
                            "profit_factor": metrics.get("simulated_profit_factor", 0),
                            "sharpe": metrics.get("simulated_sharpe", 0),
                        }
                        
                        # Increment state count
                        if expert_state in state_counts:
                            state_counts[expert_state] += 1
                        
                        # Categorize by state
                        if expert_state == "deprecated":
                            result["deprecated"].append(expert_info)
                            result["deprecated_experts"] += 1
                        elif expert_state == "suspended":
                            result["suspended"].append(expert_info)
                            result["suspended_experts"] += 1
                        elif expert_state == "probation":
                            result["probation"].append(expert_info)
                            result["probation_experts"] += 1
                        elif expert_state in ["active", "rollback"]:
                            result["experts"].append(expert_info)
                            result["active_experts"] += 1
                            if expert_state == "rollback":
                                result["rollback_experts"] += 1
                        else:
                            result["untrained_experts"] += 1
                    except Exception as e:
                        pass
            
            result["state_counts"] = state_counts
            
            # Sort experts by profit
            result["experts"].sort(key=lambda x: x["profit_pct"], reverse=True)
            result["suspended"].sort(key=lambda x: x["profit_pct"], reverse=True)
            
            # Find best expert
            if result["experts"]:
                result["best_expert"] = result["experts"][0]
            
            # Load gating info
            gating_meta_path = "data/moe_models/gating.meta.pkl"
            if os.path.exists(gating_meta_path):
                try:
                    import pickle
                    with open(gating_meta_path, "rb") as f:
                        gating_data = pickle.load(f)
                    result["gating"] = {
                        "n_experts": len(gating_data.get("expert_ids", [])),
                        "routing_accuracy": gating_data.get("metrics", {}).get("routing_accuracy", 0),
                        "profit_vs_random": gating_data.get("metrics", {}).get("profit_vs_random", 0),
                    }
                except:
                    pass
                    
        except Exception as e:
            result["error"] = str(e)
    
    return result


def get_orchestrator_status():
    """Get AI orchestrator status (training schedule)."""
    result = {
        "last_training_time": None,
        "next_training": None,
        "training_interval_hours": 6,
    }
    
    state_path = "data/ai_models/orchestrator_state.json"
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            result["last_training_time"] = state.get("last_training_time")
            
            # Calculate next training
            if result["last_training_time"]:
                last = datetime.fromisoformat(result["last_training_time"].replace('Z', '+00:00'))
                if last.tzinfo:
                    last = last.replace(tzinfo=None)
                next_train = last + timedelta(hours=result["training_interval_hours"])
                result["next_training"] = next_train.isoformat()
        except:
            pass
    
    return result


def main():
    print("\n" + "=" * 70)
    print("                    POLYB0T STATUS DASHBOARD")
    print("                    " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # Configuration
    config = get_config_status()
    placing = config['placing_orders'].lower()
    is_live = placing == 'true' and config['dry_run'].lower() != 'true'
    mode_icon = "🔴 LIVE" if is_live else "🟡 DRY RUN" if placing == 'true' else "⚪ OFF"
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Mode:             {mode_icon}")
    print(f"   Strategy:         {config['strategy_mode'].upper()}")
    print(f"   Placing Orders:   {config['placing_orders']}")
    print(f"   Dry Run:          {config['dry_run']}")
    
    # Log/Runtime Status
    log = get_log_status()
    status_icon = "🟢" if log["bot_running"] else "🔴"
    print(f"\n{status_icon} BOT STATUS:")
    print(f"   Running:          {'YES' if log['bot_running'] else 'NO'}")
    print(f"   Last Activity:    {format_time_ago(log['last_log'])}")
    print(f"   Last AI Collect:  {format_time_ago(log['last_ai_collection'])}")
    
    # Training status
    if log.get("training_in_progress"):
        print(f"   🔄 Training:       IN PROGRESS (started {format_time_ago(log['last_training_start'])})")
    elif log.get("recent_training"):
        print(f"   Last Training:    {format_time_ago(log['recent_training'])}")
    
    if log["errors_24h"] > 0:
        print(f"   ⚠️  Errors (24h):  {log['errors_24h']}")
    
    # Trading Status
    trading = get_trading_status()
    print(f"\n💰 TRADING STATUS:")
    print(f"   Total Balance:    {format_usd(trading['total_balance'])}")
    print(f"   Available:        {format_usd(trading['available_balance'])}")
    if trading["drawdown_pct"]:
        dd_icon = "🔴" if trading["drawdown_pct"] > 20 else "🟡" if trading["drawdown_pct"] > 10 else "🟢"
        print(f"   Drawdown:         {dd_icon} {trading['drawdown_pct']:.1f}%")
    print(f"   Pending Intents:  {trading['pending_intents']}")
    print(f"   Trades (24h):     {trading['recent_trades']}")
    
    # AI Status
    ai = get_ai_status()
    print(f"\n🤖 AI MODEL STATUS:")
    
    if ai["has_model"]:
        print(f"   ✅ TRAINED MODEL v{ai['model_version']}")
        print(f"   Created:          {format_time_ago(ai['model_created'])}")
        print(f"   Training Size:    {ai['training_examples']:,} examples")
        
        metrics = ai["metrics"]
        prof = metrics.get('profitable_accuracy', 0)
        dir_acc = metrics.get('directional_accuracy', 0)
        n_models = metrics.get('n_models_ensemble', 0)
        n_features = metrics.get('n_features_used', 0)
        cv_std = metrics.get('cv_std', 0)
        avg_conf = metrics.get('avg_confidence', 0)
        conf_trade_pct = metrics.get('confident_trade_pct', 0)
        
        # NEW: Profitability simulation metrics (what actually matters!)
        sim_profit = metrics.get('simulated_profit_pct', 0)
        sim_trades = metrics.get('simulated_num_trades', 0)
        sim_win_rate = metrics.get('simulated_win_rate', 0)
        sim_pf = metrics.get('simulated_profit_factor', 0)
        sim_sharpe = metrics.get('simulated_sharpe', 0)
        sim_dd = metrics.get('simulated_max_drawdown', 0)
        
        # PRIMARY: Profitability (what actually matters)
        if sim_trades > 0:
            profit_icon = "✅" if sim_profit > 0.05 else "🟢" if sim_profit > 0 else "🟡" if sim_profit > -0.05 else "❌"
            print(f"\n   💰 SIMULATED PROFITABILITY (PRIMARY METRIC):")
            print(f"   Total Profit:     {profit_icon} {sim_profit:+.2%}")
            print(f"   Trades Simulated: {sim_trades}")
            print(f"   Win Rate:         {sim_win_rate:.1%}")
            print(f"   Profit Factor:    {sim_pf:.2f}x" + (" ✅" if sim_pf > 1.5 else " 🟡" if sim_pf > 1 else " ❌"))
            print(f"   Sharpe Ratio:     {sim_sharpe:.2f}" + (" ✅" if sim_sharpe > 1 else " 🟡" if sim_sharpe > 0 else ""))
            print(f"   Max Drawdown:     {sim_dd:.2%}")
        
        # Secondary: Old accuracy metrics
        prof_icon = "✅" if prof > 0.55 else "🟡" if prof > 0.50 else "❌"
        print(f"\n   📊 ACCURACY METRICS (secondary):")
        print(f"   Profitable Acc:   {prof_icon} {format_pct(prof)}")
        print(f"   Directional Acc:  {format_pct(dir_acc)}")
        
        # Show confidence metrics if available
        if avg_conf > 0 or conf_trade_pct > 0:
            print(f"   Avg Confidence:   {format_pct(avg_conf)}")
            print(f"   Confident Trades: {format_pct(conf_trade_pct)}")
        
        if n_models > 0:
            print(f"\n   🧠 MODEL DETAILS:")
            print(f"   Ensemble:         {n_models} classifiers")
            print(f"   Features Used:    {n_features}")
            if cv_std > 0:
                consistency = "High" if cv_std < 0.05 else "Medium" if cv_std < 0.1 else "Low"
                print(f"   CV Consistency:   {consistency} (std={cv_std:.3f})")
        
        # Live/Actual Performance
        live = get_live_performance()
        has_live_data = (
            live.get("total_predictions", 0) > 0 or 
            live.get("simulated_total", 0) > 0
        )
        
        if has_live_data:
            print(f"\n   📈 ACTUAL PERFORMANCE (current model only):")
            
            # Show simulated prediction results
            sim_total = live.get("simulated_total", 0)
            if sim_total > 0:
                sim_acc = live.get("simulated_accuracy", 0)
                sim_prof = live.get("simulated_profitable_pct", 0)
                sim_prof_icon = "✅" if sim_prof > 0.55 else "🟡" if sim_prof > 0.50 else "❌"
                
                print(f"   Predictions Made: {sim_total:,}")
                print(f"   Actual Dir Acc:   {sim_acc:.1%}")
                print(f"   Actual Profit:    {sim_prof_icon} {sim_prof:.1%}")
                
                # Compare to validation
                if prof > 0:
                    diff = sim_prof - prof
                    diff_icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                    print(f"   vs Validation:    {diff_icon} {diff:+.1%}")
            else:
                # Model is new, no evaluated predictions yet
                print(f"   ⏳ Model v{ai['model_version']} is new")
                print(f"   Waiting for predictions to be evaluated...")
                print(f"   (Takes 24-48h after model deployment)")
            
            # Show category tracker results
            cat_total = live.get("total_predictions", 0)
            if cat_total > 0:
                cat_correct = live.get("correct_direction", 0)
                cat_profitable = live.get("profitable_predictions", 0)
                cat_pnl = live.get("total_pnl", 0)
                
                print(f"\n   From Category Tracker:")
                print(f"   Recorded:         {cat_total:,} predictions")
                print(f"   Correct Dir:      {cat_correct:,} ({cat_correct/cat_total:.1%})")
                print(f"   Profitable:       {cat_profitable:,} ({cat_profitable/cat_total:.1%})")
                print(f"   Simulated P&L:    {cat_pnl:+.2%}")
        else:
            print(f"\n   📈 ACTUAL PERFORMANCE:")
            print(f"   ⏳ Waiting for predictions to be evaluated...")
            print(f"   (Takes ~24h after predictions are made)")
    else:
        print(f"   ⏳ NO TRAINED MODEL YET")
        
    print(f"\n   📦 TRAINING DATA:")
    print(f"   Total Examples:   {ai['total_examples']:,}")
    print(f"   With 24h Label:   {ai['labeled_examples']:,}  (usable for training)")
    print(f"   Resolved:         {ai.get('resolved_examples', 0):,}  (market closed)")
    print(f"   Partial Labels:   {ai['partial_labels']:,}")
    print(f"   Snapshots:        {ai['snapshots']:,}")
    print(f"   Price History:    {ai['price_points']:,}")
    if ai.get("db_error"):
        print(f"   ⚠️  DB Error:      {ai['db_error']}")
    
    # ==========================================
    # MIXTURE OF EXPERTS STATUS (24 experts with state machine)
    # ==========================================
    moe = get_moe_status()
    if moe["has_moe"]:
        print(f"\n🧠 MIXTURE OF EXPERTS (24 experts):")
        
        # Show state summary
        state_counts = moe.get("state_counts", {})
        print(f"   ┌─────────────────────────────────────────┐")
        print(f"   │  EXPERT STATES:                         │")
        print(f"   │    ✅ Active:     {moe['active_experts']:3d}  (trading enabled)   │")
        print(f"   │    🔄 Probation:  {moe.get('probation_experts', 0):3d}  (proving itself)   │")
        print(f"   │    ⏸️  Suspended:  {moe.get('suspended_experts', 0):3d}  (training only)    │")
        print(f"   │    ⏪ Rollback:   {moe.get('rollback_experts', 0):3d}  (using old version)│")
        print(f"   │    ❌ Deprecated: {moe['deprecated_experts']:3d}  (permanently off)  │")
        print(f"   │    ⏳ Untrained:  {moe.get('untrained_experts', 0):3d}  (needs data)       │")
        print(f"   └─────────────────────────────────────────┘")
        
        if moe.get("training_cycles", 0) > 0:
            print(f"\n   Training Cycles:  {moe['training_cycles']}")
            if moe.get("last_training"):
                print(f"   Last Training:    {format_time_ago(moe['last_training'])}")
        
        # Show active experts (top performers)
        if moe["experts"]:
            print(f"\n   ✅ ACTIVE EXPERTS (trading enabled):")
            for expert in moe["experts"][:6]:
                profit = expert["profit_pct"]
                trades = expert["num_trades"]
                win_rate = expert["win_rate"]
                version = expert.get("current_version", "?")
                conf = expert.get("confidence_mult", 0.5)
                trend = expert.get("trend", 0)
                
                # Trend indicator
                trend_icon = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➡️"
                
                if trades == 0:
                    status = "⏳"
                    detail = "not trained"
                elif profit > 0.05:
                    status = "✅"
                    detail = f"{profit:+.1%}, {trades} trades, {win_rate:.0%} win"
                elif profit > 0:
                    status = "🟢"
                    detail = f"{profit:+.1%}, {trades} trades, {win_rate:.0%} win"
                elif profit > -0.05:
                    status = "🟡"
                    detail = f"{profit:+.1%}, {trades} trades, {win_rate:.0%} win"
                else:
                    status = "❌"
                    detail = f"{profit:+.1%}, {trades} trades, {win_rate:.0%} win"
                
                domain = expert["domain"]
                print(f"      {status} {domain} [v{version}] {trend_icon} {detail} (conf:{conf:.1f})")
        
        # Show suspended experts (still training)
        if moe.get("suspended"):
            print(f"\n   ⏸️  SUSPENDED EXPERTS (training, not trading):")
            for expert in moe["suspended"][:4]:
                profit = expert["profit_pct"]
                trades = expert["num_trades"]
                trend = expert.get("trend", 0)
                trend_icon = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➡️"
                print(f"      {expert['domain']}: {profit:+.1%} ({trades} trades) {trend_icon}")
        
        # Show probation experts
        if moe.get("probation"):
            print(f"\n   🔄 PROBATION EXPERTS (new, proving performance):")
            for expert in moe["probation"][:4]:
                profit = expert["profit_pct"]
                trades = expert["num_trades"]
                print(f"      {expert['domain']}: {profit:+.1%} ({trades} trades)")
        
        # Show deprecated experts
        if moe["deprecated"]:
            print(f"\n   ❌ DEPRECATED EXPERTS (permanently disabled):")
            for expert in moe["deprecated"][:3]:
                print(f"      {expert['domain']}: {expert['profit_pct']:+.1%}")
        
        # Gating network info
        if moe.get("gating"):
            gating = moe["gating"]
            routing_acc = gating.get("routing_accuracy", 0)
            profit_vs_random = gating.get("profit_vs_random", 0)
            
            print(f"\n   📡 GATING NETWORK:")
            if routing_acc > 0:
                acc_icon = "✅" if routing_acc > 0.7 else "🟡" if routing_acc > 0.5 else "❌"
                print(f"      Routing Accuracy: {acc_icon} {routing_acc:.1%}")
                print(f"      vs Random:        {profit_vs_random:+.2%}")
            else:
                print(f"      Status:           ⏳ Not trained yet")
    else:
        print(f"\n🧠 MIXTURE OF EXPERTS:")
        print(f"   Status:           ⏳ Initializing...")
        print(f"   (Will be created on first training run)")
        print(f"   (24 experts: 10 category, 3 risk, 3 time, 3 volume, 3 volatility, 2 timing)")
    
    # Category Performance
    categories = get_category_stats()
    if categories and isinstance(categories, list) and len(categories) > 0:
        print(f"\n   📂 CATEGORY PERFORMANCE:")
        
        # Sort by total predictions
        sorted_cats = sorted(categories, key=lambda x: x["total"], reverse=True)
        
        # Show top 5 categories
        for cat in sorted_cats[:5]:
            if cat["total"] < 5:
                continue
            
            status_icon = "🚫" if cat["is_avoided"] else ("✅" if cat["profitable_acc"] > 0.50 else "🟡")
            print(
                f"   {status_icon} {cat['category']}: "
                f"{cat['profitable_acc']:.0%} profit ({cat['total']} predictions)"
            )
        
        # Show avoided categories
        avoided = [c for c in categories if c["is_avoided"]]
        if avoided:
            print(f"\n   ⚠️  AVOIDED CATEGORIES: {len(avoided)}")
            for cat in avoided:
                print(f"      • {cat['category']} ({cat['profitable_acc']:.0%})")
    
    # Arbitrage Scanner Info
    print(f"\n   ⚡ ARBITRAGE SCANNER:")
    arb_state_path = "data/arbitrage_state.json"
    if os.path.exists(arb_state_path):
        try:
            with open(arb_state_path, "r") as f:
                arb_state = json.load(f)
            
            is_disabled = arb_state.get("is_disabled", False)
            stats = arb_state.get("stats", {})
            
            if is_disabled:
                print(f"   Status:           🔴 DISABLED")
                print(f"   Reason:           {arb_state.get('disable_reason', 'Unknown')}")
                print(f"   (Run 'poetry run python -c \"from polyb0t.services.arbitrage_scanner import get_arbitrage_scanner; get_arbitrage_scanner().enable()\"' to re-enable)")
            else:
                total_trades = stats.get("total_trades", 0)
                win_rate = stats.get("win_rate", 0)
                total_profit = stats.get("total_profit", 0)
                
                if total_trades > 0:
                    status_icon = "✅" if win_rate >= 0.6 else "🟡" if win_rate >= 0.5 else "❌"
                    print(f"   Status:           {status_icon} Active")
                    print(f"   Total Trades:     {total_trades}")
                    print(f"   Win Rate:         {win_rate:.1%}")
                    print(f"   Total Profit:     {total_profit:+.2%}")
                else:
                    print(f"   Status:           ✅ Active (no trades yet)")
                    
        except Exception as e:
            print(f"   Status:           ✅ Active (no trades yet)")
    else:
        print(f"   Status:           ✅ Active (no trades yet)")
    
    # Check API keys
    print(f"\n   API Configuration:")
    # Check both prefixed and non-prefixed env vars
    news_api_key = os.environ.get("POLYBOT_NEWSAPI_KEY", "") or os.environ.get("NEWSAPI_KEY", "")
    openai_key = os.environ.get("POLYBOT_OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    
    # Also try reading from .env file directly
    if not openai_key or not news_api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("POLYBOT_OPENAI_API_KEY=") and not openai_key:
                            openai_key = line.split("=", 1)[1].strip()
                        elif line.startswith("POLYBOT_NEWSAPI_KEY=") and not news_api_key:
                            news_api_key = line.split("=", 1)[1].strip()
            except:
                pass
    
    if openai_key:
        print(f"   OpenAI GPT-5.2:   ✅ Configured")
        print(f"   Model:            gpt-5.2 (latest)")
        print(f"   Cost:             ~$0.002/analysis")
    else:
        print(f"   OpenAI GPT-5.2:   ⚠️  Not configured")
        print(f"   (Set OPENAI_API_KEY for intelligent news analysis)")
    
    if news_api_key:
        print(f"   NewsAPI:          ✅ Configured (1000/day limit)")
    else:
        print(f"   NewsAPI:          ⚠️  Not configured")
        print(f"   (Set NEWSAPI_KEY for headline fetching)")
    
    print(f"\n   Safety Features:")
    print(f"   • Auto-disables if win rate < 60% (after 10+ trades)")
    print(f"   • Auto-disables if cumulative loss > 5%")
    print(f"   • Requires GPT-5.2 to confirm news meaning")
    print(f"   • 4-hour cache on LLM analysis to reduce costs")
    
    # Resolution Predictor Status
    resolution_state_path = "data/resolution_models/resolution_state.json"
    if os.path.exists(resolution_state_path):
        try:
            with open(resolution_state_path, "r") as f:
                res_state = json.load(f)
            res_metrics = res_state.get("metrics", {})
            print(f"\n   🎯 RESOLUTION PREDICTOR:")
            print(f"   Model Version:    v{res_state.get('version', 1)}")
            print(f"   Markets Trained:  {res_state.get('n_markets', 0):,}")
            print(f"   Accuracy:         {res_metrics.get('accuracy', 0):.1%}")
            print(f"   Confident Acc:    {res_metrics.get('confident_accuracy', 0):.1%}")
            print(f"   Calibration Err:  {res_metrics.get('calibration_error', 0):.3f}")
        except Exception as e:
            pass
    else:
        resolved = ai.get('resolved_examples', 0)
        if resolved >= 100:
            print(f"\n   🎯 RESOLUTION PREDICTOR:")
            print(f"   Status: Ready to train ({resolved} resolved markets)")
        else:
            print(f"\n   🎯 RESOLUTION PREDICTOR:")
            print(f"   Status: Need {100 - resolved} more resolved markets")
    
    # Training estimate
    min_examples = 500
    if not ai["has_model"] and ai["labeled_examples"] < min_examples:
        needed = min_examples - ai["labeled_examples"]
        hours = needed / 60  # ~60 per hour
        print(f"\n   ⏱️  TRAINING ETA:")
        print(f"   Need:             {needed} more labeled examples")
        print(f"   Estimated:        ~{hours:.1f} hours")
    
    # Storage
    print(f"\n💾 STORAGE:")
    print(f"   AI Database:      {ai['db_size_mb']:.1f} MB")
    print(f"   Max Allowed:      140 GB ({ai['db_size_mb']/1024/140*100:.2f}% used)")
    
    # System Resources
    sys_stats = get_system_stats()
    if sys_stats:
        print(f"\n🖥️  SYSTEM RESOURCES:")
        cpu = sys_stats.get("cpu", {})
        mem = sys_stats.get("memory", {})
        disk = sys_stats.get("disk", {})
        proc = sys_stats.get("process", {})
        training = sys_stats.get("training", {})
        
        cpu_pct = cpu.get("percent", {})
        mem_pct = mem.get("percent", {})
        
        print(f"   Uptime:           {sys_stats.get('uptime_hours', 0):.1f} hours")
        print(f"   CPU Cores:        {cpu.get('count', 'N/A')}")
        
        if cpu_pct.get("samples", 0) > 0:
            print(f"\n   📊 CPU USAGE:")
            print(f"   Current:          {cpu_pct.get('current', 0):.1f}%")
            print(f"   Average:          {cpu_pct.get('avg', 0):.1f}%")
            print(f"   Min / Max:        {cpu_pct.get('min', 0):.1f}% / {cpu_pct.get('max', 0):.1f}%")
            
            load = cpu.get("load_avg", {})
            if load.get("current", 0) > 0:
                print(f"   Load Average:     {load.get('current', 0):.2f}")
        
        if mem_pct.get("samples", 0) > 0:
            print(f"\n   📊 MEMORY USAGE:")
            print(f"   Total:            {mem.get('total_gb', 0):.1f} GB")
            print(f"   Current:          {mem_pct.get('current', 0):.1f}%")
            print(f"   Average:          {mem_pct.get('avg', 0):.1f}%")
            print(f"   Min / Max:        {mem_pct.get('min', 0):.1f}% / {mem_pct.get('max', 0):.1f}%")
        
        disk_pct = disk.get("percent", {})
        if disk_pct.get("samples", 0) > 0:
            print(f"\n   📊 DISK USAGE:")
            print(f"   Total:            {disk.get('total_gb', 0):.1f} GB")
            print(f"   Current:          {disk_pct.get('current', 0):.1f}%")
        
        proc_cpu = proc.get("cpu_percent", {})
        proc_mem = proc.get("memory_mb", {})
        if proc_cpu.get("samples", 0) > 0:
            print(f"\n   📊 BOT PROCESS:")
            print(f"   CPU (avg):        {proc_cpu.get('avg', 0):.1f}%")
            print(f"   Memory (avg):     {proc_mem.get('avg', 0):.1f} MB")
        
        train_cpu = training.get("cpu_percent", {})
        if train_cpu.get("samples", 0) > 0:
            print(f"\n   📊 DURING TRAINING:")
            print(f"   CPU (avg):        {train_cpu.get('avg', 0):.1f}%")
            print(f"   Avg Duration:     {training.get('avg_duration_seconds', 0):.1f}s")
            print(f"   Training Runs:    {training.get('training_count', 0)}")
        
        # Recommendation
        rec = get_system_recommendation(sys_stats)
        if rec:
            print(f"\n   {rec['status_icon']} SYSTEM STATUS: {rec['status'].upper()}")
            for r in rec.get("recommendations", [])[:3]:  # Top 3 recommendations
                print(f"   • {r}")
    
    # ==========================================
    # NEW: Recent Simulation Results (from logs)
    # ==========================================
    if log.get("recent_simulations"):
        print(f"\n📊 RECENT TRAINING SIMULATIONS:")
        for sim in log["recent_simulations"][:5]:
            # Parse: "SIMULATION: 2556 trades, profit=+5.2%, win_rate=48.3%, ..."
            msg = sim["message"]
            time_ago = format_time_ago(sim["time"])
            
            # Extract key metrics
            import re
            trades_match = re.search(r'(\d+) trades', msg)
            profit_match = re.search(r'profit=([+-]?\d+\.?\d*)%', msg)
            win_match = re.search(r'win_rate=(\d+\.?\d*)%', msg)
            pf_match = re.search(r'profit_factor=(\d+\.?\d*)', msg)
            
            trades = trades_match.group(1) if trades_match else "?"
            profit = profit_match.group(1) if profit_match else "?"
            win_rate = win_match.group(1) if win_match else "?"
            pf = pf_match.group(1) if pf_match else "?"
            
            # Determine emoji based on profit
            try:
                p = float(profit)
                emoji = "✅" if p > 5 else "🟢" if p > 0 else "🟡" if p > -5 else "❌"
            except:
                emoji = "•"
            
            print(f"   {emoji} {time_ago}: {trades} trades, profit={profit}%, win={win_rate}%, PF={pf}")
    
    # ==========================================
    # NEW: Training Schedule
    # ==========================================
    orch = get_orchestrator_status()
    print(f"\n⏰ TRAINING SCHEDULE:")
    print(f"   Interval:         Every 6 hours")
    if orch["last_training_time"]:
        print(f"   Last Training:    {format_time_ago(orch['last_training_time'])}")
    if orch["next_training"]:
        next_time = datetime.fromisoformat(orch["next_training"])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if next_time > now:
            delta = next_time - now
            hours = delta.total_seconds() / 3600
            print(f"   Next Training:    In {hours:.1f} hours")
        else:
            print(f"   Next Training:    Overdue (should run soon)")
    
    # ==========================================
    # NEW: Recent Errors (if any)
    # ==========================================
    if log.get("recent_errors"):
        print(f"\n⚠️  RECENT ERRORS:")
        for err in log["recent_errors"][:5]:
            time_ago = format_time_ago(err["time"])
            msg = err["message"][:100]  # Truncate
            print(f"   • [{time_ago}] {msg}")
    
    # ==========================================
    # NEW: Recent Activity Log
    # ==========================================
    if log.get("recent_activity"):
        print(f"\n📜 RECENT ACTIVITY:")
        # Filter to most interesting events
        shown = 0
        for activity in log["recent_activity"]:
            if shown >= 10:
                break
            time_ago = format_time_ago(activity["time"])
            level = activity["level"]
            msg = activity["message"]
            
            # Skip boring messages
            if any(skip in msg.lower() for skip in ["heartbeat", "checking", "sleeping"]):
                continue
            
            level_icon = "❌" if level == "ERROR" else "⚠️" if level == "WARNING" else "•"
            print(f"   {level_icon} [{time_ago}] {msg[:80]}")
            shown += 1
    
    print("\n" + "=" * 70)
    print("  This dashboard shows everything - no other commands needed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
