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
    }
    
    log_path = "live_run.log"
    if os.path.exists(log_path):
        try:
            # Read last few lines
            with open(log_path, "rb") as f:
                # Go to end and read last 50KB
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 50000))
                content = f.read().decode('utf-8', errors='ignore')
            
            lines = content.strip().split('\n')
            
            for line in reversed(lines):
                try:
                    data = json.loads(line)
                    ts = data.get("asctime")
                    
                    if result["last_log"] is None:
                        result["last_log"] = ts
                        # Check if bot is running (last log within 1 minute)
                        try:
                            log_time = datetime.fromisoformat(ts)
                            now = datetime.now(timezone.utc).replace(tzinfo=None)
                            if (now - log_time).total_seconds() < 60:
                                result["bot_running"] = True
                        except:
                            pass
                    
                    msg = data.get("message", "")
                    
                    if "AI data collection" in msg and result["last_ai_collection"] is None:
                        result["last_ai_collection"] = ts
                    
                    if data.get("levelname") == "ERROR":
                        result["errors_24h"] += 1
                        
                except:
                    continue
                    
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
    """Get actual live prediction performance (not validation, real outcomes)."""
    result = {
        "total_predictions": 0,
        "evaluated_predictions": 0,
        "correct_direction": 0,
        "profitable_predictions": 0,
        "total_pnl": 0.0,
        "by_category": {},
    }
    
    # Check category stats database for actual prediction outcomes
    db_path = "data/category_stats.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Get overall stats from predictions table
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
            
            # Get per-category stats
            cursor.execute("""
                SELECT category, 
                       COUNT(*) as total,
                       SUM(was_correct) as correct,
                       SUM(was_profitable) as profitable,
                       SUM(pnl) as pnl
                FROM predictions
                GROUP BY category
                ORDER BY total DESC
            """)
            for row in cursor.fetchall():
                cat, total, correct, profitable, pnl = row
                if total > 0:
                    result["by_category"][cat] = {
                        "total": total,
                        "correct": correct or 0,
                        "profitable": profitable or 0,
                        "pnl": pnl or 0,
                        "accuracy": (correct or 0) / total,
                        "profitable_pct": (profitable or 0) / total,
                    }
            
            conn.close()
        except Exception as e:
            result["error"] = str(e)
    
    # Also check training_examples for evaluated predictions
    ai_db_path = "data/ai_training.db"
    if os.path.exists(ai_db_path):
        try:
            conn = sqlite3.connect(ai_db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Count evaluated predictions
            cursor.execute("""
                SELECT COUNT(*) FROM training_examples 
                WHERE prediction_evaluated = 1 AND predicted_change IS NOT NULL
            """)
            row = cursor.fetchone()
            result["evaluated_predictions"] = row[0] if row else 0
            
            # Calculate actual accuracy from evaluated predictions
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


def main():
    print("\n" + "=" * 70)
    print("                    POLYB0T STATUS DASHBOARD")
    print("=" * 70)
    
    # Configuration
    config = get_config_status()
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Strategy Mode:    {config['strategy_mode'].upper()}")
    print(f"   Placing Orders:   {config['placing_orders']}")
    print(f"   Dry Run:          {config['dry_run']}")
    
    # Log/Runtime Status
    log = get_log_status()
    status_icon = "🟢" if log["bot_running"] else "🔴"
    print(f"\n{status_icon} BOT STATUS:")
    print(f"   Running:          {'YES' if log['bot_running'] else 'NO'}")
    print(f"   Last Activity:    {format_time_ago(log['last_log'])}")
    print(f"   Last AI Collect:  {format_time_ago(log['last_ai_collection'])}")
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
        
        prof_icon = "✅" if prof > 0.55 else "🟡" if prof > 0.50 else "❌"
        print(f"\n   📊 VALIDATION METRICS (from training):")
        print(f"   Profitable Acc:   {prof_icon} {format_pct(prof)}")
        print(f"   Directional Acc:  {format_pct(dir_acc)}")
        print(f"   R² Score:         {metrics.get('r2', 0):.3f}")
        if n_models > 0:
            print(f"\n   🧠 MODEL DETAILS:")
            print(f"   Ensemble:         {n_models} models")
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
            print(f"\n   📈 ACTUAL PERFORMANCE (real outcomes):")
            
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
    
    print("\n" + "=" * 70)
    print("  Run: poetry run python scripts/check_ai_status.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
