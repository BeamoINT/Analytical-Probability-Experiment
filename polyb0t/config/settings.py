"""Application settings using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix="POLYBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Mode Configuration
    # Required (fail-fast enforced by env_loader + Settings instantiation)
    mode: Literal["paper", "live"] = Field(
        ..., description="Trading mode: paper (simulated) or live (real data)"
    )
    dry_run: bool = Field(
        ...,
        description="In live mode, compute intents but never execute (safety default)",
    )
    user_address: str = Field(
        ...,
        description="Polymarket wallet address for read-only monitoring",
    )

    # Database
    db_url: str = Field(default="sqlite:///./polybot.db", description="Database connection URL")

    # Market Filtering
    resolve_min_days: int = Field(default=30, description="Minimum days until resolution")
    resolve_max_days: int = Field(default=60, description="Maximum days until resolution")
    min_liquidity: float = Field(default=1000.0, description="Minimum market liquidity")
    max_spread: float = Field(default=0.05, description="Maximum bid-ask spread (5%)")

    # Strategy
    edge_threshold: float = Field(
        default=0.05, description="Minimum raw edge to consider (mid-price, 5%)"
    )
    min_net_edge: float = Field(
        default=0.02, description="Minimum net edge after fees/slippage to trade (2%)"
    )
    paper_bankroll: float = Field(default=10000.0, description="Initial paper trading bankroll")

    # Risk Management
    max_position_pct: float = Field(
        default=2.0, description="Max position size as % of bankroll"
    )
    max_total_exposure_pct: float = Field(
        default=20.0, description="Max total exposure as % of bankroll"
    )
    max_per_category_exposure_pct: float = Field(
        default=10.0, description="Max exposure per category as % of bankroll"
    )
    drawdown_limit_pct: float = Field(
        default=15.0, description="Stop trading if drawdown exceeds this %"
    )
    max_daily_loss_pct: float = Field(
        default=10.0, description="Max daily loss as % of starting equity"
    )
    max_orders_per_hour: int = Field(
        default=20, description="Max orders allowed per hour (rate limit)"
    )
    max_notional_per_market: float = Field(
        default=1000.0, description="Max dollar exposure per market"
    )
    max_positions_per_market: int = Field(
        default=4, description="Max number of positions allowed per market (diversification limit)"
    )

    # Live Balance / Chain Configuration
    chain_id: int = Field(default=137, description="Chain ID for on-chain balance checks (Polygon=137)")
    polygon_rpc_url: str | None = Field(
        default=None, description="Polygon RPC URL for on-chain reads (required for balance checks)"
    )
    # Polygon USDC (commonly called USDC.e in some docs)
    usdce_token_address: str | None = Field(
        default="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        description="USDC token contract address on Polygon",
    )
    usdc_decimals: int = Field(default=6, description="USDC decimals (default 6)")

    # Live absolute risk/limits (configured for percentage-based sizing)
    min_order_usd: float = Field(default=1.0, description="Min USD notional per order (live)")
    max_order_usd: float = Field(default=10000.0, description="Max USD notional per order - set high to allow percentage-based sizing")
    max_total_exposure_usd: float = Field(default=100000.0, description="Max total exposure USD - set high to allow percentage-based sizing")
    max_open_orders: int = Field(default=3, description="Max open orders allowed (live)")
    # Daily notional cap: set to 0 to disable (no cap).
    max_daily_notional_usd: float = Field(default=0.0, description="Max daily notional USD (live). 0 disables.")
    kill_switch_on_errors: int = Field(
        default=5, description="Kill switch if >= N consecutive cycle errors"
    )

    # Execution
    loop_interval_seconds: int = Field(
        ..., description="Main loop interval (seconds) - live defaults to 10"
    )

    # Live scanning / rate-limit controls
    live_scan_markets_limit: int = Field(
        default=500,
        description="How many markets to fetch from Gamma in live mode (broad scan).",
    )
    live_enrich_markets_limit: int = Field(
        default=50,
        description="How many of the scanned markets to enrich with outcomes/token_ids (top by volume).",
    )
    live_clob_markets_limit: int = Field(
        default=50,
        description="How many markets to fetch orderbooks/trades for per cycle in live mode (top by volume).",
    )
    live_clob_concurrency: int = Field(
        default=6,
        description="Max concurrent CLOB HTTP requests per cycle in live mode (lower = safer on rate limits).",
    )
    live_fetch_trades: bool = Field(
        default=False,
        description="Fetch recent trades from CLOB in live mode (often 404); disable to reduce load.",
    )
    order_timeout_seconds: int = Field(default=300, description="Order timeout (seconds)")
    slippage_bps: int = Field(default=10, description="Assumed slippage in basis points")
    fee_bps: int = Field(default=20, description="Trading fee in basis points")

    # Trade Intent System
    intent_expiry_seconds: int = Field(
        default=60, description="Trade intent expiry time (seconds)"
    )
    intent_cooldown_seconds: int = Field(
        default=120,
        description="Dedup cooldown window for emitting equivalent intents (seconds)",
    )
    intent_price_round: float = Field(
        default=0.001,
        description="Price rounding used for intent fingerprinting (e.g. 0.001)",
    )
    intent_size_bucket_usd: float = Field(
        default=10.0,
        description="USD size bucket used for intent fingerprinting (e.g. $10)",
    )
    dedup_edge_delta: float = Field(
        default=0.01,
        description="Allow new intent within cooldown if edge changes by more than this",
    )
    dedup_price_delta: float = Field(
        default=0.01,
        description="Allow new intent within cooldown if price changes by more than this",
    )
    auto_approve_intents: bool = Field(
        default=False,
        description="DANGER: Auto-approve intents (only for testing, never in production)",
    )

    # Live safety: prevent the bot from proposing SELL orders unless closing bot-opened positions.
    # Many users trade manually; this avoids the bot suggesting sells on positions it didn't open.
    live_allow_open_sell_intents: bool = Field(
        default=False,
        description=(
            "Allow OPEN_POSITION intents with side=SELL in live mode. "
            "When false (default), the bot will only propose BUY intents to open positions; "
            "SELL is reserved for closing bot-managed positions."
        ),
    )

    # Exit Management
    enable_take_profit: bool = Field(default=True, description="Enable take-profit proposals")
    take_profit_pct: float = Field(default=10.0, description="Take profit at % gain")
    enable_stop_loss: bool = Field(default=True, description="Enable stop-loss proposals")
    stop_loss_pct: float = Field(default=5.0, description="Stop loss at % loss")
    enable_time_exit: bool = Field(
        default=True, description="Enable time-based exit proposals"
    )
    time_exit_days_before: int = Field(
        default=2, description="Propose exit N days before market resolution"
    )

    # Kill Switches
    max_api_error_rate_pct: float = Field(
        default=50.0, description="Halt if API error rate exceeds this %"
    )
    max_stale_data_seconds: int = Field(
        default=60, description="Halt if data is stale beyond this threshold"
    )
    max_spread_multiplier: float = Field(
        default=3.0, description="Halt if spread exceeds normal by this multiplier"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")

    # External APIs
    gamma_base_url: str = Field(
        default="https://gamma-api.polymarket.com", description="Gamma Markets API base URL"
    )
    clob_base_url: str = Field(
        default="https://clob.polymarket.com", description="CLOB API base URL"
    )

    # Live Trading Credentials (ONLY for live mode with explicit user consent)
    polygon_private_key: str | None = Field(
        default=None, description="Polygon L2 private key (NEVER commit or print)"
    )
    clob_api_key: str | None = Field(
        default=None, description="CLOB API key (if required for account access)"
    )
    clob_api_secret: str | None = Field(
        default=None, description="CLOB API secret (NEVER commit or print)"
    )
    clob_passphrase: str | None = Field(
        default=None, description="CLOB API passphrase (NEVER commit or print)"
    )
    
    # Polymarket Signature Configuration
    signature_type: int = Field(
        default=0,
        description="Signature type: 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE"
    )
    funder_address: str | None = Field(
        default=None,
        description="Funder address (usually same as user_address for EOA; different for proxy/Safe)"
    )

    # Machine Learning
    enable_ml: bool = Field(
        default=False,
        description="Enable ML predictions (requires trained model)"
    )
    ml_model_dir: str = Field(
        default="models",
        description="Directory containing ML models"
    )
    ml_data_db: str = Field(
        default="data/training_data.db",
        description="Database for training data collection"
    )
    ml_retrain_interval_hours: int = Field(
        default=6,
        description="Hours between automatic model retraining"
    )
    ml_min_training_examples: int = Field(
        default=1000,
        description="Minimum examples before first training"
    )
    ml_validation_threshold_r2: float = Field(
        default=0.03,
        description="Minimum R² to accept new model (0.03 = 3% variance explained)"
    )
    ml_prediction_blend_weight: float = Field(
        default=0.7,
        description="Weight for ML predictions (0.7 = 70% ML, 30% baseline)"
    )
    ml_use_ensemble: bool = Field(
        default=False,
        description="Use ensemble of models instead of single model"
    )
    ml_data_collection_limit: int = Field(
        default=50,
        description="Max markets to collect data from per cycle (0 = unlimited)"
    )
    ml_max_training_examples: int = Field(
        default=25_000_000,
        description="Max training examples to use (larger = more memory, better models)"
    )
    ml_data_retention_days: int = Field(
        default=1095,
        description="How long to keep training data (days). 1095 days (3 years) = ~75GB max DB size"
    )
    ml_price_snapshot_interval_minutes: int = Field(
        default=15,
        description="Collect dense price snapshots every N minutes for deep learning"
    )
    ml_enable_backfill: bool = Field(
        default=True,
        description="Backfill missing price data when bot restarts"
    )
    ml_auto_enable_threshold: int = Field(
        default=2000,
        description="Auto-enable ML when this many labeled examples are collected (0=never)"
    )
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "console"] = Field(
        default="json", description="Log output format"
    )

    @field_validator("max_spread", "edge_threshold")
    @classmethod
    def validate_percentages(cls, v: float) -> float:
        """Validate percentage values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {v}")
        return v

    @field_validator("paper_bankroll")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate bankroll is positive."""
        if v <= 0:
            raise ValueError(f"Bankroll must be positive, got {v}")
        return v

    @field_validator("loop_interval_seconds")
    @classmethod
    def validate_loop_interval(cls, v: int) -> int:
        """Validate loop interval is reasonable."""
        if v < 1:
            raise ValueError(f"Loop interval must be at least 1 second, got {v}")
        if v < 5:
            import warnings

            warnings.warn(
                f"Loop interval {v}s is very aggressive. Consider 10s+ to avoid rate limits."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

