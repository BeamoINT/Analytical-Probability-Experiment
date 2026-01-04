"""Balance and collateral availability service (live mode).

This module intentionally uses only read-only on-chain RPC calls. It never asks for,
prints, or requires private keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.storage import AccountStateDB, BalanceSnapshotDB, TradeIntentDB

logger = logging.getLogger(__name__)


BALANCE_OF_SELECTOR = "0x70a08231"  # balanceOf(address)


def _to_32byte_hex_address(addr: str) -> str:
    a = addr.lower()
    if a.startswith("0x"):
        a = a[2:]
    if len(a) != 40:
        raise ValueError("address must be 20 bytes")
    return a.rjust(64, "0")


def _hex_to_int(hex_str: str) -> int:
    if hex_str is None:
        raise ValueError("missing hex result")
    return int(hex_str, 16)


@dataclass(frozen=True)
class BalanceSnapshot:
    timestamp: datetime
    wallet_address: str
    token_address: str
    chain_id: int
    total_usdc: float
    reserved_usdc: float
    available_usdc: float
    meta: dict[str, Any]


class BalanceService:
    """Compute total/reserved/available USDC collateral for live mode sizing."""

    def __init__(self, db_session: Session, http_client: httpx.Client | None = None) -> None:
        self.db_session = db_session
        self.settings = get_settings()
        self.http = http_client or httpx.Client(timeout=20.0)

    def close(self) -> None:
        try:
            self.http.close()
        except Exception:
            pass

    def fetch_usdc_balance(self) -> BalanceSnapshot:
        """Fetch on-chain USDC balance and compute conservative available collateral."""
        s = self.settings
        if not s.polygon_rpc_url:
            raise RuntimeError("POLYBOT_POLYGON_RPC_URL is not set; cannot fetch on-chain balance")
        if not s.usdce_token_address:
            raise RuntimeError("POLYBOT_USDCE_TOKEN_ADDRESS is not set; cannot fetch on-chain balance")

        raw = self._erc20_balance_of(
            rpc_url=s.polygon_rpc_url,
            token_address=s.usdce_token_address,
            wallet_address=s.user_address,
        )
        total = raw / (10 ** int(s.usdc_decimals))

        reserved = self._compute_reserved_usdc()
        available = max(0.0, total - reserved)

        return BalanceSnapshot(
            timestamp=datetime.utcnow(),
            wallet_address=s.user_address,
            token_address=s.usdce_token_address,
            chain_id=s.chain_id,
            total_usdc=float(total),
            reserved_usdc=float(reserved),
            available_usdc=float(available),
            meta={"method": "rpc_eth_call_balanceOf", "raw": raw},
        )

    def persist_snapshot(self, cycle_id: str, snap: BalanceSnapshot) -> None:
        row = BalanceSnapshotDB(
            cycle_id=cycle_id,
            timestamp=snap.timestamp,
            wallet_address=snap.wallet_address,
            token_address=snap.token_address,
            chain_id=snap.chain_id,
            total_usdc=snap.total_usdc,
            reserved_usdc=snap.reserved_usdc,
            available_usdc=snap.available_usdc,
            meta_json=snap.meta,
        )
        self.db_session.add(row)
        self.db_session.commit()

    def _compute_reserved_usdc(self) -> float:
        """Conservative reserved amount.

        If we cannot precisely compute reserved for positions/open orders, we prefer being conservative:
        - reserve APPROVED intents that have not been executed/submitted yet
        """
        now = datetime.utcnow()
        total = 0.0

        # 1) approved but not executed intents (closest to becoming orders)
        approved = (
            self.db_session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == "APPROVED")
            .filter(TradeIntentDB.executed_at.is_(None))
            .filter(TradeIntentDB.expires_at >= now)
            .all()
        )
        for i in approved:
            v = i.size_usd if getattr(i, "size_usd", None) is not None else i.size
            if v:
                total += float(v)

        # 2) open orders notional (best-effort from most recent account state snapshot)
        last_state = (
            self.db_session.query(AccountStateDB)
            .order_by(AccountStateDB.timestamp.desc())
            .first()
        )
        if last_state and isinstance(last_state.open_orders, list):
            for o in last_state.open_orders:
                try:
                    # We don't know whether API reports `size` as USD notional or shares.
                    # Conservative: treat it as USD notional if it looks like dollars.
                    sz = float(o.get("size", 0) or 0)
                    total += max(0.0, sz)
                except Exception:
                    continue

        return float(total)

    def _erc20_balance_of(self, rpc_url: str, token_address: str, wallet_address: str) -> int:
        """Call ERC-20 balanceOf via eth_call and return integer raw units."""
        data = BALANCE_OF_SELECTOR + _to_32byte_hex_address(wallet_address)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [
                {"to": token_address, "data": data},
                "latest",
            ],
        }
        resp = self.http.post(rpc_url, json=payload)
        resp.raise_for_status()
        j = resp.json()
        if "error" in j:
            raise RuntimeError(f"RPC error: {j['error']}")
        result = j.get("result")
        return _hex_to_int(result)


