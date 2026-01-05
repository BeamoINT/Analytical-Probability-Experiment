"""Authenticated CLOB trading client (Polymarket).

IMPORTANT:
- This module is only used when `POLYBOT_MODE=live` and `POLYBOT_DRY_RUN=false`
  AND a user has explicitly approved an intent.
- It never runs automatically.
- It never prints secrets.

This implementation delegates authentication + order signing to the official `py-clob-client`
library (already included in dependencies). Any failure becomes a controlled error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs
from py_clob_client.exceptions import PolyApiException


@dataclass(frozen=True)
class CLOBOrderResult:
    success: bool
    status_code: int | None
    order_id: str | None
    message: str
    raw: dict[str, Any] | None = None


class CLOBTradingClient:
    """Authenticated client for submitting/canceling orders (best-effort)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        passphrase: str,
        polygon_private_key: str,
        chain_id: int = 137,
        signature_type: int = 0,
        funder: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        # NOTE: Never log secrets.
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.polygon_private_key = polygon_private_key
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.timeout = timeout

        self.client = ClobClient(
            host=self.base_url,
            chain_id=self.chain_id,
            key=self.polygon_private_key,
            creds=ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.passphrase,
            ),
            signature_type=self.signature_type,
            funder=self.funder,
        )

    def close(self) -> None:
        # py-clob-client doesn't expose a documented close hook; keep for parity.
        return

    def submit_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usd: float,
        fee_rate_bps: int = 0,
    ) -> CLOBOrderResult:
        """Submit a LIMIT order (best-effort).

        `py-clob-client` expects `size` in *shares* (outcome tokens), not USD.
        For Polymarket, notional USD is roughly: shares * price.
        """
        try:
            p = float(price or 0.0)
            usd = float(size_usd or 0.0)
            if p <= 0 or usd <= 0:
                return CLOBOrderResult(
                    success=False,
                    status_code=None,
                    order_id=None,
                    message="Invalid order params (price/size_usd must be > 0)",
                    raw=None,
                )

            # Convert USD notional to share quantity.
            size_shares = usd / p
            order_args = OrderArgs(
                token_id=token_id,
                price=p,
                size=float(size_shares),
                side=str(side).upper(),
                fee_rate_bps=int(fee_rate_bps or 0),
            )
            res = self.client.create_and_post_order(order_args)

            # Best-effort extraction across possible response shapes.
            order_id = None
            raw: dict[str, Any] | None = None
            if isinstance(res, dict):
                raw = res
                order_id = (
                    res.get("orderID")
                    or res.get("orderId")
                    or res.get("order_id")
                    or res.get("id")
                    or (res.get("order") or {}).get("id")  # type: ignore[union-attr]
                )

            return CLOBOrderResult(
                success=True,
                status_code=200,
                order_id=str(order_id) if order_id is not None else None,
                message="Order submitted",
                raw=raw,
            )
        except PolyApiException as e:
            # Includes status_code and parsed error payload (safe to store).
            raw = e.error_msg if isinstance(e.error_msg, dict) else {"text": str(e.error_msg)}
            status_code = getattr(e, "status_code", None)
            error_detail = str(e.error_msg) if e.error_msg else "Unknown error"
            return CLOBOrderResult(
                success=False,
                status_code=status_code,
                order_id=None,
                message=f"HTTP {status_code} from CLOB: {error_detail}",
                raw=raw if isinstance(raw, dict) else None,
            )
        except Exception as e:
            return CLOBOrderResult(
                success=False,
                status_code=None,
                order_id=None,
                message=str(e),
                raw=None,
            )

    def cancel_order(self, order_id: str) -> CLOBOrderResult:
        """Cancel an order (best-effort)."""
        try:
            res = self.client.cancel_orders([order_id])
            raw: dict[str, Any] | None = res if isinstance(res, dict) else None
            return CLOBOrderResult(
                success=True,
                status_code=200,
                order_id=order_id,
                message="cancel submitted",
                raw=raw,
            )
        except Exception as e:
            msg = str(e)
            status = 401 if "401" in msg else None
            return CLOBOrderResult(
                success=False,
                status_code=status,
                order_id=order_id,
                message=msg,
                raw=None,
            )


