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
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

# py-clob-client uses `requests` internally; we use it here as well so we can
# sign the *exact* JSON payload we send for POST /order.
import base64
import hashlib
import hmac
import json
import time

import requests
from py_clob_client.headers.headers import (
    POLY_ADDRESS,
    POLY_API_KEY,
    POLY_PASSPHRASE,
    POLY_SIGNATURE,
    POLY_TIMESTAMP,
)


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
            # Create & sign the order (Level 1 / wallet auth).
            order = self.client.create_order(order_args)

            # Build request body.
            #
            # NOTE: Polymarket's CLOB expects `owner` to be the signing address (POLY_ADDRESS),
            # not the API key. Some client versions use `owner=<api_key>`; that yields 401 on
            # modern deployments. Keep this aligned with the on-wire API.
            body_dict = {
                "order": order.dict(),
                "owner": self.client.signer.address(),
                "orderType": OrderType.GTC,
            }
            body_json = json.dumps(body_dict, separators=(",", ":"), ensure_ascii=False)

            # Level-2 HMAC signing: timestamp(seconds) + method + path + body_json
            ts = str(int(time.time()))
            sig = self._hmac_signature(
                secret=self.api_secret,
                timestamp=ts,
                method="POST",
                request_path="/order",
                body_json=body_json,
            )

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                POLY_ADDRESS: self.client.signer.address(),
                POLY_SIGNATURE: sig,
                POLY_TIMESTAMP: ts,
                POLY_API_KEY: self.api_key,
                POLY_PASSPHRASE: self.passphrase,
            }

            resp = requests.post(f"{self.base_url}/order", headers=headers, data=body_json, timeout=self.timeout)
            status = resp.status_code
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}

            if status < 200 or status >= 300:
                return CLOBOrderResult(
                    success=False,
                    status_code=status,
                    order_id=None,
                    message=f"HTTP {status} from CLOB",
                    raw=data if isinstance(data, dict) else None,
                )

            # Best-effort extraction across possible response shapes.
            order_id = None
            if isinstance(data, dict):
                order_id = (
                    data.get("orderID")
                    or data.get("orderId")
                    or data.get("order_id")
                    or data.get("id")
                    or (data.get("order") or {}).get("id")  # type: ignore[union-attr]
                )

            return CLOBOrderResult(
                success=True,
                status_code=status,
                order_id=str(order_id) if order_id is not None else None,
                message="Order submitted",
                raw=data if isinstance(data, dict) else None,
            )
        except Exception as e:
            # py-clob-client raises on non-2xx (including 401). Keep message controlled.
            msg = str(e)
            status = None
            if "401" in msg:
                status = 401
            return CLOBOrderResult(
                success=False,
                status_code=status,
                order_id=None,
                message=msg,
                raw=None,
            )

    @staticmethod
    def _hmac_signature(
        *,
        secret: str,
        timestamp: str,
        method: str,
        request_path: str,
        body_json: str | None,
    ) -> str:
        """Compute Polymarket Level-2 HMAC signature.

        The secret is urlsafe-base64 decoded before HMAC-SHA256.
        """
        base64_secret = base64.urlsafe_b64decode(secret)
        msg = f"{timestamp}{method}{request_path}"
        if body_json:
            msg += body_json
        digest = hmac.new(base64_secret, msg.encode("utf-8"), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode("utf-8")

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


