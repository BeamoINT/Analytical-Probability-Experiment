"""Best-effort authenticated CLOB trading client.

IMPORTANT:
- This module is only used when `POLYBOT_MODE=live` and `POLYBOT_DRY_RUN=false`
  AND a user has explicitly approved an intent.
- It never runs automatically.
- It never prints secrets.

This implementation uses a conservative "attempt and fail gracefully" approach because
Polymarket CLOB auth requirements may vary. Any non-2xx response becomes a controlled error.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class CLOBOrderResult:
    success: bool
    status_code: int | None
    order_id: str | None
    message: str
    raw: dict[str, Any] | None = None


class CLOBTradingClient:
    """Authenticated client for submitting/canceling LIMIT orders (best-effort)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_secret: str,
        passphrase: str,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def submit_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_usd: float,
    ) -> CLOBOrderResult:
        """Submit a LIMIT order (best-effort).

        NOTE: This may fail if Polymarket requires additional signing beyond API creds.
        """
        body = {
            "token_id": token_id,
            "side": side,
            "type": "LIMIT",
            "price": price,
            "size": size_usd,
        }
        return self._request("POST", "/order", body)

    def cancel_order(self, order_id: str) -> CLOBOrderResult:
        """Cancel an order (best-effort)."""
        return self._request("POST", "/order/cancel", {"order_id": order_id})

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> CLOBOrderResult:
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body or {}, separators=(",", ":"), sort_keys=True)
        sig = self._sign(ts, method.upper(), path, body_str)

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "X-API-Passphrase": self.passphrase,
            "X-API-Timestamp": ts,
            "X-API-Signature": sig,
        }
        try:
            resp = self.client.request(method, path, headers=headers, content=body_str)
            status = resp.status_code
            data = None
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

            # Try a few common order id fields
            order_id = None
            if isinstance(data, dict):
                order_id = (
                    data.get("order_id")
                    or data.get("id")
                    or (data.get("order") or {}).get("id")  # type: ignore[union-attr]
                )
            return CLOBOrderResult(
                success=True,
                status_code=status,
                order_id=str(order_id) if order_id is not None else None,
                message="ok",
                raw=data if isinstance(data, dict) else None,
            )
        except Exception as e:
            return CLOBOrderResult(success=False, status_code=None, order_id=None, message=str(e), raw=None)

    def _sign(self, ts: str, method: str, path: str, body: str) -> str:
        """HMAC signature (best-effort).

        Many API-key systems sign: timestamp + method + path + body.
        """
        msg = (ts + method + path + body).encode("utf-8")
        secret = self.api_secret.encode("utf-8")
        digest = hmac.new(secret, msg, hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")


