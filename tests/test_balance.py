from datetime import datetime

import httpx
import pytest

from polyb0t.services.balance import BalanceService, _to_32byte_hex_address


def test_to_32byte_hex_address():
    assert _to_32byte_hex_address("0x" + "11" * 20).endswith("11" * 20)
    with pytest.raises(ValueError):
        _to_32byte_hex_address("0x123")


def test_balance_service_parses_rpc_balance(monkeypatch, db_session):
    # Configure required env for settings
    monkeypatch.setenv("POLYBOT_MODE", "live")
    monkeypatch.setenv("POLYBOT_DRY_RUN", "true")
    monkeypatch.setenv("POLYBOT_LOOP_INTERVAL_SECONDS", "10")
    monkeypatch.setenv("POLYBOT_USER_ADDRESS", "0x" + "11" * 20)
    monkeypatch.setenv("POLYBOT_POLYGON_RPC_URL", "http://rpc.local")
    monkeypatch.setenv("POLYBOT_USDCE_TOKEN_ADDRESS", "0x" + "22" * 20)
    monkeypatch.setenv("POLYBOT_USDC_DECIMALS", "6")
    from polyb0t.config.settings import get_settings

    get_settings.cache_clear()

    # 12.345678 USDC => raw = 12_345_678
    raw = 12_345_678
    result_hex = hex(raw)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://rpc.local")
        import json

        body = json.loads(request.content.decode("utf-8"))
        assert body["method"] == "eth_call"
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": result_hex})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    svc = BalanceService(db_session=db_session, http_client=client)
    snap = svc.fetch_usdc_balance()
    assert snap.total_usdc == pytest.approx(12.345678)
    assert snap.available_usdc <= snap.total_usdc


