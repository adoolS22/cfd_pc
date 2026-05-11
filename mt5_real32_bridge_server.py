#!/usr/bin/env python3
"""
MT5 Real32 Bridge Server
========================
Minimal HTTP bridge that exposes MT5 market data to the dockerized bot.

Endpoints:
  GET /health
  GET /ticker?symbol=BTCUSDm
  GET /ohlcv?symbol=BTCUSDm&timeframe=1m&limit=300

Environment:
  MT5_LOGIN=<account number>
  MT5_PASSWORD=<account password>
  MT5_SERVER=<server name, e.g. "Exness-MT5Real32">
  MT5_PATH=<optional terminal64.exe path>
  MT5_BRIDGE_HOST=0.0.0.0
  MT5_BRIDGE_PORT=18080
  MT5_BRIDGE_TOKEN=<optional shared token>
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover
    mt5 = None  # type: ignore


_LOCK = threading.Lock()
_INITIALIZED = False


def _tf_to_mt5(timeframe: str) -> Optional[int]:
    if mt5 is None:
        return None
    tf = str(timeframe or "").strip().lower()
    mapping = {
        "1m": mt5.TIMEFRAME_M1,
        "3m": mt5.TIMEFRAME_M3,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    return mapping.get(tf)


def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, **payload}


def _err(message: str, code: int = 400) -> tuple[int, Dict[str, Any]]:
    return code, {"ok": False, "error": message}


def _ensure_mt5() -> Optional[str]:
    global _INITIALIZED
    if mt5 is None:
        return "MetaTrader5 package is not installed in this Python environment."
    if _INITIALIZED:
        return None

    login_raw = os.getenv("MT5_LOGIN", "").strip()
    password = os.getenv("MT5_PASSWORD", "").strip()
    server = os.getenv("MT5_SERVER", "").strip()
    path = os.getenv("MT5_PATH", "").strip() or None

    if not login_raw or not password or not server:
        return "Missing MT5_LOGIN / MT5_PASSWORD / MT5_SERVER."

    try:
        login = int(login_raw)
    except Exception:
        return "MT5_LOGIN must be numeric."

    kwargs: Dict[str, Any] = {}
    if path:
        kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        return f"mt5.initialize() failed: {mt5.last_error()}"

    if not mt5.login(login=login, password=password, server=server):
        return f"mt5.login() failed: {mt5.last_error()}"

    _INITIALIZED = True
    return None


def _ensure_symbol(symbol: str) -> Optional[str]:
    info = mt5.symbol_info(symbol) if mt5 is not None else None
    if info is None:
        return f"Unknown MT5 symbol: {symbol}"
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            return f"symbol_select failed for {symbol}: {mt5.last_error()}"
    return None


class _Handler(BaseHTTPRequestHandler):
    server_version = "MT5Bridge/1.0"

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        expected = os.getenv("MT5_BRIDGE_TOKEN", "").strip()
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        api_key = self.headers.get("X-API-Key", "")
        bearer = auth.replace("Bearer ", "").strip() if auth else ""
        return bearer == expected or api_key == expected

    def do_GET(self) -> None:  # noqa: N802
        if not self._auth_ok():
            status, payload = _err("Unauthorized", code=401)
            self._write_json(status, payload)
            return

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        with _LOCK:
            mt5_err = _ensure_mt5()
            if mt5_err:
                status, payload = _err(mt5_err, code=500)
                self._write_json(status, payload)
                return

            if parsed.path == "/health":
                self._write_json(200, _ok({"status": "ok"}))
                return

            if parsed.path == "/ticker":
                self._handle_ticker(query)
                return

            if parsed.path == "/ohlcv":
                self._handle_ohlcv(query)
                return

            status, payload = _err("Not found", code=404)
            self._write_json(status, payload)

    def _handle_ticker(self, query: Dict[str, Any]) -> None:
        symbol = str((query.get("symbol") or [""])[0]).strip()
        if not symbol:
            status, payload = _err("Missing symbol", code=400)
            self._write_json(status, payload)
            return

        sym_err = _ensure_symbol(symbol)
        if sym_err:
            status, payload = _err(sym_err, code=400)
            self._write_json(status, payload)
            return

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            status, payload = _err(f"symbol_info_tick failed: {mt5.last_error()}", code=500)
            self._write_json(status, payload)
            return

        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        last = float(getattr(tick, "last", 0.0) or 0.0)
        if last <= 0 and bid > 0 and ask > 0:
            last = (bid + ask) / 2.0
        if bid <= 0 and last > 0:
            bid = last
        if ask <= 0 and last > 0:
            ask = last

        payload = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last,
            "timestamp": int(getattr(tick, "time", 0) or 0),
        }
        self._write_json(200, _ok(payload))

    def _handle_ohlcv(self, query: Dict[str, Any]) -> None:
        symbol = str((query.get("symbol") or [""])[0]).strip()
        timeframe = str((query.get("timeframe") or ["1m"])[0]).strip()
        limit_raw = str((query.get("limit") or ["300"])[0]).strip()

        if not symbol:
            status, payload = _err("Missing symbol", code=400)
            self._write_json(status, payload)
            return

        tf_const = _tf_to_mt5(timeframe)
        if tf_const is None:
            status, payload = _err(f"Unsupported timeframe: {timeframe}", code=400)
            self._write_json(status, payload)
            return

        try:
            limit = max(10, min(2000, int(limit_raw)))
        except Exception:
            limit = 300

        sym_err = _ensure_symbol(symbol)
        if sym_err:
            status, payload = _err(sym_err, code=400)
            self._write_json(status, payload)
            return

        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, limit)
        if rates is None:
            status, payload = _err(f"copy_rates_from_pos failed: {mt5.last_error()}", code=500)
            self._write_json(status, payload)
            return

        candles = []
        for r in rates:
            candles.append({
                "timestamp": int(r["time"]),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["tick_volume"]),
            })

        self._write_json(200, _ok({"symbol": symbol, "timeframe": timeframe, "candles": candles}))

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep stdout clean; dockerized bot has its own logs.
        return


def main() -> None:
    host = os.getenv("MT5_BRIDGE_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port_raw = os.getenv("MT5_BRIDGE_PORT", "18080").strip()
    try:
        port = int(port_raw)
    except Exception:
        port = 18080

    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"MT5 bridge server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

