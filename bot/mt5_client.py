"""
Direct MetaTrader 5 client utilities.

This module isolates the official MetaTrader5 data-feed integration so the
existing bot can consume MT5 quotes/candles with minimal code changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import threading
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError:
    mt5 = None  # type: ignore


class MT5ClientError(Exception):
    """Raised when the MT5 client cannot provide market data."""


def _normalize_symbol_key(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("_FUTURES", "").replace("_SPOT", "")
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw.replace("/", "").replace("-", "").replace("_", "")


def _namedtuple_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    if isinstance(value, dict):
        return dict(value)
    return {}


def _format_last_error(prefix: str) -> str:
    if mt5 is None:
        return prefix
    err = mt5.last_error()
    code = err[0] if isinstance(err, tuple) and len(err) > 0 else "unknown"
    message = err[1] if isinstance(err, tuple) and len(err) > 1 else str(err)
    details = f"{prefix}: {code} {message}"
    text = str(message or "").lower()
    if "ipc" in text or "initialize" in text:
        details += " (MT5 terminal may be closed or unreachable)"
    return details


class MT5Client:
    """Small direct wrapper around the official MetaTrader5 Python package."""

    def __init__(
        self,
        *,
        login: Any,
        password: str,
        server: str,
        path: Optional[str] = None,
        timeout_seconds: float = 10.0,
        symbol_suffix: str = "",
        symbol_map: Optional[Dict[str, str]] = None,
        default_symbol: str = "",
    ) -> None:
        self.login = self._parse_login(login)
        self.password = str(password or "").strip()
        self.server = str(server or "").strip()
        self.path = str(path or "").strip() or None
        self.timeout_seconds = max(1.0, float(timeout_seconds or 10.0))
        self.symbol_suffix = str(symbol_suffix or "").strip()
        self.symbol_map = dict(symbol_map or {})
        self.default_symbol = str(default_symbol or "").strip()
        self._connected = False
        self._selected_symbols: set[str] = set()
        self._lock = threading.RLock()

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "MT5Client":
        config = dict(cfg or {})
        return cls(
            login=config.get("login") or os.getenv("MT5_LOGIN", ""),
            password=str(config.get("password") or os.getenv("MT5_PASSWORD", "")),
            server=str(config.get("server") or os.getenv("MT5_SERVER", "")),
            path=config.get("path") or os.getenv("MT5_PATH", ""),
            timeout_seconds=float(config.get("timeout_seconds", 10.0) or 10.0),
            symbol_suffix=str(config.get("symbol_suffix", "") or ""),
            symbol_map=dict(config.get("symbol_map", {}) or {}),
            default_symbol=str(config.get("default_symbol", "") or ""),
        )

    @staticmethod
    def _parse_login(value: Any) -> int:
        try:
            return int(str(value or "").strip())
        except Exception as exc:
            raise MT5ClientError("MT5 login must be numeric") from exc

    def _require_package(self) -> None:
        if mt5 is None:
            logger.error("Failed to initialize MT5: MetaTrader5 package is not installed")
            raise MT5ClientError("Failed to initialize MT5: MetaTrader5 package is not installed")

    def _resolve_symbol(self, symbol: str) -> str:
        source = str(symbol or self.default_symbol or "").strip()
        if not source:
            raise MT5ClientError("Symbol is required")
        normalized = _normalize_symbol_key(source)
        target = str(self.symbol_map.get(normalized) or normalized).strip()
        if self.symbol_suffix and not target.endswith(self.symbol_suffix):
            target = f"{target}{self.symbol_suffix}"
        return target

    def _timeframe_to_mt5(self, timeframe: Any) -> int:
        self._require_package()
        if isinstance(timeframe, int):
            return timeframe
        tf = str(timeframe or "").strip().lower()
        mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "2m": mt5.TIMEFRAME_M2,
            "3m": mt5.TIMEFRAME_M3,
            "4m": mt5.TIMEFRAME_M4,
            "5m": mt5.TIMEFRAME_M5,
            "6m": mt5.TIMEFRAME_M6,
            "10m": mt5.TIMEFRAME_M10,
            "12m": mt5.TIMEFRAME_M12,
            "15m": mt5.TIMEFRAME_M15,
            "20m": mt5.TIMEFRAME_M20,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "2h": mt5.TIMEFRAME_H2,
            "3h": mt5.TIMEFRAME_H3,
            "4h": mt5.TIMEFRAME_H4,
            "6h": mt5.TIMEFRAME_H6,
            "8h": mt5.TIMEFRAME_H8,
            "12h": mt5.TIMEFRAME_H12,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
            "1mo": mt5.TIMEFRAME_MN1,
        }
        if tf not in mapping:
            raise MT5ClientError(f"Unsupported MT5 timeframe: {timeframe}")
        return mapping[tf]

    def _preselect_symbols_from_map(self) -> None:
        """Select all symbols from symbol_map + common defaults into Market Watch.
        Must be called after a successful login so MT5 starts streaming prices."""
        import time as _time
        targets = set(self.symbol_map.values())
        # Always include common symbols
        targets.update([
            "BTCUSDm", "ETHUSDm",
            "EURUSDm", "GBPUSDm", "USDJPYm", "USDCHFm",
            "AUDUSDm", "USDCADm", "NZDUSDm",
            "XAUUSDm", "XAGUSDm", "USOILm", "US500m",
        ])
        if self.symbol_suffix:
            # Also try with suffix if map has bare names
            targets = {s if s.endswith(self.symbol_suffix) else f"{s}{self.symbol_suffix}"
                       for s in targets}

        selected = []
        failed = []
        for sym in sorted(targets):
            try:
                ok = mt5.symbol_select(sym, True)
                if ok:
                    selected.append(sym)
                    self._selected_symbols.add(sym)
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)

        if selected:
            logger.info(f"Pre-selected {len(selected)} symbols from Market Watch")
        if failed:
            logger.debug(f"Pre-select skipped (not on broker): {failed}")

        # Brief pause so MT5 terminal starts streaming data for new symbols
        if selected:
            _time.sleep(1.5)

    def connect_mt5(self) -> bool:
        with self._lock:
            self._require_package()
            if self._connected:
                return True
            if not self.password or not self.server:
                logger.error("Failed to initialize MT5: missing login/password/server")
                raise MT5ClientError("Failed to initialize MT5: missing login/password/server")

            kwargs: Dict[str, Any] = {"timeout": int(self.timeout_seconds * 1000)}
            if self.path:
                kwargs["path"] = self.path

            if not mt5.initialize(**kwargs):
                message = _format_last_error("Failed to initialize MT5")
                logger.error(message)
                raise MT5ClientError(message)

            if not mt5.login(login=self.login, password=self.password, server=self.server):
                message = _format_last_error("Failed to login to MT5")
                logger.error(message)
                mt5.shutdown()
                raise MT5ClientError(message)

            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("MT5 connected successfully, but terminal info is unavailable")
            else:
                logger.info(
                    "MT5 connected successfully "
                    f"(server={self.server}, login={self.login}, connected={getattr(terminal_info, 'connected', None)})"
                )
            self._connected = True

            # Pre-select all symbols from the symbol_map so MT5 starts
            # receiving price feeds before any scan requests data.
            self._preselect_symbols_from_map()

            return True

    def shutdown_mt5(self) -> None:
        with self._lock:
            if mt5 is not None:
                mt5.shutdown()
            self._connected = False
            self._selected_symbols.clear()
            logger.info("MT5 connection closed")

    def ensure_symbol(self, symbol: str) -> Dict[str, Any]:
        with self._lock:
            self.connect_mt5()
            mt5_symbol = self._resolve_symbol(symbol)

            # Always call symbol_select to ensure the symbol stays active.
            # MT5 can drop symbols from the feed between scans, especially
            # if another mt5.login() call was made in the same process.
            mt5.symbol_select(mt5_symbol, True)

            info = mt5.symbol_info(mt5_symbol)
            if info is None:
                logger.error(f"DEBUG: MT5 symbol_info returned None for '{mt5_symbol}'. Terminal connected: {mt5.terminal_info().connected if mt5.terminal_info() else False}")
                # Try selecting it
                mt5.symbol_select(mt5_symbol, True)
                import time as _t; _t.sleep(0.5)
                info = mt5.symbol_info(mt5_symbol)
            if info is None:
                logger.error(f"Symbol not found in MT5: {mt5_symbol}")
                raise MT5ClientError(f"Symbol not found in MT5: {mt5_symbol}")

            if mt5_symbol not in self._selected_symbols:
                logger.info(f"Symbol selected: {mt5_symbol}")
                self._selected_symbols.add(mt5_symbol)

            return _namedtuple_to_dict(info)

    def get_tick(self, symbol: str) -> Dict[str, Any]:
        with self._lock:
            info = self.ensure_symbol(symbol)
            mt5_symbol = self._resolve_symbol(symbol)
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                logger.warning(f"No tick data available for {mt5_symbol}")
                raise MT5ClientError(f"No tick data available for {mt5_symbol}")

            tick_data = _namedtuple_to_dict(tick)
            bid = float(tick_data.get("bid") or 0.0)
            ask = float(tick_data.get("ask") or 0.0)
            point = float(info.get("point") or 0.0)
            if bid <= 0.0 and ask <= 0.0:
                logger.warning(f"Market is closed or no live tick quotes are available for {mt5_symbol}")
                raise MT5ClientError(f"Market is closed or no live tick quotes are available for {mt5_symbol}")

            spread = max(0.0, ask - bid) if bid > 0.0 and ask > 0.0 else 0.0
            spread_points = (spread / point) if point > 0.0 else float(info.get("spread") or 0.0)
            tick_time = int(tick_data.get("time") or 0)
            logger.debug(f"Tick received for {mt5_symbol}: bid={bid}, ask={ask}, spread={spread}")
            return {
                "symbol": symbol,
                "mt5_symbol": mt5_symbol,
                "bid": bid or None,
                "ask": ask or None,
                "last": float(tick_data.get("last") or 0.0) or ((bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else None),
                "spread": spread,
                "spread_points": float(spread_points or 0.0),
                "tick_time": tick_time,
                "tick_time_iso": datetime.fromtimestamp(tick_time, tz=timezone.utc).isoformat() if tick_time > 0 else None,
                "volume": float(tick_data.get("volume") or tick_data.get("volume_real") or 0.0),
                "symbol_info": info,
                "raw": tick_data,
            }

    def get_rates(self, symbol: str, timeframe: Any, count: int) -> List[Dict[str, Any]]:
        with self._lock:
            self.ensure_symbol(symbol)
            mt5_symbol = self._resolve_symbol(symbol)
            tf_const = self._timeframe_to_mt5(timeframe)
            bars = max(1, int(count or 1))
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_const, 0, bars)
            if rates is None or len(rates) == 0:
                logger.warning(f"Failed to fetch candles for {mt5_symbol} {timeframe}")
                raise MT5ClientError(f"Failed to fetch candles for {mt5_symbol} {timeframe}")

            parsed: List[Dict[str, Any]] = []
            for row in rates:
                ts = int(row["time"])
                parsed.append(
                    {
                        "symbol": symbol,
                        "mt5_symbol": mt5_symbol,
                        "time": ts,
                        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "tick_volume": float(row["tick_volume"]),
                        "spread": float(row["spread"]),
                        "real_volume": float(row["real_volume"]),
                    }
                )

            logger.debug(f"Fetched {len(parsed)} MT5 candles for {mt5_symbol} {timeframe}")
            return parsed

    def get_spread(self, symbol: str) -> float:
        tick = self.get_tick(symbol)
        return float(tick.get("spread") or 0.0)

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        return self.ensure_symbol(symbol)

    def get_market_data(self, symbol: str, timeframe: Any = "1m", count: int = 100) -> Dict[str, Any]:
        tick = self.get_tick(symbol)
        rates = self.get_rates(symbol, timeframe, count)
        return {
            "symbol": symbol,
            "mt5_symbol": tick.get("mt5_symbol"),
            "bid": tick.get("bid"),
            "ask": tick.get("ask"),
            "spread": tick.get("spread"),
            "tick_time": tick.get("tick_time"),
            "symbol_info": tick.get("symbol_info"),
            "candles": rates,
        }

    def execute_trade(self, symbol: str, side: str, lot: float, sl: float, tp: float) -> Optional[Dict[str, Any]]:
        """Executes a market order with SL and TP directly on MT5."""
        with self._lock:
            info = self.ensure_symbol(symbol)
            mt5_symbol = self._resolve_symbol(symbol)

            # --- SMART VOLUME FORMATTING ---
            # Retrieve constraints for the symbol
            vol_min = info.get("volume_min", 0.01)
            vol_step = info.get("volume_step", 0.01)
            
            # Ensure volume is not below minimum limit
            lot = max(lot, vol_min)
            
            # Round volume to match the broker's step rules (e.g. 0.01 or 0.1)
            if vol_step > 0:
                lot = round(lot / vol_step) * vol_step
            # -------------------------------
            
            # Prepare MT5 order request
            action = mt5.ORDER_TYPE_BUY if side.upper() == 'LONG' else mt5.ORDER_TYPE_SELL
            
            # Fetch current tick to know exact execution price needed for formatting request properly if required,
            # though ORDER_TYPE_BUY/SELL usually uses live market.
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                logger.error(f"Cannot execute trade: No tick data for {mt5_symbol}")
                return None
                
            price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": float(lot),
                "type": action,
                "price": price,
                "sl": float(sl),
                "tp": float(tp),
                "deviation": 20,
                "magic": 202604, # Identifies bot trades
                "comment": "CryptoSignalBot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = _format_last_error(f"OrderSend failed (retcode={result.retcode})")
                logger.error(f"Trade execution failed for {mt5_symbol}: {error_msg} | Request: {request}")
                return None
                
            logger.info(f"Trade executed successfully: {side} {mt5_symbol} Lot: {lot} Entry: {price} SL: {sl} TP: {tp} Ticket: {result.order}")
            return _namedtuple_to_dict(result)

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modifies the stop loss of an open MT5 position."""
        with self._lock:
            pos = mt5.positions_get(ticket=ticket)
            if pos is None or len(pos) == 0:
                logger.error(f"Cannot modify SL: Ticket {ticket} not found")
                return False
                
            p = pos[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "symbol": p.symbol,
                "sl": float(new_sl),
                "tp": p.tp,
                "magic": p.magic,
            }
            
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"SL Modification failed for {p.symbol}: retcode={res.retcode}")
                return False
                
            logger.info(f"Successfully modified SL to {new_sl} for ticket {ticket}")
            return True

    def get_bot_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch active positions for this bot's magic number and symbol."""
        with self._lock:
            mt5_symbol = self._resolve_symbol(symbol)
            positions = mt5.positions_get(symbol=mt5_symbol)
            if positions is None:
                return []
            # filter by bot magic
            return [_namedtuple_to_dict(p) for p in positions if p.magic == 202604]

    def close_position(self, ticket: int) -> bool:
        """Close an open MT5 position by ticket using a market counter-order."""
        with self._lock:
            self._require_package()
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"close_position: ticket {ticket} not found or already closed")
                return False

            pos = positions[0]
            # Counter order: if LONG → SELL, if SHORT → BUY
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logger.error(f"close_position: no tick for {pos.symbol}, cannot close ticket {ticket}")
                return False

            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 30,
                "magic": 202604,
                "comment": "SessionClose",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                logger.error(f"close_position failed for ticket {ticket}: retcode={code}")
                return False

            logger.info(f"Position closed: ticket={ticket} symbol={pos.symbol} price={price}")
            return True

    def get_all_bot_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions placed by this bot (any symbol)."""
        with self._lock:
            self._require_package()
            self.connect_mt5()
            positions = mt5.positions_get()
            if positions is None:
                return []
            return [_namedtuple_to_dict(p) for p in positions if p.magic == 202604]

    # ═══════════════════════════════════════════════════════════════════════
    # Pending Order Management (BUY LIMIT / SELL LIMIT)
    # ═══════════════════════════════════════════════════════════════════════

    def _validate_broker_constraints(
        self, info: Dict[str, Any], order_type_str: str, price: float, sl: float, tp: float, lot: float,
        mt5_symbol: str = "",
    ) -> Dict[str, Any]:
        """Validate all broker constraints before placing a pending order.
        Returns dict with 'valid': bool and 'errors': list[str], 'lot': adjusted lot."""
        errors: List[str] = []

        # Trade mode check
        trade_mode = info.get("trade_mode", 4)
        if trade_mode == 0:
            errors.append(f"Trading disabled by broker (trade_mode=0)")
            return {"valid": False, "errors": errors, "lot": lot}

        # Volume constraints
        vol_min = float(info.get("volume_min", 0.01))
        vol_max = float(info.get("volume_max", 100.0))
        vol_step = float(info.get("volume_step", 0.01))

        adjusted_lot = max(lot, vol_min)
        adjusted_lot = min(adjusted_lot, vol_max)
        if vol_step > 0:
            adjusted_lot = round(adjusted_lot / vol_step) * vol_step

        if adjusted_lot < vol_min:
            errors.append(f"Lot {lot} below minimum {vol_min}")
        if adjusted_lot > vol_max:
            errors.append(f"Lot {lot} above maximum {vol_max}")

        # Stops level: minimum distance from price to SL/TP (in points)
        point = float(info.get("point", 0.00001))
        stops_level = int(info.get("trade_stops_level", 0))
        freeze_level = int(info.get("trade_freeze_level", 0))

        if point > 0 and stops_level > 0:
            min_distance = stops_level * point
            if sl > 0 and abs(price - sl) < min_distance:
                errors.append(
                    f"SL too close to entry: distance={abs(price - sl):.5f}, "
                    f"minimum={min_distance:.5f} (stops_level={stops_level})"
                )
            if tp > 0 and abs(price - tp) < min_distance:
                errors.append(
                    f"TP too close to entry: distance={abs(price - tp):.5f}, "
                    f"minimum={min_distance:.5f} (stops_level={stops_level})"
                )

        # Live tick checks: limit orders on the wrong side of the market are
        # rejected by MT5 with retcode 10015 (Invalid price), so catch it here
        # with a fresh tick instead of the (possibly stale) planner price.
        tick = mt5.symbol_info_tick(mt5_symbol) if mt5_symbol else None
        if tick is None:
            errors.append(f"No live tick available for '{mt5_symbol}'")
        else:
            ot = order_type_str.upper()
            current = tick.bid if "SELL" in ot else tick.ask
            if current > 0:
                if "BUY" in ot and price >= current:
                    errors.append(
                        f"BUY_LIMIT price ({price}) must be below current ask ({current})"
                    )
                elif "SELL" in ot and price <= current:
                    errors.append(
                        f"SELL_LIMIT price ({price}) must be above current bid ({current})"
                    )

                # Freeze level: minimum distance from current price to pending order price
                if point > 0 and freeze_level > 0 and abs(current - price) < freeze_level * point:
                    errors.append(
                        f"Pending order price too close to market: "
                        f"distance={abs(current - price):.5f}, "
                        f"minimum={freeze_level * point:.5f} (freeze_level={freeze_level})"
                    )

        return {"valid": len(errors) == 0, "errors": errors, "lot": adjusted_lot}

    def place_pending_order(
        self,
        symbol: str,
        order_type: str,
        lot: float,
        price: float,
        sl: float,
        tp: float,
        expiry_hours: int = 8,
    ) -> Optional[Dict[str, Any]]:
        """Place a pending limit order (BUY_LIMIT or SELL_LIMIT) with full broker validation.

        Args:
            symbol: Trading symbol
            order_type: 'BUY_LIMIT' or 'SELL_LIMIT'
            lot: Volume
            price: Pending order trigger price
            sl: Stop loss price
            tp: Take profit price
            expiry_hours: Hours until the order auto-cancels (0 = GTC)

        Returns:
            Order result dict or None on failure
        """
        with self._lock:
            self._require_package()
            info = self.ensure_symbol(symbol)
            mt5_symbol = self._resolve_symbol(symbol)

            # Map order type
            ot = order_type.upper().replace(" ", "_")
            if ot == "BUY_LIMIT":
                mt5_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif ot == "SELL_LIMIT":
                mt5_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                logger.error(f"Invalid pending order type: {order_type}")
                return None

            # Broker constraint validation (includes live-tick price side check)
            validation = self._validate_broker_constraints(info, ot, price, sl, tp, lot, mt5_symbol)
            if not validation["valid"]:
                for err in validation["errors"]:
                    logger.error(f"Broker constraint violation [{mt5_symbol}]: {err}")
                return None
            adjusted_lot = validation["lot"]

            # Build expiry
            import time as _time
            type_time = mt5.ORDER_TIME_GTC
            expiration = 0
            if expiry_hours > 0:
                type_time = mt5.ORDER_TIME_SPECIFIED
                expiration = int(_time.time()) + (expiry_hours * 3600)

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": mt5_symbol,
                "volume": float(adjusted_lot),
                "type": mt5_type,
                "price": float(price),
                "sl": float(sl),
                "tp": float(tp),
                "deviation": 0,
                "magic": 202604,
                "comment": "SMC_PendingOrder",
                "type_time": type_time,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            if expiration > 0:
                request["expiration"] = expiration

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                comment = getattr(result, "comment", "") if result else ""
                logger.error(
                    f"Pending order failed [{mt5_symbol}]: retcode={code} {comment} | "
                    f"Request: type={ot} price={price} sl={sl} tp={tp} lot={adjusted_lot}"
                )
                return None

            logger.info(
                f"Pending order placed: {ot} {mt5_symbol} Lot={adjusted_lot} "
                f"Price={price} SL={sl} TP={tp} Ticket={result.order} Expiry={expiry_hours}h"
            )
            return _namedtuple_to_dict(result)

    def cancel_pending_order(self, ticket: int) -> bool:
        """Cancel a pending order by ticket number."""
        with self._lock:
            self._require_package()
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                logger.error(f"Cancel pending order failed: ticket={ticket} retcode={code}")
                return False

            logger.info(f"Pending order cancelled: ticket={ticket}")
            return True

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active pending orders for this bot, optionally filtered by symbol."""
        with self._lock:
            self._require_package()
            self.connect_mt5()

            if symbol:
                mt5_symbol = self._resolve_symbol(symbol)
                orders = mt5.orders_get(symbol=mt5_symbol)
            else:
                orders = mt5.orders_get()

            if orders is None:
                return []

            # Filter by bot magic number
            return [_namedtuple_to_dict(o) for o in orders if o.magic == 202604]

    def get_closed_positions_history(self, lookback_days: int = 3) -> List[Dict[str, Any]]:
        """Aggregate closed bot positions (magic 202604) from MT5 deal history.

        Returns one dict per fully closed position:
        position_id, open_order_ticket, symbol, side (LONG/SHORT), volume,
        entry_price, close_price, profit (account currency, incl. swap+commission),
        open_time, close_time (unix), close_reason ('TP' | 'SL' | 'OTHER').
        """
        with self._lock:
            self._require_package()
            self.connect_mt5()

            date_from = datetime.now() - timedelta(days=max(1, lookback_days))
            date_to = datetime.now() + timedelta(days=1)
            deals = mt5.history_deals_get(date_from, date_to)
            if deals is None:
                return []

            reason_sl = getattr(mt5, "DEAL_REASON_SL", 4)
            reason_tp = getattr(mt5, "DEAL_REASON_TP", 5)
            entry_in = getattr(mt5, "DEAL_ENTRY_IN", 0)
            entry_outs = {getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)}

            positions: Dict[int, Dict[str, Any]] = {}
            for d in deals:
                if d.magic != 202604 or not d.position_id:
                    continue
                pos = positions.setdefault(int(d.position_id), {
                    "position_id": int(d.position_id),
                    "open_order_ticket": 0,
                    "symbol": d.symbol,
                    "side": "",
                    "volume": 0.0,
                    "entry_price": 0.0,
                    "close_price": 0.0,
                    "profit": 0.0,
                    "open_time": 0,
                    "close_time": 0,
                    "close_reason": "OTHER",
                    "_in_volume": 0.0,
                    "_out_volume": 0.0,
                })
                pos["profit"] += float(d.profit) + float(d.swap) + float(d.commission)
                if d.entry == entry_in:
                    pos["side"] = "LONG" if d.type == mt5.ORDER_TYPE_BUY else "SHORT"
                    pos["entry_price"] = float(d.price)
                    pos["open_time"] = int(d.time)
                    pos["open_order_ticket"] = int(d.order)
                    pos["_in_volume"] += float(d.volume)
                elif d.entry in entry_outs:
                    pos["close_price"] = float(d.price)
                    pos["close_time"] = max(pos["close_time"], int(d.time))
                    pos["_out_volume"] += float(d.volume)
                    if d.reason == reason_sl:
                        pos["close_reason"] = "SL"
                    elif d.reason == reason_tp:
                        pos["close_reason"] = "TP"

            closed = []
            for pos in positions.values():
                # Fully closed: out volume covers in volume (with float tolerance)
                if pos["_in_volume"] > 0 and pos["_out_volume"] >= pos["_in_volume"] - 1e-8:
                    pos["volume"] = pos["_in_volume"]
                    pos.pop("_in_volume", None)
                    pos.pop("_out_volume", None)
                    closed.append(pos)
            return closed

    def modify_pending_order(
        self,
        ticket: int,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Modify price, SL, or TP of an existing pending order."""
        with self._lock:
            self._require_package()
            orders = mt5.orders_get(ticket=ticket)
            if not orders:
                logger.error(f"Pending order not found: ticket={ticket}")
                return False

            order = orders[0]
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": ticket,
                "symbol": order.symbol,
                "price": float(price) if price is not None else order.price_open,
                "sl": float(sl) if sl is not None else order.sl,
                "tp": float(tp) if tp is not None else order.tp,
                "type_time": order.type_time,
                "expiration": order.time_expiration,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                code = result.retcode if result else "None"
                logger.error(f"Modify pending order failed: ticket={ticket} retcode={code}")
                return False

            logger.info(f"Pending order modified: ticket={ticket} price={request['price']} sl={request['sl']} tp={request['tp']}")
            return True


def connect_mt5(
    login: Any,
    password: str,
    server: str,
    *,
    path: Optional[str] = None,
    timeout_seconds: float = 10.0,
    symbol_suffix: str = "",
    symbol_map: Optional[Dict[str, str]] = None,
    default_symbol: str = "",
) -> MT5Client:
    client = MT5Client(
        login=login,
        password=password,
        server=server,
        path=path,
        timeout_seconds=timeout_seconds,
        symbol_suffix=symbol_suffix,
        symbol_map=symbol_map,
        default_symbol=default_symbol,
    )
    client.connect_mt5()
    return client


def ensure_symbol(client: MT5Client, symbol: str) -> Dict[str, Any]:
    return client.ensure_symbol(symbol)


def get_tick(client: MT5Client, symbol: str) -> Dict[str, Any]:
    return client.get_tick(symbol)


def get_rates(client: MT5Client, symbol: str, timeframe: Any, count: int) -> List[Dict[str, Any]]:
    return client.get_rates(symbol, timeframe, count)


def get_spread(client: MT5Client, symbol: str) -> float:
    return client.get_spread(symbol)


def shutdown_mt5(client: MT5Client) -> None:
    client.shutdown_mt5()