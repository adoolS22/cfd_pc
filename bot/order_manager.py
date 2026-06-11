"""
Order Lifecycle Manager
=======================
Manages the full lifecycle of pending orders:
  create → monitor → cancel/expire → track filled orders

Prevents conflicts (no dual BUY+SELL on same symbol),
handles auto-expiry, and coordinates with MT5 and storage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .pending_order_planner import PendingOrderDecision


class OrderLifecycleManager:
    """Manages pending order placement, cancellation, and monitoring."""

    def __init__(
        self,
        max_pending_per_symbol: int = 1,
        default_expiry_hours: int = 8,
    ):
        self.max_pending_per_symbol = max(1, max_pending_per_symbol)
        self.default_expiry_hours = max(1, default_expiry_hours)

    def process_decision(
        self,
        decision: PendingOrderDecision,
        mt5_client: Any,
        storage: Any,
        notifier: Any,
        lot: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        """Process a planner decision and execute the appropriate action.

        Args:
            decision: The validated PendingOrderDecision from the planner
            mt5_client: Connected MT5Client instance
            storage: Storage instance for persistence
            notifier: Notifier instance for Telegram alerts
            lot: Volume for the pending order

        Returns:
            Result dict or None
        """
        symbol = decision.symbol

        if decision.decision == "PLACE_BUY_LIMIT":
            return self._place_order(decision, mt5_client, storage, notifier, lot)

        elif decision.decision == "PLACE_SELL_LIMIT":
            return self._place_order(decision, mt5_client, storage, notifier, lot)

        elif decision.decision == "CANCEL_PENDING_ORDER":
            return self._cancel_all_orders(symbol, mt5_client, storage, notifier, decision.reason)

        elif decision.decision == "KEEP_PENDING_ORDER":
            logger.debug(f"OrderManager: keeping existing orders for {symbol}")
            return {"action": "keep", "symbol": symbol}

        elif decision.decision == "MANAGE_OPEN_TRADE":
            logger.info(f"OrderManager: MANAGE_OPEN_TRADE for {symbol} — {decision.reason}")
            return {"action": "manage", "symbol": symbol, "reason": decision.reason}

        else:
            # NO_TRADE or WAIT_FOR_SETUP
            logger.debug(f"OrderManager: {decision.decision} for {symbol}")
            return None

    def _place_order(
        self,
        decision: PendingOrderDecision,
        mt5_client: Any,
        storage: Any,
        notifier: Any,
        lot: float,
    ) -> Optional[Dict[str, Any]]:
        """Place a pending limit order after conflict checks."""
        symbol = decision.symbol
        order_type = decision.order_type  # BUY_LIMIT or SELL_LIMIT

        # Conflict check: cancel existing orders on same symbol first
        try:
            existing = mt5_client.get_pending_orders(symbol)
            if existing:
                # Check if an almost identical order already exists
                new_price = float(decision.entry_price or 0.0)
                new_type_code = 2 if "BUY" in order_type.upper() else 3
                
                for order in existing:
                    existing_price = float(order.get("price_open", 0.0))
                    existing_type = order.get("type", -1)
                    
                    if existing_price > 0 and new_price > 0 and existing_type == new_type_code:
                        diff_pct = abs(existing_price - new_price) / existing_price * 100.0
                        if diff_pct < 0.1:  # Within 0.1% price difference
                            logger.info(f"OrderManager: ⏸️ Skipping new {order_type} for {symbol}, similar order already exists at {existing_price} (diff {diff_pct:.3f}%)")
                            return {"action": "skipped_duplicate", "symbol": symbol, "ticket": order.get("ticket")}
                            
                if len(existing) >= self.max_pending_per_symbol:
                    logger.info(
                        f"OrderManager: cancelling {len(existing)} existing order(s) for {symbol} "
                        f"before placing new {order_type}"
                    )
                    for order in existing:
                        ticket = order.get("ticket")
                        if ticket:
                            mt5_client.cancel_pending_order(ticket)
        except Exception as e:
            logger.warning(f"OrderManager: conflict check failed for {symbol}: {e}")

        # Place the order
        expiry = decision.valid_until_hours or self.default_expiry_hours

        try:
            result = mt5_client.place_pending_order(
                symbol=symbol,
                order_type=order_type,
                lot=lot,
                price=decision.entry_price,
                sl=decision.stop_loss,
                tp=decision.take_profit_1,
                expiry_hours=expiry,
            )
        except Exception as e:
            logger.error(f"OrderManager: failed to place {order_type} for {symbol}: {e}")
            return None

        if result is None:
            logger.error(f"OrderManager: MT5 rejected {order_type} for {symbol}")
            return None

        ticket = result.get("order", 0)
        logger.info(
            f"OrderManager: ✅ {order_type} placed for {symbol} | "
            f"Entry={decision.entry_price} SL={decision.stop_loss} "
            f"TP1={decision.take_profit_1} R:R={decision.risk_to_reward} "
            f"Score={decision.score} Ticket={ticket}"
        )

        # Save to database
        try:
            if hasattr(storage, "save_pending_order"):
                storage.save_pending_order({
                    "symbol": symbol,
                    "order_type": order_type,
                    "entry_price": decision.entry_price,
                    "stop_loss": decision.stop_loss,
                    "take_profit_1": decision.take_profit_1,
                    "take_profit_2": decision.take_profit_2,
                    "lot": lot,
                    "mt5_ticket": ticket,
                    "status": "PENDING",
                    "score": decision.score,
                    "setup_quality": decision.setup_quality,
                    "reason": decision.reason,
                    "forecast_scenario": decision.full_analysis.get("forecast", {}).get("most_likely_scenario", ""),
                    "valid_until_hours": expiry,
                    "llm_analysis": decision.full_analysis,
                })
        except Exception as e:
            logger.warning(f"OrderManager: save_pending_order failed: {e}")

        # Send Telegram notification
        try:
            if notifier and hasattr(notifier, "send_text"):
                direction_emoji = "🟢" if "BUY" in order_type else "🔴"
                msg = (
                    f"{direction_emoji} <b>أمر معلق جديد: {order_type}</b>\n"
                    f"📊 الرمز: <b>{symbol}</b>\n"
                    f"💰 سعر الدخول: <code>{decision.entry_price}</code>\n"
                    f"🛑 وقف الخسارة: <code>{decision.stop_loss}</code>\n"
                    f"🎯 الهدف الأول: <code>{decision.take_profit_1}</code>\n"
                    f"📐 نسبة العائد/المخاطرة: <code>1:{decision.risk_to_reward}</code>\n"
                    f"⭐ الجودة: {decision.setup_quality} (Score: {decision.score})\n"
                    f"⏰ صالح لمدة: {expiry} ساعات\n"
                    f"🎫 التذكرة: <code>{ticket}</code>\n"
                    f"\n📝 {decision.reason[:200] if decision.reason else ''}"
                )
                notifier.send_text(msg)
        except Exception as e:
            logger.warning(f"OrderManager: Telegram notification failed: {e}")

        return {
            "action": "placed",
            "order_type": order_type,
            "symbol": symbol,
            "ticket": ticket,
            "entry_price": decision.entry_price,
            "stop_loss": decision.stop_loss,
            "take_profit_1": decision.take_profit_1,
            "risk_to_reward": decision.risk_to_reward,
            "score": decision.score,
        }

    def _cancel_all_orders(
        self,
        symbol: str,
        mt5_client: Any,
        storage: Any,
        notifier: Any,
        reason: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Cancel all pending orders for a symbol."""
        try:
            orders = mt5_client.get_pending_orders(symbol)
        except Exception as e:
            logger.error(f"OrderManager: failed to get orders for {symbol}: {e}")
            return None

        if not orders:
            logger.debug(f"OrderManager: no pending orders to cancel for {symbol}")
            return {"action": "no_orders", "symbol": symbol}

        cancelled = 0
        for order in orders:
            ticket = order.get("ticket")
            if ticket:
                try:
                    if mt5_client.cancel_pending_order(ticket):
                        cancelled += 1
                        # Update DB
                        if hasattr(storage, "update_pending_order_status"):
                            storage.update_pending_order_status(ticket, "CANCELLED")
                except Exception as e:
                    logger.error(f"OrderManager: cancel failed for ticket {ticket}: {e}")

        if cancelled > 0:
            logger.info(f"OrderManager: ❌ cancelled {cancelled} order(s) for {symbol}: {reason}")
            try:
                if notifier and hasattr(notifier, "send_text"):
                    notifier.send_text(
                        f"❌ <b>إلغاء أوامر معلقة:</b> {symbol}\n"
                        f"عدد الأوامر الملغاة: {cancelled}\n"
                        f"السبب: {reason[:200] if reason else 'غير محدد'}"
                    )
            except Exception:
                pass

        return {"action": "cancelled", "symbol": symbol, "count": cancelled, "reason": reason}

    def cleanup_expired_orders(
        self,
        mt5_client: Any,
        storage: Any,
        notifier: Any,
    ) -> int:
        """Check and cancel expired pending orders. Returns count of cancelled orders."""
        try:
            all_orders = mt5_client.get_pending_orders()
        except Exception:
            return 0

        cancelled = 0
        now = int(time.time())

        for order in all_orders:
            expiry = int(order.get("time_expiration", 0))
            # MT5 handles its own expiry, but we double-check stale orders
            # that might have time_expiration=0 (GTC) but are too old
            setup_time = int(order.get("time_setup", 0))
            if expiry == 0 and setup_time > 0:
                age_hours = (now - setup_time) / 3600
                if age_hours > 24:  # Auto-cancel GTC orders after 24 hours
                    ticket = order.get("ticket")
                    if ticket and mt5_client.cancel_pending_order(ticket):
                        cancelled += 1
                        logger.info(f"OrderManager: auto-cancelled stale GTC order ticket={ticket} (age={age_hours:.1f}h)")
                        if hasattr(storage, "update_pending_order_status"):
                            storage.update_pending_order_status(ticket, "EXPIRED")

        return cancelled
