import json
import os
from datetime import datetime, timedelta, timezone
from typing import Tuple
from loguru import logger
from .utils import PortfolioRiskConfig
from .storage import SignalStorage
import sqlite3

class EquityProtection:
    def __init__(self, config: PortfolioRiskConfig):
        self.config = config
        self.state_file = config.state_file
        self._load_state()

    def _load_state(self):
        self.peak_balance = 0.0
        self.consecutive_wins = 0
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.peak_balance = data.get('peak_balance', 0.0)
                    self.consecutive_wins = data.get('consecutive_wins', 0)
            except Exception as e:
                logger.warning(f"Failed to load equity state: {e}")

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'peak_balance': self.peak_balance,
                    'consecutive_wins': self.consecutive_wins
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save equity state: {e}")

    def track_peak_balance(self, current_balance: float) -> None:
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self._save_state()
            logger.debug(f"New peak balance recorded: {self.peak_balance}")

    def get_weekly_pnl(self, storage: SignalStorage) -> float:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        try:
            with sqlite3.connect(storage.db_path) as conn:
                result = conn.execute(
                    "SELECT SUM(pnl_pct) FROM signal_outcomes WHERE closed_at >= ? AND pnl_pct IS NOT NULL",
                    (cutoff,)
                ).fetchone()
                return float(result[0] or 0.0)
        except Exception as e:
            logger.error(f"Error calculating weekly PnL: {e}")
            return 0.0

    def get_cumulative_pnl(self, storage: SignalStorage) -> float:
        try:
            with sqlite3.connect(storage.db_path) as conn:
                result = conn.execute(
                    "SELECT SUM(pnl_pct) FROM signal_outcomes WHERE pnl_pct IS NOT NULL"
                ).fetchone()
                return float(result[0] or 0.0)
        except Exception as e:
            logger.error(f"Error calculating cumulative PnL: {e}")
            return 0.0

    def check_protection(self, storage: SignalStorage, current_balance: float = 0.0) -> Tuple[bool, str]:
        if not self.config.enabled:
            return True, ""
            
        # Use cumulative pnl as proxy for balance if actual balance not provided properly
        if current_balance <= 0.0:
            current_balance = self.get_cumulative_pnl(storage)
            
        self.track_peak_balance(current_balance)
        
        # Weekly loss limit check
        weekly_pnl = self.get_weekly_pnl(storage)
        if weekly_pnl <= self.config.weekly_loss_limit_pct:
            if self.config.pause_on_drawdown_breach:
                reason = f"Weekly loss limit breached: {weekly_pnl:.2f}% <= {self.config.weekly_loss_limit_pct}%"
                logger.warning(reason)
                return False, reason
                
        # Drawdown from peak check
        if self.peak_balance > 0 or current_balance < 0:
            drawdown = current_balance - self.peak_balance
            if abs(self.peak_balance) > 100:
                dd_pct = (drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0
            else:
                dd_pct = drawdown # it's already a pct proxy
                
            if dd_pct <= self.config.max_drawdown_from_peak_pct:
                if self.config.pause_on_drawdown_breach:
                    reason = f"Max drawdown from peak breached: {dd_pct:.2f}% <= {self.config.max_drawdown_from_peak_pct}%"
                    logger.warning(reason)
                    return False, reason

        return True, ""

    def mark_consecutive_wins(self, won: bool) -> bool:
        """Update consecutive wins and return True if we reached the resume threshold."""
        if won:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
            
        self._save_state()
        logger.debug(f"Consecutive wins updated: {self.consecutive_wins}")
        
        if self.consecutive_wins >= self.config.resume_after_consecutive_wins:
            return True
        return False

    def reset_after_breach(self) -> None:
        # Reset peak to current to allow trading again
        self.consecutive_wins = 0
        self._save_state()
