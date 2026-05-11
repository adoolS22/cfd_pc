"""
Signal Storage and Cooldown Management
=======================================
SQLite-based storage for tracking signals and enforcing cooldowns.
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict, field
from loguru import logger


@dataclass
class SignalRecord:
    """Represents a stored signal."""
    symbol: str
    side: str  # 'LONG', 'SHORT', 'EXIT'
    timestamp: datetime
    score: int
    entry: float
    stop_loss: float
    take_profit_near: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
    # ML and Order Flow tracking
    ml_win_probability: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


class SignalStorage:
    """SQLite-based signal storage and cooldown manager."""
    
    def __init__(self, db_path: str = "signals.db"):
        """
        Initialize storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    score INTEGER,
                    entry REAL,
                    stop_loss REAL,
                    take_profit_near REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    reasons TEXT
                )
            """)
            
            # Cooldown tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cooldowns (
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    last_signal_time TEXT NOT NULL,
                    PRIMARY KEY (symbol, side)
                )
            """)
            
            # Outcome tracking table (Win/Loss tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry REAL,
                    stop_loss REAL,
                    take_profit_near REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    outcome TEXT DEFAULT 'OPEN',
                    close_price REAL,
                    pnl_pct REAL,
                    closed_at TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)

            # LLM postmortem reviews (one review per closed outcome)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_trade_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    outcome_id INTEGER NOT NULL UNIQUE,
                    signal_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    outcome TEXT,
                    pnl_pct REAL,
                    verdict TEXT,
                    action TEXT,
                    confidence INTEGER,
                    penalty REAL,
                    mistake_tags TEXT,
                    summary TEXT,
                    recommendation TEXT,
                    raw_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (outcome_id) REFERENCES signal_outcomes(id),
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
                ON signals(symbol, timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_symbol 
                ON signal_outcomes(symbol, outcome)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_reviews_symbol_side_time
                ON llm_trade_reviews(symbol, side, created_at DESC)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    ideal_entry REAL NOT NULL,
                    zone_top REAL,
                    zone_bottom REAL,
                    atr REAL,
                    original_score REAL,
                    original_reasons TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_entries_symbol
                ON pending_entries(symbol)
            """)

            # Backward-compatible migrations for existing DB files.
            self._ensure_column(cursor, "signals", "take_profit_near", "REAL")
            self._ensure_column(cursor, "signal_outcomes", "take_profit_near", "REAL")
            self._ensure_column(cursor, "signal_outcomes", "tp1_touched", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "signal_outcomes", "break_even_armed", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "signal_outcomes", "trail_stop", "REAL")
            self._ensure_column(cursor, "signal_outcomes", "extreme_price", "REAL")
            self._ensure_column(cursor, "signal_outcomes", "trail_armed_at", "TEXT")
            # Partial exit columns (60/40 split at TP1)
            self._ensure_column(cursor, "signal_outcomes", "partial_exit_done", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "signal_outcomes", "tp1_partial_pnl", "REAL")
            self._ensure_column(cursor, "llm_trade_reviews", "raw_json", "TEXT")
            
            # ML & Order Flow columns
            self._ensure_column(cursor, "signals", "ml_win_probability", "REAL")
            self._ensure_column(cursor, "signals", "funding_rate", "REAL")
            self._ensure_column(cursor, "signals", "open_interest", "REAL")

            conn.commit()
        
        logger.debug(f"Initialized signal storage at {self.db_path}")

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
        """Add a column when missing (SQLite migration helper)."""
        rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
        names = {str(r[1]) for r in rows if len(r) > 1}
        if column not in names:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    
    def save_signal(self, signal: SignalRecord) -> int:
        """
        Save a signal to the database.
        
        Args:
            signal: SignalRecord to save
            
        Returns:
            ID of the inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals 
                (symbol, side, timestamp, score, entry, stop_loss, take_profit_near, take_profit_1, take_profit_2, reasons, ml_win_probability, funding_rate, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.side,
                signal.timestamp.isoformat(),
                signal.score,
                signal.entry,
                signal.stop_loss,
                signal.take_profit_near,
                signal.take_profit_1,
                signal.take_profit_2,
                json.dumps(signal.reasons),
                signal.ml_win_probability,
                signal.funding_rate,
                signal.open_interest
            ))
            # IMPORTANT: capture signal_id immediately after inserting into `signals`.
            # `lastrowid` changes after any subsequent INSERT/REPLACE (e.g. cooldown row).
            signal_id = int(cursor.lastrowid)
            
            # Update cooldown
            cursor.execute("""
                INSERT OR REPLACE INTO cooldowns (symbol, side, last_signal_time)
                VALUES (?, ?, ?)
            """, (signal.symbol, signal.side, signal.timestamp.isoformat()))

            # Register LONG/SHORT signals for outcome tracking
            if signal.side in ('LONG', 'SHORT') and signal.entry and signal.stop_loss:
                cursor.execute("""
                    INSERT INTO signal_outcomes
                    (
                        signal_id, symbol, side, entry, stop_loss,
                        take_profit_near, take_profit_1, take_profit_2,
                        outcome, tp1_touched, break_even_armed, trail_stop, extreme_price
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 0, 0, NULL, NULL)
                """, (
                    signal_id,
                    signal.symbol,
                    signal.side,
                    signal.entry,
                    signal.stop_loss,
                    signal.take_profit_near,
                        signal.take_profit_1,
                        signal.take_profit_2
                    ))
            elif signal.side in ('LONG', 'SHORT'):
                logger.warning(
                    "Skipping outcome registration for "
                    f"{signal.symbol}: entry={signal.entry} stop_loss={signal.stop_loss}"
                )
            
            conn.commit()
            
            logger.info(f"Saved {signal.side} signal for {signal.symbol}")
            return signal_id

    def repair_recent_missing_open_outcomes(self, lookback_hours: int = 2) -> int:
        """
        Backfill missing OPEN outcomes for very recent LONG/SHORT signals.

        This is a safety net for runtime glitches so outcome tracking and reports
        remain consistent with newly inserted signals.
        """
        hours = max(1, int(lookback_hours))
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO signal_outcomes
                (
                    signal_id, symbol, side, entry, stop_loss,
                    take_profit_near, take_profit_1, take_profit_2,
                    outcome, tp1_touched, break_even_armed, trail_stop, extreme_price
                )
                SELECT
                    s.id,
                    s.symbol,
                    s.side,
                    s.entry,
                    s.stop_loss,
                    COALESCE(s.take_profit_near, 0),
                    COALESCE(s.take_profit_1, 0),
                    COALESCE(s.take_profit_2, 0),
                    'OPEN',
                    0,
                    0,
                    NULL,
                    NULL
                FROM signals s
                WHERE s.side IN ('LONG', 'SHORT')
                  AND s.timestamp >= ?
                  AND COALESCE(s.entry, 0) > 0
                  AND COALESCE(s.stop_loss, 0) > 0
                  AND NOT EXISTS (
                    SELECT 1
                    FROM signal_outcomes so
                    WHERE so.signal_id = s.id
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM signals sx
                    WHERE sx.symbol = s.symbol
                      AND sx.side = 'EXIT'
                      AND sx.timestamp >= s.timestamp
                  )
                """,
                (cutoff,),
            )
            inserted = int(cursor.rowcount or 0)
            conn.commit()

        if inserted > 0:
            logger.warning(
                f"Backfilled {inserted} missing OPEN outcomes from last {hours}h "
                "(repair_recent_missing_open_outcomes)"
            )
        return inserted
    
    def check_cooldown(
        self,
        symbol: str,
        side: str,
        cooldown_minutes: int
    ) -> bool:
        """
        Check if a signal is in cooldown.
        
        Args:
            symbol: Trading symbol
            side: Signal side ('LONG', 'SHORT')
            cooldown_minutes: Cooldown period in minutes
            
        Returns:
            True if still in cooldown, False if allowed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_signal_time FROM cooldowns
                WHERE symbol = ? AND side = ?
            """, (symbol, side))
            
            result = cursor.fetchone()
            
            if not result:
                return False  # No previous signal, not in cooldown
            
            last_time = datetime.fromisoformat(result[0])
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            
            cooldown_end = last_time + timedelta(minutes=cooldown_minutes)
            now = datetime.now(timezone.utc)
            
            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds() / 60
                logger.debug(f"{symbol} {side} in cooldown for {remaining:.1f} more minutes")
                return True
            
            return False
    
    def get_last_signal(self, symbol: str, side: Optional[str] = None) -> Optional[SignalRecord]:
        """
        Get the most recent signal for a symbol.
        
        Args:
            symbol: Trading symbol
            side: Optional filter by side
            
        Returns:
            SignalRecord or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if side:
                cursor.execute("""
                    SELECT symbol, side, timestamp, score, entry, stop_loss, 
                           take_profit_near, take_profit_1, take_profit_2, reasons
                    FROM signals
                    WHERE symbol = ? AND side = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (symbol, side))
            else:
                cursor.execute("""
                    SELECT symbol, side, timestamp, score, entry, stop_loss, 
                           take_profit_near, take_profit_1, take_profit_2, reasons
                    FROM signals
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (symbol,))
            
            result = cursor.fetchone()
            
            if not result:
                return None
            
            return SignalRecord(
                symbol=result[0],
                side=result[1],
                timestamp=datetime.fromisoformat(result[2]),
                score=result[3],
                entry=result[4],
                stop_loss=result[5],
                take_profit_near=result[6] or 0.0,
                take_profit_1=result[7],
                take_profit_2=result[8],
                reasons=json.loads(result[9]) if result[9] else []
            )
    
    def get_recent_signals(self, hours: int = 24, symbol: Optional[str] = None) -> List[SignalRecord]:
        """
        Get recent signals.
        
        Args:
            hours: Lookback period in hours
            symbol: Optional filter by symbol
            
        Returns:
            List of SignalRecord objects
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("""
                    SELECT symbol, side, timestamp, score, entry, stop_loss, 
                           take_profit_near, take_profit_1, take_profit_2, reasons
                    FROM signals
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (symbol, cutoff.isoformat()))
            else:
                cursor.execute("""
                    SELECT symbol, side, timestamp, score, entry, stop_loss, 
                           take_profit_near, take_profit_1, take_profit_2, reasons
                    FROM signals
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (cutoff.isoformat(),))
            
            results = cursor.fetchall()
            
            signals = []
            for r in results:
                signals.append(SignalRecord(
                    symbol=r[0],
                    side=r[1],
                    timestamp=datetime.fromisoformat(r[2]),
                    score=r[3],
                    entry=r[4],
                    stop_loss=r[5],
                    take_profit_near=r[6] or 0.0,
                    take_profit_1=r[7],
                    take_profit_2=r[8],
                    reasons=json.loads(r[9]) if r[9] else []
                ))
            
            return signals
    
    def clear_cooldown(self, symbol: str, side: str) -> None:
        """
        Clear cooldown for a symbol/side combination.
        
        Args:
            symbol: Trading symbol
            side: Signal side
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM cooldowns WHERE symbol = ? AND side = ?
            """, (symbol, side))
            conn.commit()

    def save_pending_entry(self, symbol: str, side: str, ideal_entry: float, zone_top: float, zone_bottom: float, atr: float, score: float, reasons: List[str], expires_minutes: int) -> int:
        """Save a pending entry opportunity for pullback tracking."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=expires_minutes)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pending_entries
                (symbol, side, ideal_entry, zone_top, zone_bottom, atr, original_score, original_reasons, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, side, ideal_entry, zone_top, zone_bottom, atr, score,
                json.dumps(reasons), now.isoformat(), expires.isoformat()
            ))
            conn.commit()
            return int(cursor.lastrowid)

    def get_pending_entries(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active pending entries, cleaning up expired ones first."""
        self.cleanup_expired_pending_entries()
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM pending_entries WHERE expires_at > ?"
            params = [now]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            cursor.execute(query, tuple(params))
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                if d.get('original_reasons'):
                    try:
                        d['original_reasons'] = json.loads(d['original_reasons'])
                    except:
                        pass
                results.append(d)
            return results

    def delete_pending_entry(self, entry_id: int) -> None:
        """Delete a specific pending entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pending_entries WHERE id = ?", (entry_id,))
            conn.commit()

    def cleanup_expired_pending_entries(self) -> None:
        """Delete all expired pending entries."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pending_entries WHERE expires_at <= ?", (now,))
            conn.commit()


    
    def has_open_position(self, symbol: str) -> Optional[str]:
        """
        Check if a symbol currently has an open tracked position.

        Priority:
        1. If latest signal is EXIT -> no open position.
        2. If outcome tracking exists, trust OPEN outcomes.
        3. Legacy fallback: latest LONG/SHORT signal when no outcomes exist.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            'LONG', 'SHORT', or None
        """
        pos = self.get_open_position_details(symbol)
        return pos["side"] if pos else None

    def get_open_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get open position side and opened_at timestamp for a symbol.

        Priority:
        1. If latest signal is EXIT -> no open position.
        2. If OPEN outcome exists -> use that side + original signal timestamp.
        3. Legacy fallback: latest LONG/SHORT signal.

        Returns:
            Dict: {"side": "LONG|SHORT", "opened_at": datetime} or None
        """
        latest_side = None
        latest_ts: Optional[datetime] = None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT side, timestamp
                FROM signals
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            last_row = cursor.fetchone()
            if last_row:
                latest_side = last_row[0]
                latest_ts = datetime.fromisoformat(last_row[1])
                if latest_ts.tzinfo is None:
                    latest_ts = latest_ts.replace(tzinfo=timezone.utc)

            if latest_side == 'EXIT':
                return None

            cursor.execute("""
                SELECT so.side, s.timestamp
                FROM signal_outcomes so
                JOIN signals s ON s.id = so.signal_id
                WHERE so.symbol = ? AND so.outcome = 'OPEN'
                ORDER BY so.id DESC
                LIMIT 1
            """, (symbol,))
            open_row = cursor.fetchone()
            if open_row:
                opened_at = datetime.fromisoformat(open_row[1])
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                return {
                    "side": open_row[0],
                    "opened_at": opened_at,
                }

            cursor.execute("""
                SELECT COUNT(*)
                FROM signal_outcomes
                WHERE symbol = ?
            """, (symbol,))
            outcomes_count = cursor.fetchone()[0]
            if outcomes_count > 0:
                return None

        if latest_side in ('LONG', 'SHORT') and latest_ts is not None:
            # Legacy fallback is only safe for recent signals.
            # Very old signals (without outcome rows) should not block new entries forever.
            fallback_max_age = timedelta(hours=12)
            if (datetime.now(timezone.utc) - latest_ts) <= fallback_max_age:
                return {"side": latest_side, "opened_at": latest_ts}
            logger.debug(
                f"Ignoring stale legacy open position for {symbol}: "
                f"latest {latest_side} at {latest_ts.isoformat()}"
            )
        return None
    
    def get_open_outcomes(self) -> List[Dict]:
        """
        Get all OPEN signal outcomes for price checking.
        
        Returns:
            List of dicts with signal tracking data
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    so.id,
                    so.signal_id,
                    so.symbol,
                    so.side,
                    so.entry,
                    so.stop_loss,
                    so.take_profit_near,
                    so.take_profit_1,
                    so.take_profit_2,
                    so.tp1_touched,
                    so.break_even_armed,
                    so.trail_stop,
                    so.extreme_price,
                    so.trail_armed_at,
                    s.timestamp AS signal_timestamp,
                    s.reasons AS signal_reasons
                FROM signal_outcomes so
                LEFT JOIN signals s ON s.id = so.signal_id
                WHERE so.outcome = 'OPEN'
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_closed_outcome_context(self, outcome_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a closed outcome row with linked signal context for LLM postmortem.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT
                    so.id AS outcome_id,
                    so.signal_id,
                    so.symbol,
                    so.side,
                    so.entry,
                    so.stop_loss,
                    so.take_profit_near,
                    so.take_profit_1,
                    so.take_profit_2,
                    so.outcome,
                    so.close_price,
                    so.pnl_pct,
                    so.closed_at,
                    s.timestamp AS signal_timestamp,
                    s.score AS signal_score,
                    s.reasons AS signal_reasons
                FROM signal_outcomes so
                LEFT JOIN signals s ON s.id = so.signal_id
                WHERE so.id = ?
                  AND so.outcome != 'OPEN'
                LIMIT 1
                """,
                (int(outcome_id),),
            ).fetchone()

        if not row:
            return None

        payload = dict(row)
        reasons_raw = payload.get("signal_reasons")
        if isinstance(reasons_raw, str) and reasons_raw:
            try:
                payload["signal_reasons"] = json.loads(reasons_raw)
            except Exception:
                payload["signal_reasons"] = [reasons_raw]
        elif reasons_raw is None:
            payload["signal_reasons"] = []
        return payload

    def has_llm_review(self, outcome_id: int) -> bool:
        """Return True if an outcome already has an LLM postmortem review."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT 1 FROM llm_trade_reviews WHERE outcome_id = ? LIMIT 1",
                (int(outcome_id),),
            ).fetchone()
        return bool(row)

    def save_llm_trade_review(
        self,
        *,
        outcome_id: int,
        signal_id: Optional[int],
        symbol: str,
        side: str,
        outcome: str,
        pnl_pct: Optional[float],
        verdict: str,
        action: str,
        confidence: int,
        penalty: float,
        mistake_tags: List[str],
        summary: str,
        recommendation: str,
        raw_json: str = "",
    ) -> None:
        """Persist normalized LLM postmortem review (idempotent on outcome_id)."""
        now = datetime.now(timezone.utc).isoformat()
        tags_json = json.dumps(list(mistake_tags or []), ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO llm_trade_reviews (
                    outcome_id, signal_id, symbol, side, outcome, pnl_pct,
                    verdict, action, confidence, penalty, mistake_tags,
                    summary, recommendation, raw_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(outcome_id),
                    int(signal_id) if signal_id is not None else None,
                    str(symbol),
                    str(side).upper(),
                    str(outcome),
                    float(pnl_pct) if pnl_pct is not None else None,
                    str(verdict),
                    str(action),
                    int(confidence),
                    float(penalty),
                    tags_json,
                    str(summary or "")[:500],
                    str(recommendation or "")[:400],
                    str(raw_json or "")[:4000],
                    now,
                ),
            )
            conn.commit()

    def get_recent_llm_penalty(
        self,
        *,
        symbol: str,
        side: str,
        since_iso: str,
        min_confidence: int = 60,
    ) -> Dict[str, Any]:
        """
        Aggregate recent LLM penalties for a symbol/side profile.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT
                    COUNT(*) AS review_count,
                    COALESCE(AVG(CASE WHEN action IN ('soft_penalty', 'hard_penalty') THEN penalty ELSE 0 END), 0) AS avg_penalty,
                    COALESCE(MAX(CASE WHEN action IN ('soft_penalty', 'hard_penalty') THEN penalty ELSE 0 END), 0) AS max_penalty,
                    COALESCE(AVG(confidence), 0) AS avg_confidence
                FROM llm_trade_reviews
                WHERE symbol = ?
                  AND side = ?
                  AND created_at >= ?
                  AND confidence >= ?
                """,
                (
                    str(symbol),
                    str(side).upper(),
                    str(since_iso),
                    max(0, min(100, int(min_confidence))),
                ),
            ).fetchone()

        if not row:
            return {
                "count": 0,
                "avg_penalty": 0.0,
                "max_penalty": 0.0,
                "avg_confidence": 0.0,
            }
        return {
            "count": int(row[0] or 0),
            "avg_penalty": float(row[1] or 0.0),
            "max_penalty": float(row[2] or 0.0),
            "avg_confidence": float(row[3] or 0.0),
        }

    def get_recent_llm_tag_stats(
        self,
        *,
        symbol: str,
        side: str,
        since_iso: str,
        tag: str,
        min_confidence: int = 60,
    ) -> Dict[str, Any]:
        """
        Aggregate recent LLM review stats for a specific mistake tag
        (e.g. late_entry) for one symbol/side profile.
        """
        normalized_tag = str(tag or "").strip().lower()
        if not normalized_tag:
            return {
                "count": 0,
                "tag_count": 0,
                "tag_ratio": 0.0,
                "tag_avg_penalty": 0.0,
            }

        # JSON field format is like: ["late_entry", "ignored_news_risk"]
        tag_like = f'%"{normalized_tag}"%'

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT
                    COUNT(*) AS review_count,
                    SUM(CASE WHEN mistake_tags LIKE ? THEN 1 ELSE 0 END) AS tag_count,
                    COALESCE(AVG(
                        CASE
                            WHEN mistake_tags LIKE ? AND action IN ('soft_penalty', 'hard_penalty')
                            THEN penalty
                            ELSE NULL
                        END
                    ), 0) AS tag_avg_penalty
                FROM llm_trade_reviews
                WHERE symbol = ?
                  AND side = ?
                  AND created_at >= ?
                  AND confidence >= ?
                """,
                (
                    tag_like,
                    tag_like,
                    str(symbol),
                    str(side).upper(),
                    str(since_iso),
                    max(0, min(100, int(min_confidence))),
                ),
            ).fetchone()

        total_count = int((row[0] if row else 0) or 0)
        tag_count = int((row[1] if row else 0) or 0)
        tag_ratio = (float(tag_count) / float(total_count)) if total_count > 0 else 0.0
        return {
            "count": total_count,
            "tag_count": tag_count,
            "tag_ratio": float(tag_ratio),
            "tag_avg_penalty": float((row[2] if row else 0.0) or 0.0),
        }

    def get_outcomes_pending_llm_review(
        self,
        *,
        lookback_days: int = 21,
        only_losses: bool = True,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Get closed outcomes that do not yet have an LLM review (for startup backfill).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
        params: List[Any] = [cutoff]
        loss_clause = ""
        if only_losses:
            loss_clause = """
              AND (
                    so.outcome = 'SL_HIT'
                    OR (so.outcome IN ('TRAIL_HIT', 'BE_HIT') AND COALESCE(so.pnl_pct, 0) < 0)
                    OR (so.outcome = 'EXITED' AND COALESCE(so.pnl_pct, 0) < 0)
                  )
            """

        query = f"""
            SELECT
                so.id,
                so.signal_id,
                so.symbol,
                so.side,
                so.outcome,
                so.pnl_pct,
                so.closed_at
            FROM signal_outcomes so
            LEFT JOIN llm_trade_reviews lr ON lr.outcome_id = so.id
            WHERE so.closed_at IS NOT NULL
              AND so.outcome != 'OPEN'
              AND so.closed_at >= ?
              AND lr.outcome_id IS NULL
              {loss_clause}
            ORDER BY so.closed_at DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_llm_reviews_for_onchain_enrichment(
        self,
        *,
        lookback_days: int = 30,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent LLM reviews with linked signal reasons so we can enrich
        historical reviews with newly introduced on-chain tags.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            rows = cursor.execute(
                """
                SELECT
                    lr.outcome_id,
                    COALESCE(so.side, lr.side) AS side,
                    lr.mistake_tags,
                    s.reasons AS signal_reasons
                FROM llm_trade_reviews lr
                LEFT JOIN signal_outcomes so ON so.id = lr.outcome_id
                LEFT JOIN signals s ON s.id = COALESCE(lr.signal_id, so.signal_id)
                WHERE lr.created_at >= ?
                ORDER BY lr.created_at DESC
                LIMIT ?
                """,
                (cutoff, max(1, int(limit))),
            ).fetchall()

        payload: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)

            tags_raw = item.get("mistake_tags")
            tags_list: List[str] = []
            if isinstance(tags_raw, str) and tags_raw.strip():
                try:
                    parsed = json.loads(tags_raw)
                    if isinstance(parsed, list):
                        tags_list = [str(x).strip().lower() for x in parsed if str(x).strip()]
                except Exception:
                    tags_list = [str(tags_raw).strip().lower()]
            item["mistake_tags"] = tags_list

            reasons_raw = item.get("signal_reasons")
            reasons_list: List[str] = []
            if isinstance(reasons_raw, str) and reasons_raw.strip():
                try:
                    parsed_reasons = json.loads(reasons_raw)
                    if isinstance(parsed_reasons, list):
                        reasons_list = [str(x) for x in parsed_reasons if str(x).strip()]
                    else:
                        reasons_list = [str(reasons_raw)]
                except Exception:
                    reasons_list = [str(reasons_raw)]
            item["signal_reasons"] = reasons_list

            payload.append(item)
        return payload

    def append_llm_review_tags(self, *, outcome_id: int, tags: List[str]) -> bool:
        """
        Append new tags to an existing review. Returns True when the row changed.
        """
        normalized_new = []
        for tag in list(tags or []):
            t = str(tag or "").strip().lower()
            if t and t not in normalized_new:
                normalized_new.append(t)
        if not normalized_new:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT mistake_tags FROM llm_trade_reviews WHERE outcome_id = ? LIMIT 1",
                (int(outcome_id),),
            ).fetchone()
            if not row:
                return False

            existing_raw = row[0]
            existing: List[str] = []
            if isinstance(existing_raw, str) and existing_raw.strip():
                try:
                    parsed = json.loads(existing_raw)
                    if isinstance(parsed, list):
                        existing = [str(x).strip().lower() for x in parsed if str(x).strip()]
                    else:
                        existing = [str(existing_raw).strip().lower()]
                except Exception:
                    existing = [str(existing_raw).strip().lower()]

            merged = list(existing)
            changed = False
            for tag in normalized_new:
                if tag not in merged:
                    merged.append(tag)
                    changed = True
            if not changed:
                return False

            cursor.execute(
                "UPDATE llm_trade_reviews SET mistake_tags = ? WHERE outcome_id = ?",
                (json.dumps(merged, ensure_ascii=False), int(outcome_id)),
            )
            conn.commit()
            return True

    def get_open_positions_count(self) -> int:
        """Return number of currently OPEN tracked outcomes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT COUNT(*) FROM signal_outcomes WHERE outcome = 'OPEN'"
            ).fetchone()
            return int(row[0] if row else 0)

    def get_daily_closed_pnl(self, day_start_utc: datetime) -> Dict[str, float]:
        """
        Aggregate today's closed outcomes (P&L and counts).

        Args:
            day_start_utc: UTC start-of-day timestamp

        Returns:
            Dict with total_pnl_pct, closed_count, loss_count, win_count
        """
        cutoff = day_start_utc.isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT
                    COALESCE(SUM(COALESCE(so.pnl_pct, 0)), 0) AS total_pnl,
                    COUNT(*) AS closed_count,
                    SUM(
                        CASE
                            WHEN so.outcome = 'SL_HIT' THEN 1
                            WHEN so.outcome IN ('TRAIL_HIT', 'BE_HIT') AND COALESCE(so.pnl_pct, 0) < 0 THEN 1
                            WHEN so.outcome = 'EXITED' AND COALESCE(so.pnl_pct, 0) < 0 THEN 1
                            ELSE 0
                        END
                    ) AS loss_count,
                    SUM(
                        CASE
                            WHEN so.outcome IN ('TP_NEAR_HIT', 'TP1_HIT', 'TP2_HIT') THEN 1
                            WHEN so.outcome IN ('TRAIL_HIT', 'BE_HIT') AND COALESCE(so.pnl_pct, 0) >= 0 THEN 1
                            WHEN so.outcome = 'EXITED' AND COALESCE(so.pnl_pct, 0) > 0 THEN 1
                            ELSE 0
                        END
                    ) AS win_count
                FROM signal_outcomes so
                LEFT JOIN signals s ON s.id = so.signal_id
                WHERE so.closed_at IS NOT NULL
                  AND so.closed_at >= ?
                  AND so.outcome != 'OPEN'
                  AND (s.timestamp IS NULL OR s.timestamp >= ?)
                """,
                (cutoff, cutoff),
            ).fetchone()

        if not row:
            return {"total_pnl_pct": 0.0, "closed_count": 0, "loss_count": 0, "win_count": 0}
        return {
            "total_pnl_pct": float(row[0] or 0.0),
            "closed_count": int(row[1] or 0),
            "loss_count": int(row[2] or 0),
            "win_count": int(row[3] or 0),
        }

    def get_recent_loss_streak(self, limit: int = 30, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Return consecutive losing streak from most recent closed outcomes.
        If symbol is provided, calculates the streak for that symbol only.

        Args:
            limit: Max recent rows to inspect
            symbol: Optional trading pair

        Returns:
            Dict with loss_streak, last_closed_at, last_outcome
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT outcome, closed_at, pnl_pct
                FROM signal_outcomes
                WHERE closed_at IS NOT NULL
                  AND outcome != 'OPEN'
            """
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            query += " ORDER BY closed_at DESC LIMIT ?"
            params.append(max(1, int(limit)))
            
            rows = cursor.execute(query, tuple(params)).fetchall()

        if not rows:
            return {"loss_streak": 0, "last_closed_at": None, "last_outcome": None}

        streak = 0
        for outcome, _, pnl in rows:
            outcome_u = str(outcome or "").upper()
            if outcome_u == "SL_HIT":
                streak += 1
            elif outcome_u in {"TRAIL_HIT", "BE_HIT"} and float(pnl or 0.0) < 0.0:
                streak += 1
            elif outcome_u == "EXITED" and float(pnl or 0.0) < 0.0:
                streak += 1
            else:
                break

        last_outcome = str(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else None
        last_closed_at = None
        try:
            if rows[0][1]:
                last_closed_at = datetime.fromisoformat(str(rows[0][1]))
                if last_closed_at.tzinfo is None:
                    last_closed_at = last_closed_at.replace(tzinfo=timezone.utc)
        except Exception:
            last_closed_at = None

        return {
            "loss_streak": streak,
            "last_closed_at": last_closed_at,
            "last_outcome": last_outcome,
        }

    def delete_open_outcome(self, outcome_id: int) -> int:
        """
        Delete a stale OPEN outcome row.

        Args:
            outcome_id: ID in signal_outcomes table

        Returns:
            Number of rows deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM signal_outcomes
                WHERE id = ? AND outcome = 'OPEN'
            """, (outcome_id,))
            deleted = int(cursor.rowcount or 0)
            conn.commit()
        if deleted:
            logger.info(f"Deleted stale OPEN outcome row id={outcome_id}")
        return deleted

    def close_open_outcomes(
        self,
        symbol: str,
        close_price: Optional[float] = None,
        outcome: str = 'EXITED',
        spread_cost_pct: Optional[float] = None,
    ) -> int:
        """
        Close all OPEN outcomes for a symbol (e.g., manual EXIT signal).
        Computes PnL% from entry price for each record.

        Args:
            symbol: Trading symbol key used in storage
            close_price: Optional close price to store
            outcome: Outcome label for closed records
            spread_cost_pct: Optional spread cost (%) to subtract from PnL

        Returns:
            Number of rows updated
        """
        now = datetime.now(timezone.utc).isoformat()
        spread_cost = max(0.0, float(spread_cost_pct or 0.0))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if close_price is not None:
                # Fetch open outcomes to compute PnL individually
                open_rows = cursor.execute("""
                    SELECT id, side, entry FROM signal_outcomes
                    WHERE symbol = ? AND outcome = 'OPEN'
                """, (symbol,)).fetchall()

                for row_id, side, entry in open_rows:
                    pnl_pct = None
                    if entry and entry > 0:
                        if str(side).upper() == 'LONG':
                            pnl_pct = (close_price - entry) / entry * 100
                        elif str(side).upper() == 'SHORT':
                            pnl_pct = (entry - close_price) / entry * 100
                    if pnl_pct is not None and spread_cost > 0:
                        pnl_pct -= spread_cost
                    cursor.execute("""
                        UPDATE signal_outcomes
                        SET outcome = ?, close_price = ?, pnl_pct = ?, closed_at = ?
                        WHERE id = ? AND outcome = 'OPEN'
                    """, (outcome, close_price, pnl_pct, now, row_id))
                updated = len(open_rows)
            else:
                cursor.execute("""
                    UPDATE signal_outcomes
                    SET outcome = ?, closed_at = ?
                    WHERE symbol = ? AND outcome = 'OPEN'
                """, (outcome, now, symbol))
                updated = cursor.rowcount

            conn.commit()

        if updated > 0:
            logger.info(f"Closed {updated} OPEN outcomes for {symbol} as {outcome}")

        return updated
    
    def update_signal_outcome(
        self,
        outcome_id: int,
        outcome: str,
        close_price: float,
        pnl_pct: float
    ) -> None:
        """
        Update the outcome of a tracked signal (TP_NEAR_HIT, TP1_HIT, TP2_HIT, SL_HIT).
        
        Args:
            outcome_id: ID in signal_outcomes table
            outcome: 'TP_NEAR_HIT', 'TP1_HIT', 'TP2_HIT', 'SL_HIT'
            close_price: Price when outcome was detected
            pnl_pct: Profit/loss percentage
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE signal_outcomes
                SET outcome = ?, close_price = ?, pnl_pct = ?, closed_at = ?
                WHERE id = ?
            """, (outcome, close_price, pnl_pct, now, outcome_id))
            conn.commit()
        logger.info(f"Outcome updated: ID={outcome_id} → {outcome} @ {close_price:.4f} ({pnl_pct:+.2f}%)")

    def update_open_outcome_progress(
        self,
        outcome_id: int,
        *,
        tp1_touched: Optional[bool] = None,
        break_even_armed: Optional[bool] = None,
        trail_stop: Optional[float] = None,
        extreme_price: Optional[float] = None,
        trail_armed_at: Optional[datetime] = None,
        partial_exit_done: Optional[int] = None,
        tp1_partial_pnl: Optional[float] = None,
    ) -> None:
        """
        Update progress fields for an OPEN tracked outcome (TP1/BE/trailing/partial-exit state).
        """
        fields: List[str] = []
        params: List[Any] = []

        if tp1_touched is not None:
            fields.append("tp1_touched = ?")
            params.append(1 if tp1_touched else 0)
        if break_even_armed is not None:
            fields.append("break_even_armed = ?")
            params.append(1 if break_even_armed else 0)
        if trail_stop is not None:
            fields.append("trail_stop = ?")
            params.append(float(trail_stop))
        if extreme_price is not None:
            fields.append("extreme_price = ?")
            params.append(float(extreme_price))
        if trail_armed_at is not None:
            if isinstance(trail_armed_at, datetime):
                fields.append("trail_armed_at = ?")
                params.append(trail_armed_at.isoformat())
            else:
                fields.append("trail_armed_at = ?")
                params.append(str(trail_armed_at))
        if partial_exit_done is not None:
            fields.append("partial_exit_done = ?")
            params.append(int(partial_exit_done))
        if tp1_partial_pnl is not None:
            fields.append("tp1_partial_pnl = ?")
            params.append(float(tp1_partial_pnl))

        if not fields:
            return

        params.append(int(outcome_id))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE signal_outcomes SET {', '.join(fields)} WHERE id = ? AND outcome = 'OPEN'",
                params,
            )
            conn.commit()

    def count_recent_signals(
        self,
        side: str,
        since: str,
        asset_class: str = 'crypto',
        exclude_symbol: Optional[str] = None,
    ) -> int:
        """
        Count recent signals of a given side for a given asset class.
        Used by the altcoin correlation filter.

        Args:
            side: 'LONG' or 'SHORT'
            since: ISO timestamp cutoff (count signals AFTER this time)
            asset_class: 'crypto' or 'macro'
            exclude_symbol: Symbol to exclude from count (the one being evaluated)

        Returns:
            Count of matching signals
        """
        _MACRO_KEYS = ('XAU', 'XAG', 'OIL', 'WTI', 'BRENT', 'SNP500', 'SPX500', 'EURUSD', 'EUR/USD')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT symbol FROM signals
                WHERE side = ?
                  AND timestamp >= ?
            """, (side, since)).fetchall()

        count = 0
        for (sym,) in rows:
            if exclude_symbol and sym == exclude_symbol:
                continue
            is_macro = any(k in str(sym).upper() for k in _MACRO_KEYS)
            sym_class = 'macro' if is_macro else 'crypto'
            if sym_class == asset_class:
                count += 1
        return count

    def get_winrate_stats(self, days: int = 7) -> Dict:
        """
        Calculate WinRate statistics for the last N days.
        
        Args:
            days: Lookback period in days
            
        Returns:
            Dict with stats: total, tp1_hit, tp2_hit, sl_hit, open, winrate_pct
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT outcome, pnl_pct
                FROM signal_outcomes
                WHERE (
                    -- Closed outcomes are measured by close time (true performance window).
                    (outcome != 'OPEN' AND COALESCE(closed_at, '') > ?)
                    OR
                    -- OPEN rows are measured by originating signal time.
                    (outcome = 'OPEN' AND id IN (
                        SELECT so.id
                        FROM signal_outcomes so
                        JOIN signals s ON so.signal_id = s.id
                        WHERE s.timestamp > ?
                    ))
                )
                """,
                (cutoff, cutoff),
            )
            rows = cursor.fetchall()
        
        stats = {
            'total': 0,
            'tp_near_hit': 0,
            'tp1_hit': 0,
            'tp2_hit': 0,
            'trail_hit': 0,
            'be_hit': 0,
            'sl_hit': 0,
            'exited': 0,
            'open': 0,
            'avg_pnl': 0.0
        }
        total_pnl = 0.0
        closed = 0
        wins = 0
        losses = 0
        
        for outcome, pnl_pct in rows:
            out = str(outcome or "").upper()
            stats['total'] += 1

            pnl_val = None
            try:
                if pnl_pct is not None:
                    pnl_val = float(pnl_pct)
            except Exception:
                pnl_val = None

            if out == 'OPEN':
                stats['open'] += 1
                continue

            if out == 'TP_NEAR_HIT':
                stats['tp_near_hit'] += 1
                wins += 1
            elif out == 'TP1_HIT':
                stats['tp1_hit'] += 1
                wins += 1
            elif out == 'TP2_HIT':
                stats['tp2_hit'] += 1
                wins += 1
            elif out == 'TRAIL_HIT':
                stats['trail_hit'] += 1
                if pnl_val is not None and pnl_val < 0:
                    losses += 1
                else:
                    wins += 1
            elif out == 'BE_HIT':
                stats['be_hit'] += 1
                if pnl_val is not None and pnl_val < 0:
                    losses += 1
                else:
                    wins += 1
            elif out == 'SL_HIT':
                stats['sl_hit'] += 1
                losses += 1
            elif out == 'EXITED':
                stats['exited'] += 1
                if pnl_val is not None:
                    if pnl_val > 0:
                        wins += 1
                    elif pnl_val < 0:
                        losses += 1

            if pnl_val is not None:
                total_pnl += pnl_val
                closed += 1

        total_closed = wins + losses
        stats['winrate_pct'] = (wins / total_closed * 100) if total_closed > 0 else 0.0
        stats['avg_pnl'] = (total_pnl / closed) if closed > 0 else 0.0
        
        return stats
