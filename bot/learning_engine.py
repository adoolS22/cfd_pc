"""
Adaptive Learning Engine
========================
Learns from historical signal outcomes (last N days) and applies a
lightweight quality filter to new candidate signals.
"""

from __future__ import annotations

import sqlite3
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Any

from loguru import logger

from .utils import LearningConfig


WIN_OUTCOMES = {"TP_NEAR_HIT", "TP1_HIT", "TP2_HIT", "TRAIL_HIT", "BE_HIT"}
LOSS_OUTCOMES = {"SL_HIT"}
# Any trade closed before this date is treated as seeded/historical, not live.
# This guards against old outcomes that lack proper source tags.
FIRST_LIVE_CUTOFF = "2026-02-25T00:00:00+00:00"
SEED_REASON_TAG = "HIST_SEED_REGIME_V1"
REPLAY_REASON_TAG = "HIST_REPLAY_REAL_V1"
NON_LIVE_REASON_TAGS = (
    "HIST_SEED_REGIME_V1",
    "HIST_REPLAY_REAL_V1",
    "source:historical_seed",
    "source:real_ohlcv_replay",
)
ENTRY_TF_RE = re.compile(r"ctx:entry_tf=([0-9]+[mhd])", re.IGNORECASE)


@dataclass
class _Bucket:
    n: int = 0
    weight_sum: float = 0.0
    win_weight_sum: float = 0.0
    pnl_weight_sum: float = 0.0

    def add(self, label: int, pnl_val: float, weight: float) -> None:
        w = max(0.01, float(weight))
        self.n += 1
        self.weight_sum += w
        self.win_weight_sum += w * float(label)
        self.pnl_weight_sum += w * float(pnl_val)

    @property
    def winrate(self) -> float:
        if self.weight_sum <= 0:
            return 0.0
        return self.win_weight_sum / self.weight_sum

    @property
    def avg_pnl(self) -> float:
        if self.weight_sum <= 0:
            return 0.0
        return self.pnl_weight_sum / self.weight_sum


@dataclass
class LearningDecision:
    allow: bool
    score_adjustment: float
    expected_winrate: float
    expected_pnl_pct: float
    sample_size: int
    reason: str


@dataclass
class _LearningSnapshot:
    trained_at: datetime
    total_closed: int
    global_bucket: _Bucket
    side_buckets: Dict[str, _Bucket]
    asset_side_buckets: Dict[Tuple[str, str], _Bucket]
    asset_tf_side_buckets: Dict[Tuple[str, str, str], _Bucket]
    symbol_buckets: Dict[str, _Bucket]
    symbol_side_buckets: Dict[Tuple[str, str], _Bucket]
    regime_side_buckets: Dict[Tuple[str, str], _Bucket]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_label(outcome: str, pnl_pct_net: Optional[float]) -> Optional[int]:
    if outcome in LOSS_OUTCOMES:
        return 0
    if pnl_pct_net is None:
        return None
    # TRAIL_HIT / BE_HIT: judge by actual PnL, not by outcome name.
    if outcome in {"TRAIL_HIT", "BE_HIT"}:
        return 1 if pnl_pct_net > 0 else 0
    if pnl_pct_net > 0:
        return 1
    if pnl_pct_net < 0:
        return 0
    if outcome in WIN_OUTCOMES:
        # Win outcome that does not beat costs -> treat as non-winning for learning.
        return 0
    return None


def _shrink(value: float, n: int, prior: float, k: int) -> float:
    if n <= 0:
        return prior
    reliability = n / float(n + max(1, k))
    return reliability * value + (1.0 - reliability) * prior


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _recency_weight(closed_at: Optional[str], half_life_days: float) -> float:
    dt = _parse_iso_dt(closed_at)
    if dt is None:
        return 1.0
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    hl = max(1.0, float(half_life_days))
    # 0.5 every half-life; keep a floor so old trades still contribute slightly.
    w = math.pow(0.5, age_days / hl)
    return max(0.15, min(1.0, w))


def _infer_regime_from_reasons(reasons_text: Optional[str]) -> str:
    text = str(reasons_text or "").lower()
    if not text:
        return "sideways"
    if "regime:high_volatility" in text or "high volatility" in text:
        return "high_volatility"
    if "regime:uptrend" in text or "regime uptrend" in text:
        return "uptrend"
    if "regime:downtrend" in text or "regime downtrend" in text:
        return "downtrend"
    if "regime:sideways" in text or "regime sideways" in text:
        return "sideways"
    return "sideways"


def _is_non_live_reason_text(reasons_text: Optional[str]) -> bool:
    text = str(reasons_text or "").lower()
    if not text:
        # Unknown provenance -> strict mode treats this as non-live.
        return True
    return any(tag.lower() in text for tag in NON_LIVE_REASON_TAGS)


def _infer_asset_class(symbol: Optional[str], reasons_text: Optional[str] = None) -> str:
    text = str(symbol or "").upper()
    reasons = str(reasons_text or "").lower()
    if "ctx:asset_class=macro" in reasons:
        return "macro"
    if "ctx:asset_class=crypto" in reasons:
        return "crypto"
    macro_keys = ("XAU", "XAG", "OIL", "WTI", "BRENT", "SNP500", "SPX500", "S&P500", "SP500", "EURUSD", "EUR/USD")
    if any(k in text for k in macro_keys):
        return "macro"
    return "crypto"


def _extract_entry_tf(reasons_text: Optional[str]) -> str:
    text = str(reasons_text or "")
    m = ENTRY_TF_RE.search(text)
    if not m:
        return "unknown"
    return str(m.group(1)).lower()


def _entry_tf_bucket(entry_tf: Optional[str]) -> str:
    tf = str(entry_tf or "").strip().lower()
    if not tf or tf == "unknown":
        return "unknown"
    try:
        unit = tf[-1]
        val = int(tf[:-1])
    except Exception:
        return "unknown"
    minutes = 0
    if unit == "m":
        minutes = val
    elif unit == "h":
        minutes = val * 60
    elif unit == "d":
        minutes = val * 1440
    else:
        return "unknown"
    if minutes <= 5:
        return "scalp"
    if minutes <= 30:
        return "intraday"
    if minutes <= 240:
        return "swing"
    return "position"


def _roundtrip_cost_pct(asset_class: str, config: LearningConfig) -> float:
    if asset_class == "macro":
        return max(0.0, float(getattr(config, "estimated_roundtrip_cost_pct_macro", 0.03)))
    return max(0.0, float(getattr(config, "estimated_roundtrip_cost_pct_crypto", 0.08)))


class AdaptiveLearningEngine:
    """Caches and serves adaptive signal-quality estimates from SQLite outcomes."""

    def __init__(self) -> None:
        self._snapshot_cache: Dict[str, _LearningSnapshot] = {}

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        return bool(row)

    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        try:
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        except Exception:
            return False
        for r in rows:
            if len(r) > 1 and str(r[1]) == column_name:
                return True
        return False

    def _train(self, db_path: str, config: LearningConfig) -> _LearningSnapshot:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=max(1, int(config.lookback_days)))).isoformat()

        global_bucket = _Bucket()
        side_buckets: Dict[str, _Bucket] = {"LONG": _Bucket(), "SHORT": _Bucket()}
        asset_side_buckets: Dict[Tuple[str, str], _Bucket] = {}
        asset_tf_side_buckets: Dict[Tuple[str, str, str], _Bucket] = {}
        symbol_buckets: Dict[str, _Bucket] = {}
        symbol_side_buckets: Dict[Tuple[str, str], _Bucket] = {}
        regime_side_buckets: Dict[Tuple[str, str], _Bucket] = {}

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            exclude_seeded = bool(getattr(config, "exclude_seeded_outcomes", False))
            include_only_real = bool(getattr(config, "include_only_real_outcomes", True))
            has_signals_table = self._table_exists(conn, "signals")
            has_signal_id = self._column_exists(conn, "signal_outcomes", "signal_id")
            if has_signals_table and has_signal_id:
                rows = cursor.execute(
                    """
                    SELECT so.symbol, so.side, so.outcome, so.pnl_pct, so.closed_at, s.reasons
                    FROM signal_outcomes so
                    LEFT JOIN signals s ON s.id = so.signal_id
                    WHERE so.closed_at IS NOT NULL
                      AND so.closed_at >= ?
                      AND so.outcome != 'OPEN'
                    """,
                    (cutoff,),
                ).fetchall()
            else:
                rows = cursor.execute(
                    """
                    SELECT so.symbol, so.side, so.outcome, so.pnl_pct, so.closed_at, NULL as reasons
                    FROM signal_outcomes so
                    WHERE so.closed_at IS NOT NULL
                      AND so.closed_at >= ?
                      AND so.outcome != 'OPEN'
                    """,
                    (cutoff,),
                ).fetchall()
                if include_only_real:
                    logger.warning(
                        "Learning strict real-only mode requested but reasons metadata unavailable in db={}, "
                        "falling back to all outcomes from signal_outcomes.",
                        db_path,
                    )

        half_life_days = float(getattr(config, "recency_half_life_days", 45.0))
        seeded_weight = float(getattr(config, "seeded_weight", 0.25))
        include_only_real = bool(getattr(config, "include_only_real_outcomes", False))
        exclude_seeded = bool(getattr(config, "exclude_seeded_outcomes", False))
        count_live = 0
        count_seeded = 0
        filtered_missing_pnl = 0

        for symbol, side, outcome, pnl_pct, closed_at, reasons in rows:
            reasons_text = str(reasons or "")

            if pnl_pct is None:
                filtered_missing_pnl += 1
                continue

            # Determine if this trade is live or seeded/historical.
            is_seeded = False
            if _is_non_live_reason_text(reasons_text):
                is_seeded = True
            elif closed_at:
                closed_dt = _parse_iso_dt(str(closed_at))
                cutoff_dt = _parse_iso_dt(FIRST_LIVE_CUTOFF)
                if closed_dt and cutoff_dt and closed_dt < cutoff_dt:
                    is_seeded = True

            # In strict mode, skip seeded entirely. In blended mode, downweight.
            if is_seeded:
                if include_only_real or exclude_seeded:
                    count_seeded += 1
                    continue  # strict mode: skip
                source_weight = max(0.05, seeded_weight)
                count_seeded += 1
            else:
                source_weight = 1.0
                count_live += 1

            side_key = str(side or "").upper()
            if side_key not in ("LONG", "SHORT"):
                continue

            asset_class = _infer_asset_class(symbol, reasons_text)
            entry_tf = _extract_entry_tf(reasons_text)
            tf_bucket = _entry_tf_bucket(entry_tf)
            roundtrip_cost_pct = _roundtrip_cost_pct(asset_class, config)
            pnl_val_gross = float(pnl_pct)
            pnl_val = pnl_val_gross - roundtrip_cost_pct
            label = _to_label(str(outcome or ""), pnl_val)
            if label is None:
                continue
            weight = _recency_weight(closed_at, half_life_days) * source_weight
            regime = _infer_regime_from_reasons(reasons_text)

            global_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            side_bucket = side_buckets.setdefault(side_key, _Bucket())
            side_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            asset_side_key = (asset_class, side_key)
            asset_side_bucket = asset_side_buckets.setdefault(asset_side_key, _Bucket())
            asset_side_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            asset_tf_side_key = (asset_class, tf_bucket, side_key)
            asset_tf_side_bucket = asset_tf_side_buckets.setdefault(asset_tf_side_key, _Bucket())
            asset_tf_side_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            sym_bucket = symbol_buckets.setdefault(symbol, _Bucket())
            sym_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            key = (symbol, side_key)
            sym_side_bucket = symbol_side_buckets.setdefault(key, _Bucket())
            sym_side_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

            regime_key = (regime, side_key)
            regime_bucket = regime_side_buckets.setdefault(regime_key, _Bucket())
            regime_bucket.add(label=label, pnl_val=pnl_val, weight=weight)

        # ── Integrate shadow-tracked rejected signal outcomes ──────────
        # Rejected signals that would have WON are fed back as "missed wins"
        # to nudge the learning engine toward being less conservative.
        # Rejected signals that would have LOST reinforce the filter.
        shadow_integrated = 0
        try:
            with sqlite3.connect(db_path) as conn_shadow:
                if self._table_exists(conn_shadow, "rejected_signals"):
                    shadow_rows = conn_shadow.execute(
                        """
                        SELECT symbol, side, outcome, pnl_pct, closed_at,
                               would_have_won, market_regime
                        FROM rejected_signals
                        WHERE closed_at IS NOT NULL
                          AND closed_at >= ?
                          AND outcome != 'TRACKING'
                          AND pnl_pct IS NOT NULL
                        """,
                        (cutoff,),
                    ).fetchall()

                    # Shadow outcomes get a reduced weight (0.3x) so they inform
                    # but don't dominate over actual live trade data.
                    shadow_weight_mult = 0.3

                    for sym, side_val, outcome, pnl_pct_val, closed_at_val, won, regime_val in shadow_rows:
                        if pnl_pct_val is None:
                            continue
                        side_k = str(side_val or "").upper()
                        if side_k not in ("LONG", "SHORT"):
                            continue

                        pnl_val = float(pnl_pct_val)
                        label = 1 if won else 0
                        weight = _recency_weight(closed_at_val, half_life_days) * shadow_weight_mult

                        asset_cls = _infer_asset_class(sym)
                        regime_name = str(regime_val or "sideways").strip().lower() or "sideways"

                        global_bucket.add(label=label, pnl_val=pnl_val, weight=weight)
                        side_buckets.setdefault(side_k, _Bucket()).add(label=label, pnl_val=pnl_val, weight=weight)
                        asset_side_buckets.setdefault((asset_cls, side_k), _Bucket()).add(label=label, pnl_val=pnl_val, weight=weight)
                        symbol_buckets.setdefault(sym, _Bucket()).add(label=label, pnl_val=pnl_val, weight=weight)
                        symbol_side_buckets.setdefault((sym, side_k), _Bucket()).add(label=label, pnl_val=pnl_val, weight=weight)
                        regime_side_buckets.setdefault((regime_name, side_k), _Bucket()).add(label=label, pnl_val=pnl_val, weight=weight)

                        shadow_integrated += 1
        except Exception as shadow_err:
            logger.debug(f"Learning shadow integration skipped: {shadow_err}")

        snapshot = _LearningSnapshot(
            trained_at=now,
            total_closed=global_bucket.n,
            global_bucket=global_bucket,
            side_buckets=side_buckets,
            asset_side_buckets=asset_side_buckets,
            asset_tf_side_buckets=asset_tf_side_buckets,
            symbol_buckets=symbol_buckets,
            symbol_side_buckets=symbol_side_buckets,
            regime_side_buckets=regime_side_buckets,
        )

        logger.info(
            "Learning trained: closed={} live={} seeded={} seeded_weight={} lookback_days={} "
            "half_life_days={} cost_crypto={} cost_macro={} filtered_no_pnl={} shadow_integrated={}",
            snapshot.total_closed,
            count_live,
            count_seeded,
            seeded_weight,
            config.lookback_days,
            half_life_days,
            float(getattr(config, "estimated_roundtrip_cost_pct_crypto", 0.08)),
            float(getattr(config, "estimated_roundtrip_cost_pct_macro", 0.03)),
            filtered_missing_pnl,
            shadow_integrated,
        )
        return snapshot

    def _get_snapshot(self, db_path: str, config: LearningConfig) -> _LearningSnapshot:
        now = datetime.now(timezone.utc)
        cached = self._snapshot_cache.get(db_path)
        refresh_minutes = max(1, int(config.retrain_interval_minutes))

        if cached:
            age = now - cached.trained_at
            if age < timedelta(minutes=refresh_minutes):
                return cached

        trained = self._train(db_path, config)
        self._snapshot_cache[db_path] = trained
        return trained

    def evaluate(
        self,
        db_path: str,
        symbol: str,
        side: str,
        base_score: float,
        threshold: float,
        config: LearningConfig,
        market_regime: Optional[str] = None,
        entry_tf: Optional[str] = None,
    ) -> LearningDecision:
        if not config.enabled:
            return LearningDecision(
                allow=True,
                score_adjustment=0.0,
                expected_winrate=0.5,
                expected_pnl_pct=0.0,
                sample_size=0,
                reason="Learning: disabled",
            )

        snapshot = self._get_snapshot(db_path, config)
        if snapshot.total_closed < max(1, int(config.min_closed_trades)):
            return LearningDecision(
                allow=True,
                score_adjustment=0.0,
                expected_winrate=0.5,
                expected_pnl_pct=0.0,
                sample_size=snapshot.total_closed,
                reason=f"Learning: warmup {snapshot.total_closed}/{config.min_closed_trades}",
            )

        side_key = (side or "").upper()
        global_bucket = snapshot.global_bucket
        side_bucket = snapshot.side_buckets.get(side_key, _Bucket())
        asset_class = _infer_asset_class(symbol)
        tf_bucket = _entry_tf_bucket(entry_tf)
        asset_side_bucket = snapshot.asset_side_buckets.get((asset_class, side_key), _Bucket())
        asset_tf_side_bucket = snapshot.asset_tf_side_buckets.get((asset_class, tf_bucket, side_key), _Bucket())
        sym_bucket = snapshot.symbol_buckets.get(symbol, _Bucket())
        sym_side_bucket = snapshot.symbol_side_buckets.get((symbol, side_key), _Bucket())

        global_wr = global_bucket.winrate if global_bucket.n > 0 else 0.5
        global_pnl = global_bucket.avg_pnl if global_bucket.n > 0 else 0.0

        side_wr = _shrink(side_bucket.winrate, side_bucket.n, global_wr, k=40)
        side_pnl = _shrink(side_bucket.avg_pnl, side_bucket.n, global_pnl, k=40)

        asset_wr = _shrink(asset_side_bucket.winrate, asset_side_bucket.n, side_wr, k=30)
        asset_pnl = _shrink(asset_side_bucket.avg_pnl, asset_side_bucket.n, side_pnl, k=30)

        tf_wr = _shrink(asset_tf_side_bucket.winrate, asset_tf_side_bucket.n, asset_wr, k=18)
        tf_pnl = _shrink(asset_tf_side_bucket.avg_pnl, asset_tf_side_bucket.n, asset_pnl, k=18)

        sym_wr = _shrink(sym_bucket.winrate, sym_bucket.n, tf_wr, k=25)
        sym_pnl = _shrink(sym_bucket.avg_pnl, sym_bucket.n, tf_pnl, k=25)

        k = max(3, int(config.min_symbol_side_trades))
        expected_wr = _shrink(sym_side_bucket.winrate, sym_side_bucket.n, sym_wr, k=k)
        expected_pnl = _shrink(sym_side_bucket.avg_pnl, sym_side_bucket.n, sym_pnl, k=k)

        regime_note = ""
        if bool(getattr(config, "use_regime_profile", True)):
            regime_key_name = str(market_regime or "").strip().lower()
            if regime_key_name:
                regime_bucket = snapshot.regime_side_buckets.get((regime_key_name, side_key), _Bucket())
                min_regime_samples = max(1, int(getattr(config, "regime_min_samples", 8)))
                if regime_bucket.n >= min_regime_samples:
                    regime_wr = _shrink(regime_bucket.winrate, regime_bucket.n, side_wr, k=max(4, min_regime_samples))
                    regime_pnl = _shrink(regime_bucket.avg_pnl, regime_bucket.n, side_pnl, k=max(4, min_regime_samples))
                    blend = _clip(float(getattr(config, "regime_blend_strength", 0.35)), 0.0, 0.8)
                    expected_wr = ((1.0 - blend) * expected_wr) + (blend * regime_wr)
                    expected_pnl = ((1.0 - blend) * expected_pnl) + (blend * regime_pnl)
                    regime_note = f" regime={regime_key_name} n_reg={regime_bucket.n}"

        # ── Shadow-informed intelligence: learn from rejected signals ──
        # Query shadow outcomes for this symbol/side to see if the filter
        # has been consistently wrong. If many rejected signals would have
        # won, correct the expected winrate/pnl UPWARD based on evidence.
        shadow_correction_wr = 0.0
        shadow_correction_pnl = 0.0
        shadow_note = ""
        try:
            shadow_stats = self._get_shadow_correction(db_path, symbol, side_key)
            if shadow_stats["total"] >= 5:  # need at least 5 shadow outcomes
                miss_rate = shadow_stats["missed_rate"]
                if miss_rate > 0.15:  # >15% of rejections were missed opportunities
                    # Apply a correction proportional to how wrong the filter is.
                    # Max correction: +10% winrate, +0.15% pnl
                    shadow_correction_wr = _clip(miss_rate * 0.5, 0.0, 0.10)
                    shadow_correction_pnl = _clip(
                        shadow_stats["avg_missed_pnl"] * 0.3, 0.0, 0.15
                    )
                    expected_wr += shadow_correction_wr
                    expected_pnl += shadow_correction_pnl
                    shadow_note = (
                        f" shadow_corr(wr=+{shadow_correction_wr*100:.1f}%"
                        f" pnl=+{shadow_correction_pnl:.2f}%"
                        f" miss={miss_rate*100:.0f}% n={shadow_stats['total']})"
                    )
        except Exception:
            pass

        # Map expected quality to a bounded score adjustment.
        # Expectancy-oriented mode gives more influence to expected net PnL.
        wr_delta = expected_wr - float(config.min_expected_winrate)
        pnl_delta = expected_pnl - float(config.min_expected_pnl_pct)
        optimize_for_expectancy = bool(getattr(config, "optimize_for_expectancy", True))
        expectancy_weight = max(0.1, float(getattr(config, "expectancy_weight", 1.0)))
        pnl_weight = 1.2 * (1.0 + (0.7 * (expectancy_weight - 1.0))) if optimize_for_expectancy else 1.2
        raw_adjustment = (wr_delta * 8.0) + (pnl_delta * pnl_weight)
        adjustment = _clip(
            raw_adjustment,
            -abs(float(config.score_penalty_max)),
            abs(float(config.score_bonus_max)),
        )

        predicted_score = float(base_score) + adjustment
        allow = True

        if bool(config.block_on_low_quality):
            min_local = max(1, int(config.min_symbol_side_trades))
            enough_local = sym_side_bucket.n >= min_local
            moderate_local = sym_side_bucket.n >= max(4, min_local // 2)
            require_full_local_for_hard_block = bool(
                getattr(config, "hard_block_requires_full_local_samples", True)
            )
            very_weak = (
                expected_wr < float(config.hard_block_winrate)
                and expected_pnl <= float(config.min_expected_pnl_pct)
            )
            below_threshold_after_adjustment = predicted_score < float(threshold)
            enough_for_threshold_block = enough_local if require_full_local_for_hard_block else moderate_local
            if (enough_local and very_weak) or (enough_for_threshold_block and below_threshold_after_adjustment):
                allow = False

        reason = (
            f"Learning: wr={expected_wr*100:.1f}% net_pnl={expected_pnl:+.2f}% "
            f"samples={sym_side_bucket.n} asset={asset_class} tf_bucket={tf_bucket} "
            f"n_asset={asset_side_bucket.n} n_asset_tf={asset_tf_side_bucket.n} "
            f"adj={adjustment:+.2f} pnl_w={pnl_weight:.2f}{regime_note}{shadow_note}"
        )
        if not allow:
            reason = f"{reason} -> blocked"

        return LearningDecision(
            allow=allow,
            score_adjustment=adjustment,
            expected_winrate=expected_wr,
            expected_pnl_pct=expected_pnl,
            sample_size=sym_side_bucket.n,
            reason=reason,
        )

    @staticmethod
    def _get_shadow_correction(
        db_path: str,
        symbol: str,
        side: str,
    ) -> Dict[str, Any]:
        """
        Compute per-symbol/side correction from shadow-tracked rejected signals.

        Returns dict with:
            total: number of closed shadow outcomes
            missed_rate: fraction that would have won
            avg_missed_pnl: average PnL of missed opportunities
            correct_rate: fraction that were correctly rejected
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
        empty = {"total": 0, "missed_rate": 0.0, "avg_missed_pnl": 0.0, "correct_rate": 1.0}

        try:
            with sqlite3.connect(db_path) as conn:
                # Check table exists
                if not AdaptiveLearningEngine._table_exists(conn, "rejected_signals"):
                    return empty

                rows = conn.execute(
                    """
                    SELECT would_have_won, pnl_pct
                    FROM rejected_signals
                    WHERE symbol = ? AND side = ?
                      AND outcome != 'TRACKING'
                      AND closed_at IS NOT NULL
                      AND created_at >= ?
                    """,
                    (str(symbol), str(side).upper(), cutoff),
                ).fetchall()

            if not rows:
                return empty

            total = len(rows)
            won_rows = [r for r in rows if r[0] == 1]
            missed = len(won_rows)
            missed_pnls = [float(r[1] or 0) for r in won_rows if r[1] is not None]

            return {
                "total": total,
                "missed_rate": missed / total if total > 0 else 0.0,
                "avg_missed_pnl": (
                    sum(missed_pnls) / len(missed_pnls)
                    if missed_pnls
                    else 0.0
                ),
                "correct_rate": (total - missed) / total if total > 0 else 1.0,
            }
        except Exception:
            return empty


_ENGINE = AdaptiveLearningEngine()


def evaluate_learning_signal(
    db_path: str,
    symbol: str,
    side: str,
    base_score: float,
    threshold: float,
    config: LearningConfig,
    market_regime: Optional[str] = None,
    entry_tf: Optional[str] = None,
) -> LearningDecision:
    """Facade for evaluating one candidate signal against the adaptive model."""
    return _ENGINE.evaluate(
        db_path=db_path,
        symbol=symbol,
        side=side,
        base_score=base_score,
        threshold=threshold,
        config=config,
        market_regime=market_regime,
        entry_tf=entry_tf,
    )

def compute_symbol_atr_calibration(storage: Any, symbol: str) -> float:
    """
    Compute a dynamic ATR stop multiplier for a specific symbol based on historical
    'stop_too_tight' LLM mistakes. Increases the multiplier if the symbol frequently
    stops out prematurely due to noise.
    """
    try:
        from datetime import datetime, timezone, timedelta
        since_iso = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
        
        # Check LONG
        long_stats = storage.get_recent_llm_tag_stats(
            symbol=symbol, side="LONG", since_iso=since_iso, tag="stop_too_tight", min_confidence=60
        )
        # Check SHORT
        short_stats = storage.get_recent_llm_tag_stats(
            symbol=symbol, side="SHORT", since_iso=since_iso, tag="stop_too_tight", min_confidence=60
        )
        
        total_reviews = long_stats.get('count', 0) + short_stats.get('count', 0)
        if total_reviews < 3: # require at least 3 reviews to calibrate
            return 1.0
            
        total_tags = long_stats.get('tag_count', 0) + short_stats.get('tag_count', 0)
        tag_ratio = total_tags / total_reviews
        
        # If the ratio of stop_too_tight is greater than 15%, scale up the multiplier
        if tag_ratio >= 0.15:
            # Scale from 1.0 up to 1.5 based on tag_ratio (0.15 to 0.50 max)
            scale_factor = 1.0 + min(0.5, (tag_ratio - 0.15) * 1.5)
            logger.debug(f"{symbol} ATR calibration: ratio={tag_ratio:.2f}, scale={scale_factor:.2f}")
            return scale_factor
            
    except Exception as e:
        logger.debug(f"Error computing ATR calibration for {symbol}: {e}")
        
    return 1.0

