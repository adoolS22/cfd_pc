"""
Utility Functions
=================
Helper functions for configuration loading, logging, and formatting.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from loguru import logger
import sys


@dataclass
class TimeStopConfig:
    """Time-based stop loss configuration."""
    enabled: bool
    max_hold_minutes: int
    min_progress_to_continue_pct: float
    progress_check_at_pct: float

@dataclass
class RiskConfig:
    """Risk management configuration."""
    buffer_pct: float
    rr_tp1: float
    rr_tp2: float
    quick_tp_pct: float
    quick_tp_min_pct: float
    quick_tp1_fraction: float
    quick_tp_outcome_enabled: bool
    break_even_after_tp1: bool
    break_even_buffer_pct: float
    trailing_after_tp1: bool
    trailing_rr_from_risk: float
    trailing_min_move_pct: float
    atr_stop_mult: float
    atr_buffer_mult: float
    tp2_atr_mult: float
    min_rr_tp1: float
    max_sl_pct_crypto: float
    max_sl_pct_macro: float
    timeframe_quick_tp_scale: Dict[str, float]
    timeframe_tp2_rr_scale: Dict[str, float]
    time_stop: TimeStopConfig


@dataclass
class ScoringConfig:
    """Scoring configuration."""
    base_threshold: int
    base_threshold_crypto: float
    base_threshold_macro: float
    add_timing_to_score: bool
    max_timing_points_used: int
    dynamic_threshold_enabled: bool
    dynamic_low_vol_atr_pct: float
    dynamic_high_vol_atr_pct: float
    dynamic_low_vol_adjust: float
    dynamic_high_vol_adjust: float
    dynamic_strong_trend_adx_min: float
    dynamic_strong_trend_adjust: float
    min_threshold: float


@dataclass
class RegimeConfig:
    """Market regime classification + score shaping."""
    enabled: bool
    adx_trend_min: float
    ema200_slope_abs_min: float
    high_vol_atr_pct: float
    sideways_threshold_add: int
    high_vol_threshold_add: int
    aligned_score_bonus: float
    countertrend_score_penalty: float
    quick_tp_scale_sideways: float
    quick_tp_scale_high_vol: float


@dataclass
class TimeAnalysisConfig:
    """Time analysis configuration."""
    enabled: bool
    gann_angles: bool
    square9: bool
    fibonacci: bool
    cycle_52: bool
    lunar: bool
    fomc_filter: bool
    cpi_filter: bool
    nfp_filter: bool
    powell_filter: bool
    fomc_minutes_filter: bool
    sentiment_filter: bool
    social_sentiment_filter: bool
    lunar_window_hours: int
    fomc_high_vol_days: int
    cpi_high_vol_days: int
    nfp_high_vol_days: int
    powell_high_vol_hours: int
    fomc_minutes_high_vol_days: int
    sentiment_extreme_fear: int
    sentiment_extreme_greed: int
    social_sentiment_min_posts: int
    social_sentiment_caution_ratio: float
    social_sentiment_extreme_ratio: float
    zone_proximity_pct: float
    expert_advisor: bool
    expert_advisor_timeout_seconds: int
    manager_advisor: bool
    manager_advisor_timeout_seconds: int
    expert_use_as_filter: bool
    expert_min_confidence: int
    expert_block_on_wait: bool
    expert_require_side_alignment: bool
    exit_min_hold_minutes: int
    exit_min_pattern_strength: int
    exit_reverse_confirmation_bars: int


@dataclass
class TelegramConfig:
    """Telegram configuration."""
    enabled: bool
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass
class OpenAIConfig:
    """OpenAI configuration."""
    enabled: bool
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None


@dataclass
class OllamaConfig:
    """Ollama configuration."""
    enabled: bool
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"


@dataclass
class SignalModeConfig:
    """Configuration for a signal mode (futures)."""
    enabled: bool
    trend_tf: str
    entry_tf: str
    sr_tf: str
    htf: str = ""  # Higher timeframe for MTF confirmation (e.g. "1h", "4h")


@dataclass
class LearningConfig:
    """Adaptive learning configuration (trained from historical outcomes)."""
    enabled: bool
    lookback_days: int
    retrain_interval_minutes: int
    min_closed_trades: int
    min_symbol_side_trades: int
    min_expected_winrate: float
    min_expected_pnl_pct: float
    hard_block_winrate: float
    score_bonus_max: float
    score_penalty_max: float
    block_on_low_quality: bool
    decision_mode: str
    decision_min_samples: int
    decision_min_winrate: float
    decision_min_pnl_pct: float
    decision_min_edge_pct: float
    exclude_seeded_outcomes: bool
    include_only_real_outcomes: bool
    recency_half_life_days: float
    estimated_roundtrip_cost_pct_crypto: float
    estimated_roundtrip_cost_pct_macro: float
    use_regime_profile: bool
    regime_blend_strength: float
    regime_min_samples: int
    hard_block_requires_full_local_samples: bool
    optimize_for_expectancy: bool
    expectancy_weight: float
    decision_min_expectancy_pct: float
    hybrid_min_expectancy_pct: float
    seeded_weight: float = 0.25  # weight for seeded/historical trades (0.0-1.0)


@dataclass
class QualityFilterConfig:
    """Entry quality filters (spread/volume/news alignment)."""
    enabled: bool
    max_spread_pct: float
    min_volume_ratio: float
    min_volume_ratio_macro: float
    block_during_high_impact_news: bool
    high_impact_news_mode: str
    high_impact_threshold_add: float
    block_on_opposing_news: bool
    opposing_news_score: float
    opposing_news_score_macro: float
    require_learning_alignment: bool
    learning_min_samples: int
    max_negative_learning_adjustment: float
    relaxed_volume_mode: bool
    min_volume_ratio_relaxed: float
    volume_relax_adx_min: float
    volume_relax_max_spread_pct: float
    volume_relax_min_score_margin: float
    anti_chasing_atr_mult: float
    anti_chasing_lookback_bars: int
    macro_session_filter_enabled: bool
    macro_session_mode: str
    macro_session_threshold_add: float
    macro_session_utc_windows: List[str]
    llm_adapter_enabled: bool
    llm_adapter_shadow_mode: bool
    llm_adapter_lookback_days: int
    llm_adapter_min_reviews: int
    llm_adapter_min_confidence: int
    llm_adapter_late_entry_ratio: float
    llm_adapter_stop_tight_ratio: float
    llm_adapter_news_ratio: float
    llm_adapter_regime_ratio: float
    llm_adapter_onchain_against_ratio: float
    llm_adapter_onchain_low_reliability_ratio: float
    llm_adapter_threshold_add_late: float
    llm_adapter_threshold_add_stop: float
    llm_adapter_threshold_add_news: float
    llm_adapter_threshold_add_regime: float
    llm_adapter_threshold_add_onchain_against: float
    llm_adapter_threshold_add_onchain_reliability: float
    llm_adapter_onchain_min_reliability: float
    llm_adapter_onchain_max_age_minutes: int
    htf_context_enabled: bool
    htf_structure_lookback: int
    htf_raid_lookback: int
    htf_poi_max_distance_pct: float
    htf_structure_bonus: float
    htf_structure_penalty: float
    htf_poi_bonus: float
    htf_poi_penalty: float
    htf_raid_bonus: float
    htf_raid_penalty: float
    ltf_confirmation_min_signals: int
    ltf_confirmation_bonus: float
    ltf_no_confirmation_penalty: float
    altcoin_correlation_filter_enabled: bool
    max_correlated_altcoin_same_dir: int
    correlation_window_minutes: int


@dataclass
class TradeFiltersConfig:
    """Spread and specific trade filters."""
    min_tp1_to_spread_ratio: float
    min_quick_tp_to_spread_ratio: float

@dataclass
class OnChainConfig:
    """On-chain analysis configuration."""
    enabled: bool
    advisory_only: bool
    provider: str
    cache_ttl_minutes: int
    lookback_days: int
    positive_change_threshold_pct: float
    score_boost_max: float
    request_timeout_seconds: int
    use_reliability_weighting: bool
    min_reliability_to_score: float
    min_data_points: int
    freshness_half_life_minutes: int


@dataclass
class LLMPostmortemConfig:
    """LLM-driven postmortem review for losing trades."""
    enabled: bool
    only_losses: bool
    advisory_only: bool
    backfill_existing_on_startup: bool
    startup_max_reviews: int
    lookback_days: int
    min_confidence: int
    min_reviews_for_penalty: int
    penalty_max: float
    timeout_seconds: int


@dataclass
class PortfolioRiskConfig:
    """Portfolio-level risk controls for signal throttling."""
    enabled: bool
    daily_loss_limit_pct: float
    max_open_positions: int
    max_consecutive_losses: int
    loss_streak_cooldown_minutes: int
    weekly_loss_limit_pct: float
    max_drawdown_from_peak_pct: float
    pause_on_drawdown_breach: bool
    resume_after_consecutive_wins: int
    state_file: str


@dataclass
class QualityFirstConfig:
    """Strict shadow policy for comparing high-quality setups vs live logic."""
    enabled: bool
    name_ar: str
    htf_adx_min: float
    require_ema_alignment: bool
    allow_counter_trend: bool
    require_zone_proximity: bool
    max_zone_distance_pct: float
    require_candle_confirmation: bool
    min_pattern_strength: int
    min_volume_ratio: float
    max_spread_pct: float
    avoid_high_impact_news: bool
    min_rr_tp1: float
    min_rr_tp2: float


@dataclass
class Config:
    """Main configuration container."""
    exchange_name: str
    market_type: str
    symbols: list
    futures: SignalModeConfig
    futures_macro: SignalModeConfig
    limit: int
    scan_interval_seconds: int
    cooldown_minutes: int
    risk: RiskConfig
    regime: RegimeConfig
    scoring: ScoringConfig
    time_analysis: TimeAnalysisConfig
    learning: LearningConfig
    quality_filter: QualityFilterConfig
    trade_filters: TradeFiltersConfig
    onchain: OnChainConfig
    llm_postmortem: LLMPostmortemConfig
    portfolio_risk: PortfolioRiskConfig
    quality_first: QualityFirstConfig
    telegram: TelegramConfig
    openai: OpenAIConfig
    ollama: OllamaConfig
    mt5: Dict[str, Any] = field(default_factory=dict)
    mt5_bridge: Dict[str, Any] = field(default_factory=dict)
    schools: Dict[str, Any] = field(default_factory=dict)
    logging_level: str = "INFO"
    logging_file: str = "signal_bot.log"
    
    # Legacy accessors for backward compatibility
    @property
    def trend_tf(self) -> str:
        return self.futures.trend_tf
    
    @property
    def entry_tf(self) -> str:
        return self.futures.entry_tf
    
    @property
    def sr_tf(self) -> str:
        return self.futures.sr_tf

    @property
    def htf(self) -> str:
        return self.futures.htf


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Config dataclass with all settings
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    env_candidates = []
    seen_env_paths = set()
    for candidate in [path.parent / ".env", *[parent / ".env" for parent in path.parent.parents], Path.cwd() / ".env"]:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_env_paths or not resolved_candidate.exists():
            continue
        seen_env_paths.add(resolved_candidate)
        env_candidates.append(resolved_candidate)
    for env_path in env_candidates:
        load_dotenv(env_path, override=False)
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse nested configs
    risk_data = data.get('risk', {})
    risk_config = RiskConfig(
        buffer_pct=float(risk_data.get('buffer_pct', 0.35)),
        rr_tp1=float(risk_data.get('rr_tp1', 1.0)),
        rr_tp2=float(risk_data.get('rr_tp2', 2.0)),
        quick_tp_pct=float(risk_data.get('quick_tp_pct', 0.35)),
        quick_tp_min_pct=float(risk_data.get('quick_tp_min_pct', 0.12)),
        quick_tp1_fraction=float(risk_data.get('quick_tp1_fraction', 0.35)),
        quick_tp_outcome_enabled=bool(risk_data.get('quick_tp_outcome_enabled', True)),
        break_even_after_tp1=bool(risk_data.get('break_even_after_tp1', True)),
        break_even_buffer_pct=float(risk_data.get('break_even_buffer_pct', 0.02)),
        trailing_after_tp1=bool(risk_data.get('trailing_after_tp1', True)),
        trailing_rr_from_risk=float(risk_data.get('trailing_rr_from_risk', 0.70)),
        trailing_min_move_pct=float(risk_data.get('trailing_min_move_pct', 0.10)),
        atr_stop_mult=float(risk_data.get('atr_stop_mult', 1.8)),
        atr_buffer_mult=float(risk_data.get('atr_buffer_mult', 0.30)),
        tp2_atr_mult=float(risk_data.get('tp2_atr_mult', 4.0)),
        min_rr_tp1=float(risk_data.get('min_rr_tp1', 1.0)),
        max_sl_pct_crypto=float(risk_data.get('max_sl_pct_crypto', 1.8)),
        max_sl_pct_macro=float(risk_data.get('max_sl_pct_macro', 2.2)),
        timeframe_quick_tp_scale=dict(risk_data.get('timeframe_quick_tp_scale', {}) or {}),
        timeframe_tp2_rr_scale=dict(risk_data.get('timeframe_tp2_rr_scale', {}) or {}),
        time_stop=TimeStopConfig(
            enabled=bool(risk_data.get('time_stop', {}).get('enabled', True)),
            max_hold_minutes=int(risk_data.get('time_stop', {}).get('max_hold_minutes', 480)),
            min_progress_to_continue_pct=float(risk_data.get('time_stop', {}).get('min_progress_to_continue_pct', 0.30)),
            progress_check_at_pct=float(risk_data.get('time_stop', {}).get('progress_check_at_pct', 0.50)),
        )
    )

    regime_data = data.get('regime', {})
    regime_config = RegimeConfig(
        enabled=bool(regime_data.get('enabled', True)),
        adx_trend_min=float(regime_data.get('adx_trend_min', 22.0)),
        ema200_slope_abs_min=float(regime_data.get('ema200_slope_abs_min', 0.0015)),
        high_vol_atr_pct=float(regime_data.get('high_vol_atr_pct', 1.2)),
        sideways_threshold_add=int(regime_data.get('sideways_threshold_add', 1)),
        high_vol_threshold_add=int(regime_data.get('high_vol_threshold_add', 1)),
        aligned_score_bonus=float(regime_data.get('aligned_score_bonus', 0.6)),
        countertrend_score_penalty=float(regime_data.get('countertrend_score_penalty', 1.0)),
        quick_tp_scale_sideways=float(regime_data.get('quick_tp_scale_sideways', 0.8)),
        quick_tp_scale_high_vol=float(regime_data.get('quick_tp_scale_high_vol', 0.75)),
    )
    
    scoring_data = data.get('scoring', {})
    base_threshold_default = float(scoring_data.get('base_threshold', 6))
    scoring_config = ScoringConfig(
        base_threshold=int(base_threshold_default),
        base_threshold_crypto=float(scoring_data.get('base_threshold_crypto', base_threshold_default)),
        base_threshold_macro=float(scoring_data.get('base_threshold_macro', base_threshold_default)),
        add_timing_to_score=bool(scoring_data.get('add_timing_to_score', True)),
        max_timing_points_used=int(scoring_data.get('max_timing_points_used', 4)),
        dynamic_threshold_enabled=bool(scoring_data.get('dynamic_threshold_enabled', True)),
        dynamic_low_vol_atr_pct=float(scoring_data.get('dynamic_low_vol_atr_pct', 0.45)),
        dynamic_high_vol_atr_pct=float(scoring_data.get('dynamic_high_vol_atr_pct', 1.25)),
        dynamic_low_vol_adjust=float(scoring_data.get('dynamic_low_vol_adjust', -0.5)),
        dynamic_high_vol_adjust=float(scoring_data.get('dynamic_high_vol_adjust', 1.0)),
        dynamic_strong_trend_adx_min=float(scoring_data.get('dynamic_strong_trend_adx_min', 28.0)),
        dynamic_strong_trend_adjust=float(scoring_data.get('dynamic_strong_trend_adjust', -0.5)),
        min_threshold=float(scoring_data.get('min_threshold', 3.0)),
    )
    
    time_config = TimeAnalysisConfig(
        enabled=data['time_analysis']['enabled'],
        gann_angles=data['time_analysis']['gann_angles'],
        square9=data['time_analysis']['square9'],
        fibonacci=data.get('time_analysis', {}).get('fibonacci', False),
        cycle_52=data['time_analysis']['cycle_52'],
        lunar=data['time_analysis']['lunar'],
        fomc_filter=data['time_analysis']['fomc_filter'],
        cpi_filter=data.get('time_analysis', {}).get('cpi_filter', False),
        nfp_filter=data.get('time_analysis', {}).get('nfp_filter', False),
        powell_filter=data.get('time_analysis', {}).get('powell_filter', False),
        fomc_minutes_filter=data.get('time_analysis', {}).get('fomc_minutes_filter', False),
        sentiment_filter=data.get('time_analysis', {}).get('sentiment_filter', False),
        social_sentiment_filter=data.get('time_analysis', {}).get('social_sentiment_filter', False),
        lunar_window_hours=data['time_analysis']['lunar_window_hours'],
        fomc_high_vol_days=data['time_analysis']['fomc_high_vol_days'],
        cpi_high_vol_days=data.get('time_analysis', {}).get('cpi_high_vol_days', 1),
        nfp_high_vol_days=data.get('time_analysis', {}).get('nfp_high_vol_days', 1),
        powell_high_vol_hours=data.get('time_analysis', {}).get('powell_high_vol_hours', 24),
        fomc_minutes_high_vol_days=data.get('time_analysis', {}).get('fomc_minutes_high_vol_days', 1),
        sentiment_extreme_fear=data.get('time_analysis', {}).get('sentiment_extreme_fear', 25),
        sentiment_extreme_greed=data.get('time_analysis', {}).get('sentiment_extreme_greed', 75),
        social_sentiment_min_posts=data.get('time_analysis', {}).get('social_sentiment_min_posts', 20),
        social_sentiment_caution_ratio=data.get('time_analysis', {}).get('social_sentiment_caution_ratio', 0.52),
        social_sentiment_extreme_ratio=data.get('time_analysis', {}).get('social_sentiment_extreme_ratio', 0.60),
        zone_proximity_pct=data['time_analysis']['zone_proximity_pct'],
        expert_advisor=data.get('time_analysis', {}).get('expert_advisor', True),
        expert_advisor_timeout_seconds=data.get('time_analysis', {}).get('expert_advisor_timeout_seconds', 75),
        manager_advisor=data.get('time_analysis', {}).get('manager_advisor', True),
        manager_advisor_timeout_seconds=data.get('time_analysis', {}).get('manager_advisor_timeout_seconds', 75),
        expert_use_as_filter=data.get('time_analysis', {}).get('expert_use_as_filter', False),
        expert_min_confidence=data.get('time_analysis', {}).get('expert_min_confidence', 60),
        expert_block_on_wait=data.get('time_analysis', {}).get('expert_block_on_wait', True),
        expert_require_side_alignment=data.get('time_analysis', {}).get('expert_require_side_alignment', True),
        exit_min_hold_minutes=data.get('time_analysis', {}).get('exit_min_hold_minutes', 5),
        exit_min_pattern_strength=data.get('time_analysis', {}).get('exit_min_pattern_strength', 3),
        exit_reverse_confirmation_bars=data.get('time_analysis', {}).get('exit_reverse_confirmation_bars', 2),
    )

    learning_data = data.get('learning', {})
    learning_config = LearningConfig(
        enabled=learning_data.get('enabled', True),
        lookback_days=int(learning_data.get('lookback_days', 365)),
        retrain_interval_minutes=int(learning_data.get('retrain_interval_minutes', 15)),
        min_closed_trades=int(learning_data.get('min_closed_trades', 80)),
        min_symbol_side_trades=int(learning_data.get('min_symbol_side_trades', 12)),
        min_expected_winrate=float(learning_data.get('min_expected_winrate', 0.47)),
        min_expected_pnl_pct=float(learning_data.get('min_expected_pnl_pct', -0.05)),
        hard_block_winrate=float(learning_data.get('hard_block_winrate', 0.35)),
        score_bonus_max=float(learning_data.get('score_bonus_max', 1.0)),
        score_penalty_max=float(learning_data.get('score_penalty_max', 1.5)),
        block_on_low_quality=bool(learning_data.get('block_on_low_quality', True)),
        decision_mode=str(learning_data.get('decision_mode', 'hybrid')).strip().lower(),
        decision_min_samples=int(learning_data.get('decision_min_samples', 30)),
        decision_min_winrate=float(learning_data.get('decision_min_winrate', 0.50)),
        decision_min_pnl_pct=float(learning_data.get('decision_min_pnl_pct', 0.00)),
        decision_min_edge_pct=float(learning_data.get('decision_min_edge_pct', 0.01)),
        exclude_seeded_outcomes=bool(learning_data.get('exclude_seeded_outcomes', True)),
        include_only_real_outcomes=bool(learning_data.get('include_only_real_outcomes', True)),
        recency_half_life_days=float(learning_data.get('recency_half_life_days', 45.0)),
        estimated_roundtrip_cost_pct_crypto=float(
            learning_data.get('estimated_roundtrip_cost_pct_crypto', 0.08)
        ),
        estimated_roundtrip_cost_pct_macro=float(
            learning_data.get('estimated_roundtrip_cost_pct_macro', 0.03)
        ),
        use_regime_profile=bool(learning_data.get('use_regime_profile', True)),
        regime_blend_strength=float(learning_data.get('regime_blend_strength', 0.35)),
        regime_min_samples=int(learning_data.get('regime_min_samples', 8)),
        hard_block_requires_full_local_samples=bool(
            learning_data.get('hard_block_requires_full_local_samples', True)
        ),
        optimize_for_expectancy=bool(learning_data.get('optimize_for_expectancy', True)),
        expectancy_weight=float(learning_data.get('expectancy_weight', 1.0)),
        decision_min_expectancy_pct=float(learning_data.get('decision_min_expectancy_pct', 0.00)),
        hybrid_min_expectancy_pct=float(learning_data.get('hybrid_min_expectancy_pct', -0.08)),
        seeded_weight=float(learning_data.get('seeded_weight', 0.25)),
    )

    quality_data = data.get('quality_filter', {})
    raw_session_windows = quality_data.get(
        'macro_session_utc_windows',
        ['06:00-16:30', '13:00-20:00'],
    )
    if isinstance(raw_session_windows, (list, tuple)):
        macro_session_windows = [str(x).strip() for x in raw_session_windows if str(x).strip()]
    elif isinstance(raw_session_windows, str):
        macro_session_windows = [x.strip() for x in raw_session_windows.split(',') if x.strip()]
    else:
        macro_session_windows = []
    if not macro_session_windows:
        macro_session_windows = ['06:00-16:30', '13:00-20:00']

    quality_filter_config = QualityFilterConfig(
        enabled=bool(quality_data.get('enabled', True)),
        max_spread_pct=float(quality_data.get('max_spread_pct', 0.22)),
        min_volume_ratio=float(quality_data.get('min_volume_ratio', 0.65)),
        min_volume_ratio_macro=float(quality_data.get('min_volume_ratio_macro', 0.0)),
        block_during_high_impact_news=bool(quality_data.get('block_during_high_impact_news', True)),
        high_impact_news_mode=str(quality_data.get('high_impact_news_mode', 'block')).strip().lower(),
        high_impact_threshold_add=float(quality_data.get('high_impact_threshold_add', 0.8)),
        block_on_opposing_news=bool(quality_data.get('block_on_opposing_news', True)),
        opposing_news_score=float(quality_data.get('opposing_news_score', 0.9)),
        opposing_news_score_macro=float(quality_data.get('opposing_news_score_macro', 1.6)),
        require_learning_alignment=bool(quality_data.get('require_learning_alignment', True)),
        learning_min_samples=int(quality_data.get('learning_min_samples', 10)),
        max_negative_learning_adjustment=float(quality_data.get('max_negative_learning_adjustment', -0.4)),
        relaxed_volume_mode=bool(quality_data.get('relaxed_volume_mode', True)),
        min_volume_ratio_relaxed=float(quality_data.get('min_volume_ratio_relaxed', 0.45)),
        volume_relax_adx_min=float(quality_data.get('volume_relax_adx_min', 24.0)),
        volume_relax_max_spread_pct=float(quality_data.get('volume_relax_max_spread_pct', 0.16)),
        volume_relax_min_score_margin=float(quality_data.get('volume_relax_min_score_margin', -0.2)),
        anti_chasing_atr_mult=float(quality_data.get('anti_chasing_atr_mult', 3.0)),
        anti_chasing_lookback_bars=int(quality_data.get('anti_chasing_lookback_bars', 3)),
        macro_session_filter_enabled=bool(quality_data.get('macro_session_filter_enabled', True)),
        macro_session_mode=str(quality_data.get('macro_session_mode', 'cautious')).strip().lower(),
        macro_session_threshold_add=float(quality_data.get('macro_session_threshold_add', 0.8)),
        macro_session_utc_windows=macro_session_windows,
        llm_adapter_enabled=bool(quality_data.get('llm_adapter_enabled', True)),
        llm_adapter_shadow_mode=bool(quality_data.get('llm_adapter_shadow_mode', False)),
        llm_adapter_lookback_days=int(quality_data.get('llm_adapter_lookback_days', 21)),
        llm_adapter_min_reviews=int(quality_data.get('llm_adapter_min_reviews', 3)),
        llm_adapter_min_confidence=int(quality_data.get('llm_adapter_min_confidence', 60)),
        llm_adapter_late_entry_ratio=float(quality_data.get('llm_adapter_late_entry_ratio', 0.45)),
        llm_adapter_stop_tight_ratio=float(quality_data.get('llm_adapter_stop_tight_ratio', 0.45)),
        llm_adapter_news_ratio=float(quality_data.get('llm_adapter_news_ratio', 0.35)),
        llm_adapter_regime_ratio=float(quality_data.get('llm_adapter_regime_ratio', 0.35)),
        llm_adapter_onchain_against_ratio=float(quality_data.get('llm_adapter_onchain_against_ratio', 0.35)),
        llm_adapter_onchain_low_reliability_ratio=float(
            quality_data.get('llm_adapter_onchain_low_reliability_ratio', 0.45)
        ),
        llm_adapter_threshold_add_late=float(quality_data.get('llm_adapter_threshold_add_late', 0.35)),
        llm_adapter_threshold_add_stop=float(quality_data.get('llm_adapter_threshold_add_stop', 0.20)),
        llm_adapter_threshold_add_news=float(quality_data.get('llm_adapter_threshold_add_news', 0.30)),
        llm_adapter_threshold_add_regime=float(quality_data.get('llm_adapter_threshold_add_regime', 0.25)),
        llm_adapter_threshold_add_onchain_against=float(
            quality_data.get('llm_adapter_threshold_add_onchain_against', 0.30)
        ),
        llm_adapter_threshold_add_onchain_reliability=float(
            quality_data.get('llm_adapter_threshold_add_onchain_reliability', 0.20)
        ),
        llm_adapter_onchain_min_reliability=float(
            quality_data.get('llm_adapter_onchain_min_reliability', 0.45)
        ),
        llm_adapter_onchain_max_age_minutes=max(
            30, int(quality_data.get('llm_adapter_onchain_max_age_minutes', 360))
        ),
        htf_context_enabled=bool(quality_data.get('htf_context_enabled', True)),
        htf_structure_lookback=max(20, int(quality_data.get('htf_structure_lookback', 60))),
        htf_raid_lookback=max(10, int(quality_data.get('htf_raid_lookback', 40))),
        htf_poi_max_distance_pct=max(0.05, float(quality_data.get('htf_poi_max_distance_pct', 0.35))),
        htf_structure_bonus=float(quality_data.get('htf_structure_bonus', 1.0)),
        htf_structure_penalty=float(quality_data.get('htf_structure_penalty', 1.3)),
        htf_poi_bonus=float(quality_data.get('htf_poi_bonus', 1.2)),
        htf_poi_penalty=float(quality_data.get('htf_poi_penalty', 1.2)),
        htf_raid_bonus=float(quality_data.get('htf_raid_bonus', 0.8)),
        htf_raid_penalty=float(quality_data.get('htf_raid_penalty', 0.6)),
        ltf_confirmation_min_signals=max(1, int(quality_data.get('ltf_confirmation_min_signals', 1))),
        ltf_confirmation_bonus=float(quality_data.get('ltf_confirmation_bonus', 0.8)),
        ltf_no_confirmation_penalty=float(quality_data.get('ltf_no_confirmation_penalty', 1.0)),
        altcoin_correlation_filter_enabled=bool(quality_data.get('altcoin_correlation_filter_enabled', True)),
        max_correlated_altcoin_same_dir=max(1, int(quality_data.get('max_correlated_altcoin_same_dir', 2))),
        correlation_window_minutes=max(5, int(quality_data.get('correlation_window_minutes', 30))),
    )

    trade_filters_data = data.get('trade_filters', {})
    trade_filters_config = TradeFiltersConfig(
        min_tp1_to_spread_ratio=float(trade_filters_data.get('min_tp1_to_spread_ratio', 3.0)),
        min_quick_tp_to_spread_ratio=float(trade_filters_data.get('min_quick_tp_to_spread_ratio', 4.0)),
    )

    onchain_data = data.get('onchain', {})
    onchain_config = OnChainConfig(
        enabled=bool(onchain_data.get('enabled', False)),
        advisory_only=bool(onchain_data.get('advisory_only', True)),
        provider=str(onchain_data.get('provider', 'coinmetrics')).strip().lower(),
        cache_ttl_minutes=max(1, int(onchain_data.get('cache_ttl_minutes', 20))),
        lookback_days=max(8, int(onchain_data.get('lookback_days', 8))),
        positive_change_threshold_pct=max(0.1, float(onchain_data.get('positive_change_threshold_pct', 5.0))),
        score_boost_max=max(0.1, float(onchain_data.get('score_boost_max', 1.2))),
        request_timeout_seconds=max(2, int(onchain_data.get('request_timeout_seconds', 10))),
        use_reliability_weighting=bool(onchain_data.get('use_reliability_weighting', True)),
        min_reliability_to_score=max(0.0, min(1.0, float(onchain_data.get('min_reliability_to_score', 0.45)))),
        min_data_points=max(4, int(onchain_data.get('min_data_points', 8))),
        freshness_half_life_minutes=max(10, int(onchain_data.get('freshness_half_life_minutes', 180))),
    )

    llm_postmortem_data = data.get('llm_postmortem', {})
    llm_postmortem_config = LLMPostmortemConfig(
        enabled=bool(llm_postmortem_data.get('enabled', False)),
        only_losses=bool(llm_postmortem_data.get('only_losses', True)),
        advisory_only=bool(llm_postmortem_data.get('advisory_only', False)),
        backfill_existing_on_startup=bool(llm_postmortem_data.get('backfill_existing_on_startup', True)),
        startup_max_reviews=max(1, int(llm_postmortem_data.get('startup_max_reviews', 40))),
        lookback_days=max(1, int(llm_postmortem_data.get('lookback_days', 21))),
        min_confidence=max(0, min(100, int(llm_postmortem_data.get('min_confidence', 60)))),
        min_reviews_for_penalty=max(1, int(llm_postmortem_data.get('min_reviews_for_penalty', 2))),
        penalty_max=max(0.0, float(llm_postmortem_data.get('penalty_max', 0.8))),
        timeout_seconds=max(10, int(llm_postmortem_data.get('timeout_seconds', 45))),
    )

    portfolio_risk_data = data.get('portfolio_risk', {})
    portfolio_risk_config = PortfolioRiskConfig(
        enabled=bool(portfolio_risk_data.get('enabled', True)),
        daily_loss_limit_pct=float(portfolio_risk_data.get('daily_loss_limit_pct', -3.0)),
        max_open_positions=int(portfolio_risk_data.get('max_open_positions', 3)),
        max_consecutive_losses=int(portfolio_risk_data.get('max_consecutive_losses', 4)),
        loss_streak_cooldown_minutes=int(portfolio_risk_data.get('loss_streak_cooldown_minutes', 90)),
        weekly_loss_limit_pct=float(portfolio_risk_data.get('weekly_loss_limit_pct', -7.0)),
        max_drawdown_from_peak_pct=float(portfolio_risk_data.get('max_drawdown_from_peak_pct', -10.0)),
        pause_on_drawdown_breach=bool(portfolio_risk_data.get('pause_on_drawdown_breach', True)),
        resume_after_consecutive_wins=int(portfolio_risk_data.get('resume_after_consecutive_wins', 2)),
        state_file=str(portfolio_risk_data.get('state_file', 'equity_state.json')).strip(),
    )

    quality_first_data = data.get('quality_first', {})
    quality_first_config = QualityFirstConfig(
        enabled=bool(quality_first_data.get('enabled', True)),
        name_ar=str(quality_first_data.get('name_ar', 'الجودة أهم')).strip() or 'الجودة أهم',
        htf_adx_min=float(quality_first_data.get('htf_adx_min', 22.0)),
        require_ema_alignment=bool(quality_first_data.get('require_ema_alignment', True)),
        allow_counter_trend=bool(quality_first_data.get('allow_counter_trend', False)),
        require_zone_proximity=bool(quality_first_data.get('require_zone_proximity', True)),
        max_zone_distance_pct=float(quality_first_data.get('max_zone_distance_pct', 0.25)),
        require_candle_confirmation=bool(quality_first_data.get('require_candle_confirmation', True)),
        min_pattern_strength=int(quality_first_data.get('min_pattern_strength', 2)),
        min_volume_ratio=float(quality_first_data.get('min_volume_ratio', 1.10)),
        max_spread_pct=float(quality_first_data.get('max_spread_pct', 0.18)),
        avoid_high_impact_news=bool(quality_first_data.get('avoid_high_impact_news', True)),
        min_rr_tp1=float(quality_first_data.get('min_rr_tp1', 1.0)),
        min_rr_tp2=float(quality_first_data.get('min_rr_tp2', 1.8)),
    )
    
    # Load Telegram credentials from environment
    telegram_config = TelegramConfig(
        enabled=data['telegram']['enabled'],
        bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )
    
    openai_data = data.get('openai', {})

    openai_config = OpenAIConfig(
        enabled=openai_data.get('enabled', False),
        api_key=os.getenv('OPENAI_API_KEY') or "dummy",
        model=openai_data.get('model', 'gpt-4o-mini'),
        base_url=openai_data.get('base_url', None)
    )

    ollama_data = data.get('ollama', {})

    ollama_config = OllamaConfig(
        enabled=ollama_data.get('enabled', False),
        model=ollama_data.get('model', 'qwen2.5:7b'),
        base_url=ollama_data.get('base_url', 'http://localhost:11434')
    )

    logging_data = data.get('logging', {})
    
    # Signal mode configurations
    futures_config = SignalModeConfig(
        enabled=data.get('futures', {}).get('enabled', True),
        trend_tf=data.get('futures', {}).get('trend_tf', '4h'),
        entry_tf=data.get('futures', {}).get('entry_tf', '15m'),
        sr_tf=data.get('futures', {}).get('sr_tf', '1h'),
        htf=str(data.get('futures', {}).get('htf', '1h')).strip(),
    )

    # Macro-specific overrides (Gold, Oil, FX, Indices via Yahoo Finance)
    macro_data = data.get('futures_macro', {})
    if macro_data:
        futures_macro_config = SignalModeConfig(
            enabled=bool(macro_data.get('enabled', True)),
            trend_tf=str(macro_data.get('trend_tf', '1h')).strip(),
            entry_tf=str(macro_data.get('entry_tf', '15m')).strip(),
            sr_tf=str(macro_data.get('sr_tf', '30m')).strip(),
            htf=str(macro_data.get('htf', '4h')).strip(),
        )
    else:
        # Default: macro uses larger timeframes than crypto
        futures_macro_config = SignalModeConfig(
            enabled=futures_config.enabled,
            trend_tf='1h',
            entry_tf='15m',
            sr_tf='30m',
            htf='4h',
        )
    
    return Config(
        exchange_name=data['exchange_name'],
        market_type=data['market_type'],
        symbols=data['symbols'],
        futures=futures_config,
        futures_macro=futures_macro_config,
        limit=data['limit'],
        scan_interval_seconds=data['scan_interval_seconds'],
        cooldown_minutes=data['cooldown_minutes'],
        risk=risk_config,
        regime=regime_config,
        scoring=scoring_config,
        time_analysis=time_config,
        learning=learning_config,
        quality_filter=quality_filter_config,
        trade_filters=trade_filters_config,
        onchain=onchain_config,
        llm_postmortem=llm_postmortem_config,
        portfolio_risk=portfolio_risk_config,
        quality_first=quality_first_config,
        telegram=telegram_config,
        openai=openai_config,
        ollama=ollama_config,
        mt5=dict(data.get('mt5', {}) or {}),
        mt5_bridge=dict(data.get('mt5_bridge', {}) or {}),
        schools=dict(data.get('schools', {}) or {}),
        logging_level=logging_data.get('level', 'INFO'),
        logging_file=logging_data.get('file', 'signal_bot.log')
    )


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure loguru logger.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )


def format_price(price: float, symbol: str = "") -> str:
    """
    Format price with adaptive decimal places.
    
    Args:
        price: The price to format
        symbol: Optional symbol for context-based formatting
        
    Returns:
        Formatted price string
    """
    if price >= 1000:
        return f"{price:.2f}"
    elif price >= 100:
        return f"{price:.3f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.01:
        return f"{price:.5f}"
    else:
        return f"{price:.6f}"


def get_decimal_places(price: float) -> int:
    """
    Get appropriate decimal places for a price.
    
    Args:
        price: The price to analyze
        
    Returns:
        Number of decimal places to use
    """
    if price >= 1000:
        return 2
    elif price >= 100:
        return 3
    elif price >= 1:
        return 4
    elif price >= 0.01:
        return 5
    else:
        return 6


def pct_change(current: float, reference: float) -> float:
    """
    Calculate percentage change.
    
    Args:
        current: Current value
        reference: Reference value
        
    Returns:
        Percentage change as decimal (e.g., 0.05 for 5%)
    """
    if reference == 0:
        return 0.0
    return (current - reference) / reference


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """
    Round price to nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum price increment
        
    Returns:
        Rounded price
    """
    return round(price / tick_size) * tick_size
