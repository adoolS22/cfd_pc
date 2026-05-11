"""
Tests for signal scoring logic.
"""

import pandas as pd

from bot.signals import calculate_technical_score
from bot.zones import Zone, ZoneProximity
from bot.patterns import CandlePattern


class TestTechnicalScoring:
    """Tests for technical scoring logic."""
    
    def test_trend_match_long(self):
        """Test trend match scoring for LONG."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        score, reasons = calculate_technical_score(
            trend='up',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result={},
            rsi_divergence=None
        )
        
        assert score >= 2  # Trend match
        assert any('Trend' in r for r in reasons)
    
    def test_trend_match_short(self):
        """Test trend match scoring for SHORT."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        score, reasons = calculate_technical_score(
            trend='down',
            side='SHORT',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result={},
            rsi_divergence=None
        )
        
        assert score >= 2  # Trend match
    
    def test_zone_score(self):
        """Test zone proximity scoring."""
        zone = Zone(
            type='support',
            level=100,
            upper=101,
            lower=99,
            strength=3,
            timestamp=pd.Timestamp.now(tz='UTC')
        )
        zone_result = ZoneProximity(
            in_zone=True,
            zone=zone,
            distance_pct=0.05,
            zone_type='support'
        )
        
        score, reasons = calculate_technical_score(
            trend='neutral',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result={},
            rsi_divergence=None
        )
        
        assert score >= 2  # Zone match
        assert any('zone' in r.lower() for r in reasons)
    
    def test_pattern_score(self):
        """Test candlestick pattern scoring."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        pattern = CandlePattern(
            name='bullish_engulfing',
            direction='bullish',
            strength=2,
            timestamp=pd.Timestamp.now(tz='UTC'),
            details={}
        )
        
        score, reasons = calculate_technical_score(
            trend='neutral',
            side='LONG',
            zone_result=zone_result,
            pattern=pattern,
            volume_spike=False,
            wave_result={},
            rsi_divergence=None
        )
        
        assert score >= 2  # Pattern
        assert any('Engulfing' in r for r in reasons)
    
    def test_volume_spike_score(self):
        """Test volume spike scoring."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        score, reasons = calculate_technical_score(
            trend='neutral',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=True,
            wave_result={},
            rsi_divergence=None
        )
        
        assert score >= 2  # Volume spike
        assert any('Volume' in r for r in reasons)
    
    def test_wave_trigger_score(self):
        """Test wave 3 trigger scoring."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        wave_result = {'triggered': True, 'score': 2}
        
        score, reasons = calculate_technical_score(
            trend='neutral',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result=wave_result,
            rsi_divergence=None
        )
        
        assert score >= 2  # Wave
        assert any('Wave' in r for r in reasons)
    
    def test_rsi_divergence_penalty(self):
        """Test RSI divergence penalty."""
        zone_result = ZoneProximity(
            in_zone=False,
            zone=None,
            distance_pct=5.0,
            zone_type=None
        )
        
        # Base score with trend match
        base_score, _ = calculate_technical_score(
            trend='up',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result={},
            rsi_divergence=None
        )
        
        # Score with bearish divergence
        penalized_score, reasons = calculate_technical_score(
            trend='up',
            side='LONG',
            zone_result=zone_result,
            pattern=None,
            volume_spike=False,
            wave_result={},
            rsi_divergence='bearish'
        )
        
        assert penalized_score < base_score
        assert any('divergence' in r.lower() for r in reasons)
    
    def test_full_score_combination(self):
        """Test combination of all scoring factors."""
        zone = Zone(
            type='support',
            level=100,
            upper=101,
            lower=99,
            strength=3,
            timestamp=pd.Timestamp.now(tz='UTC')
        )
        zone_result = ZoneProximity(
            in_zone=True,
            zone=zone,
            distance_pct=0.05,
            zone_type='support'
        )
        
        pattern = CandlePattern(
            name='hammer',
            direction='bullish',
            strength=2,
            timestamp=pd.Timestamp.now(tz='UTC'),
            details={}
        )
        
        wave_result = {'triggered': True, 'score': 2}
        
        score, reasons = calculate_technical_score(
            trend='up',
            side='LONG',
            zone_result=zone_result,
            pattern=pattern,
            volume_spike=True,
            wave_result=wave_result,
            rsi_divergence=None
        )
        
        # Should have high score: 2(trend) + 2(zone) + 2(pattern) + 2(volume) + 2(wave) = 10
        assert score >= 10
        assert len(reasons) >= 5
