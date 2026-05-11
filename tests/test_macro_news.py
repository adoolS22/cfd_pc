"""
Tests for additional macro news analyzers: NFP, Powell speeches, FOMC minutes.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from bot.calendar_events import (
    NFPRelease,
    PowellSpeech,
    FOMCMeeting,
    _parse_named_dates_with_year,
    analyze_nfp,
    analyze_powell_speeches,
    analyze_fomc_minutes,
    fetch_fomc_minutes_dates,
)


class TestNFPParsing:
    """Tests for NFP date parsing."""

    def test_parse_named_dates_with_year(self):
        text = "Employment Situation: March 7, 2026 and April 3, 2026"
        rows = _parse_named_dates_with_year(text, label_prefix="NFP", keep_since_days=800)

        assert len(rows) >= 2
        assert rows[0]['datetime'].tzinfo == timezone.utc


class TestNFPAnalysis:
    """Tests for NFP timing analysis."""

    @patch('bot.calendar_events.fetch_nfp_dates')
    def test_analyze_nfp_in_window(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [NFPRelease(release_datetime=now + timedelta(hours=3), label="NFP")]

        result = analyze_nfp(high_vol_days=1, dt=now)
        assert result.in_high_vol_window
        assert result.nfp_score < 0

    @patch('bot.calendar_events.fetch_nfp_dates')
    def test_analyze_nfp_outside_window(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [NFPRelease(release_datetime=now + timedelta(days=4), label="NFP")]

        result = analyze_nfp(high_vol_days=1, dt=now)
        assert not result.in_high_vol_window
        assert result.nfp_score == 0


class TestPowellAnalysis:
    """Tests for Powell speech timing analysis."""

    @patch('bot.calendar_events.fetch_powell_speeches')
    def test_analyze_powell_in_window(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [
            PowellSpeech(event_datetime=now + timedelta(hours=2), title="Chair Powell speech")
        ]

        result = analyze_powell_speeches(high_vol_hours=24, dt=now)
        assert result.in_high_vol_window
        assert result.powell_score < 0

    @patch('bot.calendar_events.fetch_powell_speeches')
    def test_analyze_powell_outside_window(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [
            PowellSpeech(event_datetime=now + timedelta(days=5), title="Chair Powell speech")
        ]

        result = analyze_powell_speeches(high_vol_hours=24, dt=now)
        assert not result.in_high_vol_window
        assert result.powell_score == 0


class TestFOMCMinutes:
    """Tests for FOMC minutes generation and analysis."""

    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_fetch_fomc_minutes_dates(self, mock_fetch):
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=datetime(2026, 3, 17, tzinfo=timezone.utc),
                end_date=datetime(2026, 3, 18, tzinfo=timezone.utc),
                is_scheduled=True,
                has_projections=False
            )
        ]

        releases = fetch_fomc_minutes_dates()
        assert len(releases) == 1
        assert releases[0].release_datetime > mock_fetch.return_value[0].end_date

    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_analyze_fomc_minutes_in_window(self, mock_fetch):
        meeting_end = datetime(2026, 3, 18, tzinfo=timezone.utc)
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=meeting_end - timedelta(days=1),
                end_date=meeting_end,
                is_scheduled=True,
                has_projections=False
            )
        ]

        dt = meeting_end + timedelta(days=21, hours=1)
        result = analyze_fomc_minutes(high_vol_days=1, dt=dt)
        assert result.in_high_vol_window
        assert result.minutes_score < 0
