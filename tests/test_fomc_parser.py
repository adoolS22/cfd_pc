"""
Tests for FOMC calendar parsing.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from bot.calendar_events import (
    fetch_fomc_dates,
    get_next_fomc_meeting,
    analyze_fomc,
    _parse_dates_from_text,
    _month_to_num,
    FOMCMeeting,
    FOMCAnalysis
)


# Sample HTML content that mimics Federal Reserve page structure
SAMPLE_FOMC_HTML = """
<html>
<body>
<div class="fomc-meeting">
    <p>January 28-29, 2025 *</p>
</div>
<div class="fomc-meeting">
    <p>March 18-19, 2025</p>
</div>
<div class="fomc-meeting">
    <p>May 6-7, 2025</p>
</div>
<div class="fomc-meeting">
    <p>June 17-18, 2025 *</p>
</div>
</body>
</html>
"""


class TestMonthParsing:
    """Tests for month name parsing."""
    
    def test_month_to_num_valid(self):
        """Test valid month conversions."""
        assert _month_to_num('January') == 1
        assert _month_to_num('february') == 2
        assert _month_to_num('MARCH') == 3
        assert _month_to_num('December') == 12
    
    def test_month_to_num_invalid(self):
        """Test invalid month returns 1."""
        assert _month_to_num('Invalid') == 1
        assert _month_to_num('') == 1


class TestDateParsing:
    """Tests for date parsing from text."""
    
    def test_parse_dates_from_text(self):
        """Test parsing FOMC dates from HTML text."""
        meetings = _parse_dates_from_text(SAMPLE_FOMC_HTML)
        
        assert len(meetings) >= 4
        
        # Check first meeting
        jan_meetings = [m for m in meetings if m.start_date.month == 1 and m.start_date.year == 2025]
        assert len(jan_meetings) > 0
        assert jan_meetings[0].start_date.day == 28
        assert jan_meetings[0].end_date.day == 29
    
    def test_parse_dates_removes_duplicates(self):
        """Test that duplicate dates are removed."""
        html_with_dups = """
        January 28-29, 2025
        January 28-29, 2025
        March 18-19, 2025
        """
        meetings = _parse_dates_from_text(html_with_dups)
        
        # Should have no duplicates
        end_dates = [m.end_date.date() for m in meetings]
        assert len(end_dates) == len(set(end_dates))
    
    def test_parse_dates_handles_missing_year(self):
        """Test parsing when year is missing."""
        html = "January 28-29"
        meetings = _parse_dates_from_text(html)
        
        # Should assume current year or next
        assert len(meetings) >= 1


class TestFOMCAnalysis:
    """Tests for FOMC analysis."""
    
    def test_analyze_fomc_returns_valid_result(self):
        """Test that analyze_fomc returns valid result."""
        result = analyze_fomc(high_vol_days=2)
        
        assert isinstance(result, FOMCAnalysis)
        assert -1 <= result.fomc_score <= 0
    
    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_analyze_fomc_in_window(self, mock_fetch):
        """Test FOMC analysis when in high volatility window."""
        # Mock a meeting happening tomorrow
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=tomorrow - timedelta(days=1),
                end_date=tomorrow,
                is_scheduled=True,
                has_projections=False
            )
        ]
        
        result = analyze_fomc(high_vol_days=2)
        
        assert result.in_high_vol_window
        assert result.fomc_score < 0
        assert result.days_to_next <= 2
    
    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_analyze_fomc_outside_window(self, mock_fetch):
        """Test FOMC analysis when outside high volatility window."""
        # Mock a meeting happening in 10 days
        future = datetime.now(timezone.utc) + timedelta(days=10)
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=future - timedelta(days=1),
                end_date=future,
                is_scheduled=True,
                has_projections=False
            )
        ]
        
        result = analyze_fomc(high_vol_days=2)
        
        assert not result.in_high_vol_window
        assert result.fomc_score == 0

    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_analyze_fomc_post_meeting_window(self, mock_fetch):
        """Meeting just ended: should still be inside post-FOMC high-vol window."""
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=now - timedelta(days=2),
                end_date=now - timedelta(days=1),
                is_scheduled=True,
                has_projections=False
            )
        ]

        result = analyze_fomc(high_vol_days=2)

        assert result.in_high_vol_window
        assert result.fomc_score < 0
        assert result.days_to_next <= 0
    
    @patch('bot.calendar_events.fetch_fomc_dates')
    def test_analyze_fomc_no_meetings(self, mock_fetch):
        """Test FOMC analysis when no meetings found."""
        mock_fetch.return_value = []
        
        result = analyze_fomc()
        
        assert result.next_meeting_end is None
        assert result.days_to_next == 999
        assert not result.in_high_vol_window


class TestFOMCMeeting:
    """Tests for FOMCMeeting dataclass."""
    
    def test_fomc_meeting_creation(self):
        """Test FOMCMeeting dataclass creation."""
        meeting = FOMCMeeting(
            start_date=datetime(2025, 1, 28, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 29, tzinfo=timezone.utc),
            is_scheduled=True,
            has_projections=True
        )
        
        assert meeting.start_date.day == 28
        assert meeting.end_date.day == 29
        assert meeting.is_scheduled
        assert meeting.has_projections


class TestCaching:
    """Tests for FOMC cache functionality."""
    
    @patch('bot.calendar_events._get_cache_path')
    @patch('bot.calendar_events._fetch_from_fed')
    def test_fetches_when_cache_missing(self, mock_fetch, mock_path):
        """Test that data is fetched when cache is missing."""
        import tempfile
        import os
        
        # Use a non-existent path
        mock_path.return_value = type('Path', (), {'exists': lambda self: False})()
        mock_fetch.return_value = [
            FOMCMeeting(
                start_date=datetime(2025, 3, 18, tzinfo=timezone.utc),
                end_date=datetime(2025, 3, 19, tzinfo=timezone.utc),
                is_scheduled=True,
                has_projections=False
            )
        ]
        
        # This would normally fetch from network
        # We mock it to test the logic flow
        result = fetch_fomc_dates()
        
        mock_fetch.assert_called_once()
