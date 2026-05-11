"""
Tests for CPI calendar parsing and analysis.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from bot.calendar_events import (
    CPIRelease,
    CPIAnalysis,
    _parse_cpi_dates_from_text,
    analyze_cpi,
)


SAMPLE_CPI_HTML = """
<html>
<body>
Consumer Price Index, March 12, 2026
Consumer Price Index, April 10, 2026
</body>
</html>
"""


class TestCPIParsing:
    """Tests for CPI date extraction."""

    def test_parse_cpi_dates_from_text(self):
        """Should extract CPI dates and convert to UTC datetimes."""
        releases = _parse_cpi_dates_from_text(SAMPLE_CPI_HTML)

        assert len(releases) >= 2
        assert releases[0].release_datetime.tzinfo is not None
        assert releases[0].release_datetime.tzinfo == timezone.utc


class TestCPIAnalysis:
    """Tests for CPI risk analysis."""

    def test_analyze_cpi_returns_valid_result(self):
        """analyze_cpi should return a valid analysis object."""
        result = analyze_cpi(high_vol_days=1)

        assert isinstance(result, CPIAnalysis)
        assert -1 <= result.cpi_score <= 0

    @patch('bot.calendar_events.fetch_cpi_dates')
    def test_analyze_cpi_in_window(self, mock_fetch):
        """Release near now should be in high-vol window."""
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [
            CPIRelease(release_datetime=now + timedelta(hours=2), label="CPI")
        ]

        result = analyze_cpi(high_vol_days=1, dt=now)

        assert result.in_high_vol_window
        assert result.cpi_score < 0
        assert result.hours_to_next <= 2.1

    @patch('bot.calendar_events.fetch_cpi_dates')
    def test_analyze_cpi_outside_window(self, mock_fetch):
        """Far release should not trigger high-vol penalty."""
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = [
            CPIRelease(release_datetime=now + timedelta(days=5), label="CPI")
        ]

        result = analyze_cpi(high_vol_days=1, dt=now)

        assert not result.in_high_vol_window
        assert result.cpi_score == 0
