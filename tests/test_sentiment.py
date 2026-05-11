"""
Tests for market sentiment analysis (Fear & Greed).
"""

from datetime import datetime, timezone
from unittest.mock import patch

from bot.calendar_events import SentimentSnapshot, analyze_market_sentiment


class TestSentimentAnalysis:
    """Tests for sentiment timing analysis."""

    @patch('bot.calendar_events.fetch_market_sentiment')
    def test_extreme_sentiment_penalty(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = SentimentSnapshot(
            value=85,
            classification="Extreme Greed",
            updated_at=now
        )

        result = analyze_market_sentiment(extreme_fear=25, extreme_greed=75)
        assert result.in_extreme_zone
        assert result.sentiment_score == -0.8

    @patch('bot.calendar_events.fetch_market_sentiment')
    def test_caution_sentiment_penalty(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = SentimentSnapshot(
            value=70,
            classification="Greed",
            updated_at=now
        )

        result = analyze_market_sentiment(extreme_fear=25, extreme_greed=75)
        assert not result.in_extreme_zone
        assert result.sentiment_score == -0.4

    @patch('bot.calendar_events.fetch_market_sentiment')
    def test_neutral_sentiment_no_penalty(self, mock_fetch):
        now = datetime.now(timezone.utc)
        mock_fetch.return_value = SentimentSnapshot(
            value=50,
            classification="Neutral",
            updated_at=now
        )

        result = analyze_market_sentiment(extreme_fear=25, extreme_greed=75)
        assert not result.in_extreme_zone
        assert result.sentiment_score == 0.0

    @patch('bot.calendar_events.fetch_market_sentiment')
    def test_missing_sentiment_data(self, mock_fetch):
        mock_fetch.return_value = None

        result = analyze_market_sentiment(extreme_fear=25, extreme_greed=75)
        assert result.value is None
        assert result.sentiment_score == 0.0
