"""
Tests for social sentiment analysis from Reddit headlines.
"""

from unittest.mock import patch

from bot.calendar_events import analyze_social_sentiment


class TestSocialSentimentAnalysis:
    """Tests for Reddit-based crowd sentiment scoring."""

    @patch('bot.calendar_events.fetch_reddit_hot_titles')
    def test_extreme_bullish_crowd_penalty(self, mock_fetch):
        mock_fetch.return_value = [
            "BTC breakout moon long setup",
            "Mega bull rally incoming",
            "We only go up buy now",
            "Another ATH is coming",
        ] * 8  # 32 posts

        result = analyze_social_sentiment(min_posts=20, caution_ratio=0.52, extreme_ratio=0.60)
        assert result.in_extreme_zone
        assert result.social_score == -0.6
        assert result.dominant_side == "bullish"

    @patch('bot.calendar_events.fetch_reddit_hot_titles')
    def test_caution_bearish_crowd_penalty(self, mock_fetch):
        bearish = [
            "Crypto crash panic sell warning",
            "Short setup after rejection",
            "Bearish breakdown confirmed",
            "More downside expected",
            "Liquidation wave hits market",
            "Downtrend continues",
        ] * 3  # 18
        neutral = [
            "What is your favorite wallet?",
            "Best exchange UX this year?",
            "Any thoughts on staking fees?",
            "Portfolio allocation discussion",
            "General macro thread",
            "Weekend market recap",
            "Stablecoin question",
            "Security tips for beginners",
            "Random altcoin discussion",
            "Learning technical analysis",
            "How to manage risk",
            "Cold storage guide",
        ]  # 12
        mock_fetch.return_value = bearish + neutral  # 30 posts, bearish ratio = 0.6

        result = analyze_social_sentiment(min_posts=20, caution_ratio=0.52, extreme_ratio=0.65)
        assert not result.in_extreme_zone
        assert result.social_score == -0.3
        assert result.dominant_side == "bearish"

    @patch('bot.calendar_events.fetch_reddit_hot_titles')
    def test_mixed_crowd_no_penalty(self, mock_fetch):
        mock_fetch.return_value = [
            "Bullish breakout setup",
            "Bearish rejection zone",
            "Market update today",
            "Risk management tips",
        ] * 10  # 40 posts balanced/mixed

        result = analyze_social_sentiment(min_posts=20, caution_ratio=0.52, extreme_ratio=0.60)
        assert not result.in_extreme_zone
        assert result.social_score == 0.0

    @patch('bot.calendar_events.fetch_reddit_hot_titles')
    def test_not_enough_posts_no_penalty(self, mock_fetch):
        mock_fetch.return_value = [
            "Bullish breakout moon long setup",
            "Mega bull rally incoming",
            "We only go up buy now",
            "Another ATH is coming",
        ] * 2  # 8 posts, below min_posts

        result = analyze_social_sentiment(min_posts=20, caution_ratio=0.52, extreme_ratio=0.60)
        assert not result.in_extreme_zone
        assert result.social_score == 0.0
