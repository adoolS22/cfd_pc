"""
Calendar Events - Macro + Sentiment
===================================
Fetches and caches high-impact macro events and market sentiment.
"""

import re
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup
from loguru import logger


# Federal Reserve FOMC calendar URL
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
# BLS CPI calendar URL
CPI_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
# BLS Employment Situation (NFP) release calendar
NFP_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"
# Federal Reserve speeches RSS (used to detect Powell speeches)
POWELL_SPEECHES_RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"
# Fear & Greed sentiment index (crowd sentiment proxy)
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
# Reddit social feed (crowd sentiment proxy from retail discussion)
REDDIT_HOT_URL = "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=100"

# Cache settings
CACHE_FILE = "fomc_cache.json"
CPI_CACHE_FILE = "cpi_cache.json"
NFP_CACHE_FILE = "nfp_cache.json"
POWELL_CACHE_FILE = "powell_speeches_cache.json"
SENTIMENT_CACHE_FILE = "sentiment_cache.json"
SOCIAL_SENTIMENT_CACHE_FILE = "social_sentiment_cache.json"
CACHE_TTL_DAYS = 7
SENTIMENT_CACHE_TTL_HOURS = 6
SOCIAL_SENTIMENT_CACHE_TTL_MINUTES = 30
SENTIMENT_API_TIMEOUT_SECONDS = 8
REDDIT_API_TIMEOUT_SECONDS = 6
SENTIMENT_FAILURE_BACKOFF_SECONDS = 300
SOCIAL_SENTIMENT_FAILURE_BACKOFF_SECONDS = 180

# Runtime backoff to prevent repeated blocking calls when external APIs are unstable.
_LAST_SENTIMENT_FETCH_FAILURE_AT: Optional[datetime] = None
_LAST_SOCIAL_FETCH_FAILURE_AT: Optional[datetime] = None


@dataclass
class FOMCMeeting:
    """Represents an FOMC meeting."""
    start_date: datetime
    end_date: datetime  # Decision day
    is_scheduled: bool
    has_projections: bool


@dataclass 
class FOMCAnalysis:
    """Result of FOMC timing analysis."""
    next_meeting_start: Optional[datetime]
    next_meeting_end: Optional[datetime]  # Decision day
    days_to_next: int
    in_high_vol_window: bool
    fomc_score: float  # -1 to 0 (risk filter)


@dataclass
class CPIRelease:
    """Represents a CPI release timestamp (UTC)."""
    release_datetime: datetime
    label: Optional[str] = None


@dataclass
class CPIAnalysis:
    """Result of CPI timing analysis."""
    next_release: Optional[datetime]
    hours_to_next: float
    in_high_vol_window: bool
    cpi_score: float  # -1 to 0 (risk filter)


@dataclass
class NFPRelease:
    """Represents an NFP release timestamp (UTC)."""
    release_datetime: datetime
    label: Optional[str] = None


@dataclass
class NFPAnalysis:
    """Result of NFP timing analysis."""
    next_release: Optional[datetime]
    hours_to_next: float
    in_high_vol_window: bool
    nfp_score: float  # -1 to 0 (risk filter)


@dataclass
class PowellSpeech:
    """Represents a Powell speech timestamp (UTC)."""
    event_datetime: datetime
    title: str
    source_url: Optional[str] = None


@dataclass
class PowellAnalysis:
    """Result of Powell speech timing analysis."""
    next_event: Optional[datetime]
    hours_to_next: float
    in_high_vol_window: bool
    powell_score: float  # -1 to 0 (risk filter)


@dataclass
class FOMCMinutesRelease:
    """Represents an FOMC minutes release timestamp (UTC)."""
    release_datetime: datetime
    meeting_end_date: datetime


@dataclass
class FOMCMinutesAnalysis:
    """Result of FOMC minutes timing analysis."""
    next_release: Optional[datetime]
    hours_to_next: float
    in_high_vol_window: bool
    minutes_score: float  # -1 to 0 (risk filter)


@dataclass
class SentimentSnapshot:
    """Latest market sentiment snapshot."""
    value: int
    classification: str
    updated_at: datetime
    source: str = "alternative.me/fng"


@dataclass
class SentimentAnalysis:
    """Result of sentiment risk analysis."""
    value: Optional[int]
    classification: str
    updated_at: Optional[datetime]
    in_extreme_zone: bool
    sentiment_score: float  # 0 to negative values


@dataclass
class SocialSentimentAnalysis:
    """Result of Reddit-based social sentiment analysis."""
    posts_scanned: int
    bullish_ratio: float
    bearish_ratio: float
    dominant_side: str
    in_extreme_zone: bool
    social_score: float  # 0 to negative values
    updated_at: Optional[datetime]
    source: str = "reddit:r/CryptoCurrency"


def _get_cache_path() -> Path:
    """Get path to cache file."""
    # Use script directory for cache
    return Path(__file__).parent.parent / CACHE_FILE


def _get_cpi_cache_path() -> Path:
    """Get path to CPI cache file."""
    return Path(__file__).parent.parent / CPI_CACHE_FILE


def _get_nfp_cache_path() -> Path:
    """Get path to NFP cache file."""
    return Path(__file__).parent.parent / NFP_CACHE_FILE


def _get_powell_cache_path() -> Path:
    """Get path to Powell speeches cache file."""
    return Path(__file__).parent.parent / POWELL_CACHE_FILE


def _get_sentiment_cache_path() -> Path:
    """Get path to sentiment cache file."""
    return Path(__file__).parent.parent / SENTIMENT_CACHE_FILE


def _get_social_sentiment_cache_path() -> Path:
    """Get path to social sentiment cache file."""
    return Path(__file__).parent.parent / SOCIAL_SENTIMENT_CACHE_FILE


def _load_cache() -> Optional[Dict]:
    """Load FOMC cache if valid."""
    cache_path = _get_cache_path()
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        
        # Check if cache is still valid
        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)
        
        age = datetime.now(timezone.utc) - cached_time
        
        if age.days < CACHE_TTL_DAYS:
            logger.debug(f"Using cached FOMC data ({age.days} days old)")
            return cache
        
        logger.debug("FOMC cache expired")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to load FOMC cache: {e}")
        return None


def _save_cache(meetings: List[Dict]) -> None:
    """Save FOMC data to cache."""
    cache_path = _get_cache_path()
    
    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'meetings': meetings
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        logger.debug(f"Saved FOMC cache with {len(meetings)} meetings")
        
    except Exception as e:
        logger.warning(f"Failed to save FOMC cache: {e}")


def _load_cpi_cache() -> Optional[Dict]:
    """Load CPI cache if valid."""
    cache_path = _get_cpi_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        if age.days < CACHE_TTL_DAYS:
            logger.debug(f"Using cached CPI data ({age.days} days old)")
            return cache

        logger.debug("CPI cache expired")
        return None

    except Exception as e:
        logger.warning(f"Failed to load CPI cache: {e}")
        return None


def _save_cpi_cache(releases: List[Dict]) -> None:
    """Save CPI releases to cache."""
    cache_path = _get_cpi_cache_path()

    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'releases': releases
        }
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)

        logger.debug(f"Saved CPI cache with {len(releases)} releases")

    except Exception as e:
        logger.warning(f"Failed to save CPI cache: {e}")


def _load_nfp_cache() -> Optional[Dict]:
    """Load NFP cache if valid."""
    cache_path = _get_nfp_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        if age.days < CACHE_TTL_DAYS:
            logger.debug(f"Using cached NFP data ({age.days} days old)")
            return cache

    except Exception as e:
        logger.warning(f"Failed to load NFP cache: {e}")

    return None


def _save_nfp_cache(releases: List[Dict]) -> None:
    """Save NFP releases to cache."""
    cache_path = _get_nfp_cache_path()
    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'releases': releases
        }
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.debug(f"Saved NFP cache with {len(releases)} releases")
    except Exception as e:
        logger.warning(f"Failed to save NFP cache: {e}")


def _load_powell_cache() -> Optional[Dict]:
    """Load Powell speeches cache if valid."""
    cache_path = _get_powell_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        if age.days < CACHE_TTL_DAYS:
            logger.debug(f"Using cached Powell speeches ({age.days} days old)")
            return cache

    except Exception as e:
        logger.warning(f"Failed to load Powell cache: {e}")

    return None


def _save_powell_cache(events: List[Dict]) -> None:
    """Save Powell speeches to cache."""
    cache_path = _get_powell_cache_path()
    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'events': events
        }
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.debug(f"Saved Powell speeches cache with {len(events)} items")
    except Exception as e:
        logger.warning(f"Failed to save Powell cache: {e}")


def _load_sentiment_cache(allow_stale: bool = False) -> Optional[Dict]:
    """Load sentiment cache if still fresh (or stale when explicitly allowed)."""
    cache_path = _get_sentiment_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        if age < timedelta(hours=SENTIMENT_CACHE_TTL_HOURS):
            logger.debug(f"Using cached sentiment data ({age.total_seconds() / 3600:.1f}h old)")
            return cache
        if allow_stale:
            logger.debug(
                f"Using stale sentiment cache ({age.total_seconds() / 3600:.1f}h old) due to API/backoff"
            )
            return cache

    except Exception as e:
        logger.warning(f"Failed to load sentiment cache: {e}")

    return None


def _save_sentiment_cache(snapshot: Dict) -> None:
    """Save sentiment snapshot to cache."""
    cache_path = _get_sentiment_cache_path()
    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'snapshot': snapshot
        }
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.debug("Saved sentiment cache")
    except Exception as e:
        logger.warning(f"Failed to save sentiment cache: {e}")


def _load_social_sentiment_cache(allow_stale: bool = False) -> Optional[Dict]:
    """Load social sentiment cache if still fresh (or stale when explicitly allowed)."""
    cache_path = _get_social_sentiment_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - cached_time
        if age < timedelta(minutes=SOCIAL_SENTIMENT_CACHE_TTL_MINUTES):
            logger.debug(f"Using cached social sentiment ({age.total_seconds() / 60:.1f}m old)")
            return cache
        if allow_stale:
            logger.debug(
                f"Using stale social sentiment cache ({age.total_seconds() / 60:.1f}m old) due to API/backoff"
            )
            return cache

    except Exception as e:
        logger.warning(f"Failed to load social sentiment cache: {e}")

    return None


def _snapshot_from_sentiment_cache(cache: Optional[Dict]) -> Optional["SentimentSnapshot"]:
    """Parse SentimentSnapshot from cache dict."""
    if not cache:
        return None
    item = cache.get('snapshot', {})
    try:
        updated_at = datetime.fromisoformat(item['updated_at'])
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return SentimentSnapshot(
            value=int(item['value']),
            classification=str(item.get('classification', 'Unknown')),
            updated_at=updated_at,
            source=str(item.get('source', 'alternative.me/fng'))
        )
    except Exception:
        return None


def _titles_from_social_cache(cache: Optional[Dict]) -> List[str]:
    """Extract normalized title list from cache payload."""
    if not cache:
        return []
    titles = cache.get('titles', [])
    if not isinstance(titles, list):
        return []
    return [str(t).strip() for t in titles if str(t).strip()]


def _save_social_sentiment_cache(titles: List[str], updated_at: datetime) -> None:
    """Save Reddit titles used for social sentiment analysis."""
    cache_path = _get_social_sentiment_cache_path()
    try:
        cache = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': updated_at.isoformat(),
            'titles': titles
        }
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.debug(f"Saved social sentiment cache with {len(titles)} titles")
    except Exception as e:
        logger.warning(f"Failed to save social sentiment cache: {e}")


def fetch_fomc_dates() -> List[FOMCMeeting]:
    """
    Fetch FOMC meeting dates from Federal Reserve website.
    
    Uses local caching to avoid repeated requests.
    
    Returns:
        List of FOMCMeeting objects
    """
    # Try cache first
    cache = _load_cache()
    if cache:
        meetings = []
        for m in cache.get('meetings', []):
            try:
                start = datetime.fromisoformat(m['start_date'])
                end = datetime.fromisoformat(m['end_date'])
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                meetings.append(FOMCMeeting(
                    start_date=start,
                    end_date=end,
                    is_scheduled=m.get('is_scheduled', True),
                    has_projections=m.get('has_projections', False)
                ))
            except Exception:
                continue
        
        if meetings:
            return meetings
    
    # Fetch from Federal Reserve
    meetings = _fetch_from_fed()
    
    # Cache the results
    if meetings:
        cache_data = [
            {
                'start_date': m.start_date.isoformat(),
                'end_date': m.end_date.isoformat(),
                'is_scheduled': m.is_scheduled,
                'has_projections': m.has_projections
            }
            for m in meetings
        ]
        _save_cache(cache_data)
    
    return meetings


def _fetch_from_fed() -> List[FOMCMeeting]:
    """Fetch FOMC dates from Federal Reserve website."""
    meetings = []
    
    try:
        logger.info(f"Fetching FOMC calendar from {FOMC_URL}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'
        }
        
        response = requests.get(FOMC_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find meeting date panels
        # The Fed website uses panels with class "fomc-meeting"
        panels = soup.find_all('div', class_='fomc-meeting')
        
        if not panels:
            # Alternative: look for date patterns in the page
            logger.debug("No fomc-meeting panels found, trying text parsing")
            meetings = _parse_dates_from_text(response.text)
        else:
            for panel in panels:
                try:
                    meeting = _parse_meeting_panel(panel)
                    if meeting:
                        meetings.append(meeting)
                except Exception as e:
                    logger.debug(f"Failed to parse meeting panel: {e}")
                    continue
        
        logger.info(f"Found {len(meetings)} FOMC meetings")
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch FOMC calendar: {e}")
    except Exception as e:
        logger.error(f"Error parsing FOMC calendar: {e}")
    
    return meetings


def _parse_meeting_panel(panel) -> Optional[FOMCMeeting]:
    """Parse a meeting panel from the Fed website."""
    # Try to find date text
    date_text = panel.get_text()
    
    # Common patterns: "January 28-29" or "March 18-19, 2025"
    # Also handles single-day meetings: "January 29"
    
    current_year = datetime.now().year
    
    # Pattern for date ranges
    range_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})-(\d{1,2})(?:,?\s*(\d{4}))?'
    
    match = re.search(range_pattern, date_text, re.IGNORECASE)
    
    if match:
        month_str = match.group(1)
        start_day = int(match.group(2))
        end_day = int(match.group(3))
        year = int(match.group(4)) if match.group(4) else current_year
        
        month = _month_to_num(month_str)
        
        start_date = datetime(year, month, start_day, tzinfo=timezone.utc)
        end_date = datetime(year, month, end_day, tzinfo=timezone.utc)
        
        # Check for projections indicator
        has_proj = 'projection' in date_text.lower() or '*' in date_text
        
        return FOMCMeeting(
            start_date=start_date,
            end_date=end_date,
            is_scheduled=True,
            has_projections=has_proj
        )
    
    return None


def _parse_dates_from_text(html_text: str) -> List[FOMCMeeting]:
    """Fallback: parse FOMC dates from raw HTML text."""
    meetings = []
    current_year = datetime.now().year
    
    # Pattern for date ranges with year
    pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[-–](\d{1,2})(?:[,\s]*(\d{4}))?'
    
    for match in re.finditer(pattern, html_text, re.IGNORECASE):
        try:
            month_str = match.group(1)
            start_day = int(match.group(2))
            end_day = int(match.group(3))
            year = int(match.group(4)) if match.group(4) else current_year
            
            # Handle dates that might be next year
            month = _month_to_num(month_str)
            if not match.group(4) and month < datetime.now().month:
                year = current_year + 1
            
            start_date = datetime(year, month, start_day, tzinfo=timezone.utc)
            end_date = datetime(year, month, end_day, tzinfo=timezone.utc)
            
            meetings.append(FOMCMeeting(
                start_date=start_date,
                end_date=end_date,
                is_scheduled=True,
                has_projections=False
            ))
        except Exception:
            continue
    
    # Remove duplicates by end_date
    seen = set()
    unique = []
    for m in meetings:
        key = m.end_date.date()
        if key not in seen:
            seen.add(key)
            unique.append(m)
    
    return unique


def _month_to_num(month: str) -> int:
    """Convert month name to number."""
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    return months.get(month.lower(), 1)


def get_next_fomc_meeting(dt: Optional[datetime] = None) -> Optional[FOMCMeeting]:
    """
    Get the next scheduled FOMC meeting.
    
    Args:
        dt: Reference datetime (default: now)
        
    Returns:
        Next FOMCMeeting or None
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    meetings = fetch_fomc_dates()
    
    for meeting in sorted(meetings, key=lambda m: m.start_date):
        # Include ongoing meetings; they are still relevant for volatility analysis.
        if meeting.end_date >= dt:
            return meeting
    
    return None


def _get_relevant_fomc_meeting(
    meetings: List[FOMCMeeting],
    dt: datetime,
    high_vol_days: int
) -> Optional[FOMCMeeting]:
    """
    Select the meeting most relevant for current volatility context.

    Priority:
    1. A meeting whose high-volatility window currently contains ``dt``.
    2. Otherwise, the nearest ongoing/upcoming meeting.
    """
    for meeting in meetings:
        window_start = meeting.start_date - timedelta(days=high_vol_days)
        window_end = meeting.end_date + timedelta(days=high_vol_days)
        if window_start <= dt <= window_end:
            return meeting

    for meeting in meetings:
        if meeting.end_date >= dt:
            return meeting

    return None


def analyze_fomc(
    high_vol_days: int = 2,
    dt: Optional[datetime] = None
) -> FOMCAnalysis:
    """
    Analyze FOMC timing impact.
    
    The days surrounding FOMC decisions typically have increased volatility.
    This analysis provides a risk filter score.
    
    Args:
        high_vol_days: Days before/after meeting to consider high volatility
        dt: Reference datetime (default: now)
        
    Returns:
        FOMCAnalysis with timing info and score
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    meetings = sorted(fetch_fomc_dates(), key=lambda m: m.start_date)
    next_meeting = _get_relevant_fomc_meeting(meetings, dt, high_vol_days)
    
    if not next_meeting:
        return FOMCAnalysis(
            next_meeting_start=None,
            next_meeting_end=None,
            days_to_next=999,
            in_high_vol_window=False,
            fomc_score=0
        )
    
    # Calculate days to decision day (end of meeting)
    days_to_decision = (next_meeting.end_date.date() - dt.date()).days
    
    # Check if in high volatility window
    # Window is high_vol_days before meeting start and after meeting end
    window_start = next_meeting.start_date - timedelta(days=high_vol_days)
    window_end = next_meeting.end_date + timedelta(days=high_vol_days)
    
    in_window = window_start <= dt <= window_end
    
    # Calculate score (negative = risk)
    if in_window:
        # Closer to decision = higher risk. Post-decision window remains risky but lower.
        if 0 <= days_to_decision <= 1:
            score = -1.0
        elif days_to_decision < 0:
            score = -0.25
        elif days_to_decision <= high_vol_days:
            score = -0.5
        else:
            score = -0.25
    else:
        score = 0.0  # No FOMC risk
    
    return FOMCAnalysis(
        next_meeting_start=next_meeting.start_date,
        next_meeting_end=next_meeting.end_date,
        days_to_next=days_to_decision,
        in_high_vol_window=in_window,
        fomc_score=score
    )


def format_fomc_analysis(analysis: FOMCAnalysis) -> str:
    """Format FOMC analysis for display."""
    if analysis.next_meeting_end is None:
        return "FOMC: No upcoming meetings found"
    
    date_str = analysis.next_meeting_end.strftime('%b %d')
    window_status = "⚠️ HIGH VOL WINDOW" if analysis.in_high_vol_window else "Normal"
    
    return (f"Next Decision: {date_str} ({analysis.days_to_next} days) | "
            f"{window_status} | Score: {analysis.fomc_score:.1f}")


def _to_cpi_release_datetime_utc(year: int, month: int, day: int) -> datetime:
    """Convert CPI release date (8:30 AM New York time) to UTC."""
    ny_tz = ZoneInfo("America/New_York")
    local = datetime(year, month, day, 8, 30, tzinfo=ny_tz)
    return local.astimezone(timezone.utc)


def _parse_cpi_dates_from_text(text: str) -> List[CPIRelease]:
    """
    Parse CPI release dates from page text.

    Example date token expected: "March 12, 2026".
    """
    releases: List[CPIRelease] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)

    pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})'

    seen = set()
    for match in re.finditer(pattern, text, re.IGNORECASE):
        try:
            month = _month_to_num(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            release_dt = _to_cpi_release_datetime_utc(year, month, day)

            if release_dt < cutoff:
                continue

            date_key = release_dt.date().isoformat()
            if date_key in seen:
                continue
            seen.add(date_key)

            releases.append(CPIRelease(
                release_datetime=release_dt,
                label=f"CPI {release_dt.strftime('%b %Y')}"
            ))
        except Exception:
            continue

    releases.sort(key=lambda r: r.release_datetime)
    return releases


def _fetch_cpi_from_bls() -> List[CPIRelease]:
    """Fetch CPI release dates from BLS schedule page."""
    try:
        logger.info(f"Fetching CPI schedule from {CPI_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'}
        response = requests.get(CPI_URL, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(" ", strip=True)
        releases = _parse_cpi_dates_from_text(text)

        logger.info(f"Found {len(releases)} CPI releases")
        return releases

    except requests.RequestException as e:
        logger.error(f"Failed to fetch CPI schedule: {e}")
    except Exception as e:
        logger.error(f"Error parsing CPI schedule: {e}")

    return []


def fetch_cpi_dates() -> List[CPIRelease]:
    """Fetch CPI release dates with cache."""
    cache = _load_cpi_cache()
    if cache:
        releases: List[CPIRelease] = []
        for item in cache.get('releases', []):
            try:
                dt = datetime.fromisoformat(item['release_datetime'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                releases.append(CPIRelease(
                    release_datetime=dt,
                    label=item.get('label')
                ))
            except Exception:
                continue

        if releases:
            releases.sort(key=lambda r: r.release_datetime)
            return releases

    releases = _fetch_cpi_from_bls()
    if releases:
        cache_data = [
            {
                'release_datetime': r.release_datetime.isoformat(),
                'label': r.label
            }
            for r in releases
        ]
        _save_cpi_cache(cache_data)

    return releases


def get_next_cpi_release(dt: Optional[datetime] = None) -> Optional[CPIRelease]:
    """Get nearest upcoming CPI release."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    releases = fetch_cpi_dates()
    for release in sorted(releases, key=lambda r: r.release_datetime):
        if release.release_datetime >= dt:
            return release
    return None


def _get_relevant_cpi_release(
    releases: List[CPIRelease],
    dt: datetime,
    high_vol_days: int
) -> Optional[CPIRelease]:
    """Find CPI release relevant to current high-volatility window."""
    for release in releases:
        window_start = release.release_datetime - timedelta(days=high_vol_days)
        window_end = release.release_datetime + timedelta(days=high_vol_days)
        if window_start <= dt <= window_end:
            return release

    for release in releases:
        if release.release_datetime >= dt:
            return release

    return None


def analyze_cpi(
    high_vol_days: int = 1,
    dt: Optional[datetime] = None
) -> CPIAnalysis:
    """
    Analyze CPI timing impact.

    CPI releases are high-impact macro prints for risk assets and crypto.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    releases = sorted(fetch_cpi_dates(), key=lambda r: r.release_datetime)
    next_release = _get_relevant_cpi_release(releases, dt, high_vol_days)

    if not next_release:
        return CPIAnalysis(
            next_release=None,
            hours_to_next=999.0,
            in_high_vol_window=False,
            cpi_score=0.0
        )

    hours_to_release = (next_release.release_datetime - dt).total_seconds() / 3600
    window_start = next_release.release_datetime - timedelta(days=high_vol_days)
    window_end = next_release.release_datetime + timedelta(days=high_vol_days)
    in_window = window_start <= dt <= window_end

    if in_window:
        abs_hours = abs(hours_to_release)
        if abs_hours <= 6:
            score = -1.0
        elif abs_hours <= 24:
            score = -0.7
        else:
            score = -0.4
    else:
        score = 0.0

    return CPIAnalysis(
        next_release=next_release.release_datetime,
        hours_to_next=hours_to_release,
        in_high_vol_window=in_window,
        cpi_score=score
    )


def format_cpi_analysis(analysis: CPIAnalysis) -> str:
    """Format CPI analysis for display."""
    if analysis.next_release is None:
        return "CPI: No upcoming releases found"

    date_str = analysis.next_release.strftime('%b %d %H:%M UTC')
    hours = f"{analysis.hours_to_next:.1f}h"
    window_status = "⚠️ HIGH VOL WINDOW" if analysis.in_high_vol_window else "Normal"
    return f"Next CPI: {date_str} ({hours}) | {window_status} | Score: {analysis.cpi_score:.1f}"


def _to_macro_release_datetime_utc(year: int, month: int, day: int, hour: int = 8, minute: int = 30) -> datetime:
    """Convert US macro release time in New York timezone to UTC."""
    ny_tz = ZoneInfo("America/New_York")
    local = datetime(year, month, day, hour, minute, tzinfo=ny_tz)
    return local.astimezone(timezone.utc)


def _parse_named_dates_with_year(text: str, label_prefix: str, keep_since_days: int = 365) -> List[Dict]:
    """Parse dates like 'March 12, 2026' from text."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=keep_since_days)
    pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})'

    out: List[Dict] = []
    seen = set()
    for match in re.finditer(pattern, text, re.IGNORECASE):
        try:
            month = _month_to_num(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            dt = _to_macro_release_datetime_utc(year, month, day)
            if dt < cutoff:
                continue
            k = dt.date().isoformat()
            if k in seen:
                continue
            seen.add(k)
            out.append({
                'datetime': dt,
                'label': f"{label_prefix} {dt.strftime('%b %Y')}"
            })
        except Exception:
            continue

    out.sort(key=lambda x: x['datetime'])
    return out


def _fetch_nfp_from_bls() -> List[NFPRelease]:
    """Fetch NFP release dates from BLS employment situation schedule."""
    try:
        logger.info(f"Fetching NFP schedule from {NFP_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'}
        response = requests.get(NFP_URL, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(" ", strip=True)
        parsed = _parse_named_dates_with_year(text, label_prefix="NFP", keep_since_days=365)

        releases = [NFPRelease(release_datetime=p['datetime'], label=p['label']) for p in parsed]
        logger.info(f"Found {len(releases)} NFP releases")
        return releases

    except requests.RequestException as e:
        logger.error(f"Failed to fetch NFP schedule: {e}")
    except Exception as e:
        logger.error(f"Error parsing NFP schedule: {e}")

    return []


def fetch_nfp_dates() -> List[NFPRelease]:
    """Fetch NFP release dates with cache."""
    cache = _load_nfp_cache()
    if cache:
        releases: List[NFPRelease] = []
        for item in cache.get('releases', []):
            try:
                dt = datetime.fromisoformat(item['release_datetime'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                releases.append(NFPRelease(release_datetime=dt, label=item.get('label')))
            except Exception:
                continue
        if releases:
            releases.sort(key=lambda r: r.release_datetime)
            return releases

    releases = _fetch_nfp_from_bls()
    if releases:
        cache_data = [
            {
                'release_datetime': r.release_datetime.isoformat(),
                'label': r.label
            }
            for r in releases
        ]
        _save_nfp_cache(cache_data)
    return releases


def _get_relevant_nfp_release(releases: List[NFPRelease], dt: datetime, high_vol_days: int) -> Optional[NFPRelease]:
    """Find NFP release relevant to the current high-volatility window."""
    for release in releases:
        window_start = release.release_datetime - timedelta(days=high_vol_days)
        window_end = release.release_datetime + timedelta(days=high_vol_days)
        if window_start <= dt <= window_end:
            return release
    for release in releases:
        if release.release_datetime >= dt:
            return release
    return None


def analyze_nfp(high_vol_days: int = 1, dt: Optional[datetime] = None) -> NFPAnalysis:
    """Analyze NFP timing impact."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    releases = sorted(fetch_nfp_dates(), key=lambda r: r.release_datetime)
    next_release = _get_relevant_nfp_release(releases, dt, high_vol_days)
    if not next_release:
        return NFPAnalysis(
            next_release=None,
            hours_to_next=999.0,
            in_high_vol_window=False,
            nfp_score=0.0
        )

    hours_to_release = (next_release.release_datetime - dt).total_seconds() / 3600
    window_start = next_release.release_datetime - timedelta(days=high_vol_days)
    window_end = next_release.release_datetime + timedelta(days=high_vol_days)
    in_window = window_start <= dt <= window_end

    if in_window:
        abs_hours = abs(hours_to_release)
        if abs_hours <= 6:
            score = -1.0
        elif abs_hours <= 24:
            score = -0.7
        else:
            score = -0.4
    else:
        score = 0.0

    return NFPAnalysis(
        next_release=next_release.release_datetime,
        hours_to_next=hours_to_release,
        in_high_vol_window=in_window,
        nfp_score=score
    )


def format_nfp_analysis(analysis: NFPAnalysis) -> str:
    """Format NFP analysis for display."""
    if analysis.next_release is None:
        return "NFP: No upcoming releases found"
    date_str = analysis.next_release.strftime('%b %d %H:%M UTC')
    window_status = "⚠️ HIGH VOL WINDOW" if analysis.in_high_vol_window else "Normal"
    return f"Next NFP: {date_str} ({analysis.hours_to_next:.1f}h) | {window_status} | Score: {analysis.nfp_score:.1f}"


def _fetch_powell_speeches_from_rss() -> List[PowellSpeech]:
    """Fetch Powell speech timestamps from Federal Reserve RSS feed."""
    try:
        logger.info(f"Fetching Powell speeches from {POWELL_SPEECHES_RSS_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'}
        response = requests.get(POWELL_SPEECHES_RSS_URL, headers=headers, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        events: List[PowellSpeech] = []
        seen = set()

        for item in root.findall('.//item'):
            title = (item.findtext('title') or '').strip()
            description = (item.findtext('description') or '').strip()
            link = (item.findtext('link') or '').strip()
            pub_date = (item.findtext('pubDate') or '').strip()

            text_blob = f"{title} {description}".lower()
            if 'powell' not in text_blob:
                continue
            if not pub_date:
                continue

            try:
                dt = parsedate_to_datetime(pub_date)
            except Exception:
                continue

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)

            key = (dt.isoformat(), title)
            if key in seen:
                continue
            seen.add(key)

            events.append(PowellSpeech(
                event_datetime=dt,
                title=title or "Powell speech",
                source_url=link or None
            ))

        events.sort(key=lambda e: e.event_datetime)
        logger.info(f"Found {len(events)} Powell speech entries")
        return events

    except requests.RequestException as e:
        logger.error(f"Failed to fetch Powell speeches: {e}")
    except Exception as e:
        logger.error(f"Error parsing Powell speeches: {e}")

    return []


def fetch_powell_speeches() -> List[PowellSpeech]:
    """Fetch Powell speeches with cache."""
    cache = _load_powell_cache()
    if cache:
        events: List[PowellSpeech] = []
        for item in cache.get('events', []):
            try:
                dt = datetime.fromisoformat(item['event_datetime'])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                events.append(PowellSpeech(
                    event_datetime=dt,
                    title=item.get('title', 'Powell speech'),
                    source_url=item.get('source_url')
                ))
            except Exception:
                continue
        if events:
            events.sort(key=lambda e: e.event_datetime)
            return events

    events = _fetch_powell_speeches_from_rss()
    if events:
        cache_data = [
            {
                'event_datetime': e.event_datetime.isoformat(),
                'title': e.title,
                'source_url': e.source_url
            }
            for e in events
        ]
        _save_powell_cache(cache_data)
    return events


def _get_relevant_powell_event(events: List[PowellSpeech], dt: datetime, high_vol_hours: int) -> Optional[PowellSpeech]:
    """Find Powell event relevant to the current high-volatility window."""
    for event in events:
        window_start = event.event_datetime - timedelta(hours=high_vol_hours)
        window_end = event.event_datetime + timedelta(hours=high_vol_hours)
        if window_start <= dt <= window_end:
            return event
    for event in events:
        if event.event_datetime >= dt:
            return event
    return None


def analyze_powell_speeches(high_vol_hours: int = 24, dt: Optional[datetime] = None) -> PowellAnalysis:
    """Analyze Powell speech timing impact."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    events = sorted(fetch_powell_speeches(), key=lambda e: e.event_datetime)
    next_event = _get_relevant_powell_event(events, dt, high_vol_hours)
    if not next_event:
        return PowellAnalysis(
            next_event=None,
            hours_to_next=999.0,
            in_high_vol_window=False,
            powell_score=0.0
        )

    hours_to_event = (next_event.event_datetime - dt).total_seconds() / 3600
    window_start = next_event.event_datetime - timedelta(hours=high_vol_hours)
    window_end = next_event.event_datetime + timedelta(hours=high_vol_hours)
    in_window = window_start <= dt <= window_end

    if in_window:
        abs_hours = abs(hours_to_event)
        if abs_hours <= 3:
            score = -1.0
        elif abs_hours <= 12:
            score = -0.7
        else:
            score = -0.4
    else:
        score = 0.0

    return PowellAnalysis(
        next_event=next_event.event_datetime,
        hours_to_next=hours_to_event,
        in_high_vol_window=in_window,
        powell_score=score
    )


def format_powell_analysis(analysis: PowellAnalysis) -> str:
    """Format Powell speech analysis for display."""
    if analysis.next_event is None:
        return "Powell: No upcoming speeches found"
    date_str = analysis.next_event.strftime('%b %d %H:%M UTC')
    window_status = "⚠️ HIGH VOL WINDOW" if analysis.in_high_vol_window else "Normal"
    return f"Next Powell: {date_str} ({analysis.hours_to_next:.1f}h) | {window_status} | Score: {analysis.powell_score:.1f}"


def fetch_fomc_minutes_dates() -> List[FOMCMinutesRelease]:
    """
    Build estimated FOMC minutes release timestamps from FOMC meetings.

    Typical release is around 3 weeks after the meeting end day at 14:00 New York time.
    """
    meetings = fetch_fomc_dates()
    releases: List[FOMCMinutesRelease] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)

    seen = set()
    for m in meetings:
        try:
            release_day = (m.end_date + timedelta(days=21)).date()
            release_dt = _to_macro_release_datetime_utc(
                release_day.year, release_day.month, release_day.day, hour=14, minute=0
            )
            if release_dt < cutoff:
                continue
            key = release_dt.date().isoformat()
            if key in seen:
                continue
            seen.add(key)
            releases.append(FOMCMinutesRelease(
                release_datetime=release_dt,
                meeting_end_date=m.end_date
            ))
        except Exception:
            continue

    releases.sort(key=lambda r: r.release_datetime)
    return releases


def _get_relevant_fomc_minutes_release(
    releases: List[FOMCMinutesRelease],
    dt: datetime,
    high_vol_days: int
) -> Optional[FOMCMinutesRelease]:
    """Find minutes release relevant to current high-volatility window."""
    for release in releases:
        window_start = release.release_datetime - timedelta(days=high_vol_days)
        window_end = release.release_datetime + timedelta(days=high_vol_days)
        if window_start <= dt <= window_end:
            return release
    for release in releases:
        if release.release_datetime >= dt:
            return release
    return None


def analyze_fomc_minutes(high_vol_days: int = 1, dt: Optional[datetime] = None) -> FOMCMinutesAnalysis:
    """Analyze FOMC minutes timing impact."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    releases = fetch_fomc_minutes_dates()
    next_release = _get_relevant_fomc_minutes_release(releases, dt, high_vol_days)
    if not next_release:
        return FOMCMinutesAnalysis(
            next_release=None,
            hours_to_next=999.0,
            in_high_vol_window=False,
            minutes_score=0.0
        )

    hours_to_release = (next_release.release_datetime - dt).total_seconds() / 3600
    window_start = next_release.release_datetime - timedelta(days=high_vol_days)
    window_end = next_release.release_datetime + timedelta(days=high_vol_days)
    in_window = window_start <= dt <= window_end

    if in_window:
        abs_hours = abs(hours_to_release)
        if abs_hours <= 6:
            score = -1.0
        elif abs_hours <= 24:
            score = -0.7
        else:
            score = -0.4
    else:
        score = 0.0

    return FOMCMinutesAnalysis(
        next_release=next_release.release_datetime,
        hours_to_next=hours_to_release,
        in_high_vol_window=in_window,
        minutes_score=score
    )


def format_fomc_minutes_analysis(analysis: FOMCMinutesAnalysis) -> str:
    """Format FOMC minutes analysis for display."""
    if analysis.next_release is None:
        return "FOMC Minutes: No upcoming releases found"
    date_str = analysis.next_release.strftime('%b %d %H:%M UTC')
    window_status = "⚠️ HIGH VOL WINDOW" if analysis.in_high_vol_window else "Normal"
    return f"Next Minutes: {date_str} ({analysis.hours_to_next:.1f}h) | {window_status} | Score: {analysis.minutes_score:.1f}"


def fetch_market_sentiment() -> Optional[SentimentSnapshot]:
    """Fetch latest Fear & Greed market sentiment snapshot."""
    global _LAST_SENTIMENT_FETCH_FAILURE_AT

    cache = _load_sentiment_cache()
    cached_snapshot = _snapshot_from_sentiment_cache(cache)
    if cached_snapshot:
        return cached_snapshot

    stale_snapshot = _snapshot_from_sentiment_cache(_load_sentiment_cache(allow_stale=True))
    now = datetime.now(timezone.utc)
    if _LAST_SENTIMENT_FETCH_FAILURE_AT:
        delta = (now - _LAST_SENTIMENT_FETCH_FAILURE_AT).total_seconds()
        if delta < SENTIMENT_FAILURE_BACKOFF_SECONDS:
            if stale_snapshot:
                return stale_snapshot
            return None

    try:
        logger.info(f"Fetching market sentiment from {FEAR_GREED_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'}
        response = requests.get(FEAR_GREED_URL, headers=headers, timeout=SENTIMENT_API_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

        rows = payload.get('data', [])
        if not rows:
            logger.warning("Sentiment API returned no rows")
            return None

        row = rows[0]
        value = int(row.get('value'))
        classification = str(row.get('value_classification', 'Unknown'))

        ts_raw = row.get('timestamp')
        if ts_raw is not None:
            try:
                updated_at = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
            except Exception:
                updated_at = datetime.now(timezone.utc)
        else:
            updated_at = datetime.now(timezone.utc)

        snapshot = SentimentSnapshot(
            value=value,
            classification=classification,
            updated_at=updated_at
        )
        _save_sentiment_cache({
            'value': snapshot.value,
            'classification': snapshot.classification,
            'updated_at': snapshot.updated_at.isoformat(),
            'source': snapshot.source
        })
        _LAST_SENTIMENT_FETCH_FAILURE_AT = None
        return snapshot

    except requests.RequestException as e:
        _LAST_SENTIMENT_FETCH_FAILURE_AT = datetime.now(timezone.utc)
        logger.error(f"Failed to fetch market sentiment: {e}")
    except Exception as e:
        _LAST_SENTIMENT_FETCH_FAILURE_AT = datetime.now(timezone.utc)
        logger.error(f"Error parsing market sentiment: {e}")

    if stale_snapshot:
        logger.warning("Using stale sentiment cache after fetch failure")
        return stale_snapshot

    return None


def analyze_market_sentiment(
    extreme_fear: int = 25,
    extreme_greed: int = 75
) -> SentimentAnalysis:
    """
    Analyze crowd sentiment risk using Fear & Greed index.

    Extreme sentiment tends to increase reversal risk and noise.
    """
    snapshot = fetch_market_sentiment()
    if not snapshot:
        return SentimentAnalysis(
            value=None,
            classification="unknown",
            updated_at=None,
            in_extreme_zone=False,
            sentiment_score=0.0
        )

    low = min(extreme_fear, extreme_greed)
    high = max(extreme_fear, extreme_greed)
    value = snapshot.value

    in_extreme = value <= low or value >= high
    in_caution = value <= (low + 10) or value >= (high - 10)

    if in_extreme:
        score = -0.8
    elif in_caution:
        score = -0.4
    else:
        score = 0.0

    return SentimentAnalysis(
        value=value,
        classification=snapshot.classification,
        updated_at=snapshot.updated_at,
        in_extreme_zone=in_extreme,
        sentiment_score=score
    )


def format_market_sentiment(analysis: SentimentAnalysis) -> str:
    """Format sentiment analysis for display."""
    if analysis.value is None:
        return "Sentiment: unavailable"
    status = "⚠️ EXTREME" if analysis.in_extreme_zone else "Normal"
    return f"Sentiment: {analysis.classification} ({analysis.value}) | {status} | Score: {analysis.sentiment_score:.1f}"


def fetch_reddit_hot_titles() -> List[str]:
    """Fetch latest discussion titles from r/CryptoCurrency."""
    global _LAST_SOCIAL_FETCH_FAILURE_AT

    cache = _load_social_sentiment_cache()
    cached_titles = _titles_from_social_cache(cache)
    if cached_titles:
        return cached_titles

    stale_titles = _titles_from_social_cache(_load_social_sentiment_cache(allow_stale=True))
    now = datetime.now(timezone.utc)
    if _LAST_SOCIAL_FETCH_FAILURE_AT:
        delta = (now - _LAST_SOCIAL_FETCH_FAILURE_AT).total_seconds()
        if delta < SOCIAL_SENTIMENT_FAILURE_BACKOFF_SECONDS:
            if stale_titles:
                return stale_titles
            return []

    try:
        logger.info(f"Fetching social sentiment titles from {REDDIT_HOT_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CryptoSignalBot/1.0)'}
        response = requests.get(REDDIT_HOT_URL, headers=headers, timeout=REDDIT_API_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

        children = payload.get('data', {}).get('children', [])
        titles: List[str] = []
        for row in children:
            data = row.get('data', {})
            title = str(data.get('title', '')).strip()
            if title:
                titles.append(title)

        if titles:
            _save_social_sentiment_cache(titles=titles, updated_at=datetime.now(timezone.utc))
            _LAST_SOCIAL_FETCH_FAILURE_AT = None
            return titles

        if stale_titles:
            logger.warning("Reddit returned empty titles; using stale social sentiment cache")
            return stale_titles
        return titles

    except requests.RequestException as e:
        _LAST_SOCIAL_FETCH_FAILURE_AT = datetime.now(timezone.utc)
        logger.error(f"Failed to fetch Reddit titles: {e}")
    except Exception as e:
        _LAST_SOCIAL_FETCH_FAILURE_AT = datetime.now(timezone.utc)
        logger.error(f"Error parsing Reddit titles: {e}")

    if stale_titles:
        logger.warning("Using stale social sentiment cache after Reddit fetch failure")
        return stale_titles

    return []


def analyze_social_sentiment(
    min_posts: int = 20,
    caution_ratio: float = 0.52,
    extreme_ratio: float = 0.60
) -> SocialSentimentAnalysis:
    """
    Analyze social sentiment from Reddit headlines.

    Uses contrarian logic:
    - one-sided crowd sentiment increases risk of fakeouts/reversals.
    """
    titles = fetch_reddit_hot_titles()
    total_posts = len(titles)

    if not titles:
        return SocialSentimentAnalysis(
            posts_scanned=0,
            bullish_ratio=0.0,
            bearish_ratio=0.0,
            dominant_side="unknown",
            in_extreme_zone=False,
            social_score=0.0,
            updated_at=None
        )

    bullish_keywords = (
        "bull", "bullish", "long", "buy", "breakout", "moon", "pump",
        "rally", "uptrend", "all-time high", "ath", "accumulate"
    )
    bearish_keywords = (
        "bear", "bearish", "short", "sell", "breakdown", "dump", "crash",
        "downtrend", "downside", "rejection", "capitulation", "panic", "liquidation"
    )

    bullish_posts = 0
    bearish_posts = 0

    for title in titles:
        text = title.lower()
        bull_hits = sum(1 for kw in bullish_keywords if kw in text)
        bear_hits = sum(1 for kw in bearish_keywords if kw in text)

        if bull_hits > bear_hits and bull_hits > 0:
            bullish_posts += 1
        elif bear_hits > bull_hits and bear_hits > 0:
            bearish_posts += 1

    bullish_ratio = bullish_posts / max(1, total_posts)
    bearish_ratio = bearish_posts / max(1, total_posts)
    dominant_side = "bullish" if bullish_ratio >= bearish_ratio else "bearish"
    dominant_ratio = max(bullish_ratio, bearish_ratio)

    has_enough_posts = total_posts >= max(1, min_posts)
    in_extreme = has_enough_posts and dominant_ratio >= extreme_ratio
    in_caution = has_enough_posts and dominant_ratio >= caution_ratio

    if in_extreme:
        social_score = -0.6
    elif in_caution:
        social_score = -0.3
    else:
        social_score = 0.0

    return SocialSentimentAnalysis(
        posts_scanned=total_posts,
        bullish_ratio=bullish_ratio,
        bearish_ratio=bearish_ratio,
        dominant_side=dominant_side if abs(bullish_ratio - bearish_ratio) >= 0.1 else "mixed",
        in_extreme_zone=in_extreme,
        social_score=social_score,
        updated_at=datetime.now(timezone.utc)
    )


def format_social_sentiment(analysis: SocialSentimentAnalysis) -> str:
    """Format Reddit social sentiment analysis for display."""
    if analysis.posts_scanned == 0:
        return "Social sentiment: unavailable"
    status = "⚠️ EXTREME" if analysis.in_extreme_zone else "Normal"
    return (
        f"Social: {analysis.dominant_side} | "
        f"Bull {analysis.bullish_ratio * 100:.0f}% / Bear {analysis.bearish_ratio * 100:.0f}% | "
        f"Posts: {analysis.posts_scanned} | {status} | Score: {analysis.social_score:.1f}"
    )
