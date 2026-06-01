"""Shared helpers for TVmaze-backed tools (public API, no key required).

Centralizes HTTP access, title parsing and HTML cleaning so each tool file
stays small and focused on its own responsibility.
"""

import re
from typing import List, Optional

import requests

BASE_URL = "https://api.tvmaze.com"
TIMEOUT = 10


def parse_title(args: str) -> str:
    """Extract a clean show title from the raw action argument string.

    Handles inputs like: title='Breaking Bad', "Breaking Bad",
    title=Breaking Bad, or just Breaking Bad.
    """
    if not args:
        return ""
    text = args.strip()

    match = re.match(r"^\s*(?:title|name|show|query|q)\s*=\s*(.*)$", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    text = text.strip().strip("()").strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1]

    return text.strip()


def strip_html(text: Optional[str], max_len: int = 300) -> str:
    """Remove HTML tags (TVmaze summaries are wrapped in <p>...</p>) and truncate."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text).strip()
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip() + "..."
    return clean


def fetch_show(title: str) -> Optional[dict]:
    """Return the best-matching show dict for a title, or None if not found."""
    resp = requests.get(
        f"{BASE_URL}/singlesearch/shows",
        params={"q": title},
        timeout=TIMEOUT,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_episodes(show_id: int) -> List[dict]:
    """Return the list of all episodes for a show id."""
    resp = requests.get(f"{BASE_URL}/shows/{show_id}/episodes", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_cast(show_id: int) -> List[dict]:
    """Return the cast list for a show id."""
    resp = requests.get(f"{BASE_URL}/shows/{show_id}/cast", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()
