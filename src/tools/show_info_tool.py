"""Tool: get_show_info -- rich metadata for a TV show from TVmaze."""

import json

import requests

from src.tools._client import fetch_show, parse_title, strip_html


def get_show_info(args: str) -> str:
    """Return rich real metadata about a TV show as a compact JSON string."""
    title = parse_title(args)
    if not title:
        return "INVALID_INPUT: provide a show title, e.g. get_show_info(title='Breaking Bad')"

    try:
        show = fetch_show(title)
    except requests.RequestException as e:
        return f"API_ERROR: {e}"

    if not show:
        return "NOT_FOUND"

    premiered = show.get("premiered") or ""
    ended = show.get("ended") or ""
    rating = (show.get("rating") or {}).get("average")
    channel = show.get("network") or show.get("webChannel") or {}

    info = {
        "name": show.get("name"),
        "type": show.get("type"),
        "language": show.get("language"),
        "premiered_year": int(premiered[:4]) if premiered[:4].isdigit() else None,
        "ended_year": int(ended[:4]) if ended[:4].isdigit() else None,
        "status": show.get("status"),
        "genres": show.get("genres") or [],
        "rating": rating if rating is not None else -1,
        "average_runtime_minutes": show.get("averageRuntime") or show.get("runtime") or -1,
        "network": channel.get("name"),
        "official_site": show.get("officialSite"),
        "summary": strip_html(show.get("summary")),
    }
    return json.dumps(info, ensure_ascii=False)


TOOL = {
    "name": "get_show_info",
    "description": (
        "Fetches rich real metadata for a TV show/series from TVmaze. "
        "Argument format: get_show_info(title='Breaking Bad'). "
        "Returns a JSON string with: name, type, language, premiered_year (int), "
        "ended_year (int or null), status, genres (list), rating (float 0-10), "
        "average_runtime_minutes (int), network, official_site, summary. "
        "Returns 'NOT_FOUND' if no match."
    ),
    "func": get_show_info,
}
