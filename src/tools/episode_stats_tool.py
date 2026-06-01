"""Tool: get_episode_stats -- season/episode/runtime statistics for a TV show."""

import json

import requests

from src.tools._client import fetch_episodes, fetch_show, parse_title


def get_episode_stats(args: str) -> str:
    """Return season/episode counts and total runtime for a show as JSON."""
    title = parse_title(args)
    if not title:
        return "INVALID_INPUT: provide a show title, e.g. get_episode_stats(title='Breaking Bad')"

    try:
        show = fetch_show(title)
        if not show:
            return "NOT_FOUND"
        episodes = fetch_episodes(show.get("id"))
    except requests.RequestException as e:
        return f"API_ERROR: {e}"

    seasons = sorted({ep.get("season") for ep in episodes if ep.get("season") is not None})
    total_runtime = sum((ep.get("runtime") or 0) for ep in episodes)

    stats = {
        "name": show.get("name"),
        "season_count": len(seasons),
        "total_episodes": len(episodes),
        "total_runtime_minutes": total_runtime,
        "total_runtime_hours": round(total_runtime / 60, 1),
    }
    return json.dumps(stats, ensure_ascii=False)


TOOL = {
    "name": "get_episode_stats",
    "description": (
        "Returns season and episode statistics for a TV show from TVmaze. "
        "Argument format: get_episode_stats(title='Breaking Bad'). "
        "Returns a JSON string with: name, season_count (int), total_episodes (int), "
        "total_runtime_minutes (int), total_runtime_hours (float). "
        "Returns 'NOT_FOUND' if the show does not exist."
    ),
    "func": get_episode_stats,
}
