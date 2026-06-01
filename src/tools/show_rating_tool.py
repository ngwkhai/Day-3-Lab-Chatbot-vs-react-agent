"""Tool: get_show_rating -- numeric TVmaze rating for a TV show."""

import requests

from src.tools._client import fetch_show, parse_title


def get_show_rating(args: str) -> str:
    """Return the TVmaze average rating (float, 0-10) of a show, or -1 if unrated."""
    title = parse_title(args)
    if not title:
        return "INVALID_INPUT: provide a show title, e.g. get_show_rating(title='Game of Thrones')"

    try:
        show = fetch_show(title)
    except requests.RequestException as e:
        return f"API_ERROR: {e}"

    if not show:
        return "NOT_FOUND"

    rating = (show.get("rating") or {}).get("average")
    if rating is None:
        return "-1"
    return str(rating)


TOOL = {
    "name": "get_show_rating",
    "description": (
        "Returns the TVmaze average rating (float, 0-10) of a TV show. "
        "Argument format: get_show_rating(title='Game of Thrones'). "
        "Returns -1 if the show has no rating, or 'NOT_FOUND' if it does not exist."
    ),
    "func": get_show_rating,
}
