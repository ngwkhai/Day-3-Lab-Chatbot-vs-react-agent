"""Tool: get_cast -- main cast (actor -> character) for a TV show from TVmaze."""

import requests

from src.tools._client import fetch_cast, fetch_show, parse_title

MAX_CAST = 8


def get_cast(args: str) -> str:
    """Return the top-billed cast as 'Actor as Character' lines."""
    title = parse_title(args)
    if not title:
        return "INVALID_INPUT: provide a show title, e.g. get_cast(title='Breaking Bad')"

    try:
        show = fetch_show(title)
        if not show:
            return "NOT_FOUND"
        cast = fetch_cast(show.get("id"))
    except requests.RequestException as e:
        return f"API_ERROR: {e}"

    if not cast:
        return "NO_CAST"

    lines = []
    for member in cast[:MAX_CAST]:
        actor = (member.get("person") or {}).get("name")
        character = (member.get("character") or {}).get("name")
        if actor and character:
            lines.append(f"{actor} as {character}")
        elif actor:
            lines.append(actor)
    return "; ".join(lines)


TOOL = {
    "name": "get_cast",
    "description": (
        "Returns the main cast of a TV show from TVmaze as a list of "
        "'Actor as Character' entries separated by '; '. "
        "Argument format: get_cast(title='Breaking Bad'). "
        f"Returns up to {MAX_CAST} top-billed cast members, "
        "'NO_CAST' if none, or 'NOT_FOUND' if the show does not exist."
    ),
    "func": get_cast,
}
