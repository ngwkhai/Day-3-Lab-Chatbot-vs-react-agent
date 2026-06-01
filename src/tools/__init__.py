"""Tool registry for the ReAct agent.

Each tool lives in its own module and exposes a `TOOL` dict shaped exactly how
`src/agent/agent.py` expects:
    { "name": str, "description": str, "func": Callable[[str], str] }

Import in the agent / runner with:
    from src.tools import TOOLS
"""

from src.tools.cast_tool import TOOL as CAST_TOOL
from src.tools.episode_stats_tool import TOOL as EPISODE_STATS_TOOL
from src.tools.show_info_tool import TOOL as SHOW_INFO_TOOL
from src.tools.show_rating_tool import TOOL as SHOW_RATING_TOOL

TOOLS = [
    SHOW_INFO_TOOL,
    SHOW_RATING_TOOL,
    EPISODE_STATS_TOOL,
    CAST_TOOL,
]

__all__ = ["TOOLS"]
