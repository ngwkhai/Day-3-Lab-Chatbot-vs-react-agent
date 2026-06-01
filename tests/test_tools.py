import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import TOOLS


def test_tools():
    print("--- Testing TVmaze Tools (real data, no API key) ---")
    print(f"Registered tools: {[t['name'] for t in TOOLS]}\n")

    by_name = {t["name"]: t["func"] for t in TOOLS}

    cases = [
        ("get_show_info", "title='Breaking Bad'"),
        ("get_show_rating", "title='Game of Thrones'"),
        ("get_episode_stats", "title='Breaking Bad'"),
        ("get_cast", "title='Breaking Bad'"),
        ("get_show_rating", "title='asdkjfhqwenxyz-not-a-show'"),  # NOT_FOUND path
    ]

    for name, args in cases:
        result = by_name[name](args)
        print(f"{name}({args}) ->\n  {result}\n")

    print("Tools are working correctly!")


if __name__ == "__main__":
    test_tools()
