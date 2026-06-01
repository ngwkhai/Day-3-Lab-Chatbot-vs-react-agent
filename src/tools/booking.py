"""Movie-ticket booking tools (mock cinema, in-memory state).

These tools turn the ReAct agent into a cinema booking assistant. There is no
external booking API, so showtimes, seats and reservations live in process
memory. State resets when the serverless container restarts -- fine for a demo.

Every tool follows the registry contract used by ``src/agent/agent.py``:
    func(args: str) -> str
Arguments arrive as a single string such as::
    book_ticket(movie='Mai', time='19:30', seats=2, name='Khai')
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Mock cinema catalogue. Seats are a 5x10 grid (rows A-E, columns 1-10).
# --------------------------------------------------------------------------- #
_ROWS = ["A", "B", "C", "D", "E"]
_COLS = list(range(1, 11))


def _all_seats() -> List[str]:
    return [f"{r}{c}" for r in _ROWS for c in _COLS]


def _make_showtime(showtime_id: str, time: str, room: str, price: int) -> Dict:
    return {
        "id": showtime_id,
        "time": time,
        "room": room,
        "price": price,            # VND
        "booked": set(),           # set of seat labels already taken
    }


# Each movie: title, genre, duration (min), age rating, list of showtimes today.
MOVIES: Dict[str, Dict] = {
    "mai": {
        "title": "Mai",
        "genre": "Tâm lý, Tình cảm",
        "duration": 131,
        "rated": "T18",
        "showtimes": [
            _make_showtime("MAI-1", "14:00", "Phòng 1", 75000),
            _make_showtime("MAI-2", "17:30", "Phòng 1", 85000),
            _make_showtime("MAI-3", "20:45", "Phòng 3", 95000),
        ],
    },
    "dune": {
        "title": "Dune: Part Two",
        "genre": "Khoa học viễn tưởng",
        "duration": 166,
        "rated": "T13",
        "showtimes": [
            _make_showtime("DUNE-1", "13:15", "Phòng 2 (IMAX)", 120000),
            _make_showtime("DUNE-2", "19:00", "Phòng 2 (IMAX)", 140000),
        ],
    },
    "inside out 2": {
        "title": "Inside Out 2",
        "genre": "Hoạt hình, Gia đình",
        "duration": 96,
        "rated": "P",
        "showtimes": [
            _make_showtime("IO2-1", "10:00", "Phòng 4", 70000),
            _make_showtime("IO2-2", "15:45", "Phòng 4", 80000),
            _make_showtime("IO2-3", "18:30", "Phòng 5", 80000),
        ],
    },
}

# Confirmation code -> booking record.
BOOKINGS: Dict[str, Dict] = {}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
_KWARG_RE = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,]+))")


def _parse_kwargs(args: str) -> Dict[str, str]:
    """Parse ``key='value', key2=value`` style argument strings into a dict."""
    out: Dict[str, str] = {}
    if not args:
        return out
    for m in _KWARG_RE.finditer(args):
        key = m.group(1).lower()
        val = m.group(2) or m.group(3) or (m.group(4) or "")
        out[key] = val.strip()
    return out


def _find_movie(query: str) -> Dict | None:
    """Match a movie by case-insensitive substring on key or title."""
    if not query:
        return None
    q = query.strip().strip("'\"").lower()
    for key, movie in MOVIES.items():
        if q in key or q in movie["title"].lower():
            return movie
    return None


def _find_showtime(movie: Dict, time_or_id: str) -> Dict | None:
    if not time_or_id:
        return None
    needle = time_or_id.strip().strip("'\"").lower()
    for st in movie["showtimes"]:
        if needle == st["id"].lower() or needle == st["time"].lower():
            return st
    return None


def _vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + "đ"


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def list_movies(args: str = "") -> str:
    """List all movies currently showing today."""
    lines = ["Phim đang chiếu hôm nay:"]
    for movie in MOVIES.values():
        times = ", ".join(st["time"] for st in movie["showtimes"])
        lines.append(
            f"- {movie['title']} ({movie['genre']}, {movie['duration']} phút, "
            f"{movie['rated']}) — suất: {times}"
        )
    return "\n".join(lines)


def get_showtimes(args: str) -> str:
    """List showtimes (with price and free seats) for one movie."""
    kw = _parse_kwargs(args)
    query = kw.get("movie") or kw.get("title") or args
    movie = _find_movie(query)
    if movie is None:
        return "NOT_FOUND: không có phim này. Dùng list_movies() để xem danh sách."
    lines = [f"Suất chiếu cho '{movie['title']}':"]
    for st in movie["showtimes"]:
        free = len(_all_seats()) - len(st["booked"])
        lines.append(
            f"- {st['time']} | {st['room']} | {_vnd(st['price'])} | còn {free} ghế "
            f"(mã suất: {st['id']})"
        )
    return "\n".join(lines)


def check_seats(args: str) -> str:
    """Show available seats for a movie + showtime (time or showtime id)."""
    kw = _parse_kwargs(args)
    movie = _find_movie(kw.get("movie") or kw.get("title") or "")
    if movie is None:
        return "NOT_FOUND: thiếu hoặc sai tên phim. Ví dụ: check_seats(movie='Mai', time='17:30')"
    st = _find_showtime(movie, kw.get("time") or kw.get("showtime") or kw.get("id") or "")
    if st is None:
        return (
            "NOT_FOUND: thiếu hoặc sai suất chiếu. "
            f"Các suất của '{movie['title']}': "
            + ", ".join(s["time"] for s in movie["showtimes"])
        )
    available = [s for s in _all_seats() if s not in st["booked"]]
    preview = ", ".join(available[:20])
    return (
        f"'{movie['title']}' suất {st['time']} ({st['room']}): còn {len(available)}/"
        f"{len(_all_seats())} ghế. Giá {_vnd(st['price'])}/ghế. "
        f"Ghế trống (ví dụ): {preview}"
    )


def book_ticket(args: str) -> str:
    """Book seats for a showtime.

    Required: movie, time (or showtime id), name.
    seats can be specific labels ("A1,A2") or a count ("2").
    """
    kw = _parse_kwargs(args)
    movie = _find_movie(kw.get("movie") or kw.get("title") or "")
    if movie is None:
        return (
            "INVALID_INPUT: thiếu tên phim. Ví dụ: "
            "book_ticket(movie='Mai', time='17:30', seats=2, name='Khai')"
        )
    st = _find_showtime(movie, kw.get("time") or kw.get("showtime") or kw.get("id") or "")
    if st is None:
        return (
            "INVALID_INPUT: thiếu/sai suất chiếu. Các suất: "
            + ", ".join(s["time"] for s in movie["showtimes"])
        )
    name = kw.get("name") or kw.get("customer")
    if not name:
        return "INVALID_INPUT: cần tên khách hàng (name='...')."

    seats_arg = (kw.get("seats") or kw.get("seat") or "").strip()
    available = [s for s in _all_seats() if s not in st["booked"]]

    if not seats_arg:
        return "INVALID_INPUT: cần số ghế hoặc danh sách ghế, ví dụ seats=2 hoặc seats='A1,A2'."

    # Numeric -> auto-assign that many seats. Otherwise treat as explicit labels.
    if re.fullmatch(r"\d+", seats_arg):
        count = int(seats_arg)
        if count < 1:
            return "INVALID_INPUT: số ghế phải >= 1."
        if count > len(available):
            return f"SOLD_OUT: chỉ còn {len(available)} ghế cho suất này."
        chosen = available[:count]
    else:
        chosen = [s.strip().upper() for s in seats_arg.split(",") if s.strip()]
        taken = [s for s in chosen if s in st["booked"]]
        invalid = [s for s in chosen if s not in _all_seats()]
        if invalid:
            return f"INVALID_INPUT: ghế không tồn tại: {', '.join(invalid)} (ghế hợp lệ A1–E10)."
        if taken:
            return f"SEAT_TAKEN: ghế đã có người đặt: {', '.join(taken)}. Chọn ghế khác."

    total = st["price"] * len(chosen)
    code = "BK-" + uuid.uuid4().hex[:6].upper()
    st["booked"].update(chosen)
    BOOKINGS[code] = {
        "code": code,
        "movie": movie["title"],
        "time": st["time"],
        "room": st["room"],
        "seats": chosen,
        "name": name,
        "total": total,
    }
    return (
        f"BOOKING_CONFIRMED | Mã đặt vé: {code} | Phim: {movie['title']} | "
        f"Suất: {st['time']} | {st['room']} | Ghế: {', '.join(chosen)} | "
        f"Khách: {name} | Tổng tiền: {_vnd(total)}. "
        "Vui lòng đến quầy xuất trình mã trước giờ chiếu 15 phút."
    )


def get_booking(args: str) -> str:
    """Look up an existing booking by its confirmation code."""
    kw = _parse_kwargs(args)
    code = (kw.get("code") or args or "").strip().strip("'\"").upper()
    record = BOOKINGS.get(code)
    if not record:
        return "NOT_FOUND: không tìm thấy mã đặt vé này."
    return (
        f"Mã {record['code']}: {record['movie']} | suất {record['time']} | "
        f"{record['room']} | ghế {', '.join(record['seats'])} | "
        f"khách {record['name']} | tổng {_vnd(record['total'])}."
    )


LIST_MOVIES_TOOL = {
    "name": "list_movies",
    "description": (
        "Liệt kê tất cả phim đang chiếu hôm nay kèm thể loại, thời lượng và các suất. "
        "Không cần tham số: list_movies()."
    ),
    "func": list_movies,
}

GET_SHOWTIMES_TOOL = {
    "name": "get_showtimes",
    "description": (
        "Xem các suất chiếu, giá vé và số ghế trống của MỘT phim. "
        "Cú pháp: get_showtimes(movie='Mai')."
    ),
    "func": get_showtimes,
}

CHECK_SEATS_TOOL = {
    "name": "check_seats",
    "description": (
        "Kiểm tra ghế trống của một suất chiếu cụ thể. "
        "Cú pháp: check_seats(movie='Mai', time='17:30')."
    ),
    "func": check_seats,
}

BOOK_TICKET_TOOL = {
    "name": "book_ticket",
    "description": (
        "Đặt vé và trả về mã xác nhận. Bắt buộc có movie, time (hoặc mã suất), name; "
        "seats là số lượng (vd seats=2) hoặc danh sách ghế (vd seats='A1,A2'). "
        "Cú pháp: book_ticket(movie='Mai', time='17:30', seats=2, name='Khai')."
    ),
    "func": book_ticket,
}

GET_BOOKING_TOOL = {
    "name": "get_booking",
    "description": (
        "Tra cứu thông tin vé đã đặt bằng mã xác nhận. "
        "Cú pháp: get_booking(code='BK-AB12CD')."
    ),
    "func": get_booking,
}

BOOKING_TOOLS = [
    LIST_MOVIES_TOOL,
    GET_SHOWTIMES_TOOL,
    CHECK_SEATS_TOOL,
    BOOK_TICKET_TOOL,
    GET_BOOKING_TOOL,
]
