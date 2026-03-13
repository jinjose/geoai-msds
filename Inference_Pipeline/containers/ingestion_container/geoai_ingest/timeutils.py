from datetime import datetime, timedelta, timezone

def parse_dt(s: str) -> datetime:
    # Accept YYYY-MM-DD OR ISO8601 with Z
    if len(s) == 10:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def floor_to(dt: datetime, gran: str) -> datetime:
    if gran == "hourly":
        return dt.replace(minute=0, second=0, microsecond=0)
    if gran == "daily":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if gran == "weekly":
        d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return d - timedelta(days=d.weekday())
    if gran == "monthly":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if gran == "yearly":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(gran)

def step(dt: datetime, gran: str) -> datetime:
    if gran == "hourly":
        return dt + timedelta(hours=1)
    if gran == "daily":
        return dt + timedelta(days=1)
    if gran == "weekly":
        return dt + timedelta(days=7)
    if gran == "monthly":
        y, m = dt.year, dt.month + 1
        if m == 13:
            y, m = y + 1, 1
        return dt.replace(year=y, month=m, day=1)
    if gran == "yearly":
        return dt.replace(year=dt.year + 1, month=1, day=1)
    raise ValueError(gran)

def iter_windows(start: datetime, end: datetime, gran: str):
    cur = floor_to(start, gran)
    while cur < end:
        nxt = step(cur, gran)
        yield cur, min(nxt, end)
        cur = nxt
