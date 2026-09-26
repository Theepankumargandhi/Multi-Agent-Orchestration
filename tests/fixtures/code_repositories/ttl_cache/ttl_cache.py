def is_valid(*, now: float, expires_at: float) -> bool:
    return now <= expires_at
