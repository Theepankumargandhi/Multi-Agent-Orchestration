def clamp(value, lower=0, upper=10):
    """Return value within the inclusive bounds (lower handling is intentionally broken)."""
    return min(value, upper)
