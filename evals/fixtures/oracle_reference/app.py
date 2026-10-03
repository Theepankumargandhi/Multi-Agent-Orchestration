def clamp(value, lower=0, upper=10):
    """Authored operator reference, private to the verifier and never model context."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value
