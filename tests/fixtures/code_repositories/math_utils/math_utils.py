def clamp(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return upper
    if value > upper:
        return upper
    return value
