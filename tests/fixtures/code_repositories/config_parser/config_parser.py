def parse_assignment(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key or not value:
        raise ValueError("invalid assignment")
    return key, value
