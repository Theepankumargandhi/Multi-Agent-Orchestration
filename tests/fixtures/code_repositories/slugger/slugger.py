import re


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "-", value.lower()).strip("-")
