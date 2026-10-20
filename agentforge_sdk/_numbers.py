"""Shared bounded-number grammar and Decimal arithmetic for evidence tools."""

import re
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

NUMBER = re.compile(r"(?:(USD|EUR|GBP) |([$€£]))?([+-]?(?:0|[1-9][0-9]{0,29})(?:\.[0-9]{1,12})?)(%)?")


def parse_number(text: str) -> tuple[Decimal, str] | None:
    match = NUMBER.fullmatch(text)
    if match is None or (match[4] and (match[1] or match[2])):
        return None
    unit = match[1] or {"$": "USD", "€": "EUR", "£": "GBP"}.get(match[2]) or match[4] or "scalar"
    return Decimal(match[3]), unit


def format_number(value: Decimal) -> str:
    if value == 0:
        return "0"
    formatted = format(value, "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def calculate_numbers(operation: str, values: list[Decimal], unit: str) -> tuple[str, str]:
    """Callers validate arity, units and denominators before entering this local tool."""
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_EVEN
        if operation == "sum":
            value = sum(values, Decimal(0))
        elif operation == "difference":
            value = values[0] - values[1]
        elif operation == "ratio":
            value = values[0] / values[1]
        elif operation == "percent_change":
            value = (values[1] - values[0]) / abs(values[0]) * 100
        else:
            raise ValueError("unsupported numeric operation")
        result_unit = "%" if operation == "percent_change" else "scalar" if operation == "ratio" else unit
        return format_number(value), result_unit
