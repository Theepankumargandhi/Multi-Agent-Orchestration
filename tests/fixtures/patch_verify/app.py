"""Authored regression controls. Not a historical upstream application."""


def fee(quantity):
    return max(0, quantity) * 10


def discount(quantity):
    return quantity // 10
