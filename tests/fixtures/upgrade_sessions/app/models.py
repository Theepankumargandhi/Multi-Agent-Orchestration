"""Authored local transaction control, not a real repository migration."""
from typing import Union

from pydantic import BaseModel


class Item(BaseModel):
    quantity: Union[int, str]


class Basket:
    def __init__(self):
        self.items = []

    def add(self, quantity):
        self.items.append(Item(quantity=quantity))

    def clear(self):
        self.items.clear()

    def snapshot(self):
        return {"values": [row.quantity for row in self.items],
            "types": [type(row.quantity).__name__ for row in self.items]}


def create_basket():
    return Basket()


def empty_basket():
    return create_basket().snapshot()
