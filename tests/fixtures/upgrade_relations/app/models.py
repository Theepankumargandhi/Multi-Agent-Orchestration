"""Authored migration controls, not a production financial service."""
from typing import Union

from pydantic import BaseModel


class Deposit(BaseModel):
    event: Union[int, str]
    amount: int


class Wallet:
    def __init__(self):
        self.balance = 0
        self.seen = set()

    def deposit(self, event, amount):
        request = Deposit(event=event, amount=amount)
        # Legacy code assumes numeric event identifiers were coerced by validation.
        if isinstance(request.event, int) and request.event in self.seen:
            return
        self.seen.add(request.event)
        self.balance += request.amount

    def reject(self, amount):
        raise ValueError("rejected without applying the operation")

    def snapshot(self):
        return {"balance": self.balance, "events": len(self.seen)}


class BrokenRollbackWallet(Wallet):
    def reject(self, amount):
        self.balance += amount
        raise ValueError("same bug under both dependencies")


def create_wallet():
    return Wallet()


def create_broken_wallet():
    return BrokenRollbackWallet()


def empty_wallet():
    return create_wallet().snapshot()
