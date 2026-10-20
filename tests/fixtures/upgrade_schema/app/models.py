from typing import Optional, Union

from pydantic import BaseModel, Field


class Item(BaseModel):
    quantity: Union[int, str]


class Request(BaseModel):
    items: list[Item] = Field(default_factory=list)


class NullableProfile(BaseModel):
    label: Optional[str]


def normalize(payload):
    request = Request(**payload)
    return {"values": [item.quantity for item in request.items], "types": [type(item.quantity).__name__ for item in request.items]}
