from typing import Union

from pydantic import BaseModel


class Request(BaseModel):
    value: Union[int, str]


def normalize(payload):
    model = Request(**payload)
    return {"value": model.value, "type": type(model.value).__name__}
