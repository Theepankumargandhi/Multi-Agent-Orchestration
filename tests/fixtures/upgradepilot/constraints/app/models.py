from pydantic import BaseModel, Field


class Request(BaseModel):
    code: str = Field(regex=r"^[A-Z]{2}$")


def normalize(payload):
    return Request(**payload).dict()
