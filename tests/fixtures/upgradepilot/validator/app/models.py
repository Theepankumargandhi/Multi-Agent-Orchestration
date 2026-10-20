from pydantic import BaseModel, validator


class Request(BaseModel):
    count: int

    @validator("count")
    def validate_count(cls, value, field):
        if field.name == "count" and value < 0:
            raise ValueError("count must be nonnegative")
        return value


def normalize(payload):
    return {"count": Request(**payload).count}
