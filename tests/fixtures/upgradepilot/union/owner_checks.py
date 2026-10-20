from app.models import normalize


def test_integer_input():
    assert normalize({"value": 7}) == {"value": 7, "type": "int"}
