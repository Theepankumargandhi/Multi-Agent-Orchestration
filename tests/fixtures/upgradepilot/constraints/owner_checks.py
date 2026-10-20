from app.models import normalize


def test_valid_code():
    assert normalize({"code": "AB"}) == {"code": "AB"}
