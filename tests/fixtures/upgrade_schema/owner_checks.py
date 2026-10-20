from app.models import normalize


def test_empty_order():
    assert normalize({"items": []}) == {"values": [], "types": []}
