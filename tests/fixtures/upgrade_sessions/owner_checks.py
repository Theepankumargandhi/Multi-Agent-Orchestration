from app.models import empty_basket


def test_empty_basket():
    assert empty_basket() == {"values": [], "types": []}
