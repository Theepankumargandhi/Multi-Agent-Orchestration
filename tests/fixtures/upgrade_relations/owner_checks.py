from app.models import empty_wallet


def test_empty_wallet():
    assert empty_wallet() == {"balance": 0, "events": 0}
