from app.models import normalize


def test_positive_count():
    assert normalize({"count": 3}) == {"count": 3}
