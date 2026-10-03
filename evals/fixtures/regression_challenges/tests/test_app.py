from app import clamp


def test_existing_happy_path():
    assert clamp(5) == 5
    assert clamp(12) == 10
