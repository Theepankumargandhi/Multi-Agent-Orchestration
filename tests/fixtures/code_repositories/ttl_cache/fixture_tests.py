from ttl_cache import is_valid


def test_exact_expiration_is_expired():
    assert is_valid(now=10.0, expires_at=10.0) is False


def test_before_expiration_is_valid():
    assert is_valid(now=9.9, expires_at=10.0) is True
