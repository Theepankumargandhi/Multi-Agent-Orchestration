from app import discount, fee


def test_existing_fee():
    assert fee(2) == 20


def test_existing_discount():
    assert discount(20) == 2
