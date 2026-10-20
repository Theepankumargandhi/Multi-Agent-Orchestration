from app import total


def test_ordinary_cart():
    assert total({"items": [{"quantity": 2, "price": 5}]}) == 10


def test_empty_cart():
    assert total({"items": []}) == 0
