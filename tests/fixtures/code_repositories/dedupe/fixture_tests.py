from dedupe import unique_items


def test_preserves_first_seen_order():
    assert unique_items(["z", "a", "z", "b", "a"]) == ["z", "a", "b"]


def test_empty_input():
    assert unique_items([]) == []
