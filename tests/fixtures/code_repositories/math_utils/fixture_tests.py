from math_utils import clamp


def test_clamp_below_lower_bound():
    assert clamp(-5, 0, 10) == 0


def test_clamp_inside_range():
    assert clamp(4, 0, 10) == 4
