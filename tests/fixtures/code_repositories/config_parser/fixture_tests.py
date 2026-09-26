import pytest
from config_parser import parse_assignment


def test_empty_value_is_valid():
    assert parse_assignment("TOKEN=") == ("TOKEN", "")


def test_empty_key_is_invalid():
    with pytest.raises(ValueError):
        parse_assignment("=secret")
