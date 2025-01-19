from src.utils.xml_parser import is_float, parse_type


def test_is_float_cases():
    """_summary_
    Test is_float functionality

    """

    assert is_float("160") is True
    assert is_float("165.5") is True
    assert is_float("16.5.5.0.1") is False


def test_parse_type():
    """Teste parse_type functionality"""

    assert parse_type("true") is True
    assert parse_type("false") is False
    assert isinstance(parse_type("160.5"), float)
    assert isinstance(parse_type("160"), float)
    assert parse_type("      0.193,           0,           0") == [0.193, 0, 0]
