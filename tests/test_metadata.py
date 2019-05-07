from pyplate import metadata

def test_str_int():
    s = '1'
    result = metadata.str_to_num(s)
    assert isinstance(result, int)
    assert result == 1

def test_str_float():
    s = '1.0'
    result = metadata.str_to_num(s)
    assert isinstance(result, float)
    assert result == 1.0

def test_plate_header():
    h = metadata.PlateHeader()
    h.populate()
    assert 'DATEORIG' in h

