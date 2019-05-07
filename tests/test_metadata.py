import pytest
import os
import sys

tests_dir = os.path.dirname(__file__)
data_dir = os.path.join(tests_dir, 'data')
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))

sys.path.insert(0, root_dir)
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

@pytest.fixture(scope='module')
def my_archive():
    archive = metadata.Archive()
    archive.assign_conf(os.path.join(data_dir, 'my_archive.conf'))
    archive.conf.set('Files', 'csv_dir', data_dir)
    archive.read_csv()
    return archive

def test_read_archive(my_archive):
    assert len(my_archive.get_platelist()) == 3
    assert len(my_archive.get_scanlist()) == 3

@pytest.mark.parametrize('plate_id,time_start', [('plate1', ['21:11:30']),
                                                 ('plate2', ['21:30:00', '22:10:00|22:17:00'])])
def test_plate_time(my_archive, plate_id, time_start):
    plate = my_archive.get_platemeta(plate_id)
    assert plate['tms_orig'] == time_start

@pytest.mark.parametrize('plate_id,ut_start', [('plate1', ['1956-03-11T20:11:30']),
                                               ('plate2', ['1956-03-11T21:30:00', '1956-03-11T22:10:00'])])
def test_plate_ut(my_archive, plate_id, ut_start):
    plate = my_archive.get_platemeta(plate_id)
    plate.calculate()
    assert plate['date_obs'] == ut_start

@pytest.mark.parametrize('plate_id,exptime', [('plate1', [600]), 
                                              ('plate2', [1800, 600])])
def test_plate_exptime(my_archive, plate_id, exptime):
    plate = my_archive.get_platemeta(plate_id)
    assert plate['exptime'] == exptime

@pytest.mark.parametrize('plate_id,jd_avg', [('plate1', [2435544.34479]), 
                                             ('plate2', [2435544.40625, 2435544.42778])])
def test_plate_jd(my_archive, plate_id, jd_avg):
    plate = my_archive.get_platemeta(plate_id)
    plate.calculate()
    assert pytest.approx(plate['jd_avg'], abs=1e-5) == jd_avg
    #assert plate['jd_avg'] == jd_avg


