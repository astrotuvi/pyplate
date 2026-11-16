"""Readers, exposures, sub-exposures and time computations of the metadata
subpackage, with the archive in tests/data."""
import dataclasses
import os

import pytest

from pyplate.config import config_from_dict, load_config
from pyplate.metadata import ArchiveReader, PlateHeader
from pyplate.metadata.readers.wfpdb import str_to_num

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_str_int():
    result = str_to_num('1')
    assert isinstance(result, int)
    assert result == 1


def test_str_float():
    result = str_to_num('1.0')
    assert isinstance(result, float)
    assert result == 1.0


@pytest.fixture(scope='module')
def my_archive():
    cfg = load_config(os.path.join(DATA_DIR, 'my_archive.toml'))
    cfg = dataclasses.replace(cfg, files=dataclasses.replace(
        cfg.files, csv_dir=DATA_DIR))
    return ArchiveReader(cfg)


def test_read_archive(my_archive):
    assert len(list(my_archive.iter_plates())) == 3
    assert len(list(my_archive.iter_scans())) == 4


@pytest.mark.parametrize('plate_num,time_start', [
    ('plate1', ['21:11:30']),
    ('plate2', ['21:30:00', '22:10:00|22:17:00']),
    ('plate3', ['12:00:00'])])
def test_plate_time(my_archive, plate_num, time_start):
    plate = my_archive.get_plate(plate_num)
    assert [e.time_orig_start for e in plate.exposures] == time_start


@pytest.mark.parametrize('plate_num,ut_start', [
    ('plate1', ['1956-03-11T20:11:30']),
    ('plate2', ['1956-03-11T21:30:00', '1956-03-11T22:10:00']),
    ('plate3', ['1931-03-12T23:49:08'])])
def test_plate_ut(my_archive, plate_num, ut_start):
    plate = my_archive.get_plate(plate_num)
    plate.compute_values()
    assert [e.ut_start for e in plate.exposures] == ut_start


@pytest.mark.parametrize('plate_num,numexp', [('plate1', 1), ('plate2', 2),
                                              ('plate3', 1)])
def test_plate_numexp(my_archive, plate_num, numexp):
    plate = my_archive.get_plate(plate_num)
    assert plate.numexp == numexp == len(plate.exposures)


@pytest.mark.parametrize('plate_num,exptime', [('plate1', [600]),
                                               ('plate2', [1800, 600]),
                                               ('plate3', [1200])])
def test_plate_exptime(my_archive, plate_num, exptime):
    plate = my_archive.get_plate(plate_num)
    assert [e.exptime for e in plate.exposures] == exptime


@pytest.mark.parametrize('plate_num,jd_mid', [
    ('plate1', [2435544.34479]),
    ('plate2', [2435544.40625, 2435544.42778]),
    ('plate3', [2426413.49940])])
def test_plate_jd(my_archive, plate_num, jd_mid):
    plate = my_archive.get_plate(plate_num)
    plate.compute_values()
    assert [e.jd_mid for e in plate.exposures] == pytest.approx(jd_mid,
                                                                abs=1e-5)


def test_subexposures(my_archive):
    plate = my_archive.get_plate('plate2')
    plate.compute_values()
    first, second = plate.exposures
    assert first.subexposures == [] and first.num_sub == 1
    assert second.num_sub == len(second.subexposures) == 2
    assert [s.time_orig_start for s in second.subexposures] == \
        ['22:10:00', '22:17:00']
    assert [s.ut_start for s in second.subexposures] == \
        ['1956-03-11T22:10:00', '1956-03-11T22:17:00']
    assert [s.exptime for s in second.subexposures] == pytest.approx([300., 300.])

    # The exposure spans the sub-exposures; its weighted mid-point uses
    # their exposure times
    assert second.ut_start == second.subexposures[0].ut_start
    assert second.ut_end == second.subexposures[1].ut_end
    assert second.jd_weighted == pytest.approx(
        (second.subexposures[0].jd_mid + second.subexposures[1].jd_mid) / 2)


@pytest.mark.parametrize('filename,plate_num', [
    ('plate_0001.fits', 'plate1'), ('plate_0002.fits', 'plate2'),
    ('plate_0003_1.fits', 'plate3'), ('plate_0003_2.fits', 'plate3')])
def test_scan_plate(my_archive, filename, plate_num):
    plate = my_archive.get_plate(filename_scan=filename)
    assert plate.plate_num == plate_num
    assert filename in [s.filename_scan for s in plate.scans]


def test_plate_scans(my_archive):
    plate = my_archive.get_plate('plate3')
    assert [s.filename_scan for s in plate.scans] == \
        ['plate_0003_1.fits', 'plate_0003_2.fits']
    assert [s.scan_author for s in plate.scans] == ['Scan Author'] * 2


def test_plateheader_jd(my_archive):
    plate = my_archive.get_plate('plate1')
    plate.compute_values()
    h = PlateHeader.from_metadata(plate, plate.scans[0], my_archive.cfg)
    assert h['JD-AVG'] == 2435544.34479
    assert h.comments['JD-AVG'] == 'Julian date at the mid-point of exposure 1'
    assert h['OBSERVAT'] == 'Astrophysikalische Observatorium Potsdam'
    assert h['TZ-ORIG'] == 'Europe/Berlin'


def csv_reader(tmp_path, rows, columns, **csv_settings):
    """ArchiveReader of a plate CSV file with the given rows."""
    (tmp_path / 'plates.csv').write_text(
        '\n'.join(','.join(r) for r in rows) + '\n')
    data = {'Files': {'csv_dir': str(tmp_path), 'plate_csv': ['plates.csv']},
            'CSV': {'plates.csv': {'csv_list_delimiter': ';',
                                   **csv_settings, **columns}}}
    return ArchiveReader(config_from_dict(data))


# DESIGN.md section 4.3
@pytest.mark.parametrize('numexp,times,date,expected', [
    ('1', '20:00;20:30;21:00', '1955-04-01', 3),
    ('4', '20:00;21:00', '1955-04-01', 2),
    ('4', '20:00', '1955-04-01', 4),
    ('4', '', '1955-04-01', 4),
    ('2', '20:00;20:30;21:00', '', 2),
    ('', '20:00', '1955-04-01', 1)])
def test_numexp_rule(tmp_path, numexp, times, date, expected):
    reader = csv_reader(tmp_path, [['p1', numexp, times, date]],
                        {'plate_num': 1, 'numexp': 2, 'time_orig_start': 3,
                         'date_orig_start': 4})
    plate = reader.get_plate('p1')
    assert plate.numexp == len(plate.exposures) == expected


def test_spreading_values(tmp_path):
    # One start time, exposure time and object for three exposures: the
    # first two go to the first exposure only, the object to all
    reader = csv_reader(tmp_path,
                        [['p1', '3', '20:00', '600', 'M 31', '1955-04-01']],
                        {'plate_num': 1, 'numexp': 2, 'time_orig_start': 3,
                         'exptime': 4, 'object_name': 5,
                         'date_orig_start': 6})
    plate = reader.get_plate('p1')
    assert [e.time_orig_start for e in plate.exposures] == ['20:00', None,
                                                            None]
    assert [e.exptime for e in plate.exposures] == [600, None, None]
    assert [e.object_name for e in plate.exposures] == ['M 31'] * 3

    # Values by position, missing ones None
    reader = csv_reader(tmp_path,
                        [['p1', '20:00;20:30;21:00', 'A;B', '1955-04-01']],
                        {'plate_num': 1, 'time_orig_start': 2,
                         'object_name': 3, 'date_orig_start': 4})
    plate = reader.get_plate('p1')
    assert [e.object_name for e in plate.exposures] == ['A', 'B', None]


def test_exposure_columns(tmp_path):
    # method_code goes to the plate; exposure.method_code to the exposures
    reader = csv_reader(tmp_path, [['p1', '2', '1', '1955-04-01']],
                        {'plate_num': 1, 'method_code': 2,
                         'date_orig_start': 4, 'exposure': {'method_code': 3}})
    plate = reader.get_plate('p1')
    assert plate.method_code == 2
    assert plate.exposures[0].method_code == 1


@pytest.mark.parametrize('row_tz,expected', [('Europe/Berlin', 'Europe/Berlin'),
                                             ('', '')])
def test_tz_orig_csv_row_wins(tmp_path, row_tz, expected):
    reader = csv_reader(tmp_path, [['p1', row_tz]],
                        {'plate_num': 1, 'tz_orig': 2},
                        csv_timezone='UT')
    assert reader.get_plate('p1').tz_orig == expected


def test_tz_orig_csv_default(tmp_path):
    reader = csv_reader(tmp_path, [['p1']], {'plate_num': 1},
                        csv_timezone='st')
    assert reader.get_plate('p1').tz_orig == 'ST'


def test_tz_orig_wfpdb_default(tmp_path):
    (tmp_path / 'main.txt').write_text(
        'POT015 004252 121736+142634 19170423      MSA 80               '
        'F 0401      HauffUltRap         202001001\n')
    data = {'Files': {'wfpdb_dir': str(tmp_path),
                      'wfpdb_maindata': 'main.txt'},
            'WFPDB': {'maindata_timezone': 'ST'}}
    reader = ArchiveReader(config_from_dict(data))
    (plate,) = reader.iter_plates()
    assert plate.tz_orig == 'ST'


def test_jd_mid_orig(tmp_path):
    reader = csv_reader(tmp_path, [['p1', '2433284.40625', '1950-01-02']],
                        {'plate_num': 1, 'jd_mid_orig': 2,
                         'date_orig_start': 3})
    plate = reader.get_plate('p1')
    assert plate.exposures[0].jd_mid_orig == 2433284.40625
