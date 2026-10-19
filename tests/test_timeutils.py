"""Time functions of the metadata computations (DESIGN.md section 4.4)."""
import pytest

from pyplate.metadata.timeutils import (combine_orig_times,
                                        compute_exposure_times,
                                        heliocentric_jd, normalise_tz,
                                        sidereal_to_ut, split_orig_times)

POTSDAM = (52.38056, 13.06417, 107)
MOCK_HILL = (58.27, 26.72, 70)


def test_sidereal_to_ut():
    # POT015 000317 (tests/ver4/metadata/expected.json)
    assert sidereal_to_ut('1910-08-02', '19:52:08', *POTSDAM) == \
        '1910-08-02T22:17:23'


def test_sidereal_time_of_24_hours_or_more():
    # POT015 000620: ST 24:24:16 is in the next day
    t = compute_exposure_times('1910-12-06', 'ST 24:24:16', None, None,
                               1800., POTSDAM, False)
    assert t.ut_start.startswith('1910-12-07')


def test_ut_and_local_time():
    ut = compute_exposure_times('1956-03-11', '21:30:00', '22:00:00', 'UT',
                                None, POTSDAM, False)
    assert (ut.ut_start, ut.ut_mid, ut.ut_end) == \
        ('1956-03-11T21:30:00', '1956-03-11T21:45:00', '1956-03-11T22:00:00')
    assert ut.duration == pytest.approx(1800.)

    local = compute_exposure_times('1956-03-11', '21:11:30', None,
                                   'Europe/Berlin', 600., POTSDAM, False)
    assert local.ut_start == '1956-03-11T20:11:30'
    assert local.ut_mid == '1956-03-11T20:16:30'


def test_evening_date():
    # With evening dates, a time before noon is in the next day
    t = compute_exposure_times('1960-05-03', '01:15:00', '01:45:00',
                               'Europe/Tallinn', None, MOCK_HILL, True)
    assert t.ut_start == '1960-05-03T22:15:00'  # UTC+3 in 1960
    t = compute_exposure_times('1960-05-03', '01:15:00', '01:45:00',
                               'Europe/Tallinn', None, MOCK_HILL, False)
    assert t.ut_start == '1960-05-02T22:15:00'


def test_end_after_midnight():
    t = compute_exposure_times('1960-05-01', 'UT 23:30:00', 'UT 00:10:00',
                               'UT', None, MOCK_HILL, False)
    assert (t.ut_start, t.ut_end) == ('1960-05-01T23:30:00',
                                      '1960-05-02T00:10:00')


def test_date_only_and_rounding():
    t = compute_exposure_times('1955-04-04', None, None, 'UT', None,
                               MOCK_HILL, False)
    assert t.ut_start == '1955-04-04'
    assert t.jd_start == 2435202.0
    assert t.ut_mid is None
    t = compute_exposure_times('1955-04-01', '01:30:00', None, 'UT', 20.,
                               MOCK_HILL, False)
    assert t.jd_start == round(t.jd_start, 5)
    assert t.year_mid == round(t.year_mid, 8)


def test_invalid_date():
    assert compute_exposure_times('1955-13-45', '20:00', None, 'UT', None,
                                  MOCK_HILL, False) is None
    assert compute_exposure_times(None, '20:00', None, 'UT', None,
                                  MOCK_HILL, False) is None


def test_heliocentric_jd():
    # POT015 000317 (expected.json): JD 2418886.43916 -> HJD 2418886.44364
    assert heliocentric_jd(2418886.43916, 289.9581, 15.3889,
                           *POTSDAM) == 2418886.44364


def test_split_and_combine():
    assert split_orig_times('22:00:00, UT 20:00:00') == ('22:00:00',
                                                         '20:00:00', None)
    assert split_orig_times('ST 14:00') == (None, None, '14:00')
    assert combine_orig_times(['22:00:00'], ['20:00:00'], []) == \
        ['22:00:00, UT 20:00:00']
    assert combine_orig_times(['UT 01:30'], ['01:30'], []) == ['UT 01:30']
    assert combine_orig_times([], [], ['14:00', '15:00']) == \
        ['ST 14:00', 'ST 15:00']


def test_normalise_tz():
    assert [normalise_tz(t) for t in ['st', 'Ut', 'Europe/Berlin', None]] == \
        ['ST', 'UT', 'Europe/Berlin', None]
