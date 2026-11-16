"""Time and date functions for the metadata computations.

The functions reproduce the computations of the old ``Plate.compute_values()``
exactly, including its rounding: Julian dates to 5 decimals, years to 8.
"""
import datetime as dt
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import ephem
import numpy as np
import pytimeparse
import pytz
import unidecode
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time, TimeDelta


def utc_now() -> dt.datetime:
    """Current UTC time without time zone information."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def format_time(t: dt.datetime) -> str:
    """ISO time to whole seconds (``2001-02-03T04:05:06``)."""
    return t.strftime('%Y-%m-%dT%H:%M:%S')


def round_jd(jd: float) -> float:
    return float('{:.5f}'.format(jd))


def round_year(year: float) -> float:
    return float('{:.8f}'.format(year))


@contextmanager
def no_erfa_warnings():
    """Suppress the ERFA warnings for dates far in the past."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Tried to get polar motions')
        warnings.filterwarnings('ignore', message='ERFA function')
        yield


def split_orig_times(combined: str) -> tuple[str | None, str | None,
                                             str | None]:
    """Split a recorded time string into (time, UT time, sidereal time).

    The string holds comma-separated times; a time with the prefix ``UT``
    or ``ST`` is a UT or sidereal time.
    """
    orig_time = None
    ut_time = None
    st_time = None

    for tm_str in [s.strip() for s in
                   unidecode.unidecode(combined).split(',')]:
        if tm_str[:2] == 'ST':
            st_time = tm_str[3:].strip()
        elif tm_str[:2] == 'UT':
            ut_time = tm_str[3:].strip()
        else:
            orig_time = tm_str.strip()

    return orig_time, ut_time, st_time


def combine_orig_times(orig: list, ut: list, st: list) -> list[str]:
    """Combine recorded times with UT and sidereal times given separately.

    For each exposure, the result holds the comma-separated times, with the
    prefix ``UT`` or ``ST`` for the separate ones; a separate time replaces
    a recorded time with the same prefix.
    """
    tms = []

    for i in range(max(len(orig), len(ut), len(st))):
        tm = []

        if i < len(orig) and orig[i]:
            tm.append(orig[i])

        for prefix, times in [('UT', ut), ('ST', st)]:
            if i < len(times):
                tm_str = '{} {}'.format(prefix, times[i])

                if tm and tm[0][:2] == prefix:
                    tm[0] = tm_str
                else:
                    tm.append(tm_str)

        tms.append(', '.join(tm))

    return tms


def normalise_tz(tz: str | None) -> str | None:
    """Time zone with ``UT`` and ``ST`` in upper case."""
    if tz is not None and tz.upper() in ('UT', 'ST'):
        return tz.upper()

    return tz


def _complete_time(t: str) -> str:
    """Add missing minutes and seconds; remove a final colon."""
    if t.count(':') == 0:
        t += ':00:00'
    elif t.count(':') == 1:
        t += ':00'

    if t.endswith(':'):
        t = t[:-1]

    return t


def _next_day(t_date: Time, t: str, evening_date: bool
              ) -> tuple[str, str] | None:
    """New (date, time) if the time is 24 h or more after the date.

    With ``evening_date``, a time before noon belongs to the next day.
    """
    tsec = pytimeparse.parse(t)

    if evening_date and tsec < 43200:
        tsec += 86400

    if tsec < 86400:
        return None

    t_new = t_date + TimeDelta(tsec, format='sec')
    date_new = Time(t_new, out_subfmt='date').iso

    if '.' in t:
        return date_new, Time(t_new).iso.split()[-1]

    return date_new, Time(t_new, precision=0).iso.split()[-1]


def sidereal_to_ut(date: str, st: str, latitude: float, longitude: float,
                   elevation: float | None) -> str:
    """UT (ISO) of the sidereal time ``st`` in the night of ``date``.

    Uses the next transit after noon of a star at the right ascension of
    the sidereal time (pyEphem).
    """
    loc = ephem.Observer()
    loc.lat = str(latitude)
    loc.lon = str(longitude)

    if elevation:
        loc.elevation = elevation

    loc.date = '%s %s' % (date, '12:00:00')
    star = ephem.FixedBody()
    star._dec = 0.
    star._epoch = date
    star._ra = ephem.hours(st)
    star.compute()
    return '%04d-%02d-%02dT%02d:%02d:%02d' % loc.next_transit(star).tuple()


def _local_to_ut(date: str, t: str, tz: str) -> tuple[str, str]:
    """Convert a local date and time in the time zone ``tz`` to UT."""
    dt_local = Time('{} {}'.format(date, t), scale='tai',
                    format='iso').datetime
    dt_utc = pytz.timezone(tz).localize(dt_local).astimezone(pytz.utc)
    return dt_utc.strftime('%Y-%m-%d'), dt_utc.strftime('%H:%M:%S')


def _ut_isot(date: str, t: str, tz: str) -> tuple[str, str]:
    """ISO UT time of a UT or local time; also returns the UT time."""
    if tz != 'UT':
        date, t = _local_to_ut(date, t, tz)

    if ':' not in t:
        t += ':00'

    return '{}T{}'.format(date, t), t


def _whole_seconds(isot: str) -> str:
    return Time(isot, format='isot', scale='ut1', precision=0).isot


def heliocentric_jd(jd: float, ra_deg: float, dec_deg: float,
                    latitude: float, longitude: float,
                    elevation: float | None) -> float:
    """Heliocentric Julian date (rounded to 5 decimals)."""
    loc = EarthLocation.from_geodetic(longitude, latitude, elevation or 0.)
    coord = SkyCoord(ra_deg, dec_deg, unit='deg')
    t = Time(jd, format='jd', scale='ut1')
    ltt = t.light_travel_time(coord, 'heliocentric', location=loc)
    return round_jd((t + ltt).jd)


@dataclass
class ExposureTimes:
    """Times computed for an exposure or a sub-exposure.

    ``duration`` is the time between the start and the end in seconds.
    """

    ut_start: str | None = None
    ut_mid: str | None = None
    ut_weighted: str | None = None
    ut_end: str | None = None
    jd_start: float | None = None
    jd_mid: float | None = None
    jd_weighted: float | None = None
    jd_end: float | None = None
    year_start: float | None = None
    year_mid: float | None = None
    year_weighted: float | None = None
    year_end: float | None = None
    duration: float | None = None


def _recorded(combined: str | None, tz: str | None
              ) -> tuple[str | None, str | None]:
    """Time and time zone from a recorded time string (UT or ST prefix)."""
    if not combined:
        return None, tz

    t, ut, st = split_orig_times(combined)

    if not t and ut:
        return ut, 'UT'

    if not t and st:
        return st, 'ST'

    return t, tz


def compute_exposure_times(date_orig: str | None, time_start: str | None,
                           time_end: str | None, tz_orig: str | None,
                           exptime: float | None, site: tuple,
                           evening_date: bool) -> ExposureTimes | None:
    """Compute UT times, Julian dates and years from the recorded times.

    Parameters
    ----------
    date_orig
        Recorded date (``date_orig_start``).
    time_start, time_end
        Recorded start and end times, possibly with comma-separated UT
        (``UT ...``) and sidereal (``ST ...``) times.
    tz_orig
        Time zone of the recorded times: ``UT``, ``ST`` or an IANA name.
    exptime
        Exposure time in seconds, used for the mid-point without an end.
    site
        (latitude, longitude, elevation) of the observatory, for sidereal
        times.
    evening_date
        The recorded date is the date of the evening.

    Returns
    -------
    ExposureTimes | None
        The times, or None if the date is not valid.
    """
    if not isinstance(date_orig, str):
        return None

    date_orig = unidecode.unidecode(date_orig)

    try:
        t_date = Time(date_orig, scale='tai')
    except ValueError:
        return None

    date_start = date_orig
    date_end = date_orig
    tms, tz = _recorded(time_start, tz_orig)
    tme, tz = _recorded(time_end, tz)
    ut_start = None
    ut_end = None

    # Times of 24 h or more, and times after midnight with evening dates
    if tme:
        tme = _complete_time(tme)
        shifted = _next_day(t_date, tme, evening_date)

        if shifted:
            date_orig, tme = shifted
            date_end = date_orig

    if tms:
        tms = _complete_time(tms)
        shifted = _next_day(t_date, tms, evening_date)

        if shifted:
            date_orig, tms = shifted
            date_start = date_orig

    latitude, longitude, elevation = site

    if tz == 'ST' and (tms or tme) and latitude and longitude:
        if tms:
            ut_start = sidereal_to_ut(date_orig, tms, latitude, longitude,
                                      elevation)

        if tme:
            ut_end = sidereal_to_ut(date_orig, tme, latitude, longitude,
                                    elevation)
    elif tz == 'UT' or tz in pytz.all_timezones:
        if tms:
            ut_start, t = _ut_isot(date_start, tms, tz)

            if '.' not in t:
                ut_start = _whole_seconds(ut_start)

        if tme:
            ut_end, t = _ut_isot(date_end, tme, tz)

            # End after midnight: next date
            if ut_start and ut_end < ut_start:
                next_day = (Time(ut_end, format='isot', scale='ut1',
                                 precision=0) + TimeDelta(1, format='jd'))
                ut_end = '{}T{}'.format(next_day.isot.split('T')[0], t)

            if '.' not in t:
                ut_end = _whole_seconds(ut_end)

    if not tms:
        ut_start = Time(date_orig, format='iso', scale='ut1',
                        out_subfmt='date').iso

    out = ExposureTimes()

    if ut_start:
        out.ut_start = ut_start
        t_start = Time(ut_start, format='isot', scale='ut1')

        if tms:
            out.jd_start = round_jd(t_start.jd)
            out.year_start = round_year(t_start.jyear)
        else:
            out.jd_start = float('%.0f' % t_start.jd)
            out.year_start = float('%.3f' % t_start.jyear)

    if ut_end:
        out.ut_end = ut_end
        t_end = Time(ut_end, format='isot', scale='ut1')

        if tme:
            out.jd_end = round_jd(t_end.jd)
            out.year_end = round_year(t_end.jyear)

    t_mid = None

    if ut_start and ut_end:
        jd_mid = np.mean([t_start.jd, t_end.jd])
        year_mid = np.mean([t_start.jyear, t_end.jyear])
        t_mid = Time(jd_mid, format='jd', scale='ut1', precision=0)
        out.jd_mid = round_jd(jd_mid)
        out.year_mid = round_year(year_mid)
    elif ut_start and exptime:
        t_mid = Time(t_start.jd + 0.5 * exptime / 86400., format='jd',
                     scale='ut1', precision=0)
        out.jd_mid = round_jd(t_mid.jd)
        out.year_mid = round_year(t_mid.jyear)

    if t_mid is not None:
        out.ut_mid = t_mid.isot
        out.ut_weighted = out.ut_mid
        out.jd_weighted = out.jd_mid
        out.year_weighted = out.year_mid

    if ut_start and ut_end:
        out.duration = (Time(ut_end, format='isot', scale='tai') -
                        Time(ut_start, format='isot', scale='tai')).sec

    return out
