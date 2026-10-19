"""Exposure: a row of the ``exposure`` table."""
from dataclasses import dataclass

import ephem
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

from . import codes
from .base import DbRecord, related
from .subexposure import SubExposure
from .timeutils import (compute_exposure_times, heliocentric_jd, round_jd,
                        round_year)


@dataclass
class Exposure(DbRecord):
    """An exposure of a plate (``exposure`` table).

    The columns are described in ``applause_dr5.yaml``. Attributes without
    a column: ``subexposures`` (when the exposure has more than one; the
    database has no sub-exposure rows for a single one), and the
    sexagesimal right ascension and declination computed with
    :meth:`compute_coords` (see :attr:`ra`, :attr:`dec`).
    """

    table = 'exposure'

    exposure_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    exposure_num: int | None = None
    object_name: str | None = None
    object_type_code: str | None = None
    ra_orig: str | None = None
    dec_orig: str | None = None
    flag_coord: str | None = None
    ra_icrs: float | None = None
    dec_icrs: float | None = None
    date_orig_start: str | None = None
    date_orig_end: str | None = None
    time_orig_start: str | None = None
    time_orig_end: str | None = None
    jd_mid_orig: float | None = None
    flag_time: str | None = None
    ut_start: str | None = None
    ut_mid: str | None = None
    ut_weighted: str | None = None
    ut_end: str | None = None
    year_start: float | None = None
    year_mid: float | None = None
    year_weighted: float | None = None
    year_end: float | None = None
    jd_start: float | None = None
    jd_mid: float | None = None
    jd_weighted: float | None = None
    jd_end: float | None = None
    hjd_mid: float | None = None
    hjd_weighted: float | None = None
    exptime: float | None = None
    num_sub: int | None = None
    method_code: int | None = None
    focus: float | None = None
    subexposures: list[SubExposure] = related()
    ra_hms: str | None = related(default=None)
    dec_dms: str | None = related(default=None)

    @property
    def object_type(self) -> str | None:
        """Object-type text of ``object_type_code``."""
        return codes.object_type(self.object_type_code)

    @property
    def ra(self) -> str | None:
        """Right ascension (ICRS) as ``hh:mm:ss.s``.

        Computed with the unrounded coordinates by :meth:`compute_coords`,
        otherwise from ``ra_icrs``.
        """
        if self.ra_hms is not None or self.ra_icrs is None:
            return self.ra_hms

        return Angle(self.ra_icrs, u.deg).to_string(unit=u.hour, sep=':',
                                                    precision=1, pad=True)

    @property
    def dec(self) -> str | None:
        """Declination (ICRS) as ``dd:mm:ss.s``, as :attr:`ra`."""
        if self.dec_dms is not None or self.dec_icrs is None:
            return self.dec_dms

        return Angle(self.dec_icrs, u.deg).to_string(unit=u.deg, sep=':',
                                                     precision=1, pad=True)

    def compute_times(self, tz_orig: str | None, site: tuple,
                      evening_date: bool) -> float | None:
        """Compute the UT times, Julian dates and years of the exposure.

        With sub-exposures, the start comes from the first, the end from
        the last, and the weighted mid-point from all of them, weighted
        with their exposure times.

        Parameters
        ----------
        tz_orig
            Time zone of the recorded times (``plate.tz_orig``).
        site
            (latitude, longitude, elevation) of the observatory.
        evening_date
            The recorded date is the date of the evening.

        Returns
        -------
        float | None
            The exposure time computed from the start and end times (the
            sum over the sub-exposures), or None.
        """
        if self.subexposures:
            return self._times_from_subexposures(tz_orig, site, evening_date)

        times = compute_exposure_times(self.date_orig_start,
                                       self.time_orig_start,
                                       self.time_orig_end, tz_orig,
                                       self.exptime, site, evening_date)

        if times is None:
            return None

        for name in ['ut_start', 'ut_mid', 'ut_weighted', 'ut_end',
                     'jd_start', 'jd_mid', 'jd_weighted', 'jd_end',
                     'year_start', 'year_mid', 'year_weighted', 'year_end']:
            setattr(self, name, getattr(times, name))

        self.num_sub = 1
        return times.duration

    def _times_from_subexposures(self, tz_orig, site, evening_date):
        subs = self.subexposures
        computed = [s.compute_times(tz_orig, site, evening_date)
                    for s in subs]

        if any(c is None for c in computed):
            return None

        first, last = subs[0], subs[-1]
        self.num_sub = len(subs)
        self.ut_start = first.ut_start
        self.jd_start = first.jd_start
        self.year_start = first.year_start
        self.ut_end = last.ut_end
        self.jd_end = last.jd_end
        self.year_end = last.year_end
        exptime = None

        if len([s.exptime for s in subs if s.exptime]) == len(subs):
            weights = [s.exptime for s in subs]
            exptime = sum(weights)
            self.jd_weighted = np.average([s.jd_mid for s in subs],
                                          weights=weights)
            self.year_weighted = np.average([s.year_mid for s in subs],
                                            weights=weights)
            self.ut_weighted = Time(self.jd_weighted, format='jd',
                                    scale='ut1', precision=0).isot
        else:
            self.jd_weighted = None
            self.year_weighted = None
            self.ut_weighted = None

        jd_mid = np.mean([first.jd_start, last.jd_end])
        year_mid = np.mean([first.year_start, last.year_end])
        self.ut_mid = Time(jd_mid, format='jd', scale='ut1',
                           precision=0).isot
        self.jd_mid = round_jd(jd_mid)
        self.year_mid = round_year(year_mid)
        return exptime

    def compute_coords(self, site: tuple) -> None:
        """Compute ICRS coordinates from the recorded ones.

        The recorded coordinates refer to the equinox of the recorded date.
        With a known site, also computes the heliocentric Julian dates of
        the mid-points.

        Parameters
        ----------
        site
            (latitude, longitude, elevation) of the observatory.
        """
        if not (self.ra_orig and self.dec_orig and self.date_orig_start):
            return

        pointing = ephem.FixedBody()
        pointing._ra = ephem.hours(str(self.ra_orig))
        pointing._dec = ephem.degrees(str(self.dec_orig))
        pointing._epoch = str(self.date_orig_start)
        pointing.compute(ephem.J2000)

        ra = Angle(pointing.ra, u.radian)
        dec = Angle(pointing.dec, u.radian)
        self.ra_hms = ra.to_string(unit=u.hour, sep=':', precision=1,
                                   pad=True)
        self.dec_dms = dec.to_string(unit=u.deg, sep=':', precision=1,
                                     pad=True)
        self.ra_icrs = float(ra.to_string(unit=u.deg, decimal=True,
                                          precision=4))
        self.dec_icrs = float(dec.to_string(unit=u.deg, decimal=True,
                                            precision=4))
        latitude, longitude, elevation = site

        if latitude and longitude:
            if self.jd_mid:
                self.hjd_mid = heliocentric_jd(self.jd_mid, self.ra_icrs,
                                               self.dec_icrs, latitude,
                                               longitude, elevation)
            else:
                self.hjd_mid = None

            if self.jd_weighted:
                self.hjd_weighted = heliocentric_jd(self.jd_weighted,
                                                    self.ra_icrs,
                                                    self.dec_icrs, latitude,
                                                    longitude, elevation)
            else:
                self.hjd_weighted = None
