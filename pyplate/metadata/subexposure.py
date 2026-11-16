"""SubExposure: a row of the ``exposure_sub`` table."""
from dataclasses import dataclass

from .base import DbRecord, related
from .timeutils import ExposureTimes, compute_exposure_times


@dataclass
class SubExposure(DbRecord):
    """A sub-exposure of an exposure (``exposure_sub`` table).

    The columns are described in ``applause_dr5.yaml``. ``year_start``,
    ``year_mid`` and ``year_end`` have no column; they are computed with the
    other times and used for the years of the exposure.
    """

    table = 'exposure_sub'

    subexposure_id: int | None = None
    exposure_id: int | None = None
    plate_id: int | None = None
    exposure_num: int | None = None
    subexposure_num: int | None = None
    date_orig_start: str | None = None
    time_orig_start: str | None = None
    time_orig_end: str | None = None
    ut_start: str | None = None
    ut_mid: str | None = None
    ut_end: str | None = None
    jd_start: float | None = None
    jd_mid: float | None = None
    jd_end: float | None = None
    exptime: float | None = None
    year_start: float | None = related(default=None)
    year_mid: float | None = related(default=None)
    year_end: float | None = related(default=None)

    def compute_times(self, tz_orig: str | None, site: tuple,
                      evening_date: bool) -> ExposureTimes | None:
        """Compute the UT times and Julian dates from the recorded times.

        As in the old code, the time zone is that of the plate, and the
        exposure time is the difference between the end and the start.

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
        ExposureTimes | None
            The computed times, or None without a valid date.
        """
        times = compute_exposure_times(self.date_orig_start,
                                       self.time_orig_start,
                                       self.time_orig_end, tz_orig, None,
                                       site, evening_date)

        if times is None:
            return None

        self.ut_start = times.ut_start
        self.ut_mid = times.ut_mid
        self.ut_end = times.ut_end
        self.jd_start = times.jd_start
        self.jd_mid = times.jd_mid
        self.jd_end = times.jd_end
        self.year_start = times.year_start
        self.year_mid = times.year_mid
        self.year_end = times.year_end
        self.exptime = times.duration
        return times
