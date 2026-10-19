"""Plate: a row of the ``plate`` table, with its related records."""
from dataclasses import dataclass

from . import codes
from .base import DbRecord, related
from .exposure import Exposure
from .logpage import Logpage
from .preview import Preview
from .scan import Scan
from .timeutils import no_erfa_warnings


@dataclass
class Plate(DbRecord):
    """A photographic plate (``plate`` table).

    The columns are described in ``applause_dr5.yaml``. Attributes without
    a column: the related records ``exposures``, ``scans``, ``previews`` and
    ``logpages`` (cover first; the list position is the ``logpage_order``
    of the plate–logpage link).
    """

    table = 'plate'

    plate_id: int | None = None
    archive_id: int | None = None
    plate_num: str | None = None
    plate_num_orig: str | None = None
    wfpdb_id: str | None = None
    series: str | None = None
    plate_format: str | None = None
    plate_size1: float | None = None
    plate_size2: float | None = None
    emulsion: str | None = None
    filter: str | None = None
    development: str | None = None
    plate_quality: str | None = None
    plate_notes: str | None = None
    date_orig: str | None = None
    tz_orig: str | None = None
    numexp: int | None = None
    observatory: str | None = None
    site_name: str | None = None
    site_longitude: float | None = None
    site_latitude: float | None = None
    site_elevation: float | None = None
    telescope: str | None = None
    ota_name: str | None = None
    ota_diameter: float | None = None
    ota_aperture: float | None = None
    ota_foclen: float | None = None
    ota_scale: float | None = None
    instrument: str | None = None
    method_code: int | None = None
    prism: str | None = None
    prism_angle: str | None = None
    dispersion: float | None = None
    grating: str | None = None
    air_temperature: float | None = None
    sky_calmness: str | None = None
    sky_sharpness: str | None = None
    sky_transparency: str | None = None
    sky_conditions: str | None = None
    observer: str | None = None
    observer_notes: str | None = None
    notes: str | None = None
    bibcode: str | None = None
    exposures: list[Exposure] = related()
    scans: list[Scan] = related()
    previews: list[Preview] = related()
    logpages: list[Logpage] = related()

    @property
    def method(self) -> str | None:
        """Method text of ``method_code``."""
        return codes.method(self.method_code)

    @property
    def site(self) -> tuple:
        """(latitude, longitude, elevation) of the observatory."""
        return self.site_latitude, self.site_longitude, self.site_elevation

    def compute_values(self, evening_date: bool = False) -> None:
        """Compute times and coordinates of the exposures.

        As in the old code, the times of an exposure after the first are
        computed only if a start time is given for it; if only the first
        exposure is computed, its number of sub-exposures applies to all.
        If no exposure has an exposure time, the exposure times are taken
        from the start and end times.

        Parameters
        ----------
        evening_date
            The recorded dates are dates of the evening
            (``[Metadata] evening_date``).
        """
        with no_erfa_warnings():
            durations = {}

            for i, exp in enumerate(self.exposures):
                if i == 0 or exp.time_orig_start is not None:
                    durations[i] = exp.compute_times(self.tz_orig, self.site,
                                                     evening_date)

            filled = (all(e.exptime is None for e in self.exposures) and
                      any(d for d in durations.values()))

            if filled:
                for i, d in durations.items():
                    self.exposures[i].exptime = d

            # One computed value is shared by all exposures
            if list(durations) == [0]:
                first = self.exposures[0]

                for exp in self.exposures[1:]:
                    exp.num_sub = first.num_sub

                    if filled:
                        exp.exptime = first.exptime

            for exp in self.exposures:
                exp.compute_coords(self.site)
