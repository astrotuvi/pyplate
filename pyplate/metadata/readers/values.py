"""Values of one plate collected from the input files.

The readers write into a :class:`PlateValues` in the order of precedence
(DESIGN.md section 4.2); :class:`pyplate.metadata.ArchiveReader` then
builds the records from it. Per-exposure values are lists, as in the input
files; they are spread over the exposures at the end (section 4.3).
"""
from ..timeutils import combine_orig_times

# Exposure columns given as lists, one value per exposure
EXPOSURE_LIST_COLUMNS = (
    'object_name', 'object_type_code', 'ra_orig', 'dec_orig', 'ra_icrs',
    'dec_icrs', 'date_orig_start', 'date_orig_end', 'time_orig_start',
    'time_orig_end', 'jd_mid_orig', 'ut_start', 'ut_mid', 'ut_weighted',
    'ut_end', 'year_start', 'year_mid', 'year_weighted', 'year_end',
    'jd_start', 'jd_mid', 'jd_weighted', 'jd_end', 'hjd_mid', 'hjd_weighted',
    'exptime', 'num_sub', 'focus')

# Start and end times given as UT or sidereal time in their own columns
TIME_KEYS = ('ut_start_orig', 'ut_end_orig', 'st_start_orig', 'st_end_orig')

# Links to scans and logpages given as lists
LIST_LINK_KEYS = ('fn_scan', 'fn_log')

LIST_KEYS = EXPOSURE_LIST_COLUMNS + TIME_KEYS + LIST_LINK_KEYS


class PlateValues(dict):
    """Collected values by key: column names, time keys and link keys.

    Keys in :data:`LIST_KEYS` hold lists. ``exposure`` holds the values
    given explicitly for the exposures (``exposure.<column>``), which are
    not shared with the plate.
    """

    def __init__(self):
        super().__init__({k: [] for k in LIST_KEYS})
        self['numexp'] = 1
        self.exposure = {}

    def combine_times(self) -> None:
        """Combine UT and sidereal times given separately into the
        recorded start and end times (``UT ...``, ``ST ...``)."""
        self['time_orig_start'] = combine_orig_times(
            self['time_orig_start'], self['ut_start_orig'],
            self['st_start_orig'])
        self['time_orig_end'] = combine_orig_times(
            self['time_orig_end'], self['ut_end_orig'], self['st_end_orig'])
