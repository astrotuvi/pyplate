"""PlateHeader: the FITS header of a plate scan, built from the metadata."""
from __future__ import annotations
import datetime as dt
import os
import re
import textwrap
from typing import NamedTuple, Self

import numpy as np
import unidecode
from astropy.io import fits

from .._version import __version__
from ..config import Config
from .plate import Plate
from .scan import Scan
from .timeutils import format_time, utc_now


class Keyword(NamedTuple):
    """A FITS keyword written from the metadata.

    ``nkey`` is the numbered keyword for the exposures (``n`` is replaced by
    the exposure number). ``source`` says where the value comes from:

    - ``plate``, ``scan``: a column of the plate or the scan;
    - ``exposure``: a column of each exposure; ``exposure1``: a column of
      the first exposure;
    - ``derived``: a property of the plate or the exposures;
    - ``raw``: the FITS structure (:func:`fits_structure`);
    - ``header``: the ``[Header]`` configuration;
    - ``links``: file names of related records; ``date``: the current time;
      ``history``: existing HISTORY lines.
    """

    key: str | None
    nkey: str | None
    source: str
    field: str
    type: type


# In the order of the old _keyword_meta, which is the order in which the
# keywords are written before the header is sorted
KEYWORDS = [
    Keyword('SIMPLE', None, 'raw', 'SIMPLE', bool),
    Keyword('BITPIX', None, 'raw', 'BITPIX', int),
    Keyword('NAXIS', None, 'raw', 'NAXIS', int),
    Keyword('NAXIS1', None, 'scan', 'naxis1', int),
    Keyword('NAXIS2', None, 'scan', 'naxis2', int),
    Keyword('BSCALE', None, 'raw', 'BSCALE', float),
    Keyword('BZERO', None, 'raw', 'BZERO', int),
    Keyword('MINVAL', None, 'scan', 'minval', float),
    Keyword('MAXVAL', None, 'scan', 'maxval', float),
    Keyword('EXTEND', None, 'raw', 'EXTEND', bool),
    Keyword('CHECKSUM', None, 'scan', 'fits_checksum', str),
    Keyword('DATASUM', None, 'scan', 'fits_datasum', str),
    Keyword('DATEORIG', 'DATEORn', 'exposure', 'date_orig_start', str),
    Keyword('TMS-ORIG', 'TMS-ORn', 'exposure', 'time_orig_start', str),
    Keyword('TME-ORIG', 'TME-ORn', 'exposure', 'time_orig_end', str),
    Keyword('TZ-ORIG', None, 'plate', 'tz_orig', str),
    Keyword('JDA-ORIG', 'JDA-ORn', 'exposure', 'jd_mid_orig', float),
    Keyword('TIMEFLAG', None, 'exposure1', 'flag_time', str),
    Keyword('RA-ORIG', 'RA-ORn', 'exposure', 'ra_orig', str),
    Keyword('DEC-ORIG', 'DEC-ORn', 'exposure', 'dec_orig', str),
    Keyword('COORFLAG', None, 'exposure1', 'flag_coord', str),
    Keyword('OBJECT', 'OBJECTn', 'exposure', 'object_name', str),
    Keyword('OBJTYPE', 'OBJTYPn', 'derived', 'object_type', str),
    Keyword('NUMEXP', None, 'plate', 'numexp', int),
    Keyword('EXPTIME', 'EXPTIMn', 'exposure', 'exptime', float),
    Keyword('OBSERVAT', None, 'plate', 'observatory', str),
    Keyword('SITENAME', None, 'plate', 'site_name', str),
    Keyword('SITELAT', None, 'plate', 'site_latitude', float),
    Keyword('SITELONG', None, 'plate', 'site_longitude', float),
    Keyword('SITEELEV', None, 'plate', 'site_elevation', float),
    Keyword('TELESCOP', None, 'plate', 'telescope', str),
    Keyword('OTA-NAME', None, 'plate', 'ota_name', str),
    Keyword('OTA-DIAM', None, 'plate', 'ota_diameter', float),
    Keyword('OTA-APER', None, 'plate', 'ota_aperture', float),
    Keyword('FOCLEN', None, 'plate', 'ota_foclen', float),
    Keyword('PLTSCALE', None, 'plate', 'ota_scale', float),
    Keyword('INSTRUME', None, 'plate', 'instrument', str),
    Keyword('DETNAM', None, 'header', 'detector', str),
    Keyword('METHOD', None, 'derived', 'method', str),
    Keyword('FILTER', None, 'plate', 'filter', str),
    Keyword('PRISM', None, 'plate', 'prism', str),
    Keyword('PRISMANG', None, 'plate', 'prism_angle', str),
    Keyword('DISPERS', None, 'plate', 'dispersion', float),
    Keyword('GRATING', None, 'plate', 'grating', str),
    Keyword('FOCUS', 'FOCUSn', 'exposure', 'focus', float),
    Keyword('TEMPERAT', None, 'plate', 'air_temperature', float),
    Keyword('CALMNESS', None, 'plate', 'sky_calmness', str),
    Keyword('SHARPNES', None, 'plate', 'sky_sharpness', str),
    Keyword('TRANSPAR', None, 'plate', 'sky_transparency', str),
    Keyword('SKYCOND', None, 'plate', 'sky_conditions', str),
    Keyword('OBSERVER', None, 'plate', 'observer', str),
    Keyword('OBSNOTES', None, 'plate', 'observer_notes', str),
    Keyword('NOTES', None, 'plate', 'notes', str),
    Keyword('BIBCODE', None, 'plate', 'bibcode', str),
    Keyword('PLATENUM', None, 'plate', 'plate_num', str),
    Keyword('PNUMORIG', None, 'plate', 'plate_num_orig', str),
    Keyword('WFPDB-ID', None, 'plate', 'wfpdb_id', str),
    Keyword('SERIES', None, 'plate', 'series', str),
    Keyword('PLATEFMT', None, 'plate', 'plate_format', str),
    Keyword('PLATESZ1', None, 'plate', 'plate_size1', float),
    Keyword('PLATESZ2', None, 'plate', 'plate_size2', float),
    Keyword('EMULSION', None, 'plate', 'emulsion', str),
    Keyword('DEVELOP', None, 'plate', 'development', str),
    Keyword('PQUALITY', None, 'plate', 'plate_quality', str),
    Keyword('PLATNOTE', None, 'plate', 'plate_notes', str),
    Keyword('DATE-OBS', 'DT-OBSn', 'exposure', 'ut_start', str),
    Keyword('DATE-AVG', 'DT-AVGn', 'exposure', 'ut_mid', str),
    Keyword('DATE-END', 'DT-ENDn', 'exposure', 'ut_end', str),
    Keyword('JD', 'JDn', 'exposure', 'jd_start', float),
    Keyword('JD-AVG', 'JD-AVGn', 'exposure', 'jd_mid', float),
    Keyword('JD-END', 'JD-ENDn', 'exposure', 'jd_end', float),
    Keyword('HJD-AVG', 'HJD-AVn', 'exposure', 'hjd_mid', float),
    Keyword('YEAR', 'YEARn', 'exposure', 'year_start', float),
    Keyword('YEAR-AVG', 'YR-AVGn', 'exposure', 'year_mid', float),
    Keyword('YEAR-END', 'YR-ENDn', 'exposure', 'year_end', float),
    Keyword('RA', 'RAn', 'derived', 'ra', str),
    Keyword('DEC', 'DECn', 'derived', 'dec', str),
    Keyword('RA_DEG', 'RA_DEGn', 'exposure', 'ra_icrs', str),
    Keyword('DEC_DEG', 'DEC_DEn', 'exposure', 'dec_icrs', str),
    Keyword('SCANNER', None, 'scan', 'scanner', str),
    Keyword('SCANRES1', None, 'scan', 'scan_res1', int),
    Keyword('SCANRES2', None, 'scan', 'scan_res2', int),
    Keyword('PIXSIZE1', None, 'scan', 'pixel_size1', float),
    Keyword('PIXSIZE2', None, 'scan', 'pixel_size2', float),
    Keyword('SCANSOFT', None, 'scan', 'scan_software', str),
    Keyword('SCANGAM', None, 'scan', 'scan_gamma', float),
    Keyword('SCANFOC', None, 'scan', 'scan_focus', str),
    Keyword('WEDGE', None, 'scan', 'wedge', str),
    Keyword('DATESCAN', None, 'scan', 'scan_date', str),
    Keyword('SCANAUTH', None, 'scan', 'scan_author', str),
    Keyword('SCANNOTE', None, 'scan', 'scan_notes', str),
    Keyword('FILENAME', None, 'scan', 'filename_scan', str),
    Keyword(None, 'FN-SCNn', 'links', 'scans', str),
    Keyword('FN-WEDGE', None, 'scan', 'filename_wedge', str),
    Keyword('FN-PRE', None, 'links', 'preview', str),
    Keyword('FN-COVER', None, 'links', 'cover', str),
    Keyword(None, 'FN-LOGn', 'links', 'logpages', str),
    Keyword('ORIGIN', None, 'scan', 'origin', str),
    Keyword('LICENCE', None, 'header', 'licence', str),
    Keyword('DATE', None, 'date', 'date', str),
    Keyword('HISTORY', None, 'history', 'HISTORY', str),
]

# Values of the FITS structure when there is no FITS file
DEFAULT_STRUCTURE = {'SIMPLE': True, 'BITPIX': None, 'NAXIS': None,
                     'BSCALE': None, 'BZERO': None, 'EXTEND': True,
                     'HISTORY': []}

DEFAULT_COMMENTS = {
    'SIMPLE': 'file conforms to FITS standard',
    'BITPIX': 'number of bits per data pixel',
    'NAXIS': 'number of data axes',
    'NAXIS1': 'length of data axis 1',
    'NAXIS2': 'length of data axis 2',
    'BSCALE': 'physical_value = BZERO + BSCALE * array_value',
    'BZERO': 'physical_value = BZERO + BSCALE * array_value',
    'MINVAL': 'minimum image value',
    'MAXVAL': 'maximum image value',
    'EXTEND': 'file may contain extensions',
    'DATEORIG': 'recorded date of the observation',
    'DATEORn': 'recorded date of exposure {}',
    'TMS-ORIG': 'recorded time of the start of exposure 1',
    'TMS-ORn': 'recorded time of the start of exposure {}',
    'TME-ORIG': 'recorded time of the end of exposure 1',
    'TME-ORn': 'recorded time of the end of exposure {}',
    'TZ-ORIG': 'time zone of the recorded time (ST = sidereal)',
    'JDA-ORIG': 'recorded Julian date, mid-point of exposure 1',
    'JDA-ORn': 'recorded Julian date, mid-point of exposure {}',
    'TIMEFLAG': 'quality flag of recorded time',
    'RA-ORIG': 'recorded right ascension of exposure 1',
    'DEC-ORIG': 'recorded declination of exposure 1',
    'RA-ORn': 'recorded right ascension of exposure {}',
    'DEC-ORn': 'recorded declination of exposure {}',
    'COORFLAG': 'quality flag of recorded coordinates',
    'OBJECT': 'observed object or field (exposure 1)',
    'OBJECTn': 'observed object or field (exposure {})',
    'OBJTYPE': 'object type',
    'OBJTYPn': 'object type (exposure {})',
    'EXPTIME': '[s] exposure time of exposure 1',
    'EXPTIMn': '[s] exposure time of exposure {}',
    'NUMEXP': 'number of exposures of the plate',
    'OBSERVAT': 'observatory name',
    'SITENAME': 'observatory site',
    'SITELONG': '[deg] East longitude of the observatory',
    'SITELAT': '[deg] latitude of the observatory',
    'SITEELEV': '[m] elevation of the observatory',
    'TELESCOP': 'telescope name',
    'OTA-NAME': 'optical tube assembly (OTA)',
    'OTA-DIAM': '[m] diameter of the OTA',
    'OTA-APER': '[m] clear aperture of the OTA',
    'FOCLEN': '[m] focal length of the OTA',
    'PLTSCALE': '[arcsec/mm] plate scale of the OTA',
    'INSTRUME': 'instrument',
    'DETNAM': 'detector',
    'METHOD': 'method of observation',
    'FILTER': 'filter type',
    'PRISM': 'objective prism',
    'PRISMANG': 'prism angle "deg:min"',
    'DISPERS': '[Angstrom/mm] dispersion',
    'GRATING': 'grating',
    'FOCUS': 'focus position',
    'FOCUSn': 'focus position (exposure {})',
    'TEMPERAT': '[deg C] air temperature (degrees Celsius)',
    'CALMNESS': 'sky calmness (scale 1-5)',
    'SHARPNES': 'sky sharpness (scale 1-5)',
    'TRANSPAR': 'sky transparency (scale 1-5)',
    'SKYCOND': 'other sky conditions',
    'OBSERVER': 'observer name',
    'OBSNOTES': 'observer notes',
    'NOTES': 'miscellaneous notes',
    'BIBCODE': 'bibcode of a related paper',
    'PLATENUM': 'plate number in archive',
    'PNUMORIG': 'original plate number in archive',
    'WFPDB-ID': 'plate identification in the WFPDB',
    'SERIES': 'plate series',
    'PLATEFMT': 'plate format in cm',
    'PLATESZ1': '[cm] plate size along axis 1',
    'PLATESZ2': '[cm] plate size along axis 2',
    'FOV1': '[deg] field of view along axis 1',
    'FOV2': '[deg] field of view along axis 2',
    'EMULSION': 'photographic emulsion type',
    'DEVELOP': 'plate development details',
    'PQUALITY': 'quality of plate',
    'PLATNOTE': 'plate notes',
    'DATE-OBS': 'UT date of the start of exposure 1',
    'DT-OBSn': 'UT date of the start of exposure {}',
    'DATE-AVG': 'UT date of the mid-point of exposure 1',
    'DT-AVGn': 'UT date of the mid-point of exposure {}',
    'DATE-END': 'UT date of the end of exposure 1',
    'DT-ENDn': 'UT date of the end of exposure {}',
    'YEAR': 'decimal year of the start of exposure 1',
    'YEARn': 'decimal year of the start of exposure {}',
    'YEAR-AVG': 'decimal year of the mid-point of exposure 1',
    'YR-AVGn': 'decimal year of the mid-point of exposure {}',
    'YEAR-END': 'decimal year of the end of exposure 1',
    'YR-ENDn': 'decimal year of the end of exposure {}',
    'JD': 'Julian date at the start of exposure 1',
    'JDn': 'Julian date at the start of exposure {}',
    'JD-AVG': 'Julian date at the mid-point of exposure 1',
    'JD-AVGn': 'Julian date at the mid-point of exposure {}',
    'JD-END': 'Julian date at the end of exposure 1',
    'JD-ENDn': 'Julian date at the end of exposure {}',
    'HJD-AVG': 'heliocentric JD at the mid-point of exposure 1',
    'HJD-AVn': 'heliocentric JD at the mid-point of exposure {}',
    'RA': 'right ascension of pointing (J2000) "h:m:s"',
    'DEC': 'declination of pointing (J2000) "d:m:s"',
    'RAn': 'right ascension of exposure {} (J2000) "h:m:s"',
    'DECn': 'declination of exposure {} (J2000) "d:m:s"',
    'RA_DEG': '[deg] right ascension of pointing (J2000)',
    'DEC_DEG': '[deg] declination of pointing (J2000)',
    'RA_DEGn': '[deg] right ascension of exposure {} (J2000)',
    'DEC_DEn': '[deg] declination of exposure {} (J2000)',
    'SCANNER': 'scanner name',
    'SCANRES1': '[dpi] scan resolution along axis 1',
    'SCANRES2': '[dpi] scan resolution along axis 2',
    'PIXSIZE1': '[um] pixel size along axis 1',
    'PIXSIZE2': '[um] pixel size along axis 2',
    'SCANSOFT': 'scanning software',
    'SCANHCUT': 'scan high-cut value',
    'SCANLCUT': 'scan low-cut value',
    'SCANGAM': 'scan gamma value',
    'SCANFOC': 'scan focus',
    'WEDGE': 'type of photometric step-wedge',
    'DATESCAN': 'scan date and time',
    'SCANAUTH': 'author of scan',
    'SCANNOTE': 'scan notes',
    'FILENAME': 'filename of the plate scan',
    'FN-SCNn': 'filename of scan {}',
    'FN-WEDGE': 'filename of the wedge scan',
    'FN-PRE': 'filename of the preview image',
    'FN-COVER': 'filename of the plate cover image',
    'FN-LOGn': 'filename of logbook image {}',
    'DATE': 'last change of this file',
    'WCSAXES': 'number of axes in the WCS description',
    'RADESYS': 'name of the reference frame',
    'EQUINOX': 'epoch of the mean equator and equinox in years',
    'CTYPE1': 'TAN (gnomonic) projection',
    'CTYPE2': 'TAN (gnomonic) projection',
    'CUNIT1': 'physical units of CRVAL and CDELT for axis 1',
    'CUNIT2': 'physical units of CRVAL and CDELT for axis 2',
    'CRPIX1': 'reference pixel for axis 1',
    'CRPIX2': 'reference pixel for axis 2',
    'CRVAL1': 'right ascension at the reference point',
    'CRVAL2': 'declination at the reference point',
    'CD1_1': 'transformation matrix',
    'CD1_2': 'transformation matrix',
    'CD2_1': 'transformation matrix',
    'CD2_2': 'transformation matrix',
    'LONPOLE': 'native longitude of the celestial pole'}

DEFAULT_ORDER = [
    'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'BSCALE', 'BZERO',
    'MINVAL', 'MAXVAL', 'EXTEND',
    'sep:Original data of the observation',
    'DATEORIG', 'DATEORn', 'TMS-ORIG', 'TMS-ORn', 'TME-ORIG', 'TME-ORn',
    'TZ-ORIG', 'JDA-ORIG', 'JDA-ORn', 'TIMEFLAG', 'RA-ORIG', 'DEC-ORIG',
    'RA-ORn', 'DEC-ORn', 'COORFLAG', 'OBJECT', 'OBJECTn', 'OBJTYPE',
    'OBJTYPn', 'EXPTIME', 'NUMEXP', 'EXPTIMn', 'OBSERVAT', 'SITENAME',
    'SITELONG', 'SITELAT', 'SITEELEV', 'TELESCOP', 'OTA-NAME', 'OTA-DIAM',
    'OTA-APER', 'FOCLEN', 'PLTSCALE', 'INSTRUME', 'DETNAM', 'METHOD',
    'FILTER', 'PRISM', 'PRISMANG', 'DISPERS', 'GRATING', 'FOCUS', 'FOCUSn',
    'TEMPERAT', 'CALMNESS', 'SHARPNES', 'TRANSPAR', 'SKYCOND', 'OBSERVER',
    'OBSNOTES', 'NOTES', 'BIBCODE',
    'sep:Photographic plate',
    'PLATENUM', 'PNUMORIG', 'WFPDB-ID', 'SERIES', 'PLATEFMT', 'PLATESZ1',
    'PLATESZ2', 'FOV1', 'FOV2', 'EMULSION', 'DEVELOP', 'PQUALITY',
    'PLATNOTE',
    'sep:Computed data of the observation',
    'DATE-OBS', 'DT-OBSn', 'DATE-AVG', 'DT-AVGn', 'DATE-END', 'DT-ENDn',
    'YEAR', 'YEARn', 'YEAR-AVG', 'YR-AVGn', 'YEAR-END', 'YR-ENDn', 'JD',
    'JDn', 'JD-AVG', 'JD-AVGn', 'JD-END', 'JD-ENDn', 'HJD-AVG', 'HJD-AVn',
    'RA', 'DEC', 'RAn', 'DECn', 'RA_DEG', 'DEC_DEG', 'RA_DEGn', 'DEC_DEn',
    'sep:Scan',
    'SCANNER', 'SCANRES1', 'SCANRES2', 'PIXSIZE1', 'PIXSIZE2', 'SCANSOFT',
    'SCANHCUT', 'SCANLCUT', 'SCANGAM', 'SCANFOC', 'WEDGE', 'DATESCAN',
    'SCANAUTH', 'SCANNOTE',
    'sep:Data files',
    'FILENAME', 'FN-SCNn', 'FN-WEDGE', 'FN-PRE', 'FN-COVER', 'FN-LOGn',
    'ORIGIN', 'DATE',
    'sep:WCS',
    'sep:Licence',
    'LICENCE',
    'sep:Acknowledgements',
    'sep:History',
    'HISTORY',
    'sep:Checksums',
    'CHECKSUM', 'DATASUM',
    'sep:']


def fits_structure(header: fits.Header | None) -> dict:
    """SIMPLE, BITPIX, NAXIS, BSCALE, BZERO, EXTEND and HISTORY of a FITS
    header (defaults without a header).

    As in the old code, BSCALE and BZERO are set to the values for unsigned
    integers of the given BITPIX when the file has a BZERO.
    """
    out = dict(DEFAULT_STRUCTURE)

    if header is None:
        return out

    for key in ['SIMPLE', 'BITPIX', 'NAXIS', 'BSCALE', 'BZERO', 'EXTEND']:
        if key in header:
            val = header[key]
            out[key] = None if isinstance(val, fits.card.Undefined) else val

    if 'HISTORY' in header:
        out['HISTORY'] = list(header['HISTORY'])

    zeros = {8: -128, 16: 32768, 32: 2147483648, 64: 9223372036854775808}

    if out['BZERO'] and out['BITPIX'] in zeros:
        out['BSCALE'] = 1.0
        out['BZERO'] = zeros[out['BITPIX']]

    return out


def _empty_card(key: str) -> fits.Card:
    return fits.Card.fromstring('{:8s}='.format(key))


class PlateHeader(fits.Header):
    """FITS header of a plate scan.

    Build it with :meth:`from_metadata`; :meth:`insert_wcs` adds the
    astrometric solution, and :meth:`output_to_file` and
    :meth:`output_to_fits` write it out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fits_dir = ''
        self.write_fits_dir = ''
        self.write_header_dir = ''
        self.create_checksum = False
        self.german_transliteration = False
        self.keyword_comments = {}
        self.keyword_order = None
        self.fits_datetime = None
        self.fits_size = None
        self.fits_checksum = None
        self.fits_datasum = None

    @classmethod
    def from_metadata(cls, plate: Plate, scan: Scan | None, cfg: Config,
                      fits_header: fits.Header | None = None,
                      version: str | None = None,
                      now: dt.datetime | None = None) -> Self:
        """Build the header of a scan from the plate and scan metadata.

        Parameters
        ----------
        plate
            The plate, with its exposures and related records.
        scan
            The scan (FILENAME, scanner values, image size and range).
        cfg
            Configuration: ``[Header]``, the fixed keyword values and
            comments, ``[Metadata] german_transliteration`` and the
            directories in ``[Files]``.
        fits_header
            Header of the scan file, for the FITS structure (SIMPLE,
            BITPIX, NAXIS, BSCALE, BZERO, EXTEND) and its HISTORY lines.
        version
            PyPlate version written into HISTORY (default: the package
            version).
        now
            Time written into DATE and HISTORY (default: the current UTC
            time).
        """
        h = cls()
        h.fits_dir = cfg.files.fits_dir
        h.write_fits_dir = cfg.files.write_fits_dir
        h.write_header_dir = cfg.files.write_header_dir
        h.create_checksum = cfg.files.create_checksum
        h.german_transliteration = cfg.metadata.german_transliteration
        h.keyword_comments = cfg.header.keyword_comments
        h.keyword_order = cfg.header.keyword_order
        version = __version__ if version is None else version
        now = format_time(utc_now() if now is None else now)

        h._update_from_metadata(plate, scan, cfg, fits_structure(fits_header),
                                now)
        h.add_history('Header updated with PyPlate v{} at {}'
                      .format(version, now))
        h.format()
        h._assign_values(cfg)
        h.format()
        return h

    def _values(self, kw: Keyword, plate: Plate, scan: Scan | None,
                cfg: Config, raw: dict, now: str):
        """Value of a keyword: a list for per-exposure keywords and links."""
        if kw.source == 'plate':
            return getattr(plate, kw.field)
        if kw.source == 'scan':
            return getattr(scan, kw.field) if scan is not None else None
        if kw.source == 'exposure1':
            return (getattr(plate.exposures[0], kw.field)
                    if plate.exposures else None)
        if kw.source in ('exposure', 'derived') and kw.nkey:
            values = [getattr(e, kw.field) for e in plate.exposures]
            return [] if all(v is None for v in values) else values
        if kw.source == 'derived':
            return getattr(plate, kw.field)
        if kw.source in ('raw', 'history'):
            return raw[kw.field]
        if kw.source == 'header':
            return getattr(cfg.header, kw.field)
        if kw.source == 'date':
            return now

        # Links
        if kw.field == 'scans':
            return [s.filename_scan for s in plate.scans]
        if kw.field == 'preview':
            return plate.previews[0].filename if plate.previews else None
        if kw.field == 'cover':
            return (plate.logpages[0].filename
                    if plate.logpages and plate.logpages[0] else None)
        if kw.field == 'logpages':
            return [lp.filename if lp else None for lp in plate.logpages[1:]]

        raise ValueError(kw)

    def _update_from_metadata(self, plate, scan, cfg, raw, now):
        for kw in KEYWORDS:
            value = self._values(kw, plate, scan, cfg, raw, now)

            if kw.key == 'HISTORY':
                # Written before HISTORY, as in the old code
                self._add_acknowledgements(cfg.header.acknowledgements)

            if kw.nkey or kw.key == 'HISTORY':
                self._update_keyword_list(kw.key, kw.nkey, kw.type, value)
            else:
                self._update_keyword(kw.key, kw.type, value)

    def _add_acknowledgements(self, ack):
        """Acknowledgements as COMMENT cards in their own section."""
        ack = ack or ''
        ack = '\n\n'.join([textwrap.fill(para, 72)
                           for para in ack.split('\n\n')])
        ack_sep = ' Acknowledgements'.rjust(72, '-')
        self.append(('', ack_sep), end=True)

        for line in ack.split('\n'):
            self.append(('COMMENT', line), end=True)

        self.append(('', '-' * 72), end=True)

    def _update_keyword(self, key, valtype, value):
        if isinstance(value, np.generic):
            value = value.item()

        if value or (valtype in (int, float) and value == 0):
            if isinstance(value, str):
                if self.german_transliteration:
                    value = (value.replace('ä', 'ae').replace('Ä', 'Ae')
                             .replace('ö', 'oe').replace('Ö', 'Oe')
                             .replace('ü', 'ue').replace('Ü', 'Ue')
                             .replace('ß', 'ss'))

                value = unidecode.unidecode(value)

                # Workaround for Astropy treating a colon in a string as
                # the assignment of a numerical value
                if ':' in value:
                    self.set(key, '')

            self.set(key, value)
        elif key not in self:
            if valtype is str:
                self.set(key, '')
            else:
                self.append(_empty_card(key))

    def _update_keyword_list(self, skey, nkey, valtype, value):
        if value:
            if isinstance(value, list):
                if skey:
                    if skey == 'HISTORY':
                        for item in value:
                            self.add_history(item)
                    else:
                        self.set(skey, value[0])

                if (len(value) > 1 or (len(value) == 1 and not skey)) and nkey:
                    for i, val in enumerate(value):
                        keyname = nkey.replace('n', '{}').format(i + 1)

                        if val:
                            self.set(keyname, val)
                        else:
                            self.append(_empty_card(keyname))
            elif skey:
                self.set(skey, value)
        elif skey and skey not in self:
            if valtype is str:
                self.set(skey, '')
            else:
                self.append(_empty_card(skey))

    def _assign_values(self, cfg: Config):
        """Fixed keyword values from the configuration."""
        for values in [cfg.header.integer_values, cfg.header.float_values,
                       cfg.header.string_values]:
            for key, value in values.items():
                self.set(key, value)

    def update_comments(self):
        """Default comments, then the comments from the configuration."""
        for key in self:
            if key in DEFAULT_COMMENTS:
                self.comments[key] = DEFAULT_COMMENTS[key]
            elif key and key[-1].isdigit():
                # EXPTIM1 -> EXPTIMn
                num = re.findall(r'\d+', key)[-1]
                nkey = key.replace(num, 'n')

                if nkey in DEFAULT_COMMENTS:
                    self.comments[key] = DEFAULT_COMMENTS[nkey].format(num)

        for key, comment in self.keyword_comments.items():
            if key in self:
                self.comments[key] = comment

    def rewrite(self):
        """Rewrite the cards with proper formatting."""
        h = self.copy()
        self.clear()

        for k, v, c in h.cards:
            # Remove empty COMMENT and HISTORY cards
            if k in ('COMMENT', 'HISTORY') and not v:
                continue

            # A blank keyword that is not a separator becomes COMMENT
            if not k and not re.match('---', v):
                k = 'COMMENT'

            # Null values separately, for proper comment alignment
            if isinstance(v, fits.card.Undefined):
                self.append(fits.Card.fromstring(k.ljust(8) + '='.ljust(22)
                                                 + ' / ' + c))
            elif (isinstance(v, str) and k not in ('COMMENT', 'HISTORY')
                  and ':' in v):
                # Workaround for Astropy treating a colon in a string as the
                # assignment of a numerical value
                self.append((k, '', c), bottom=True)
                self[k] = v
            else:
                self.append((k, v, c), bottom=True)

            # Pad empty strings
            if (isinstance(v, str) and not v and k
                    and k not in ('COMMENT', 'HISTORY')):
                self[k] = 'a'
                self[k] = ' '

    def reorder(self):
        """Sort the cards in the order of ``["FITS keyword order"]`` or
        :data:`DEFAULT_ORDER`, with section separators."""
        orderkeys = self.keyword_order or DEFAULT_ORDER
        h = self.copy()
        self.clear()

        for key in orderkeys:
            if key[:3] == 'sep':
                if key[4:] == '':
                    sepstr = '-' * 72
                else:
                    sepstr = (' ' + key[4:]).rjust(72, '-')

                self.append(('', sepstr), end=True)

                # Copy the WCS block
                wcs_sep = ' WCS'.rjust(72, '-')

                if key.strip().endswith('WCS') and wcs_sep in h.values():
                    wcs_ind = list(h.values()).index(wcs_sep) + 1

                    for c in h.cards[wcs_ind:]:
                        if c[0]:
                            self.append(c, end=True)
                            del h[wcs_ind]
                        else:
                            break

                # Copy the acknowledgements
                ack_sep = ' Acknowledgements'.rjust(72, '-')

                if (key.strip().endswith('Acknowledgements') and
                        ack_sep in h.values()):
                    ack_ind = list(h.values()).index(ack_sep) + 1

                    for c in h.cards[ack_ind:]:
                        if c[0] == 'COMMENT':
                            self.append(c, end=True)
                            del h[ack_ind]
                        else:
                            break
            elif 'n' in key:
                for n in range(1, 100):
                    keystr = key.replace('n', str(n))

                    if keystr in h:
                        self.append(h.cards[h.index(keystr)], end=True)
                        h.remove(keystr)
            elif key == 'HISTORY':
                if key in h:
                    for item in h[key]:
                        self.append(('HISTORY', item), end=True)

                    del h[key]
            elif key in h:
                self.append(h.cards[h.index(key)], end=True)
                h.remove(key)

        # Copy the remaining cards that are not existing separators
        for c in h.cards:
            k, v, comment = c

            if k or v not in self['']:
                self.append(c, bottom=True)

    def format(self):
        """Update comments, rewrite and sort the cards."""
        self.update_comments()
        self.rewrite()
        self.reorder()

    def insert_wcs(self, wcshead: fits.Header, version: str | None = None,
                   now: dt.datetime | None = None) -> None:
        """Insert WCS cards after the WCS separator.

        HISTORY cards are copied, COMMENT cards dropped, and keywords that
        are already in the header are not inserted. Adds a HISTORY line and
        sets DATE.

        Parameters
        ----------
        wcshead
            Header with the WCS.
        version, now
            Version and time for HISTORY and DATE, as in
            :meth:`from_metadata`.
        """
        version = __version__ if version is None else version
        now = format_time(utc_now() if now is None else now)
        wcs_sep = ' WCS'.rjust(72, '-')

        if wcs_sep in self.values():
            wcs_ind = list(self.values()).index(wcs_sep) + 1

            for c in wcshead.cards:
                if c[0] == 'HISTORY' or c[0] == '':
                    self.insert(wcs_ind, c)
                    wcs_ind += 1
                elif c[0] == 'COMMENT':
                    pass
                elif c[0] not in self:
                    self.insert(wcs_ind, c)
                    wcs_ind += 1

            self.add_history('WCS added with PyPlate v{} at {}'
                             .format(version, now))
            self.set('DATE', now)

    def output_to_fits(self, filename: str, checksum: bool | None = None):
        """Write the header to the FITS file in ``[Files] write_fits_dir``,
        copying the file from ``fits_dir`` if it is not there yet."""
        if checksum is None:
            checksum = self.create_checksum

        fn_fits = os.path.join(self.fits_dir, filename)
        fn_out = os.path.join(self.write_fits_dir, filename)

        if os.path.exists(fn_out):
            fitsfile = fits.open(fn_out, mode='update',
                                 do_not_scale_image_data=True,
                                 ignore_missing_end=True)
            fitsfile[0].header = self.copy()
            fitsfile.flush()
        else:
            if not os.path.exists(fn_fits):
                print('File does not exist: {}'.format(fn_fits))

            fitsfile = fits.open(fn_fits, do_not_scale_image_data=True,
                                 ignore_missing_end=True)
            fitsfile[0].header = self.copy()
            os.makedirs(self.write_fits_dir, exist_ok=True)

            try:
                fitsfile.writeto(fn_out, checksum=checksum)
            except IOError:
                print('Could not write to {}'.format(fn_out))

        fitsfile.close()
        del fitsfile

        # Size and time of the updated FITS file
        try:
            mtime = dt.datetime.fromtimestamp(os.path.getmtime(fn_out),
                                              dt.timezone.utc)
            self.fits_datetime = format_time(mtime)
            self.fits_size = os.path.getsize(fn_out)

            if checksum:
                h = fits.getheader(fn_out)
                self.fits_checksum = h['CHECKSUM']
                self.fits_datasum = h['DATASUM']
        except IOError:
            pass

    def output_to_file(self, filename: str):
        """Write the header to a text file in ``[Files] write_header_dir``
        (FITS blocks, without separators)."""
        os.makedirs(self.write_header_dir or '.', exist_ok=True)
        fn_out = os.path.join(self.write_header_dir, filename)

        if os.path.exists(fn_out):
            os.remove(fn_out)

        try:
            self.tofile(fn_out, sep='', endcard=True, padding=True)
        except IOError:
            print('Error writing {}'.format(fn_out))
