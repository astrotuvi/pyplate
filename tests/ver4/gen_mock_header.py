"""Generate a plate FITS header from mock metadata into tests/ver4/header/
(no FITS file needed).

Follows the header steps of PlatePipeline.single_image(): PlateHeader with
assign_conf(), assign_platemeta(), update_from_platemeta(), assign_values(),
update_comments(), rewrite() and reorder(), written as header.txt; then
insert_wcs() with a mock WCS header, written as header_wcs.txt. The files
contain the header cards, one per line (Header.tostring(sep='\\n')).

Every metadata key that has a FITS keyword gets a value; the plate has two
exposures. Values that the redesigned code will derive (RA/DEC strings,
OBJTYPE, METHOD) are made consistent with their sources. The pyplate version
and the current time, which the header code writes into HISTORY and DATE,
are replaced by fixed values.

The inputs are written to tests/ver4/header/mock_header.json, in database
column names where a column exists. See README.md.

Runs against the pyplate code as of commit ed2a193 (DR4 schema, before the
redesign described in DESIGN.md). Run from the repository root:

    PYTHONPATH=. python tests/ver4/gen_mock_header.py
"""
import configparser
import datetime
import json
import os
from types import SimpleNamespace

from astropy import units
from astropy.coordinates import Angle
from astropy.io import fits

from pyplate import metadata
from pyplate.database.db_pgsql import DB_pgsql
from sqlrecord import ROOT, recording_platedb

OUT_DIR = os.path.join(ROOT, 'tests', 'ver4', 'header')
PYPLATE_VERSION = 'mock-version'
UTCNOW = datetime.datetime(2001, 2, 3, 4, 5, 6)


class FixedDatetime(datetime.datetime):
    @classmethod
    def utcnow(cls):
        return UTCNOW


# Fixed values instead of the package version and the current time
metadata.__version__ = PYPLATE_VERSION
metadata.dt = SimpleNamespace(datetime=FixedDatetime)

# Metadata keys that differ from column names, as in PlateDB.write_plate()
PLATE_KEYS = {'flag_coord': 'coord_flag', 'ra_icrs': 'ra_deg',
              'dec_icrs': 'dec_deg', 'date_orig_start': 'date_orig',
              'date_orig_end': 'date_orig_end', 'time_orig_start': 'tms_orig',
              'time_orig_end': 'tme_orig', 'flag_time': 'time_flag',
              'ut_start': 'date_obs', 'ut_mid': 'date_avg',
              'ut_weighted': 'date_weighted', 'ut_end': 'date_end',
              'year_start': 'year', 'year_mid': 'year_avg', 'jd_start': 'jd',
              'jd_mid': 'jd_avg', 'hjd_mid': 'hjd_avg', 'num_sub': 'numsub'}

# Metadata keys that differ from column names, as in PlateDB.write_scan()
SCAN_KEYS = {'filename_scan': 'filename', 'filename_wedge': 'fn_wedge',
             'naxis1': 'fits_naxis1', 'naxis2': 'fits_naxis2',
             'minval': 'fits_minval', 'maxval': 'fits_maxval',
             'pixel_size1': 'pix_size1', 'pixel_size2': 'pix_size2',
             'scan_date': 'datescan', 'file_datetime': 'fits_datetime',
             'file_size': 'fits_size'}

# Explicit mock values; other keys get generated values
NUMEXP = 2
VALUES = {
    'fits_simple': True, 'fits_bitpix': 16, 'fits_naxis': 2,
    'fits_naxis1': 10000, 'fits_naxis2': 12000, 'fits_bscale': 1.0,
    'fits_bzero': 32768, 'fits_minval': 120.0, 'fits_maxval': 65000.0,
    'fits_extend': True,
    'date_orig': ['1950-01-02', '1950-01-02'],
    'tms_orig': ['21:30:00', '22:10:00|22:17:00'],
    'tme_orig': ['22:00:00', '22:15:00|22:22:00'],
    'jd_avg_orig': [2433284.40625, 2433284.42778],
    'ra_orig': ['10:00:00', '11:30:00'],
    'dec_orig': ['+20:00:00', '-05:30:00'],
    'ra_deg': [150.1235, 172.8417],
    'dec_deg': [19.9012, -5.6234],
    'object_name': ['M 81', 'Field 2'],
    'object_type_code': ['G1', 'F'],
    'numexp': NUMEXP,
    'exptime': [1800.0, 600.0],
    'method_code': 1,
    'focus': [12.5, None],
    'date_obs': ['1950-01-02T21:30:00', '1950-01-02T22:10:00'],
    'date_avg': ['1950-01-02T21:45:00', '1950-01-02T22:16:00'],
    'date_end': ['1950-01-02T22:00:00', '1950-01-02T22:22:00'],
    'jd': [2433284.39583, 2433284.42361],
    'jd_avg': [2433284.40625, 2433284.42778],
    'jd_end': [2433284.41667, 2433284.43194],
    'hjd_avg': [2433284.41012, 2433284.43165],
    'year': [1950.00527, 1950.00535],
    'year_avg': [1950.0053, 1950.00536],
    'year_end': [1950.00533, 1950.00537],
    'observer': 'Jõgi Müller',
    'date': '2001-01-01T00:00:00',
    'fits_history': ['First history line', 'Second history line'],
    'fits_acknowledgements': (
        'This is the first paragraph of the acknowledgements. It is long '
        'enough to be wrapped into more than one line of the FITS header.'
        '\n\nThis is the second paragraph.'),
}

MOCK_CONF = {
    'Integer values': {'MOCKINT': '7'},
    'Float values': {'MOCKFLT': '2.5'},
    'String values': {'MOCKSTR': 'mock string value'},
    'FITS keyword comments': {'OBSERVER': 'observer (comment from config)'},
}


_counter = [0]


def mock_value(key, valtype, i=0):
    """Distinct generated value: 101, 102, ... or 201.25, 202.25, ..."""
    _counter[0] += 1

    if valtype is int:
        return 100 + _counter[0]
    if valtype is float:
        return 200.25 + _counter[0]
    if valtype is bool:
        return True
    return '{}{}'.format(key, i)


def derived(meta):
    """Set values derived from other values, as the old compute_values()."""
    meta['ra'] = [Angle(d, units.deg).to_string(unit=units.hour, sep=':',
                                                precision=1, pad=True)
                  for d in meta['ra_deg']]
    meta['dec'] = [Angle(d, units.deg).to_string(unit=units.deg, sep=':',
                                                 precision=1, pad=True)
                   for d in meta['dec_deg']]
    meta['object_type'] = [metadata.Plate._object_types[c]
                           for c in meta['object_type_code']]
    meta['method'] = metadata.Plate._method_codes[meta['method_code']]


def mock_plate():
    plate = metadata.Plate()

    for key, (valtype, is_list, _, skey, nkey) in \
            metadata._keyword_meta.items():
        if key in VALUES:
            plate[key] = VALUES[key]
        elif skey or nkey:
            if is_list:
                plate[key] = [mock_value(key, valtype, i)
                              for i in range(NUMEXP)]
            else:
                plate[key] = mock_value(key, valtype)

    derived(plate)
    return plate


def mock_conf():
    conf = configparser.ConfigParser()
    conf.optionxform = str

    for section, items in MOCK_CONF.items():
        conf.add_section(section)

        for k, v in items.items():
            conf.set(section, k, v)

    return conf


def mock_wcs():
    h = fits.Header()
    h['CTYPE1'] = ('RA---TAN', 'TAN projection')
    h['CTYPE2'] = ('DEC--TAN', 'TAN projection')
    h['CRVAL1'] = (150.1235, 'RA of reference point')
    h['CRVAL2'] = (19.9012, 'Dec of reference point')
    h['CRPIX1'] = (5000.5, 'X reference pixel')
    h['CRPIX2'] = (6000.5, 'Y reference pixel')
    h['CD1_1'] = (-0.000277, 'Transformation matrix')
    h['CD1_2'] = (0.0, 'Transformation matrix')
    h['CD2_1'] = (0.0, 'Transformation matrix')
    h['CD2_2'] = (0.000277, 'Transformation matrix')
    h['NAXIS1'] = 10000
    h['HISTORY'] = 'WCS created by mock solver'
    h['COMMENT'] = 'This comment is not copied'
    return h


def make_header(plate, conf, wcs):
    """Header steps of PlatePipeline.single_image()."""
    h = metadata.PlateHeader()
    h.assign_conf(conf)
    h.assign_platemeta(plate)
    h.update_from_platemeta()
    h.assign_values()
    h.update_comments()
    h.rewrite()
    h.reorder()
    text = h.tostring(sep='\n')
    h.insert_wcs(wcs)
    return text, h.tostring(sep='\n')


# Export of the inputs to JSON, in database column names

def encode(val):
    if isinstance(val, fits.Header):
        return [[c.keyword, c.value, c.comment] for c in val.cards]
    return val


def export_inputs(plate, conf, wcs):
    db = recording_platedb('pgsql', DB_pgsql)
    cols = {t: list(db.get_table_dict(t))
            for t in ['plate', 'exposure', 'scan']}
    used = set()
    out_plate = {}
    exposures = [{'exposure_num': i + 1} for i in range(NUMEXP)]
    out_scan = {}

    for col in cols['exposure']:
        key = PLATE_KEYS.get(col, col)

        if key in plate and plate[key] not in (None, []) and col != 'plate_id':
            for i, exp in enumerate(exposures):
                val = plate[key]
                exp[col] = val[i] if isinstance(val, list) else val
            used.add(key)

    for col in cols['plate']:
        key = PLATE_KEYS.get(col, col)

        if key in plate and plate[key] not in (None, []) and col != 'plate_id':
            val = plate[key]
            out_plate[col] = val[0] if isinstance(val, list) else val
            used.add(key)

    for col in cols['scan']:
        key = SCAN_KEYS.get(col, col)

        if key in plate and plate[key] not in (None, []):
            out_scan[col] = plate[key]
            used.add(key)

    out_plate['exposures'] = exposures

    # Metadata keys without a database column
    header_only = {}

    for key, meta in metadata._keyword_meta.items():
        if key not in used and plate[key] not in (None, [], ''):
            header_only[key] = {'fits_keyword': meta[3] or meta[4],
                                'value': plate[key]}

    for key in ['ra', 'dec', 'object_type', 'method']:
        header_only[key]['derived'] = True

    return {
        'description': 'Inputs of the plate header in header.txt and '
                       'header_wcs.txt, generated by gen_mock_header.py. '
                       'See README.md.',
        'pyplate_version': PYPLATE_VERSION,
        'utcnow': UTCNOW.strftime('%Y-%m-%dT%H:%M:%S'),
        'plate': out_plate,
        'scan': out_scan,
        'header_only': header_only,
        'config': MOCK_CONF,
        'wcs': encode(wcs),
    }


if __name__ == '__main__':
    plate = mock_plate()
    conf = mock_conf()
    wcs = mock_wcs()
    inputs = export_inputs(plate, conf, wcs)
    text, text_wcs = make_header(plate, conf, wcs)

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, 'header.txt'), 'w') as f:
        f.write(text + '\n')

    with open(os.path.join(OUT_DIR, 'header_wcs.txt'), 'w') as f:
        f.write(text_wcs + '\n')

    with open(os.path.join(OUT_DIR, 'mock_header.json'), 'w') as f:
        json.dump(inputs, f, indent=1, ensure_ascii=False)
        f.write('\n')

    print(len(text.split('\n')), 'and', len(text_wcs.split('\n')), 'cards')
