"""PlateHeader.from_metadata() gives the header of the old code.

The inputs are in tests/ver4/header/mock_header.json, the expected headers in
header.txt and header_wcs.txt (see tests/ver4/README.md, section "Plate
header"). Intended differences: the TZ-ORIG card, and DATE taken from the
same time as the HISTORY line.
"""
import datetime
import json
import os

from astropy.io import fits

from pyplate.config import load_config
from pyplate.metadata import (Exposure, Logpage, Plate, PlateHeader, Preview,
                              Scan)

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER_DIR = os.path.join(HERE, 'ver4', 'header')

with open(os.path.join(HEADER_DIR, 'mock_header.json')) as f:
    DATA = json.load(f)

NOW = datetime.datetime.strptime(DATA['utcnow'], '%Y-%m-%dT%H:%M:%S')
VERSION = DATA['pyplate_version']
TZ_CARD = ("TZ-ORIG = 'ST      '           / time zone of the recorded time "
           "(ST = sidereal) ")


def expected(filename):
    """Expected header with the intended differences, without the blank
    padding after END (one card more changes its length)."""
    with open(os.path.join(HEADER_DIR, filename)) as f:
        lines = f.read().split('\n')

    i = [line[:8] for line in lines].index('TME-OR2 ')
    lines.insert(i + 1, TZ_CARD)
    date = DATA['header_only']['date']['value']
    lines = [line.replace(date, DATA['utcnow']) if line.startswith('DATE    ')
             else line for line in lines]
    return '\n'.join(lines).rstrip()


def mock_records():
    only = {k: v['value'] for k, v in DATA['header_only'].items()}
    plate = Plate.from_row(DATA['plate'])
    plate.tz_orig = 'ST'

    for i, row in enumerate(DATA['plate']['exposures']):
        exp = Exposure.from_row(row)
        exp.jd_mid_orig = only['jd_avg_orig'][i]
        plate.exposures.append(exp)

    plate.scans = [Scan(filename_scan=fn) for fn in only['fn_scan']]
    plate.previews = [Preview(filename=only['fn_pre'])]
    plate.logpages = [Logpage(filename=fn)
                      for fn in [only['fn_cover']] + only['fn_log']]

    # The FITS structure and the existing HISTORY lines of the scan file
    fits_header = fits.Header()

    for key in ['fits_simple', 'fits_bitpix', 'fits_naxis', 'fits_bscale',
                'fits_bzero', 'fits_extend']:
        fits_header[DATA['header_only'][key]['fits_keyword']] = only[key]

    for line in only['fits_history']:
        fits_header.add_history(line)

    return plate, Scan.from_row(DATA['scan']), fits_header


def test_header():
    cfg = load_config(os.path.join(HEADER_DIR, 'header.toml'))
    plate, scan, fits_header = mock_records()
    h = PlateHeader.from_metadata(plate, scan, cfg, fits_header=fits_header,
                                  version=VERSION, now=NOW)
    assert h.tostring(sep='\n').rstrip() == expected('header.txt')

    wcs = fits.Header()

    for key, value, comment in DATA['wcs']:
        wcs.append((key, value, comment))

    h.insert_wcs(wcs, version=VERSION, now=NOW)
    assert h.tostring(sep='\n').rstrip() == expected('header_wcs.txt')


def test_derived_values():
    plate, _, _ = mock_records()
    only = {k: v['value'] for k, v in DATA['header_only'].items()}
    assert plate.method == only['method']
    assert [e.object_type for e in plate.exposures] == only['object_type']
    assert [e.ra for e in plate.exposures] == only['ra']
    assert [e.dec for e in plate.exposures] == only['dec']
