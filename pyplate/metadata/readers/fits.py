"""Values from the FITS file of a scan."""
import numpy as np
from astropy.io import fits

from ..header import KEYWORDS, fits_structure
from .exif import file_values


def _value(header: fits.Header, key: str):
    val = header[key]
    return None if isinstance(val, fits.card.Undefined) else val


def header_values(header: fits.Header) -> tuple[dict, dict]:
    """Plate and exposure values, and scan values, from a FITS header.

    The keywords are those that :class:`PlateHeader` writes. Per-exposure
    values are taken from the numbered keywords only (``DATEOR1``, ...), as
    in the old code; the first exposure's flags from ``TIMEFLAG`` and
    ``COORFLAG``.
    """
    plate = {}
    scan = {}

    for kw in KEYWORDS:
        if kw.key is None:
            continue

        if kw.source == 'exposure' and kw.nkey:
            plate[kw.field] = [_value(header, kw.nkey.replace('n', str(i)))
                               for i in range(1, 100)
                               if kw.nkey.replace('n', str(i)) in header]
        elif kw.key in header and kw.source in ('plate', 'exposure1'):
            plate[kw.field] = _value(header, kw.key)
        elif kw.key in header and kw.source == 'scan':
            scan[kw.field] = _value(header, kw.key)

    return plate, scan


def read_fits(path: str) -> tuple[fits.Header | None, dict, dict]:
    """Read the header and the data range of a FITS file.

    Returns
    -------
    header
        The FITS header, or None if the file cannot be read.
    plate
        Plate and exposure values from the header.
    scan
        Scan values: from the header, the file time and size, and the
        minimum and maximum pixel values (scaled with BZERO and BSCALE).
    """
    try:
        data, header = fits.getdata(path, header=True,
                                    do_not_scale_image_data=True,
                                    ignore_missing_end=True)
    except Exception:
        print('Error reading {}'.format(path))
        return None, {}, {}

    plate, scan = header_values(header)
    scan.update(file_values(path))
    raw = fits_structure(header)
    minval = data.min()
    maxval = data.max()

    try:
        minval = raw['BZERO'] + raw['BSCALE'] * minval
        maxval = raw['BZERO'] + raw['BSCALE'] * maxval
    except TypeError:
        pass

    scan['minval'] = minval.item() if isinstance(minval, np.generic) else minval
    scan['maxval'] = maxval.item() if isinstance(maxval, np.generic) else maxval
    return header, plate, scan
