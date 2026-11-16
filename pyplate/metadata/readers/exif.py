"""File and image metadata of previews and logpages."""
import datetime as dt
import os

import pytz
from PIL import Image

from ..timeutils import format_time

_FORMATS = {'JPG': 'JPEG', 'JPEG': 'JPEG', 'TIF': 'TIFF', 'TIFF': 'TIFF',
            'PNG': 'PNG'}


def format_from_name(filename: str) -> str | None:
    """Image format from the file name extension (JPEG, PNG or TIFF)."""
    return _FORMATS.get(filename.split('.')[-1].upper())


def file_values(path: str) -> dict:
    """``file_datetime`` (modification time, UTC) and ``file_size``."""
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc)
    return {'file_datetime': format_time(mtime),
            'file_size': os.path.getsize(path)}


def image_values(path: str, exif_timezone: str | None = None) -> dict:
    """File, size, format and EXIF date of an image.

    The EXIF date is written as ``YYYY-MM-DD HH:MM:SS``; with
    ``exif_timezone`` it is converted from that time zone to UTC and written
    as ``YYYY-MM-DDTHH:MM:SS``. Returns an empty dictionary if the file does
    not exist.
    """
    if not os.path.exists(path):
        return {}

    out = file_values(path)

    try:
        im = Image.open(path)
    except Exception:
        return out

    out['image_width'], out['image_height'] = im.size
    out['file_format'] = im.format
    exif_datetime = None

    if im.format in ('JPEG', 'TIFF'):
        exif_datetime = im.getexif().get(306)

    if exif_datetime:
        if exif_datetime[4] == ':':
            exif_datetime = '{} {}'.format(exif_datetime[:10]
                                           .replace(':', '-'),
                                           exif_datetime[11:])

        if exif_timezone:
            dt_exif = dt.datetime.strptime(exif_datetime, '%Y-%m-%d %H:%M:%S')

            try:
                dt_local = pytz.timezone(exif_timezone).localize(dt_exif)
                exif_datetime = format_time(dt_local.astimezone(pytz.utc))
            except pytz.exceptions.UnknownTimeZoneError:
                pass

        out['image_datetime'] = exif_datetime

    return out
