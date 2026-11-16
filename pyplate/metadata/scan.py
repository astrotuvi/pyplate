"""Scan: a row of the ``scan`` table."""
from dataclasses import dataclass

from .base import DbRecord


@dataclass
class Scan(DbRecord):
    """A scan of a plate (``scan`` table).

    The columns are described in ``applause_dr5.yaml``.
    """

    table = 'scan'

    scan_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    filename_scan: str | None = None
    filename_wedge: str | None = None
    naxis1: int | None = None
    naxis2: int | None = None
    minval: float | None = None
    maxval: float | None = None
    scanner: str | None = None
    scan_res1: int | None = None
    scan_res2: int | None = None
    pixel_size1: float | None = None
    pixel_size2: float | None = None
    scan_software: str | None = None
    scan_gamma: float | None = None
    scan_focus: str | None = None
    wedge: str | None = None
    scan_date: str | None = None
    scan_author: str | None = None
    scan_notes: str | None = None
    origin: str | None = None
    file_datetime: str | None = None
    file_size: int | None = None
    fits_checksum: str | None = None
    fits_datasum: str | None = None
