"""Logpage: a row of the ``logpage`` table."""
from dataclasses import dataclass

from .base import DbRecord, related


@dataclass
class Logpage(DbRecord):
    """An image of a logbook page or a plate cover (``logpage`` table).

    The columns are described in ``applause_dr5.yaml``. ``logbook_num``
    (no column) is the number of the logbook the page belongs to, by which
    ``logbook_id`` is looked up.
    """

    table = 'logpage'

    logpage_id: int | None = None
    logbook_id: int | None = None
    archive_id: int | None = None
    logpage_type: int | None = None
    page_num: int | None = None
    logpage_order: int | None = None
    filename: str | None = None
    file_format: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_datetime: str | None = None
    file_datetime: str | None = None
    file_size: int | None = None
    logbook_num: str | None = related(default=None)
