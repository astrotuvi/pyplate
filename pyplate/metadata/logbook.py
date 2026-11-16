"""Logbook: a row of the ``logbook`` table."""
from dataclasses import dataclass

from .base import DbRecord


@dataclass
class Logbook(DbRecord):
    """An observation logbook (``logbook`` table).

    The columns are described in ``applause_dr5.yaml``.
    """

    table = 'logbook'

    logbook_id: int | None = None
    archive_id: int | None = None
    logbook_num: str | None = None
    logbook_title: str | None = None
    logbook_type: int | None = None
    logbook_notes: str | None = None
