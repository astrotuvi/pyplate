"""Archive: a row of the ``archive`` table."""
from dataclasses import dataclass

from .base import DbRecord, related
from .logbook import Logbook


@dataclass
class Archive(DbRecord):
    """A plate archive (``archive`` table).

    The columns are described in ``applause_dr5.yaml``. ``logbooks`` (no
    column) are the logbooks of the archive.
    """

    table = 'archive'

    archive_id: int | None = None
    archive_name: str | None = None
    institute: str | None = None
    logbooks: list[Logbook] = related()
