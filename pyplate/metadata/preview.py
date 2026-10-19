"""Preview: a row of the ``preview`` table."""
from dataclasses import dataclass

from .base import DbRecord


@dataclass
class Preview(DbRecord):
    """A preview image of a plate (``preview`` table).

    The columns are described in ``applause_dr5.yaml``.
    """

    table = 'preview'

    preview_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    preview_type: int | None = None
    filename: str | None = None
    file_format: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_datetime: str | None = None
    file_datetime: str | None = None
    file_size: int | None = None
