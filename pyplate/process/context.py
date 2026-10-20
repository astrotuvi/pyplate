"""ProcessContext: the state that the steps of a plate process share."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.io import fits

from ..config import Config
from ..metadata import Plate, Scan
from .log import ProcessLog

if TYPE_CHECKING:
    from .process import Process


@dataclass
class ProcessContext:
    """State shared by the steps of a plate process (DESIGN.md 6.1).

    Steps receive the context and explicit inputs, and return result
    objects. :class:`~pyplate.process.process.PlateProcess` sets the image
    size, the plate epoch, the scratch directory and the log in
    ``setup()``.

    Attributes
    ----------
    header
        The plate header made from the metadata, or the header of the FITS
        file if that one lacks the image size.
    plate_epoch
        Epoch of the plate (decimal year).
    work_dir
        Scratch directory of the process, removed when it finishes.
    record
        The ``process`` table row, updated as steps finish.
    """

    cfg: Config
    filename: str
    record: 'Process'
    plate: Plate | None = None
    scan: Scan | None = None
    header: fits.Header | None = None
    imwidth: int | None = None
    imheight: int | None = None
    plate_epoch: float = 1950.
    work_dir: Path | None = None
    log: ProcessLog | None = None

    @property
    def basefn(self) -> str:
        """File name of the scan without extension."""
        return os.path.splitext(self.filename)[0]

    @property
    def plate_year(self) -> int:
        """Year of the plate epoch."""
        return int(self.plate_epoch)
