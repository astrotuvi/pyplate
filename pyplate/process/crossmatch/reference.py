"""Reference stars: catalogue stars moved to the plate epoch and projected
onto the plate with each astrometric solution."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import wcs
from astropy.table import Table

from ..log import ProcessLog

if TYPE_CHECKING:
    from ..solve import SolutionSet


@dataclass
class ReferenceStars:
    """Stars of a catalogue on the plate.

    On a plate with several exposures a star appears once per solution, so
    a reference star is identified by its catalogue row and its solution.

    Attributes
    ----------
    catalog
        The star catalogue (:class:`~pyplate.process.catalog.StarCatalog`).
    index
        Catalogue row of each reference star.
    solution_num
        Solution whose WCS gives the position of each reference star.
    xy
        Pixel coordinates of the reference stars (one row per star).
    """

    catalog: Table
    index: np.ndarray
    solution_num: np.ndarray
    xy: np.ndarray

    @classmethod
    def from_catalog(cls, catalog: Table, solution_set: 'SolutionSet',
                     epoch: float, log: ProcessLog) -> 'ReferenceStars':
        """Move the catalogue stars to epoch with their proper motions and
        convert them to pixel coordinates with each solution; keep the
        stars inside the image.

        If the catalogue has a ``solution_num`` column, only its stars of
        that solution are converted with each solution.
        """
        num_gaia = len(catalog)

        # Calculate RA and Dec for the plate epoch
        ra_ref = (catalog['ra']
                  + (epoch - catalog['ref_epoch'])
                  * catalog['pmra']
                  / np.cos(catalog['dec'] * np.pi / 180.) / 3600000.)
        dec_ref = (catalog['dec']
                   + (epoch - catalog['ref_epoch'])
                   * catalog['pmdec'] / 3600000.)
        xy_ref = np.empty((0, 2))
        sol_ref = np.empty((0,), dtype=np.int8)
        index_ref = np.empty((0,), dtype=np.int32)

        # Build a list of Gaia stars in image coordinates
        for i in np.arange(solution_set.num_solutions):
            solution = solution_set.solutions[i]

            # If there is a column named 'solution_num', then take only
            # reference stars with the current solution number
            if 'solution_num' in catalog.columns:
                mask_sol = catalog['solution_num'] == i + 1
            else:
                mask_sol = np.full(num_gaia, True)

            w = wcs.WCS(solution.header_wcs)

            try:
                xr,yr = w.all_world2pix(ra_ref[mask_sol], dec_ref[mask_sol], 1)
            except wcs.NoConvergence as e:
                log.write('Failed to convert sky coordinates to '
                          'pixel coordinates for solution {:d}: {}'
                          .format(i + 1, e))
                continue

            mask_inside = ((xr > 0.5) & (xr < solution_set.imwidth) &
                           (yr > 0.5) & (yr < solution_set.imheight))
            num_inside = mask_inside.sum()
            xyr = np.vstack((xr[mask_inside], yr[mask_inside])).T
            xy_ref = np.vstack((xy_ref, xyr))
            sol_ref = np.hstack((sol_ref, np.full(num_inside, i + 1)))
            index_ref = np.hstack((index_ref,
                                   np.arange(num_gaia)[mask_sol][mask_inside]))

        return cls(catalog=catalog, index=index_ref, solution_num=sol_ref,
                   xy=xy_ref)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def gaiadr4_id(self) -> np.ndarray:
        """Gaia source ID of each reference star."""
        return self.catalog['source_id'][self.index]
