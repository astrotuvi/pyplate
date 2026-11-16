"""The matcher interface and its result, Association."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from astropy.table import Table

if TYPE_CHECKING:
    from ..sources import SourceTable
    from .reference import ReferenceStars


@dataclass
class Association:
    """Which reference stars belong to which plate sources.

    Attributes
    ----------
    matches
        The best match of each matched source (sources without a match
        have no row): ``source_num``, ``solution_num`` (of the matched
        star), ``gaiadr4_id``, ``gaiadr4_dist`` (arcsec), ``match_radius``
        (arcsec), ``gaiadr4_neighbors`` (number of stars within the search
        radius), and the working columns ``source_index`` and
        ``ref_index`` (row of the source table and index of the reference
        star).
    neighbors
        Every source–star pair within the search radius, including the
        match; written to ``source_xmatch`` unchanged: ``source_num``,
        ``gaiadr4_id``, ``dist`` (pixels), ``solution_num``, ``x_gaia``,
        ``y_gaia`` (star position in pixels) and ``flag_xmatch`` (1 for the
        match of the source).
    """

    matches: Table
    neighbors: Table


class Matcher(Protocol):
    """Decides which reference stars belong to which sources."""

    def associate(self, sources: 'SourceTable', refs: 'ReferenceStars',
                  pairs: Table) -> Association:
        """Associate sources with reference stars, given all source–star
        pairs within the search radius (see
        :func:`~pyplate.process.crossmatch.pairs.find_pairs`)."""
        ...
