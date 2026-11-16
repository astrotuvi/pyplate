"""NearestMatcher: each source is matched with the nearest reference star."""
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropy.table import MaskedColumn, Table, join

from .matcher import Association
from .pairs import crossmatch_cartesian, source_coords
from .tolerance import Radii

if TYPE_CHECKING:
    from ..sources import SourceTable
    from .reference import ReferenceStars


class NearestMatcher:
    """Matches each source with the nearest reference star within the match
    radius.

    Parameters
    ----------
    radii
        The match and search radii.
    """

    def __init__(self, radii: Radii):
        self.radii = radii

    def associate(self, sources: 'SourceTable', refs: 'ReferenceStars',
                  pairs: Table) -> Association:
        """Match the sources with the reference stars (see
        :class:`~pyplate.process.crossmatch.matcher.Matcher`)."""
        ind_plate, ind_ref, ds = crossmatch_cartesian(
            source_coords(sources), refs.xy, tolerance=self.radii.match)
        dist_arcsec = (ds * u.pixel * self.radii.pixel_scale).to(u.arcsec).value

        matches = Table()
        matches['source_num'] = sources['source_num'][ind_plate]
        matches['solution_num'] = refs.solution_num[ind_ref]
        matches['gaiadr4_id'] = refs.gaiadr4_id[ind_ref]
        matches['gaiadr4_dist'] = dist_arcsec
        matches['match_radius'] = np.full(len(ind_plate),
                                          self.radii.match_arcsec)
        matches['source_index'] = ind_plate
        matches['ref_index'] = ind_ref

        neighbors = self.neighbors(sources, refs, pairs, matches)

        # Number of neighbours of the matched sources
        source_num, cnt = np.unique(neighbors['source_num'].data,
                                    return_counts=True)
        counts = dict(zip(source_num, cnt))
        matches['gaiadr4_neighbors'] = np.array(
            [counts.get(s, 0) for s in matches['source_num']], dtype=int)

        return Association(matches=matches, neighbors=neighbors)

    @staticmethod
    def neighbors(sources: 'SourceTable', refs: 'ReferenceStars',
                  pairs: Table, matches: Table) -> Table:
        """The neighbour table of the pairs, with ``flag_xmatch`` set for
        the matches."""
        if len(pairs) == 0:
            # Create empty neighbors table
            return Table(names=('source_num', 'gaiadr4_id', 'dist',
                                'solution_num', 'x_gaia', 'y_gaia',
                                'flag_xmatch'),
                         dtype=('i4', 'i8', 'f4', 'i2', 'f8', 'f8', 'i1'))

        k_plate = pairs['source_index']
        k_ref = pairs['ref_index']

        # Construct neighbors table
        nbs = Table()
        nbs['source_num'] = sources['source_num'][k_plate]
        nbs['gaiadr4_id'] = refs.gaiadr4_id[k_ref]
        nbs['dist'] = pairs['dist'].data * u.pixel
        nbs['solution_num'] = refs.solution_num[k_ref]
        nbs['x_gaia'] = refs.xy[k_ref,0]
        nbs['y_gaia'] = refs.xy[k_ref,1]

        # Create the flag_xmatch column by joining the neighbors table
        # with the matches
        tab = Table()
        tab['source_num'] = matches['source_num']
        tab['gaiadr4_id'] = matches['gaiadr4_id']
        tab['flag_xmatch'] = np.ones(len(matches), dtype=np.int8)
        jtab = join(nbs, tab, keys=('source_num', 'gaiadr4_id'),
                    join_type='left')
        jtab['flag_xmatch'] = MaskedColumn(jtab['flag_xmatch']).filled(0)
        return jtab
