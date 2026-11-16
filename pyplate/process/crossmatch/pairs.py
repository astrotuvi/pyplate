"""Source–star pairs within a radius, in pixel coordinates."""
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table

if TYPE_CHECKING:
    from ..sources import SourceTable
    from .reference import ReferenceStars

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT


def crossmatch_cartesian(coords_image: np.ndarray, coords_ref: np.ndarray,
                         tolerance: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crossmatch source coordinates with reference-star coordinates: the
    nearest reference point of each point, if it is closer than tolerance.

    Parameters
    ----------
    coords_image
        The coordinates of points to match
    coords_ref
        The coordinates of reference points to match
    tolerance
        Crossmatch distance in pixels (default: 5)

    Returns
    -------
    ind_image, ind_ref, ds
        Indices of the matched points and reference points, and their
        distances.

    """

    if tolerance is None:
        tolerance = 5.

    try:
        kdt = KDT(coords_ref, balanced_tree=False, compact_nodes=False)
    except TypeError:
        kdt = KDT(coords_ref)

    ds,ind_ref = kdt.query(coords_image, k=1, distance_upper_bound=tolerance)
    mask_xmatch = ds < tolerance
    ind_image = np.arange(len(coords_image))

    return ind_image[mask_xmatch], ind_ref[mask_xmatch], ds[mask_xmatch]


def source_coords(sources: 'SourceTable') -> np.ndarray:
    """Pixel coordinates (``x_source``, ``y_source``) of the sources."""
    return np.vstack((sources['x_source'], sources['y_source'])).T


def find_pairs(sources: 'SourceTable', refs: 'ReferenceStars',
               search_radius: float) -> Table:
    """
    Find every source–star pair closer than search_radius.

    Parameters
    ----------
    sources
        The plate sources.
    refs
        The reference stars.
    search_radius
        Radius in pixels.

    Returns
    -------
    Table
        One row per pair: ``source_index`` (row of the source table),
        ``ref_index`` (index of the reference star) and ``dist``
        (distance in pixels).

    """
    kdt_ref = KDT(refs.xy)
    kdt_plate = KDT(source_coords(sources))
    mtrx = kdt_plate.sparse_distance_matrix(kdt_ref, search_radius)
    mtrx_keys = np.array([a for a in mtrx.keys()])
    pairs = Table(names=('source_index', 'ref_index', 'dist'),
                  dtype=('i8', 'i8', 'f8'))

    if len(mtrx_keys) > 0:
        pairs = Table()
        pairs['source_index'] = mtrx_keys[:,0]
        pairs['ref_index'] = mtrx_keys[:,1]
        pairs['dist'] = np.fromiter(mtrx.values(), dtype=float)

    return pairs
