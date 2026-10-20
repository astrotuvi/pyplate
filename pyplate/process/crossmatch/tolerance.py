"""Match and search radii from the astrometric errors of the solutions."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from ..solve import SolutionSet


@dataclass
class Radii:
    """Radii of the crossmatch.

    Attributes
    ----------
    match, match_arcsec
        Match radius (5 sigma) in pixels and in arcsec: a star within it
        can be the match of a source.
    search, search_arcsec
        Search radius (20 sigma, at least 5 arcsec) in pixels and in arcsec:
        a star within it is a neighbour of a source.
    pixel_scale
        Mean pixel scale of the solutions, for converting distances.
    """

    match: float
    match_arcsec: float
    search: float
    search_arcsec: float
    pixel_scale: u.Quantity


def mean_sigma(solution_set: 'SolutionSet') -> u.Quantity:
    """Mean astrometric error of the solutions.

    The SCAMP errors of the solutions with more than 10 degrees of freedom
    are averaged, weighted by the degrees of freedom, and increased if
    there are fewer than 30 degrees of freedom in all. Without such
    solutions, or if the error is abnormally low, it is two pixels.
    """
    solutions = solution_set.solutions
    mean_pixscale = solution_set.mean_pixel_scale

    # Calculate mean astrometric error
    sigma1 = u.Quantity([sol.scamp_sigma_1 for sol in solutions
                         if sol.scamp_sigma_1 is not None])
    sigma2 = u.Quantity([sol.scamp_sigma_2 for sol in solutions
                         if sol.scamp_sigma_2 is not None])
    ndeg = np.array([sol.scamp_ndeg for sol in solutions
                     if sol.scamp_sigma_1 is not None])

    # Keep values only for solutions with scamp_ndeg > 10
    mask_ndeg = ndeg > 10
    sigma1 = sigma1[mask_ndeg]
    sigma2 = sigma2[mask_ndeg]
    ndeg = ndeg[mask_ndeg]

    if len(sigma1) > 0 and len(sigma2) > 0:
        # Calculate weighted mean of scamp_sigma over all solutions
        wmean_sigma1 = (sigma1 * ndeg).sum() / ndeg.sum()
        wmean_sigma2 = (sigma2 * ndeg).sum() / ndeg.sum()
        mean_scamp_sigma = np.sqrt(wmean_sigma1**2 + wmean_sigma2**2)

        # Increase sigma if the sum of scamp_ndeg < 30
        if ndeg.sum() < 30:
            mean_scamp_sigma = (30. / ndeg.sum()) * mean_scamp_sigma

        # Use default value if the mean sigma is abnormally low
        if (mean_scamp_sigma < 0.1 * mean_pixscale * u.pixel
            and ndeg.mean() < 20):
            mean_scamp_sigma = 2. * mean_pixscale * u.pixel
    else:
        mean_scamp_sigma = 2. * mean_pixscale * u.pixel

    return mean_scamp_sigma


def radii(solution_set: 'SolutionSet') -> Radii:
    """Match and search radii of the solution set (see :class:`Radii`)."""
    mean_pixscale = solution_set.mean_pixel_scale
    mean_scamp_sigma = mean_sigma(solution_set)
    match = ((5. * mean_scamp_sigma / mean_pixscale)
             .to(u.pixel).value)
    match_arcsec = (5. * mean_scamp_sigma).to(u.arcsec).value
    search = ((20. * mean_scamp_sigma / mean_pixscale)
              .to(u.pixel).value)

    if (20. * mean_scamp_sigma) < 5 * u.arcsec:
        search = (5 * u.arcsec / mean_pixscale).to(u.pixel).value

    search_arcsec = (search * u.pixel * mean_pixscale).to(u.arcsec).value
    return Radii(match=match, match_arcsec=match_arcsec, search=search,
                 search_arcsec=search_arcsec, pixel_scale=mean_pixscale)
