"""Crossmatching plate sources with Gaia stars (DESIGN.md 6.5).

The steps, as :meth:`PlateProcess.match_catalog
<pyplate.process.process.PlateProcess.match_catalog>` runs them::

    refs = ReferenceStars.from_catalog(catalog, solution_set, epoch, log)
    radii = tolerance.radii(solution_set)
    pairs = find_pairs(sources, refs, radii.search)
    association = NearestMatcher(radii).associate(sources, refs, pairs)
    sources.add_matches(association, refs)
"""
from . import tolerance
from .matcher import Association, Matcher
from .nearest import NearestMatcher
from .pairs import crossmatch_cartesian, find_pairs
from .reference import ReferenceStars
from .tolerance import Radii

__all__ = ['Association', 'Matcher', 'NearestMatcher', 'Radii',
           'ReferenceStars', 'crossmatch_cartesian', 'find_pairs',
           'tolerance']
