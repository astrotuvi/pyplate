"""SolveProcess: finding the astrometric solutions of a plate."""
import os
import subprocess as sp

import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from ..context import ProcessContext
from ..crossmatch.pairs import crossmatch_cartesian
from ..sources import SourceTable
from . import anet, multiexp, scamp
from .solution import Solution, SolutionSet, valid_wcs_header

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT


def select_reference_stars(ctx: ProcessContext, solution: Solution,
                           plate_epoch: float) -> Table:
    """
    Get astrometric reference stars inside the image of a solution, at the
    plate epoch, from ``gaiaedr3_pyplate.fits`` in ``[Files] gaia_dir``.

    Returns
    -------
    Table
        ``ra``, ``dec`` and ``mag`` (Gaia G) of the stars.

    """

    # Read the Gaia EDR3 catalogue
    fn_gaia = os.path.join(ctx.cfg.files.gaia_dir, 'gaiaedr3_pyplate.fits')

    tab = Table.read(fn_gaia)

    # Calculate RA and Dec for the plate epoch
    ra_ref = (tab['ra'] + (plate_epoch - 2016.0) * tab['pmra']
              / np.cos(tab['dec'] * np.pi / 180.) / 3600000.)
    dec_ref = (tab['dec'] + (plate_epoch - 2016.0)
               * tab['pmdec'] / 3600000.)
    catalog = SkyCoord(ra_ref, dec_ref, frame='icrs')

    # Query stars around the plate center
    c = SkyCoord(solution.ra_icrs * u.deg,
                 solution.dec_icrs * u.deg, frame='icrs')
    dist = catalog.separation(c)
    mask_dist = dist < solution.half_diag
    ind = np.arange(len(tab))[mask_dist]

    # Check which stars fall inside image area
    w = wcs.WCS(solution.header_wcs)
    xr,yr = w.all_world2pix(ra_ref[mask_dist], dec_ref[mask_dist], 1,
                            quiet=True)
    mask_inside = ((xr > 0) & (xr < ctx.imwidth) &
                   (yr > 0) & (yr < ctx.imheight))
    ind_ref = ind[mask_inside]

    numref = mask_inside.sum()
    ctx.log.write('Fetched {:d} entries from Gaia EDR3'
                  ''.format(numref))

    # Construct table for return
    astref_table = Table()
    astref_table['ra'] = ra_ref[ind_ref]
    astref_table['dec'] = dec_ref[ind_ref]
    astref_table['mag'] = tab[ind_ref]['phot_g_mean_mag']

    return astref_table


class SolveProcess:
    """
    Plate solve process class

    Solutions are found one at a time with astrometry.net
    (:meth:`find_next_solution`), the sources of each found solution are
    removed from the list, and finally all solutions are refined together
    with SCAMP, with the scanner pattern removed
    (:meth:`improve_solutions`).

    Parameters
    ----------
    ctx
        Context of the process.
    sources
        Extracted sources.

    """

    def __init__(self, ctx: ProcessContext, sources: SourceTable):
        self.ctx = ctx
        self.sources = sources.as_array()
        self.num_sources_sixbins = sources.num_sources_sixbins
        self.rel_area_sixbins = sources.rel_area_sixbins
        self.plate_epoch = ctx.plate_epoch

        # Sources that are left for finding solutions, and their exposure
        # numbers from the multi-exposure pattern
        self.astrom_sources = None
        self.exp_numbers = None

        self.plate_solved = False
        self.solutions = []
        self.duplicate_solutions = []
        self.astref_tables = []
        self.num_iterations = 0
        self.pattern_x = None
        self.pattern_y = None
        self.pattern_ratio = None
        self.pattern_table = None

    @property
    def numexp(self) -> int:
        """Number of exposures of the plate (1 without metadata)."""
        try:
            return self.ctx.plate.numexp
        except Exception:
            return 1

    def solve_plate(self, plate_epoch: float | None = None,
                    sip: int | None = None, skip_bright: int | None = None,
                    repeat_find: bool | None = None) -> SolutionSet:
        """
        Solve astrometry in a FITS file.

        Parameters
        ----------
        plate_epoch
            Epoch of plate in decimal years (default: the plate epoch of the
            context)
        sip
            SIP distortion order (default ``[Solve] sip``)
        skip_bright
            Number of brightest stars to skip when solving with
            Astrometry.net (default ``[Solve] skip_bright``).
        repeat_find
            If True, repeat finding astrometric solutions until none is
            found. If False, stop after finding the expected number of
            solutions (default ``[Solve] repeat_find``).

        """

        if plate_epoch is None:
            plate_epoch = self.plate_epoch
            plate_year = self.ctx.plate_year
        else:
            self.plate_epoch = plate_epoch

            try:
                plate_year = int(plate_epoch)
            except ValueError:
                plate_year = self.ctx.plate_year

        if repeat_find is None:
            repeat_find = self.ctx.cfg.solve.repeat_find

        # Log Astrometry.net and SCAMP versions
        anet_ver = (sp.check_output([self.ctx.cfg.programs.solve_field_path, '-h']).strip()
                    .decode('utf-8').split('\n')[4])
        scamp_ver = (sp.check_output([self.ctx.cfg.programs.scamp_path, '-v']).strip()
                     .decode('utf-8'))
        self.ctx.log.write('Using Astrometry.net {}'.format(anet_ver),
                           level=4, event=30, double_newline=False)
        self.ctx.log.write('Using {}'.format(scamp_ver), level=4, event=30,
                           double_newline=False)

        # Log plate epoch
        self.ctx.log.write('Using plate epoch of {:.2f}'.format(plate_epoch),
                           level=4, event=30, double_newline=False)

        if sip is None:
            sip = self.ctx.cfg.solve.sip

        if skip_bright is None:
            skip_bright = self.ctx.cfg.solve.skip_bright

        exp_factor = self.exposure_factor()
        self.astrom_sources = self.select_solve_sources(skip_bright,
                                                        exp_factor)

        if self.numexp > 4 and multiexp.have_sklearn:
            nrows = len(self.astrom_sources)
            coords = np.empty((nrows, 3))
            coords[:,0] = self.astrom_sources['x_source']
            coords[:,1] = self.astrom_sources['y_source']
            coords[:,2] = 0.
            dist_peak = multiexp.find_peak_separation(coords[:,:2])

            coords[:,2] = (self.astrom_sources['mag_auto'] * dist_peak * 10.
                           / exp_factor**2)
            self.exp_numbers = multiexp.find_pattern(self.ctx, coords,
                                                     dist_peak, self.numexp)

        # Output short-listed star data to FITS file
        xycat = Table()
        xycat['X_IMAGE'] = self.astrom_sources['x_source']
        xycat['Y_IMAGE'] = self.astrom_sources['y_source']
        xycat['MAG_AUTO'] = self.astrom_sources['mag_auto']
        xycat['FLUX'] = self.astrom_sources['flux_auto']
        xycat['X_IMAGE'].unit = 'pixel'
        xycat['Y_IMAGE'].unit = 'pixel'
        xycat['MAG_AUTO'].unit = 'mag'
        xycat['FLUX'].unit = 'count'
        fn_xy = os.path.join(self.ctx.work_dir, '{}.xy'.format(self.ctx.basefn))
        xycat.write(fn_xy, format='fits', overwrite=True)

        use_force = False

        # Repeat finding astrometric solutions until none is found
        while True:
            solution, astref_table = self.find_next_solution(
                ref_year=plate_year, sip=sip, use_force=use_force)

            if solution is None:
                if (use_force or len(self.solutions) > 0 or
                    not self.ctx.cfg.solve.allow_force):
                    break
                else:
                    use_force = True
                    continue
            else:
                use_force = False

            self.add_solution(solution, astref_table)

            # Check the number of remaining solutions
            num_remain_exp = self.numexp - len(self.solutions)

            # If the expected number of solutions is found and repeat_find is
            # False, then stop finding.
            if repeat_find == False and num_remain_exp < 1:
                break

        # Improve astrometric solutions (two iterations)
        if self.plate_solved:
            self.improve_solutions(distort=3)
            self.improve_solutions()

        return self.solution_set()

    def exposure_factor(self) -> float:
        """A factor (1 to 4) that takes into account how much the longest
        and shortest exposure times differ."""
        try:
            max_exp = np.max([e.exptime for e in self.ctx.plate.exposures])
            min_exp = np.min([e.exptime for e in self.ctx.plate.exposures])

            if min_exp > 0:
                exp_factor = np.log10(max_exp / min_exp) + 1.
            else:
                exp_factor = 1.

            exp_factor = min(max(1., exp_factor), 4.)
        except Exception:
            exp_factor = 1.

        return exp_factor

    def select_solve_sources(self, skip_bright: int,
                             exp_factor: float) -> np.ndarray:
        """Select the clean sources in annular bins 1--8 for solving, in
        brightness order, skipping the brightest ones. The number of
        sources grows with the source density, the number of exposures and
        exp_factor."""

        # By default, use 1000 sources for solving
        num_keep = 1000
        numexp = self.numexp
        num_keep = max([num_keep, int(self.num_sources_sixbins * numexp *
                                      exp_factor / 100)])

        # Limit number of stars with 100000/numexp
        num_keep = min([num_keep, int(100000/numexp)])

        self.ctx.log.write('Using max {:d} stars for astrometric solving'.format(num_keep),
                           level=4, event=30, double_newline=False)

        # Use only flag_clean, because model_prediction is not robust enough
        bclean = self.sources['flag_clean'] == 1

        indclean = np.where(bclean & (self.sources['annular_bin'] <= 8))[0]
        sb = skip_bright
        indsort = np.argsort(self.sources[indclean]['mag_auto'])[sb:sb+num_keep]
        indsel = indclean[indsort]
        nrows = len(indsel)

        self.ctx.log.write('Selected {:d} stars for astrometric solving'.format(nrows),
                           level=4, event=30)

        return self.sources[indsel]

    def add_solution(self, solution: Solution, astref_table: Table) -> None:
        """Number a found solution and add it to the solutions, or to the
        duplicate solutions if it is within 10 pixels of an earlier one."""
        unique_solution = True

        # If this is not the first solution, check if we got a unique
        # solution
        if len(self.solutions) > 0:
            sol_ra = np.array([sol.ra_icrs for sol in self.solutions])
            sol_dec = np.array([sol.dec_icrs for sol in self.solutions])
            c_sol = SkyCoord(sol_ra * u.deg, sol_dec * u.deg, frame='icrs')
            c_cur = SkyCoord(solution.ra_icrs * u.deg,
                             solution.dec_icrs * u.deg, frame='icrs')
            min_sep = c_cur.separation(c_sol).min()

            if min_sep > 1. * u.deg:
                sep_str = '{:.2f} degrees'.format(min_sep.deg)
            elif min_sep > 1. * u.arcmin:
                sep_str = '{:.2f} arcmin'.format(min_sep.arcmin)
            else:
                sep_str = '{:.2f} arcsec'.format(min_sep.arcsec)

            self.ctx.log.write('Current solution is separated from the '
                               'nearest solution by {}'.format(sep_str),
                               level=4, event=32)

            if min_sep < (10. * solution.pixel_scale * u.pixel):
                unique_solution = False
                unique_num = c_cur.separation(c_sol).argmin() + 1

        if unique_solution:
            solution.solution_num = len(self.solutions) + 1
            self.solutions.append(solution)
            self.astref_tables.append(astref_table)

            self.ctx.log.write('Current solution is unique; assigned number '
                               '{:d}'.format(solution.solution_num),
                               level=4, event=32,
                               solution_num=solution.solution_num)
        else:
            solution.solution_num = -len(self.duplicate_solutions) - 1
            solution.unique_num = unique_num
            self.duplicate_solutions.append(solution)

            self.ctx.log.write('Current solution is a duplicate of '
                               'solution {:d}; assigned number {:d}'
                               .format(unique_num, solution.solution_num),
                               level=4, event=32,
                               solution_num=solution.solution_num)

    def solution_set(self) -> SolutionSet:
        """The solution set of the found solutions, with the mean values
        and the centroid of the solutions."""
        solution_set = SolutionSet(
            num_solutions=len(self.solutions),
            num_duplicate_solutions=len(self.duplicate_solutions),
            pattern_ratio=self.pattern_ratio, plate_solved=self.plate_solved,
            solutions=self.solutions,
            duplicate_solutions=self.duplicate_solutions,
            pattern_x=self.pattern_x, pattern_y=self.pattern_y,
            pattern_table=self.pattern_table, imwidth=self.ctx.imwidth,
            imheight=self.ctx.imheight)

        if self.plate_solved:
            # Calculate mean pixel scale and FOV across all solutions
            pixscales = u.Quantity([sol.pixel_scale for sol in self.solutions])
            solution_set.mean_pixel_scale = pixscales.mean()
            solution_set.min_pixel_scale = pixscales.min()
            solution_set.max_pixel_scale = pixscales.max()
            fov1 = u.Quantity([sol.fov1 for sol in self.solutions])
            fov2 = u.Quantity([sol.fov2 for sol in self.solutions])
            solution_set.mean_fov1 = fov1.mean()
            solution_set.mean_fov2 = fov2.mean()
            dens = u.Quantity([sol.source_density for sol in self.solutions])
            solution_set.source_density = dens.mean()

            # Calculate solutions centroid
            solution_set.calculate_centroid()

        return solution_set

    def find_next_solution(self, ref_year: int | None = None,
                           sip: int | None = None, use_force: bool = False
                           ) -> tuple[Solution | None, Table | None]:
        """
        Find the next astrometric solution among the remaining sources.

        The sources are selected (:meth:`select_sources`) and solved with
        astrometry.net, the solution is refined with SCAMP using the Gaia
        reference stars, and the sources that belong to it are removed from
        the remaining sources.

        Parameters
        ----------
        ref_year
            Year number that will be used for reference coordinates
        sip
            SIP distortion order (default ``[Solve] sip``)
        use_force
            Try harder: select the sources classified as true sources in
            the inner annular bins, with a longer CPU time limit.

        Returns
        -------
        solution, astref_table
            The solution and its reference stars, or None, None.

        """

        self.ctx.log.write('Looking for an astrometric solution', level=3, event=32)

        if ref_year is None:
            ref_year = self.ctx.plate_year

        if sip is None:
            sip = self.ctx.cfg.solve.sip

        num_solutions = len(self.solutions)

        try:
            num_remain_exp = self.ctx.plate.numexp - num_solutions
        except Exception:
            num_remain_exp = 1

        self.ctx.log.write('Number of remaining solutions: {:d}'
                           .format(num_remain_exp), level=4, event=32)

        if use_force:
            self.ctx.log.write('Using force to find solution', level=4, event=32)

        # Current solution sequence number
        solution_seq = num_solutions + len(self.duplicate_solutions) + 1
        basefn_solution = '{}-{:02d}'.format(self.ctx.basefn, solution_seq)
        use_sources = self.select_sources(num_remain_exp, solution_seq,
                                          use_force)

        if num_solutions > 0:
            pixel_scale = self.solutions[0].pixel_scale
        else:
            pixel_scale = None

        # If the number of solutions is larger than 4, then accept
        # solutions with lower odds
        many_solutions = self.numexp > 4 or num_solutions > 4

        if num_remain_exp > 0:
            # Higher cpu limit for the first solution
            if use_force:
                cpulimit = 600
            elif num_solutions == 0:
                cpulimit = 300
            else:
                cpulimit = 180
        else:
            # Low limit for extra solutions (after numexp from metadata)
            cpulimit = 30

        header_anet = anet.solve_field(self.ctx, use_sources, basefn_solution,
                                       ref_year, sip, pixel_scale,
                                       many_solutions, cpulimit)

        if header_anet is not None:
            self.plate_solved = True
            self.ctx.log.write('Astrometric solution found (sequence number {:d})'
                               .format(solution_seq), level=4, event=32)
        else:
            if num_solutions > 0:
                self.ctx.log.write('Could not find additional astrometric solutions',
                                   level=4, event=32)
            else:
                self.ctx.log.write('Could not solve astrometry for the plate',
                                   level=2, event=32)
            return None, None

        # Create Solution instance and calculate parameters
        self.ctx.log.write('Calculating parameters for the initial solution '
                           '(sequence number {:d})'.format(solution_seq),
                           level=4, event=32)

        solution = Solution.from_anet(header_anet, self.ctx,
                                      self.num_sources_sixbins,
                                      self.rel_area_sixbins)
        header_wcs_anet = solution.header_wcs

        # Get reference stars
        astref_table = select_reference_stars(self.ctx, solution,
                                              self.plate_epoch)

        # Improve solution with SCAMP
        self.ctx.log.write('Improving solution and recalculating parameters '
                           '(sequence number {:d})'.format(solution_seq),
                           level=4, event=32)
        self.refine_solution(solution, astref_table, basefn_solution)

        # Remove the sources that belong to the solution
        self.remove_matched_sources(solution, astref_table, header_wcs_anet,
                                    basefn_solution)
        return solution, astref_table

    def select_sources(self, num_remain_exp: int, solution_seq: int,
                       use_force: bool) -> np.ndarray:
        """
        Select the remaining sources to find the next solution with.

        - If the sources are numbered by exposure, the sources of the
          current exposure.
        - If more than two solutions remain, the sources whose two nearest
          neighbours are within a 90-degree angle, and of these the largest
          group with the same offset between the neighbours.
        - With use_force, the sources classified as true sources in annular
          bins 1--6.
        - Otherwise all remaining sources.

        """
        num_astrom_sources = len(self.astrom_sources)

        # If sources have been numbered according to exposures,
        # then select sources that match the current exposure number
        if (self.exp_numbers is not None
            and len(self.exp_numbers) > 4
            and self.exp_numbers.max() > len(self.solutions)
            and use_force == False):
            self.ctx.log.write('Selecting sources that match the exposure number '
                               '{:d}'.format(solution_seq),
                               level=4, event=32)
            indmask = (self.exp_numbers == solution_seq)
            return self.astrom_sources[indmask]

        # If number of remaining exposures is larger than 2, then
        # select sources that have two nearest neighbours within 90-degree
        # angle
        if num_remain_exp > 2 and use_force == False:
            self.ctx.log.write('Selecting sources that have two nearest neighbours '
                               'within 90-degree angle', level=4, event=32)
            return self._select_by_neighbours(solution_seq)

        # If use_force is True, then select stars that have been classified
        # as true sources, regardless of the number of exposures.
        # Also, narrow the selection to annular bins 1-6.
        if use_force == True:
            if np.isnan(self.astrom_sources['model_prediction']).sum() == 0:
                btrue = self.astrom_sources['model_prediction'] > 0.9

                # If less than 5 sources are classified as true sources,
                # use all
                if btrue.sum() < 5:
                    btrue = np.full(num_astrom_sources, True)
            else:
                btrue = np.full(num_astrom_sources, True)

            bselect = btrue & (self.astrom_sources['annular_bin'] <= 6)

            # If less than 5 sources remain, select all
            if bselect.sum() < 5:
                bselect = np.full(num_astrom_sources, True)

            return self.astrom_sources[bselect]

        # In other cases, use all sources
        return self.astrom_sources

    def _select_by_neighbours(self, solution_seq):
        """Sources whose two nearest neighbours are within a 90-degree
        angle, in the largest compact group of the vectors between the
        neighbours (with scikit-learn)."""
        num_astrom_sources = len(self.astrom_sources)
        coords = np.empty((num_astrom_sources, 2))
        coords[:,0] = self.astrom_sources['x_source']
        coords[:,1] = self.astrom_sources['y_source']
        kdt = KDT(coords)
        ds,ind = kdt.query(coords, k=3)
        x0 = coords[:,0]
        y0 = coords[:,1]
        x1 = coords[ind[:,1],0]
        y1 = coords[ind[:,1],1]
        x2 = coords[ind[:,2],0]
        y2 = coords[ind[:,2],1]
        indmask = (x1-x0)*(x2-x0)+(y1-y0)*(y2-y0) > 0
        use_sources = self.astrom_sources[indmask]

        # Find clusters of vectors (from the nearest neighbor to the
        # second nearest) and take the largest cluster
        if multiexp.have_sklearn:
            dx = x2[indmask] - x1[indmask]
            dy = y2[indmask] - y1[indmask]
            dxdy = np.vstack((dx, dy)).T
            db = multiexp.DBSCAN(eps=2.0, min_samples=10).fit(dxdy)
            labels = db.labels_
            lab, cnt = np.unique(labels[labels>-1], return_counts=True)

            # Sort clusters by the number of members (descending order)
            if len(lab) > 1:
                ind_sort = np.argsort(cnt)[::-1]

                for i in ind_sort:
                    clumpmask = (labels == lab[i])
                    dxm = np.mean(dx[clumpmask])
                    dym = np.mean(dy[clumpmask])
                    dxs = np.std(dx[clumpmask])
                    dys = np.std(dy[clumpmask])

                    # Check if cluster is compact and away from center
                    # If yes, then interrupt the loop
                    if np.maximum(dxs, dys) < np.sqrt(dxm**2 + dym**2):
                        break

                use_sources = self.astrom_sources[indmask][clumpmask]

        # Output for checking
        t = Table()
        t['x'] = x0[indmask]
        t['y'] = y0[indmask]
        t['dx1'] = x1[indmask] - x0[indmask]
        t['dy1'] = y1[indmask] - y0[indmask]
        t['dx2'] = x2[indmask] - x0[indmask]
        t['dy2'] = y2[indmask] - y0[indmask]
        t['label'] = labels
        basefn_solution = '{}-{:02d}'.format(self.ctx.basefn, solution_seq)
        fn_out = os.path.join(self.ctx.work_dir,
                              '{}_dxy.fits'.format(basefn_solution))
        t.write(fn_out, format='fits', overwrite=True)

        return use_sources

    def refine_solution(self, solution: Solution, astref_table: Table,
                        basefn_solution: str) -> None:
        """Refine a new solution with SCAMP (distortion degree 3) and
        recalculate its parameters, if SCAMP gives a valid WCS."""

        # Create scampref file
        scamp.write_reference_catalog(self.ctx, astref_table, basefn_solution)

        # Create SCAMP .ahead files
        scamp.write_tan_ahead(self.ctx, basefn_solution)

        # Use crossid radius of 5 pixels and transform it to arcsec scale
        crossid_radius = 5. * u.pixel * solution.pixel_scale

        # Run SCAMP
        scamp.refine(self.ctx, basefn_solution, basefn_solution,
                     crossid_radius, distort=3, match=True, event=32)

        # Read SCAMP solution
        try:
            header_scamp, header_wcs, scamp_stats = scamp.parse_results(
                self.ctx, basefn_solution)
        except FileNotFoundError:
            return

        if scamp_stats['NDeg_Reference'][0] > 5:
            solution.scamp_dscale = scamp_stats['DPixelScale'][0]
            solution.scamp_dangle = scamp_stats['DPosAngle'].quantity[0]
            solution.scamp_dx = scamp_stats['DX'].quantity[0].to(u.arcsec)
            solution.scamp_dy = scamp_stats['DY'].quantity[0].to(u.arcsec)
            scamp.set_statistics(solution, scamp_stats, distort=3,
                                 iteration=0)

            # Store SCAMP solution and recalculate parameters
            solution.header_scamp = header_scamp

            if valid_wcs_header(header_wcs, self.ctx.imwidth, self.ctx.imheight):
                solution.header_wcs = header_wcs
                solution.calculate_parameters(
                    self.ctx, self.num_sources_sixbins,
                    self.rel_area_sixbins)
            else:
                self.ctx.log.write('SCAMP WCS not valid!', level=2, event=32)

    def remove_matched_sources(self, solution: Solution, astref_table: Table,
                               header_wcs_anet, basefn_solution: str) -> None:
        """Remove from the remaining sources those that match the reference
        stars of a new solution (within 20 sigma, in the WCS of
        astrometry.net) or appear in its correspondence file, and set the
        centroid of the matched sources in the solution."""
        num_astrom_sources = len(self.astrom_sources)

        # Crossmatch sources with rerefence stars and throw out
        # stars that matched

        if solution.scamp_sigma_mean is not None:
            xmatch_radius = (20. * solution.scamp_sigma_mean
                             / solution.pixel_scale
                             / u.pixel)
        else:
            xmatch_radius = 10.

        # Using WCS from Astrometry.net, as it is more robust
        w = wcs.WCS(header_wcs_anet)
        xr,yr = w.all_world2pix(astref_table['ra'], astref_table['dec'], 1,
                                quiet=True)
        coords_ref = np.vstack((xr, yr)).T
        coords_plate = np.vstack((self.astrom_sources['x_source'],
                                  self.astrom_sources['y_source'])).T
        kdt = KDT(coords_ref)
        ds,ind_ref = kdt.query(coords_plate, k=1)
        indmask = ds > xmatch_radius
        ind_plate = np.arange(num_astrom_sources)

        # Select crossmatched stars and calculate their centroid
        mask_xmatch = ds <= xmatch_radius
        matched_sources = self.astrom_sources[ind_plate[mask_xmatch]]
        solution.x_centroid = matched_sources['x_source'].mean()
        solution.y_centroid = matched_sources['y_source'].mean()
        xcenter = (self.ctx.imwidth + 1.) / 2.
        ycenter = (self.ctx.imheight + 1.) / 2.
        solution.rel_x_centroid = ((solution.x_centroid - xcenter)
                                   / self.ctx.imwidth)
        solution.rel_y_centroid = ((solution.y_centroid - ycenter)
                                   / self.ctx.imheight)
        solution.num_xmatch = mask_xmatch.sum()
        self.ctx.log.write('Centroid of matched sources: {:.2f} {:.2f}, '
                           'relative centroid: {:.3f} {:.3f}, '
                           'number of matches: {:d}'
                           .format(solution.x_centroid, solution.y_centroid,
                               solution.rel_x_centroid,
                               solution.rel_y_centroid,
                               solution.num_xmatch),
                           level=4, event=32)

        # Keep only stars that were not crossmatched
        self.astrom_sources = self.astrom_sources[ind_plate[indmask]]
        num_astrom_sources = len(self.astrom_sources)

        if self.exp_numbers is not None:
            self.exp_numbers = self.exp_numbers[ind_plate[indmask]]

        # Also, throw out stars that appear in the Astrometry.net .corr file
        corr_tab = anet.read_correspondences(self.ctx, basefn_solution)
        coords_corr = np.vstack((corr_tab['field_x'], corr_tab['field_y'])).T
        coords_plate = np.vstack((self.astrom_sources['x_source'],
                                  self.astrom_sources['y_source'])).T
        kdt = KDT(coords_corr)
        ds,ind_ref = kdt.query(coords_plate, k=1)
        indmask = ds > 1.
        ind_plate = np.arange(num_astrom_sources)
        self.astrom_sources = self.astrom_sources[ind_plate[indmask]]

        if self.exp_numbers is not None:
            self.exp_numbers = self.exp_numbers[ind_plate[indmask]]

    def improve_solutions(self, distort: int | None = None) -> None:
        """
        Improve all astrometric solutions together: find the scanner
        pattern from the isolated sources matched with isolated reference
        stars, and refine each solution with SCAMP on the pattern-corrected
        coordinates.

        Parameters
        ----------
        distort
            Distortion order for improved solution (default
            ``[Solve] distort``)

        """

        self.ctx.log.write('Improving astrometric solutions', level=3, event=33)

        if len(self.solutions) < 1:
            self.ctx.log.write('No astrometric solutions to improve!',
                               level=2, event=33)
            return

        if distort is None:
            distort = self.ctx.cfg.solve.distort

        coords_ref = self._isolated_reference_stars()

        if coords_ref is None:
            return

        isolated = self._isolated_sources()

        if isolated is None:
            return

        coords_plate, ind_sources = isolated

        # Crossmatch sources and reference stars
        ind_plate, ind_ref, _ = crossmatch_cartesian(coords_plate, coords_ref)

        if len(ind_plate) < 10:
            self.ctx.log.write('Too few crossmatches between sources and reference '
                               'stars: {:d}'.format(len(ind_plate)),
                               level=2, event=33)
            return

        ind_sources = ind_sources[ind_plate]
        coords_wobble = coords_plate[ind_plate]

        # Find scanner pattern and get pattern-subtracted coordinates
        res = multiexp.find_scanner_pattern(coords_plate[ind_plate],
                                            coords_ref[ind_ref])
        coords_dewobbled, pattern_x, pattern_y, pattern_ratio = res
        self.pattern_x = pattern_x
        self.pattern_y = pattern_y

        if np.isfinite(pattern_ratio):
            self.pattern_ratio = pattern_ratio

        self.ctx.log.write('Scanner pattern ratio (stdev_y/stdev_x): '
                           '{:.3f}'.format(pattern_ratio), level=4, event=33)
        self.pattern_table = self._pattern_table(coords_wobble)

        # Go through individual solutions and improve them by
        # running SCAMP
        for i in np.arange(len(self.solutions)):
            self._improve_solution(i, coords_dewobbled, ind_sources, distort)

        # Increase iteration count
        self.num_iterations += 1

    def _isolated_reference_stars(self):
        """Pixel coordinates of the reference stars of all solutions that
        have no other reference star within 5 pixels, in their own solution
        or in another; None if there are 10 or fewer."""

        # Create array for xy coordinates of reference stars
        coords_ref = np.zeros((0,2))

        # Go through solutions and build a list of reference stars
        for solution, astref_table in zip(self.solutions,
                                          self.astref_tables):
            if len(astref_table) > 0:
                w = wcs.WCS(solution.header_wcs)
                xr,yr = w.all_world2pix(astref_table['ra'],
                                        astref_table['dec'], 1, quiet=True)
                mask_inside = ((xr > 0.5) & (xr < self.ctx.imwidth) &
                               (yr > 0.5) & (yr < self.ctx.imheight))
                xy_ref = np.vstack((xr[mask_inside], yr[mask_inside])).T

                # Include only isolated reference stars
                kdt = KDT(xy_ref)
                ds,ind = kdt.query(xy_ref, k=2)
                mask_isolated = ds[:,1] > 5
                add_ref = xy_ref[mask_isolated]
                coords_ref = np.append(coords_ref, add_ref, axis=0)

        if len(coords_ref) <= 10:
            self.ctx.log.write('Too few astrometric reference stars: '
                               '{:d}'.format(len(coords_ref)), level=2, event=33)
            return None

        # Exclude reference stars that have close neighbours either
        # naturally or due to other solution
        kdt = KDT(coords_ref)
        ds,ind = kdt.query(coords_ref, k=2)
        mask_isolated = ds[:,1] > 5
        num_isolated = mask_isolated.sum()

        num_excluded = (ds[:,1] <= 5).sum()
        self.ctx.log.write('Excluded {:d} reference stars due to close neighbours, '
                           '{:d} stars remained.'
                           ''.format(num_excluded, num_isolated), level=4, event=33)

        if num_isolated <= 10:
            self.ctx.log.write('Too few isolated astrometric reference stars: '
                               '{:d}'.format(num_isolated), level=2, event=33)
            return None

        coords_ref = coords_ref[mask_isolated]

        # Output reference stars for debugging
        t = Table()
        t['x_ref'] = coords_ref[:,0]
        t['y_ref'] = coords_ref[:,1]
        fn_out = os.path.join(self.ctx.work_dir, '{}_ref_all.fits'.format(self.ctx.basefn))
        t.write(fn_out, format='fits', overwrite=True)

        return coords_ref

    def _isolated_sources(self):
        """Coordinates and indices of the clean sources in annular bins
        1--9 that have no other such source within 5 pixels; None if there
        are 10 or fewer."""

        # Use only flag_clean, because model_prediction is not robust enough
        bclean = self.sources['flag_clean'] == 1
        mask_bins = bclean & (self.sources['annular_bin'] <= 9)

        if mask_bins.sum() <= 10:
            self.ctx.log.write('Too few clean sources in annular bins 1--9: '
                               '{:d}'.format(mask_bins.sum()), level=2, event=33)
            return None

        coords_plate = np.vstack((self.sources[mask_bins]['x_source'],
                                  self.sources[mask_bins]['y_source'])).T
        ind_sources = np.arange(len(self.sources))[mask_bins]

        # Exclude sources that have close neighbours
        kdt = KDT(coords_plate)
        ds,ind = kdt.query(coords_plate, k=2)
        mask_isolated = ds[:,1] > 5
        num_isolated = mask_isolated.sum()

        if num_isolated <= 10:
            self.ctx.log.write('Too few isolated sources in annular bins 1--9: '
                               '{:d}'.format(num_isolated), level=2, event=33)
            return None

        return coords_plate[mask_isolated], ind_sources[mask_isolated]

    def _pattern_table(self, coords_wobble):
        """The scanner pattern sampled every 50 pixels along both axes,
        with the points outside the matched sources marked as
        extrapolated."""
        xx = np.arange(0, self.ctx.imwidth + 1, 50)
        yy = np.arange(0, self.ctx.imheight + 1, 50)
        xx_extra = ((xx < coords_wobble[:,0].min()) |
                    (xx > coords_wobble[:,0].max()))
        yy_extra = ((yy < coords_wobble[:,1].min()) |
                    (yy > coords_wobble[:,1].max()))
        xx_pattern = self.pattern_x(xx)
        yy_pattern = self.pattern_y(yy)

        t = Table()
        t['axis'] = np.append(np.full(len(xx), 1), np.full(len(yy), 2))
        t['coord'] = np.append(xx, yy)
        t['shift'] = np.append(xx_pattern, yy_pattern)
        t['extrapolated'] = np.append(xx_extra, yy_extra).astype(int)
        return t

    def _improve_solution(self, i, coords_dewobbled, ind_sources, distort):
        """Refine solution i with SCAMP on the pattern-corrected source
        coordinates and update the centroid of its matched sources."""
        solution = self.solutions[i]
        astref_table = self.astref_tables[i]
        basefn_solution = '{}-{:02d}'.format(self.ctx.basefn, i+1)
        nsrc = len(coords_dewobbled)

        fn_scampcat = os.path.join(self.ctx.work_dir,
                                   '{}.cat'.format(basefn_solution))
        cat = fits.open(fn_scampcat)
        scampdata = fits.BinTableHDU.from_columns(cat[2].columns,
                                                  nrows=nsrc).data
        scampdata.field('X_IMAGE')[:] = coords_dewobbled[:,0]
        scampdata.field('Y_IMAGE')[:] = coords_dewobbled[:,1]
        scampdata.field('ERR_A')[:] = self.sources[ind_sources]['erra_source']
        scampdata.field('ERR_B')[:] = self.sources[ind_sources]['errb_source']
        scampdata.field('FLUX')[:] = self.sources[ind_sources]['flux_auto']
        scampdata.field('FLUX_ERR')[:] = self.sources[ind_sources]['fluxerr_auto']
        scampdata.field('FLAGS')[:] = self.sources[ind_sources]['sextractor_flags']
        cat[2].data = scampdata

        fn_scampcat = os.path.join(self.ctx.work_dir,
                                   '{}_dewobbled.cat'.format(basefn_solution))
        cat.writeto(fn_scampcat, overwrite=True)

        # Write SCAMP header to .ahead file
        fn_ahead = os.path.join(self.ctx.work_dir,
                                '{}_dewobbled.ahead'.format(basefn_solution))
        header = solution.header_wcs
        header.totextfile(fn_ahead, endcard=True, overwrite=True)

        # Use crossid radius of 3 pixels and transform it to arcsec scale
        crossid_radius = 3. * u.pixel * solution.pixel_scale

        # Limit distortion degree based on the number of sources
        use_distort = distort

        if solution.scamp_ndeg is None or solution.scamp_ndeg < 100:
            use_distort = min(distort, 3)
        elif solution.scamp_ndeg < 500:
            use_distort = min(distort, 5)

        if solution.scamp_ndeg is None:
            ndeg_str = 'unknown'
        else:
            ndeg_str = '{:d}'.format(solution.scamp_ndeg)

        self.ctx.log.write('Using distortion degree {:d} (scamp_ndeg: {})'
                           .format(use_distort, ndeg_str),
                           level=3, event=33, solution_num=i+1)

        # Run SCAMP again on dewobbled pixel coordinates
        catalog = '{}_dewobbled'.format(basefn_solution)
        scamp.refine(self.ctx, basefn_solution, catalog, crossid_radius,
                     distort=use_distort, match=False, event=33)

        # Read SCAMP solution
        header_scamp, header_wcs, scamp_stats = scamp.parse_results(
            self.ctx, catalog)
        scamp_ndeg = scamp_stats['NDeg_Reference'][0]

        if scamp_ndeg > 5:
            scamp.set_statistics(solution, scamp_stats, distort=use_distort,
                                 iteration=self.num_iterations + 1)

            # Store improved solution
            solution.header_scamp = header_scamp

            if valid_wcs_header(header_wcs, self.ctx.imwidth, self.ctx.imheight):
                solution.header_wcs = header_wcs
                solution.calculate_parameters(
                    self.ctx, self.num_sources_sixbins,
                    self.rel_area_sixbins)
            else:
                self.ctx.log.write('SCAMP WCS for solution {:d} not valid!'
                                   .format(i+1),
                                   level=2, event=33, solution_num=i+1)

        # Crossmatch sources with rerefence stars
        w = wcs.WCS(header_wcs)
        xr,yr = w.wcs_world2pix(astref_table['ra'],
                                astref_table['dec'], 1)

        coords_ref_sol = np.vstack((xr, yr)).T

        if scamp_ndeg > 5:
            tolerance = ((5. * solution.scamp_sigma_mean / solution.pixel_scale)
                         .to(u.pixel).value)
        else:
            tolerance = 5.

        kdt = KDT(coords_ref_sol)
        ds,ind_ref = kdt.query(coords_dewobbled, k=1)
        mask_xmatch = ds <= tolerance
        ind_plate = np.arange(len(coords_dewobbled))

        if mask_xmatch.sum() > 0:
            # Select crossmatched stars and calculate their centroid
            matched_sources = coords_dewobbled[ind_plate[mask_xmatch]]
            x_centroid = matched_sources[:,0].mean()
            y_centroid = matched_sources[:,1].mean()
            solution.x_centroid = x_centroid
            solution.y_centroid = y_centroid
            xcenter = (self.ctx.imwidth + 1.) / 2.
            ycenter = (self.ctx.imheight + 1.) / 2.
            solution.rel_x_centroid = ((x_centroid - xcenter)
                                       / self.ctx.imwidth)
            solution.rel_y_centroid = ((y_centroid - ycenter)
                                       / self.ctx.imheight)
            solution.num_xmatch = mask_xmatch.sum()
            self.ctx.log.write('Centroid of matched sources: {:.2f} {:.2f}, '
                               'relative centroid: {:.3f} {:.3f}, '
                               'number of matches: {:d}'
                               .format(x_centroid, y_centroid,
                                   solution.rel_x_centroid,
                                   solution.rel_y_centroid,
                                   solution.num_xmatch),
                               level=4, event=33, solution_num=i+1)
        else:
            self.ctx.log.write('Cannot calculate centroid of solution '
                               'due to no matched sources!',
                               level=2, event=33, solution_num=i+1)
