"""Results of the astrometric solving: SolutionSet, Solution and
ScannerPattern."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.coordinates import ICRS, Angle, SkyCoord
from astropy.io import fits
from astropy.table import Table

from ...metadata.base import DbRecord, related
from ..context import ProcessContext
from ..log import ProcessLog


def valid_wcs_header(header, imwidth, imheight):
    """
    Validate WCS header based on FOV and image ratios.

    """

    try:
        w = wcs.WCS(header)
    except wcs._wcs.InvalidTransformError:
        return False

    pix_edge_midpoints = np.array([[1., (imheight+1.)/2.],
                                   [imwidth, (imheight+1.)/2.],
                                   [(imwidth + 1.)/2., 1.],
                                   [(imwidth + 1.)/2., imheight]])
    edge_midpoints = w.all_pix2world(pix_edge_midpoints, 1)

    c1 = SkyCoord(ra=edge_midpoints[0,0], dec=edge_midpoints[0,1],
                  unit=(u.deg, u.deg))
    c2 = SkyCoord(ra=edge_midpoints[1,0], dec=edge_midpoints[1,1],
                  unit=(u.deg, u.deg))
    c3 = SkyCoord(ra=edge_midpoints[2,0], dec=edge_midpoints[2,1],
                  unit=(u.deg, u.deg))
    c4 = SkyCoord(ra=edge_midpoints[3,0], dec=edge_midpoints[3,1],
                  unit=(u.deg, u.deg))
    fov1 = c1.separation(c2).to(u.deg).value
    fov2 = c3.separation(c4).to(u.deg).value

    if fov1 < 1e-3 or fov2 < 1e-3:
        return False

    ratio = (fov1 / fov2) / (float(imwidth) / float(imheight))

    if ratio > 0.95 and ratio < 1.05:
        return True
    else:
        return False


@dataclass
class Solution(DbRecord):
    """An astrometric solution (``solution`` table).

    The columns are described in ``applause_dr5.yaml``. Angles and scales
    are astropy quantities, headers :class:`astropy.io.fits.Header`; the
    database writer converts them.

    Attributes
    ----------
    ra_icrs_hms, dec_icrs_dms
        Coordinates of the centre as sexagesimal strings.
    half_diag
        Half of the diagonal of the field of view.
    ncp_close, scp_close
        Whether a celestial pole is within ``half_diag`` of the centre.
    skycoord_corners
        Coordinates of the image corners.
    healpix_table
        Number of sources in each HEALPix (level 1024) of the solution.
    unique_num
        For a duplicate solution, the number of the solution it duplicates.
    """

    table = 'solution'

    solution_id: int | None = None
    solutionset_id: int | None = None
    process_id: int | None = None
    scan_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    solution_num: int | None = None
    ra_icrs: float | None = None
    dec_icrs: float | None = None
    fov1: Angle | None = None
    fov2: Angle | None = None
    pixel_scale: u.Quantity | None = None
    source_density: u.Quantity | None = None
    cd1_1: float | None = None
    cd1_2: float | None = None
    cd2_1: float | None = None
    cd2_2: float | None = None
    rotation_angle: u.Quantity | None = None
    plate_mirrored: bool | None = None
    ncp_on_plate: bool | None = None
    scp_on_plate: bool | None = None
    stc_box: str | None = None
    stc_polygon: str | None = None
    header_wcs: fits.Header | None = None
    header_anet: fits.Header | None = None
    header_scamp: fits.Header | None = None
    scamp_dscale: float | None = None
    scamp_dangle: u.Quantity | None = None
    scamp_dx: u.Quantity | None = None
    scamp_dy: u.Quantity | None = None
    scamp_sigma_1: u.Quantity | None = None
    scamp_sigma_2: u.Quantity | None = None
    scamp_sigma_mean: u.Quantity | None = None
    scamp_chi2: float | None = None
    scamp_ndeg: int | None = None
    scamp_distort: int | None = None
    scamp_iteration: int | None = None
    x_centroid: float | None = None
    y_centroid: float | None = None
    rel_x_centroid: float | None = None
    rel_y_centroid: float | None = None
    num_xmatch: int | None = None
    num_gaia_dr4: int | None = None

    ra_icrs_hms: str | None = related(default=None)
    dec_icrs_dms: str | None = related(default=None)
    half_diag: Angle | None = related(default=None)
    ncp_close: bool | None = related(default=None)
    scp_close: bool | None = related(default=None)
    skycoord_corners: SkyCoord | None = related(default=None)
    healpix_table: Table | None = related(default=None)
    unique_num: int | None = related(default=None)

    @classmethod
    def from_anet(cls, header_anet: fits.Header, ctx: ProcessContext,
                  num_sources_sixbins: int,
                  rel_area_sixbins: float) -> 'Solution':
        """Make a solution from the WCS header of astrometry.net and
        calculate its parameters (see :meth:`calculate_parameters`)."""
        header_wcs = header_anet.copy()
        header_wcs.set('NAXIS', 2)
        header_wcs.set('NAXIS1', ctx.imwidth, after='NAXIS')
        header_wcs.set('NAXIS2', ctx.imheight, after='NAXIS1')
        solution = cls(header_anet=header_anet, header_wcs=header_wcs)
        solution.calculate_parameters(ctx, num_sources_sixbins,
                                      rel_area_sixbins)
        return solution

    def get_header(self, keyword=None):
        """
        Return Astrometry.net or SCAMP header.

        Parameters
        ----------
        keyword : str
            A keyword specifying which header to return

        """

        if self.header_scamp:
            header = self.header_scamp.copy()
        elif self.header_anet:
            header = self.header_anet.copy()
        else:
            header = None

        if keyword == 'header_scamp':
            if self.header_scamp:
                header = self.header_scamp.copy()
            else:
                header = None

        if keyword == 'header_anet':
            if self.header_anet:
                header = self.header_anet.copy()
            else:
                header = None

        if keyword == 'header_wcs':
            if self.header_wcs:
                header = self.header_wcs.copy()
            else:
                header = None

        return header

    def calculate_parameters(self, ctx: ProcessContext,
                             num_sources_sixbins: int,
                             rel_area_sixbins: float) -> None:
        """
        Calculate solution-related parameters from ``header_wcs``.

        Parameters
        ----------
        ctx
            Context of the process (image size and log).
        num_sources_sixbins, rel_area_sixbins
            Number of sources in the annular bins 1--6 and the area of the
            bins relative to the image, for the source density.

        """

        log = ctx.log
        imwidth = ctx.imwidth
        imheight = ctx.imheight

        if self.header_wcs is None:
            log.write('No solution found, cannot calculate '
                      'solution-related parameters',
                      level=2, event=34)
            return

        w = wcs.WCS(self.header_wcs)
        self.ra_icrs = self.header_wcs['CRVAL1']
        self.dec_icrs = self.header_wcs['CRVAL2']
        self.cd1_1 = self.header_wcs['CD1_1']
        self.cd1_2 = self.header_wcs['CD1_2']
        self.cd2_1 = self.header_wcs['CD2_1']
        self.cd2_2 = self.header_wcs['CD2_2']

        pix_edge_midpoints = np.array([[1., (imheight+1.)/2.],
                                       [imwidth, (imheight+1.)/2.],
                                       [(imwidth + 1.)/2., 1.],
                                       [(imwidth + 1.)/2., imheight]])
        edge_midpoints = w.all_pix2world(pix_edge_midpoints, 1)

        c1 = SkyCoord(ra=edge_midpoints[0,0], dec=edge_midpoints[0,1],
                      unit=(u.deg, u.deg))
        c2 = SkyCoord(ra=edge_midpoints[1,0], dec=edge_midpoints[1,1],
                      unit=(u.deg, u.deg))
        c3 = SkyCoord(ra=edge_midpoints[2,0], dec=edge_midpoints[2,1],
                      unit=(u.deg, u.deg))
        c4 = SkyCoord(ra=edge_midpoints[3,0], dec=edge_midpoints[3,1],
                      unit=(u.deg, u.deg))
        self.fov1 = c1.separation(c2)
        self.fov2 = c3.separation(c4)

        self.source_density = (num_sources_sixbins
                               / rel_area_sixbins
                               / (self.fov1*self.fov2))

        pixscale1 = self.fov1.to(u.arcsec) / (imwidth * u.pixel)
        pixscale2 = self.fov2.to(u.arcsec) / (imheight * u.pixel)
        self.pixel_scale = (pixscale1 + pixscale2) / 2.

        # Check if a celestial pole is nearby or on the plate
        self.half_diag = np.sqrt(self.fov1**2 + self.fov2**2) / 2.
        self.ncp_close = 90. - self.dec_icrs <= self.half_diag / u.deg
        self.scp_close = 90. + self.dec_icrs <= self.half_diag / u.deg
        ncp_on_plate = False
        scp_on_plate = False

        if self.ncp_close:
            ncp_pix = w.all_world2pix([[self.ra_icrs, 90.]], 1,
                                      quiet=True)

            if (ncp_pix[0,0] > 0 and ncp_pix[0,0] < imwidth
                and ncp_pix[0,1] > 0 and ncp_pix[0,1] < imheight):
                ncp_on_plate = True

        if self.scp_close:
            scp_pix = w.all_world2pix([[self.ra_icrs, -90.]], 1,
                                      quiet=True)

            if (scp_pix[0,0] > 0 and scp_pix[0,0] < imwidth
                and scp_pix[0,1] > 0 and scp_pix[0,1] < imheight):
                scp_on_plate = True

        self.ncp_on_plate = ncp_on_plate
        self.scp_on_plate = scp_on_plate

        # Construct coordinate strings
        ra_angle = Angle(self.ra_icrs, u.deg)
        dec_angle = Angle(self.dec_icrs, u.deg)

        self.ra_icrs_hms = ra_angle.to_string(unit=u.hour, sep=':',
                                              precision=1, pad=True)
        self.dec_icrs_dms = dec_angle.to_string(unit=u.deg, sep=':',
                                                precision=1, pad=True)
        self.stc_box = ('Box ICRS {:.5f} {:.5f} {:.5f} {:.5f}'
                        .format(self.header_wcs['CRVAL1'],
                                self.header_wcs['CRVAL2'],
                                self.fov1.value, self.fov2.value))

        pix_corners = np.array([[1., 1.], [imwidth, 1.],
                               [imwidth, imheight],
                               [1., imheight]])
        corners = w.all_pix2world(pix_corners, 1)
        self.stc_polygon = ('Polygon ICRS {:.5f} {:.5f} {:.5f} {:.5f} '
                            '{:.5f} {:.5f} {:.5f} {:.5f}'
                            .format(corners[0,0], corners[0,1],
                                    corners[1,0], corners[1,1],
                                    corners[2,0], corners[2,1],
                                    corners[3,0], corners[3,1]))
        self.skycoord_corners = SkyCoord(corners[:,0] * u.deg,
                                         corners[:,1] * u.deg,
                                         frame='icrs')

        # Calculate plate rotation angle
        try:
            cp = np.array([self.header_wcs['CRPIX1'],
                           self.header_wcs['CRPIX2']])

            if dec_angle.deg > 89.:
                cn = w.wcs_world2pix([[self.ra_icrs, 90.]], 1)
            else:
                cn = w.wcs_world2pix([[self.ra_icrs,
                                       self.dec_icrs + 1.]], 1)

            if ra_angle.deg > 359.:
                ce = w.wcs_world2pix([[self.ra_icrs - 359.,
                                       self.dec_icrs]], 1)
            else:
                ce = w.wcs_world2pix([[self.ra_icrs + 1.,
                                       self.dec_icrs]], 1)

            naz = 90. - np.arctan2((cn-cp)[0,1],(cn-cp)[0,0]) * 180. / np.pi
            eaz = 90. - np.arctan2((ce-cp)[0,1],(ce-cp)[0,0]) * 180. / np.pi

            if naz < 0:
                naz += 360.

            if eaz < 0:
                eaz += 360.

            rotation_angle = naz
            ne_angle = naz - eaz

            if rotation_angle > 180:
                rotation_angle -= 360.

            if ne_angle < 0:
                ne_angle += 360.

            if ne_angle > 180:
                plate_mirrored = True
            else:
                plate_mirrored = False
        except Exception:
            rotation_angle = None
            plate_mirrored = None
            log.write('Could not calculate plate rotation angle',
                      level=2, event=34)

        self.rotation_angle = rotation_angle * u.deg
        self.plate_mirrored = plate_mirrored

        log.write('Image dimensions: {:.2f} x {:.2f} degrees'
                  ''.format(self.fov1.value, self.fov2.value),
                  double_newline=False)
        log.write('Mean pixel scale: {:.3f} arcsec'
                  ''.format(self.pixel_scale.value),
                  double_newline=False)
        log.write('The image has {:.0f} stars per square degree'
                  ''.format(self.source_density.value))
        log.write('Plate rotation angle: {}'
                  .format(self.rotation_angle.value),
                  double_newline=False)
        log.write('Plate is mirrored: {}'.format(self.plate_mirrored),
                  double_newline=False)
        log.write('North Celestial Pole is on the plate: {}'
                  .format(self.ncp_on_plate),
                  double_newline=False)
        log.write('South Celestial Pole is on the plate: {}'
                  .format(self.scp_on_plate))


@dataclass
class ScannerPattern(DbRecord):
    """A point of the scanner pattern (``scanner_pattern`` table).

    The columns are described in ``applause_dr5.yaml``.
    """

    table = 'scanner_pattern'

    pattern_id: int | None = None
    solutionset_id: int | None = None
    process_id: int | None = None
    scan_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    axis: int | None = None
    coord: float | None = None
    shift: float | None = None
    extrapolated: int | None = None


@dataclass
class SolutionSet(DbRecord):
    """The astrometric solutions of a plate and the values common to them
    (``solution_set`` table).

    The columns are described in ``applause_dr5.yaml``.

    Attributes
    ----------
    plate_solved
        Whether a solution was found.
    solutions, duplicate_solutions
        The solutions, and the solutions found again (numbered -1, -2, ...).
    pattern_x, pattern_y
        Scanner pattern along each axis: the shift as a function of the
        coordinate.
    pattern_table
        The scanner pattern, sampled every 50 pixels (the rows of the
        ``scanner_pattern`` table).
    imwidth, imheight
        Image size.
    centroid, radius
        Centre of the solutions and the distance from it to the furthest
        image corner.
    """

    table = 'solution_set'

    solutionset_id: int | None = None
    process_id: int | None = None
    scan_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    num_solutions: int = 0
    num_duplicate_solutions: int = 0
    pattern_ratio: float | None = None
    mean_pixel_scale: u.Quantity | None = None
    min_pixel_scale: u.Quantity | None = None
    max_pixel_scale: u.Quantity | None = None
    mean_fov1: Angle | None = None
    mean_fov2: Angle | None = None
    source_density: u.Quantity | None = None
    max_separation: Angle | None = None
    header_wcs: fits.Header | None = None

    plate_solved: bool = related(default=False)
    solutions: list[Solution] = related()
    duplicate_solutions: list[Solution] = related()
    pattern_x: Callable | None = related(default=None)
    pattern_y: Callable | None = related(default=None)
    pattern_table: Table | None = related(default=None)
    imwidth: int | None = related(default=None)
    imheight: int | None = related(default=None)
    centroid: SkyCoord | None = related(default=None)
    radius: Angle | None = related(default=None)

    def create_wcs_header(self, log: ProcessLog) -> None:
        """
        Stack WCS keywords from solutions into one header. Use alternate
        WCS keywords specified in the FITS Standard 4.0.

        """

        if self.num_solutions == 0:
            log.write('No plate solution for the FITS header',
                      level=2, event=36)
            return

        self.header_wcs = self.solutions[0].get_header()

        if self.num_solutions > 1:
            self.header_wcs.insert(0, ('WCSNAME', 'Solution_1'))

        # Add additional solutions.
        # If there are more than 27 solutions, add the extra solutions
        # as comments.
        for i,solution in enumerate(self.solutions[1:]):
            if i < 26:
                suffix = chr(ord('A') + i)
                sep = (' WCS {} (solution {:d})'.format(suffix, i+2)
                       .rjust(72, '.'))
                self.header_wcs.append(('', sep), end=True)
                wcsname_card = ('WCSNAME{}'.format(suffix),
                                'Solution_{:d}'.format(i+2))
                self.header_wcs.append(wcsname_card, end=True)
            else:
                sep = ' WCS (solution {:d})'.format(i+2).rjust(72, '.')
                self.header_wcs.append(('', sep), end=True)
                wcsname_card = ('', 'WCSNAME = \'Solution_{:d}\''.format(i+2))
                self.header_wcs.append(wcsname_card, end=True)

            # For alternate WCS, append only WCS keywords
            for c in solution.get_header().cards:
                kw = c.keyword
                wcskeys = ['WCSAXES', 'CTYPE', 'CUNIT', 'CRVAL', 'CDELT',
                           'CRPIX', 'PC', 'CD', 'PV', 'PS', 'WCSNAME',
                           'CNAME', 'CRDER', 'CSYER', 'LONPOLE', 'LATPOLE',
                           'EQUINOX', 'RADESYS']

                for k in wcskeys:
                    if kw.startswith(k):
                        if i < 26:
                            kw_alternate = '{}{}'.format(kw, suffix)
                            newcard = fits.Card(kw_alternate, c.value,
                                                c.comment)
                        else:
                            newcard = fits.Card('', c.image.strip())

                        self.header_wcs.append(newcard, end=True)

    def calculate_centroid(self):
        """
        Calculate coordinates of a mid-point of solutions, and distance from
        centroid to the furthest plate corner.

        """

        assert self.num_solutions > 0

        # Collect center coordinates of solutions
        sol_ra = np.array([sol.ra_icrs for sol in self.solutions])
        sol_dec = np.array([sol.dec_icrs for sol in self.solutions])
        c_sol = SkyCoord(sol_ra * u.deg, sol_dec * u.deg, frame='icrs')

        # Create an offset frame based on the first solution
        aframe = c_sol[0].skyoffset_frame()
        fr_sol = c_sol.transform_to(aframe)

        # Find mean coordinates of solutions in the offset frame
        fr_cntr = SkyCoord(fr_sol.lon.mean(), fr_sol.lat.mean(), frame=aframe)
        c_cntr = fr_cntr.transform_to(ICRS)

        # Find solution that is furthest from mean coordinates
        ind_max1 = c_cntr.separation(c_sol).argmax()

        # Find solution that is furthest from the previously found solution
        sep2 = c_sol[ind_max1].separation(c_sol)
        ind_max2 = sep2.argmax()
        max_sep = sep2[ind_max2]

        # Collect corner coordinates of the two solutions
        corners1 = self.solutions[ind_max1].skycoord_corners
        corners2 = self.solutions[ind_max2].skycoord_corners
        max_sep_corners = 0 * u.deg

        # Find corners that are most distant from each other
        for c in corners1:
            sep_corners = c.separation(corners2)

            if sep_corners.max() > max_sep_corners:
                c1 = c
                c2 = corners2[sep_corners.argmax()]
                max_sep_corners = sep_corners.max()

        # Take the point between two corners as a centroid of solutions
        pos_angle = c1.position_angle(c2).to(u.deg)
        radius = max_sep_corners / 2.
        centroid = c1.directional_offset_by(pos_angle, radius)

        self.centroid = centroid
        self.radius = radius
        self.max_separation = max_sep

