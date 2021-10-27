import os
import glob
import shutil
import sys
import math
import datetime as dt
import subprocess as sp
import numpy as np
import warnings
import xml.etree.ElementTree as ET
import unidecode
from astropy import __version__ as astropy_version
from astropy import wcs
from astropy.io import fits, votable
from astropy.table import Table, vstack
from astropy.coordinates import Angle, EarthLocation, SkyCoord, ICRS, AltAz
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from scipy.linalg import lstsq
from collections import OrderedDict
from ..database.database import PlateDB
from ..conf import read_conf
from .._version import __version__
from .sources import crossmatch_cartesian

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

try:
    from sklearn.cluster import DBSCAN
    have_sklearn = True
except ImportError:
    have_sklearn = False

try:
    import MySQLdb
except ImportError:
    pass

try:
    import healpy
    have_healpy = True
except ImportError:
    have_healpy = False

try:
    import statsmodels.api as sm
    have_statsmodels = True
except ImportError:
    have_statsmodels = False


def new_scampref():
    """
    Create HDUList for SCAMP reference catalogue.

    Returns
    -------
    hdulist : an astropy.io.fits.HDUList object

    """

    hdulist = fits.HDUList()

    hdu = fits.PrimaryHDU()
    hdulist.append(hdu)

    hdummy = hdu.header.copy()
    hdummystr = hdummy.tostring()
    col = fits.Column(name='Field Header Card', format='2880A',
                      array=[hdummystr])

    try:
        tbl = fits.BinTableHDU.from_columns([col])
    except AttributeError:
        tbl = fits.new_table([col])

    tbl.header.set('EXTNAME', 'LDAC_IMHEAD', 'table name', after='TFIELDS')
    tbl.header.set('TDIM1', '(80, 36)', after='TFORM1')
    hdulist.append(tbl)

    col1 = fits.Column(name='X_WORLD', format='1D', unit='deg')
    col2 = fits.Column(name='Y_WORLD', format='1D', unit='deg')
    col3 = fits.Column(name='ERRA_WORLD', format='1E', unit='deg')
    col4 = fits.Column(name='ERRB_WORLD', format='1E', unit='deg')
    col5 = fits.Column(name='MAG', format='1E', unit='mag', disp='F8.4')
    col6 = fits.Column(name='MAGERR', format='1E', unit='mag', disp='F8.4')
    col7 = fits.Column(name='OBSDATE', format='1D', unit='yr',
                       disp='F13.8')

    try:
        tbl = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5,
                                             col6, col7])
    except AttributeError:
        tbl = fits.new_table([col1, col2, col3, col4, col5, col6, col7])

    tbl.header.set('EXTNAME', 'LDAC_OBJECTS', 'table name',
                   after='TFIELDS')
    hdulist.append(tbl)

    return hdulist


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

    ratio = (fov1 / fov2) / (np.float(imwidth) / np.float(imheight))

    if ratio > 0.95 and ratio < 1.05:
        return True
    else:
        return False


class PlateSolution:
    """
    Plate solution class for multiple astrometric solutions and parameters
    that are common to all solutions

    """

    def __init__(self):
        self.basefn = None
        self.log = None
        self.write_wcs_dir = ''

        self.plate_header = None
        self.platemeta = None
        self.imwidth = None
        self.imheight = None
        self.plate_solved = False
        self.mean_pixel_scale = None
        self.min_pixel_scale = None
        self.max_pixel_scale = None
        self.mean_fov1 = None
        self.mean_fov2 = None
        self.source_density = None
        self.num_sources = None
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None
        self.min_ra = None
        self.max_ra = None
        self.min_dec = None
        self.max_dec = None
        self.ncp_close = None
        self.scp_close = None
        self.ncp_on_plate = None
        self.scp_on_plate = None

        self.header_wcs = None
        self.solutions = None
        self.duplicate_solutions = None
        self.exp_numbers = None
        self.num_solutions = 0
        self.num_duplicate_solutions = 0
        self.num_iterations = 0
        self.pattern_x = None
        self.pattern_y = None
        self.pattern_ratio = None
        self.pattern_table = None

        self.centroid = None
        self.ra_centroid = None
        self.dec_centroid = None
        self.radius = None
        self.max_separation = None

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        try:
            self.archive_id = conf.getint('Archive', 'archive_id')
        except ValueError:
            print('Error in configuration file '
                  '([{}], {})'.format('Archive', attr))
        except configparser.Error:
            pass

        for attr in ['write_wcs_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except configparser.Error:
                pass

    def assign_header(self, header):
        """
        Assign FITS header with metadata.

        """

        self.plate_header = header

    def assign_metadata(self, platemeta):
        """
        Assign plate metadata.

        """

        self.platemeta = platemeta

        if self.platemeta['archive_id']:
            self.archive_id = self.platemeta['archive_id']

    def create_wcs_header(self):
        """
        Stack WCS keywords from solutions into one header. Use alternate
        WCS keywords specified in the FITS Standard 4.0.

        """

        if self.num_solutions == 0:
            self.log.write('No plate solution for the FITS header',
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

    def output_wcs_header(self):
        """
        Write WCS header to an ASCII file.

        """

        if self.plate_solved:
            self.log.write('Writing WCS header to a file', level=3, event=36)

            # Create output directory, if missing
            if self.write_wcs_dir and not os.path.isdir(self.write_wcs_dir):
                self.log.write('Creating WCS output directory {}'
                               ''.format(self.write_wcs_dir), level=4, event=36)
                os.makedirs(self.write_wcs_dir)

            for i in np.arange(self.num_solutions):
                fn_wcshead = '{}-{:02d}.wcs'.format(self.basefn, i+1)
                fn_wcshead = os.path.join(self.write_wcs_dir, fn_wcshead)
                self.log.write('Writing WCS output file {}'.format(fn_wcshead),
                               level=4, event=36)
                wcshead = self.solutions[i]['header_scamp']
                wcshead.tofile(fn_wcshead, overwrite=True)

    def calculate_centroid(self):
        """
        Calculate coordinates of a mid-point of solutions, and distance from
        centroid to the furthest plate corner.

        """

        assert self.num_solutions > 0

        # Collect center coordinates of solutions
        sol_ra = np.array([sol['ra_icrs'] for sol in self.solutions])
        sol_dec = np.array([sol['dec_icrs'] for sol in self.solutions])
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
        corners1 = self.solutions[ind_max1]['skycoord_corners']
        corners2 = self.solutions[ind_max2]['skycoord_corners']
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
        self.ra_centroid = centroid.ra.deg
        self.dec_centroid = centroid.dec.deg
        self.radius = radius
        self.max_separation = max_sep


class AstrometricSolution(OrderedDict):
    """
    Astrometric solution class

    """

    def __init__(self):
        self.imwidth = None
        self.imheight = None
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None
        self.wcs = None
        self.log = None

    def populate(self):
        keys = ['solution_num', 'ra_icrs', 'dec_icrs',
                'ra_icrs_hms', 'dec_icrs_dms',
                'fov1', 'fov2', 'half_diag', 'pixel_scale', 'source_density',
                'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2', 'rotation_angle',
                'plate_mirrored', 'ncp_close', 'scp_close',
                'ncp_on_plate', 'scp_on_plate', 'stc_box', 'stc_polygon',
                'header_anet', 'header_scamp', 'header_wcs',
                'skycoord_corners', 'x_centroid', 'y_centroid',
                'rel_x_centroid', 'rel_y_centroid', 'num_xmatch',
                'scamp_dscale', 'scamp_dangle', 'scamp_dx', 'scamp_dy',
                'scamp_sigma_1', 'scamp_sigma_2', 'scamp_sigma_mean',
                'scamp_chi2', 'scamp_ndeg',
                'scamp_distort', 'scamp_iteration',
                'num_gaia_edr3', 'healpix_table']

        for k in keys:
            self[k] = None

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        try:
            self.archive_id = conf.getint('Archive', 'archive_id')
        except ValueError:
            print('Error in configuration file '
                  '([{}], {})'.format('Archive', attr))
        except configparser.Error:
            pass

    @staticmethod
    def strip_header_comments(header):
        """
        Strip comment lines from header.

        """

        header_strip = fits.Header()

        for c in header.cards:
            if c[0] != 'COMMENT':
                header_strip.append(c, bottom=True)

        return header_strip

    def get_header(self, keyword=None):
        """
        Return Astrometry.net or SCAMP header.

        Parameters
        ----------
        keyword : str
            A keyword specifying which header to return

        """

        if self['header_scamp']:
            header = self['header_scamp'].copy()
        elif self['header_anet']:
            header = self['header_anet'].copy()
        else:
            header = None

        if keyword == 'header_scamp':
            if self['header_scamp']:
                header = self['header_scamp'].copy()
            else:
                header = None

        if keyword == 'header_anet':
            if self['header_anet']:
                header = self['header_anet'].copy()
            else:
                header = None

        if keyword == 'header_wcs':
            if self['header_wcs']:
                header = self['header_wcs'].copy()
            else:
                header = None

        return header

    def calculate_parameters(self):
        """
        Calculate solution-related parameters.

        """

        if self['header_wcs'] is None:
            self.log.write('No solution found, cannot calculate '
                           'solution-related parameters',
                           level=2, event=34)
            return

        self.wcs = wcs.WCS(self['header_wcs'])
        self['ra_icrs'] = self['header_wcs']['CRVAL1']
        self['dec_icrs'] = self['header_wcs']['CRVAL2']
        self['cd1_1'] = self['header_wcs']['CD1_1']
        self['cd1_2'] = self['header_wcs']['CD1_2']
        self['cd2_1'] = self['header_wcs']['CD2_1']
        self['cd2_2'] = self['header_wcs']['CD2_2']

        pix_edge_midpoints = np.array([[1., (self.imheight+1.)/2.],
                                       [self.imwidth, (self.imheight+1.)/2.],
                                       [(self.imwidth + 1.)/2., 1.],
                                       [(self.imwidth + 1.)/2., self.imheight]])
        edge_midpoints = self.wcs.all_pix2world(pix_edge_midpoints, 1)

        c1 = SkyCoord(ra=edge_midpoints[0,0], dec=edge_midpoints[0,1],
                      unit=(u.deg, u.deg))
        c2 = SkyCoord(ra=edge_midpoints[1,0], dec=edge_midpoints[1,1],
                      unit=(u.deg, u.deg))
        c3 = SkyCoord(ra=edge_midpoints[2,0], dec=edge_midpoints[2,1],
                      unit=(u.deg, u.deg))
        c4 = SkyCoord(ra=edge_midpoints[3,0], dec=edge_midpoints[3,1],
                      unit=(u.deg, u.deg))
        self['fov1'] = c1.separation(c2)
        self['fov2'] = c3.separation(c4)

        self['source_density'] = (self.num_sources_sixbins
                                  / self.rel_area_sixbins
                                  / (self['fov1']*self['fov2']))

        pixscale1 = self['fov1'].to(u.arcsec) / (self.imwidth * u.pixel)
        pixscale2 = self['fov2'].to(u.arcsec) / (self.imheight * u.pixel)
        self['pixel_scale'] = (pixscale1 + pixscale2) / 2.

        # Check if a celestial pole is nearby or on the plate
        self['half_diag'] = np.sqrt(self['fov1']**2 + self['fov2']**2) / 2.
        self['ncp_close'] = 90. - self['dec_icrs'] <= self['half_diag'] / u.deg
        self['scp_close'] = 90. + self['dec_icrs'] <= self['half_diag'] / u.deg
        ncp_on_plate = False
        scp_on_plate = False

        if self['ncp_close']:
            ncp_pix = self.wcs.all_world2pix([[self['ra_icrs'], 90.]], 1,
                                             quiet=True)

            if (ncp_pix[0,0] > 0 and ncp_pix[0,0] < self.imwidth
                and ncp_pix[0,1] > 0 and ncp_pix[0,1] < self.imheight):
                ncp_on_plate = True

        if self['scp_close']:
            scp_pix = self.wcs.all_world2pix([[self['ra_icrs'], -90.]], 1,
                                             quiet=True)

            if (scp_pix[0,0] > 0 and scp_pix[0,0] < self.imwidth
                and scp_pix[0,1] > 0 and scp_pix[0,1] < self.imheight):
                scp_on_plate = True

        self['ncp_on_plate'] = ncp_on_plate
        self['scp_on_plate'] = scp_on_plate

        # Construct coordinate strings
        ra_angle = Angle(self['ra_icrs'], u.deg)
        dec_angle = Angle(self['dec_icrs'], u.deg)

        self['ra_icrs_hms'] = ra_angle.to_string(unit=u.hour, sep=':',
                                                 precision=1, pad=True)
        self['dec_icrs_dms'] = dec_angle.to_string(unit=u.deg, sep=':',
                                                   precision=1, pad=True)
        self['stc_box'] = ('Box ICRS {:.5f} {:.5f} {:.5f} {:.5f}'
                           .format(self['header_wcs']['CRVAL1'],
                                   self['header_wcs']['CRVAL2'],
                                   self['fov1'].value, self['fov2'].value))

        pix_corners = np.array([[1., 1.], [self.imwidth, 1.],
                               [self.imwidth, self.imheight],
                               [1., self.imheight]])
        corners = self.wcs.all_pix2world(pix_corners, 1)
        self['stc_polygon'] = ('Polygon ICRS {:.5f} {:.5f} {:.5f} {:.5f} '
                               '{:.5f} {:.5f} {:.5f} {:.5f}'
                               .format(corners[0,0], corners[0,1],
                                       corners[1,0], corners[1,1],
                                       corners[2,0], corners[2,1],
                                       corners[3,0], corners[3,1]))
        self['skycoord_corners'] = SkyCoord(corners[:,0] * u.deg,
                                            corners[:,1] * u.deg,
                                            frame='icrs')

        # Calculate plate rotation angle
        try:
            cp = np.array([self['header_wcs']['CRPIX1'],
                           self['header_wcs']['CRPIX2']])

            if dec_angle.deg > 89.:
                cn = self.wcs.wcs_world2pix([[self['ra_icrs'], 90.]], 1)
            else:
                cn = self.wcs.wcs_world2pix([[self['ra_icrs'],
                                              self['dec_icrs'] + 1.]], 1)

            if ra_angle.deg > 359.:
                ce = self.wcs.wcs_world2pix([[self['ra_icrs'] - 359.,
                                              self['dec_icrs']]], 1)
            else:
                ce = self.wcs.wcs_world2pix([[self['ra_icrs'] + 1.,
                                              self['dec_icrs']]], 1)

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
            self.log.write('Could not calculate plate rotation angle',
                           level=2, event=34)

        self['rotation_angle'] = rotation_angle * u.deg
        self['plate_mirrored'] = plate_mirrored

        self.log.write('Image dimensions: {:.2f} x {:.2f} degrees'
                       ''.format(self['fov1'].value, self['fov2'].value),
                       double_newline=False)
        self.log.write('Mean pixel scale: {:.3f} arcsec'
                       ''.format(self['pixel_scale'].value),
                       double_newline=False)
        self.log.write('The image has {:.0f} stars per square degree'
                       ''.format(self['source_density'].value))
        self.log.write('Plate rotation angle: {}'
                       .format(self['rotation_angle'].value),
                       double_newline=False)
        self.log.write('Plate is mirrored: {}'.format(self['plate_mirrored']),
                       double_newline=False)
        self.log.write('North Celestial Pole is on the plate: {}'
                       .format(self['ncp_on_plate']),
                       double_newline=False)
        self.log.write('South Celestial Pole is on the plate: {}'
                       .format(self['scp_on_plate']))


class SolveProcess:
    """
    Plate solve process class

    """

    def __init__(self, filename, archive_id=None):
        self.filename = os.path.basename(filename)
        self.archive_id = archive_id
        self.basefn = ''
        self.fn_fits = ''
        self.process_id = None
        self.scan_id = None
        self.plate_id = None

        self.fits_dir = ''
        self.index_dir = ''
        self.gaia_dir = ''
        self.tycho2_dir = ''
        self.work_dir = ''
        self.write_source_dir = ''
        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''
        self.write_phot_dir = ''
        self.write_wcs_dir = ''
        self.write_log_dir = ''

        self.astref_catalog = None
        self.photref_catalog = None

        self.use_gaia_fits = False
        self.use_tycho2_fits = False

        self.use_ucac4_db = False
        self.ucac4_db_host = 'localhost'
        self.ucac4_db_user = ''
        self.ucac4_db_name = ''
        self.ucac4_db_passwd = ''
        self.ucac4_db_table = 'ucac4'

        self.use_apass_db = False
        self.apass_db_host = 'localhost'
        self.apass_db_user = ''
        self.apass_db_name = ''
        self.apass_db_passwd = ''
        self.apass_db_table = 'apass'

        self.output_db_host = 'localhost'
        self.output_db_user = ''
        self.output_db_name = ''
        self.output_db_passwd = ''
        self.write_sources_csv = False

        self.sextractor_path = 'sex'
        self.scamp_path = 'scamp'
        self.psfex_path = 'psfex'
        self.solve_field_path = 'solve-field'
        self.wcs_to_tan_path = 'wcs-to-tan'

        self.timestamp = dt.datetime.now()
        self.timestamp_str = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.scratch_dir = None
        self.enable_log = False
        self.log = None
        self.enable_db_log = False

        self.plate_epoch = 1950
        self.plate_year = int(self.plate_epoch)
        self.threshold_sigma = 4.
        self.use_filter = False
        self.filter_path = None
        self.use_psf = False
        self.psf_threshold_sigma = 20.
        self.psf_model_sigma = 20.
        self.min_model_sources = 100
        self.max_model_sources = 10000
        self.sip = 3
        self.skip_bright = 10
        self.allow_force = False
        self.repeat_find = True
        self.distort = 3
        self.subfield_distort = 1
        self.max_recursion_depth = 5
        self.force_recursion_depth = 0
        self.circular_film = False
        self.crossmatch_radius = None
        self.crossmatch_nsigma = 10.
        self.crossmatch_nlogarea = 2.
        self.crossmatch_maxradius = 20.

        self.plate_header = None
        self.platemeta = None
        self.imwidth = None
        self.imheight = None
        self.plate_solved = False
        self.mean_pixel_scale = None
        self.min_pixel_scale = None
        self.max_pixel_scale = None
        self.mean_fov1 = None
        self.mean_fov2 = None
        self.source_density = None
        self.num_sources = None
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None
        self.min_ra = None
        self.max_ra = None
        self.min_dec = None
        self.max_dec = None
        self.ncp_close = None
        self.scp_close = None
        self.ncp_on_plate = None
        self.scp_on_plate = None

        self.sources = None
        self.scampref = None
        self.scampcat = None
        self.solutions = None
        self.duplicate_solutions = None
        self.exp_numbers = None
        self.num_solutions = 0
        self.num_duplicate_solutions = 0
        self.num_iterations = 0
        self.pattern_x = None
        self.pattern_y = None
        self.pattern_ratio = None
        self.pattern_table = None
        self.astref_tables = []
        self.gaia_files = None
        self.neighbors_gaia = None
        self.sol_centroid = None
        self.sol_radius = None
        self.sol_max_sep = None
        self.astrom_sub = []
        self.phot_cterm = []
        self.phot_color = None
        self.phot_calib = []
        self.phot_calibrated = False
        self.phot_sub = []

        self.id_tyc = None
        self.id_tyc_pad = None
        self.hip_tyc = None
        self.ra_tyc = None
        self.dec_tyc = None
        self.btmag_tyc = None
        self.vtmag_tyc = None
        self.btmagerr_tyc = None
        self.vtmagerr_tyc = None
        self.num_tyc = 0

        self.id_ucac = None
        self.ra_ucac = None
        self.dec_ucac = None
        self.raerr_ucac = None
        self.decerr_ucac = None
        self.mag_ucac = None
        self.bmag_ucac = None
        self.vmag_ucac = None
        self.magerr_ucac = None
        self.bmagerr_ucac = None
        self.vmagerr_ucac = None
        self.num_ucac = 0

        self.ra_apass = None
        self.dec_apass = None
        self.bmag_apass = None
        self.vmag_apass = None
        self.berr_apass = None
        self.verr_apass = None
        self.num_apass = 0

        self.combined_ucac_apass = None

        self.ucac4_columns = OrderedDict([
            ('ucac4_ra', ('RAJ2000', 'f8')),
            ('ucac4_dec', ('DEJ2000', 'f8')),
            ('ucac4_raerr', ('e_RAJ2000', 'i')),
            ('ucac4_decerr', ('e_DEJ2000', 'i')),
            ('ucac4_mag', ('amag', 'f8')),
            ('ucac4_magerr', ('e_amag', 'f8')),
            ('ucac4_pmra', ('pmRA', 'f8')),
            ('ucac4_pmdec', ('pmDE', 'f8')),
            ('ucac4_id', ('UCAC4', 'a10')),
            ('ucac4_bmag', ('Bmag', 'f8')),
            ('ucac4_vmag', ('Vmag', 'f8')),
            ('ucac4_bmagerr', ('e_Bmag', 'f8')),
            ('ucac4_vmagerr', ('e_Vmag', 'f8'))
        ])

        self.apass_columns = OrderedDict([
            ('apass_ra', ('RAdeg', 'f8')),
            ('apass_dec', ('DEdeg', 'f8')),
            ('apass_bmag', ('B', 'f8')),
            ('apass_vmag', ('V', 'f8')),
            ('apass_bmagerr', ('e_B', 'f8')),
            ('apass_vmagerr', ('e_V', 'f8'))
        ])

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        try:
            self.archive_id = conf.getint('Archive', 'archive_id')
        except ValueError:
            print('Error in configuration file '
                  '([{}], {})'.format('Archive', attr))
        except configparser.Error:
            pass

        for attr in ['sextractor_path', 'scamp_path', 'psfex_path',
                     'solve_field_path', 'wcs_to_tan_path']:
            try:
                setattr(self, attr, conf.get('Programs', attr))
            except configparser.Error:
                pass

        for attr in ['fits_dir', 'index_dir', 'gaia_dir', 'tycho2_dir',
                     'work_dir', 'write_log_dir', 'write_phot_dir',
                     'write_source_dir', 'write_wcs_dir',
                     'write_db_source_dir', 'write_db_source_calib_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except configparser.Error:
                pass

        if self.write_log_dir:
            self.enable_log = True

        for attr in ['use_gaia_fits', 'use_tycho2_fits',
                     'use_ucac4_db', 'use_apass_db',
                     'enable_db_log', 'write_sources_csv']:
            try:
                setattr(self, attr, conf.getboolean('Database', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Database', attr))
            except configparser.Error:
                pass

        for attr in ['ucac4_db_host', 'ucac4_db_user', 'ucac4_db_name',
                     'ucac4_db_passwd', 'ucac4_db_table',
                     'apass_db_host', 'apass_db_user', 'apass_db_name',
                     'apass_db_passwd', 'apass_db_table',
                     'output_db_host', 'output_db_user',
                     'output_db_name', 'output_db_passwd']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except configparser.Error:
                pass

        for attr in ['use_filter', 'use_psf', 'circular_film',
                     'allow_force', 'repeat_find']:
            try:
                setattr(self, attr, conf.getboolean('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['plate_epoch', 'threshold_sigma',
                     'psf_threshold_sigma', 'psf_model_sigma',
                     'crossmatch_radius', 'crossmatch_nsigma',
                     'crossmatch_nlogarea', 'crossmatch_maxradius']:
            try:
                setattr(self, attr, conf.getfloat('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['sip', 'skip_bright', 'distort', 'subfield_distort',
                     'max_recursion_depth', 'force_recursion_depth',
                     'min_model_sources', 'max_model_sources']:
            try:
                setattr(self, attr, conf.getint('Solve', attr))
            except ValueError:
                print('Error in configuration file '
                      '([{}], {})'.format('Solve', attr))
            except configparser.Error:
                pass

        for attr in ['filter_path', 'astref_catalog', 'photref_catalog']:
            try:
                setattr(self, attr, conf.get('Solve', attr))
            except configparser.Error:
                pass

        # Read UCAC4 and APASS table column names from the dedicated sections,
        # named after the tables
        if conf.has_section(self.ucac4_db_table):
            for attr in self.ucac4_columns.keys():
                try:
                    colstr = conf.get(self.ucac4_db_table, attr)
                    _,typ = self.ucac4_columns[attr]
                    self.ucac4_columns[attr] = (colstr, typ)
                except configparser.Error:
                    pass

        if conf.has_section(self.apass_db_table):
            for attr in self.apass_columns.keys():
                try:
                    colstr = conf.get(self.apass_db_table, attr)
                    _,typ = self.apass_columns[attr]
                    self.apass_columns[attr] = (colstr, typ)
                except configparser.Error:
                    pass

    def find_scanner_pattern(self, coords_image, coords_ref):
        """
        Find if there is a pattern along one axis in the scan file.
        If found, fit a smooth curve to the difference between scan
        coordinates and reference coordinates along that axis.
        Return corrected image coordinates.

        Parameters
        ----------
        coords_image : array-like
            The coordinates of sources
        coords_ref : array-like
            The coordinates of reference stars

        """

        # All input arrays must have the same length
        assert len(coords_image) == len(coords_ref), 'coords_image and coords_ref must have the same length'
        assert len(coords_image) >= 10, 'Number of sources must be at least 10'

        # Prepare 1-dimensional arrays
        x_image = coords_image[:,0]
        y_image = coords_image[:,1]
        x_ref = coords_ref[:,0]
        y_ref = coords_ref[:,1]

        # Calculate differences
        dx = x_image - x_ref
        dy = y_image - y_ref

        # Make sure that lowess fraction includes at least 10 stars
        nstars = len(y_image)

        if nstars > 100:
            frac = 1. / np.sqrt(nstars)
        else:
            frac = 10. / len(y_image)

        # Find smooth curve along y-axis
        z = sm.nonparametric.lowess(dy, y_image, frac=frac, it=3,
                                    return_sorted=True)
        _,uind = np.unique(z[:,0], return_index=True)
        s_y = InterpolatedUnivariateSpline(z[uind,0], z[uind,1], k=1)

        # Find smooth curve along x-axis
        z = sm.nonparametric.lowess(dx, x_image, frac=frac, it=3,
                                    return_sorted=True)
        _,uind = np.unique(z[:,0], return_index=True)
        s_x = InterpolatedUnivariateSpline(z[uind,0], z[uind,1], k=1)

        y_range = np.max(y_image) - np.min(y_image)
        x_range = np.max(x_image) - np.min(x_image)
        yy = np.linspace(np.min(y_image) + 0.2 * y_range,
                         np.max(y_image) - 0.2 * y_range, 60)
        xx = np.linspace(np.min(x_image) + 0.2 * x_range,
                         np.max(x_image) - 0.2 * x_range, 60)

        # Scanner pattern exists if the standard deviation of
        # 60 points from the smooth curve along that axis is at least
        # 1.5 times as high as the standard deviation along the other axis
        std_ratio = s_y(yy).std() / s_x(xx).std()

        if std_ratio > 1.5:
            coords_image[:,1] = y_image - s_y(y_image)
        elif std_ratio < 2./3.:
            coords_image[:,0] = x_image - s_x(x_image)

        return coords_image, s_x, s_y, std_ratio

    def find_peak_separation(self, coords):
        """
        Find characteristic distance between nearest neighbors in coords.

        """

        n, dim = coords.shape
        assert dim == 2

        kdt = KDT(coords)
        ds,_ = kdt.query(coords, k=2)

        # Kernel density estimation (KDE) to distances
        kde = sm.nonparametric.KDEUnivariate(ds[:,1].astype(np.double))
        kde.fit()

        # Find peak in density
        a = kde.density.argmax()
        dist_peak = kde.support[a]

        # Return peak distance
        return dist_peak

    @staticmethod
    def _get_scale_rotation(xy1, xy2):
        """
        Find scale and rotation between two point clouds.

        """

        assert xy1.shape == xy2.shape
        n, dim = xy1.shape

        H = (xy1.T @ xy2) / n
        V, S, W = np.linalg.svd(H)

        if (np.linalg.det(V) * np.linalg.det(W)) < 0:
            S[-1] = -S[-1]
            V[:,-1] = -V[:,-1]

        R = V @ W
        scale = (np.sqrt((xy2**2).sum(axis=1)).sum()
                 / np.sqrt((xy1**2).sum(axis=1)).sum())
        rot_angle = np.degrees(np.arctan2(R[1,0], R[0,0]))

        if rot_angle > 90:
            rot_angle -= 180.
        elif rot_angle < -90:
            rot_angle += 180.

        return scale, rot_angle, R

    def find_multiexp_pattern(self, coords, dist_peak, numexp):
        """
        Find multi-exposure pattern in coords.

        """

        self.log.write('Finding multi-exposure pattern', level=3, event=31)

        n, dim = coords.shape
        assert dim == 3

        # Choose optimal clustering parameters
        if numexp > 20:
            eps = dist_peak * 2
            min_samples = 1
            num_select = 50 * numexp
        elif numexp > 10:
            eps = dist_peak * 3
            min_samples = 2
            num_select = 100 * numexp
        else:
            eps = dist_peak * 4
            min_samples = 4
            num_select = 200 * numexp

        coords_select = coords[:num_select]
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_select)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.log.write('Found {:d} source clusters'.format(num_clusters),
                       level=4, event=31, double_newline=False)

        # Find cluster sizes (label counts)
        lab, cnt = np.unique(labels[labels>-1], return_counts=True)

        # Group cluster sizes
        #c, cc = np.unique(cnt, return_counts=True)

        # Sort by difference between cluster size and numexp
        #ind_sort = np.argsort(np.abs(c-numexp))

        # Accept most probable cluster size (label count)
        #count_prob = c[ind_sort[0]]
        #num_accepted_clusters = cc[ind_sort[0]]
        #self.log.write('Use clusters with {:d} members'
        #               .format(count_prob), level=4, double_newline=False)
        #self.log.write('Such clusters appear {:d} times'
        #               .format(num_accepted_clusters), level=4,
        #               double_newline=False)

        # Select labels that have the accepted cluster size
        #labels_sel = lab[cnt == count_prob]
        labels_sel = lab[np.abs(cnt - numexp) < 0.1 * numexp]
        label_mask = np.isin(labels, labels_sel)
        num_accepted_clusters = len(labels_sel)

        if num_accepted_clusters < 5:
            self.log.write('Not enough source clusters found (<5)!',
                           level=2, event=31)
            return None
        else:
            self.log.write('Use {:d} source clusters'
                           .format(num_accepted_clusters), level=4, event=31,
                           double_newline=False)

        # Create array for source coordinates relative to cluster center
        xy = coords_select.copy()[:,:2]
        xy_mean = np.zeros((len(labels_sel), 2))

        # Create arrays for nearest-neighbour distances
        nnd_min = np.zeros(len(labels_sel))
        nnd_max = np.zeros(len(labels_sel))

        for i,k in enumerate(labels_sel):
            class_member_mask = (labels == k)
            #xy_mean[i,0] = np.mean(xy[class_member_mask,0])
            #xy_mean[i,1] = np.mean(xy[class_member_mask,1])

            # Calculate mean x and y of a cluster by averaging the coordinates
            # of its extreme members
            xy_mean[i,0] = (np.min(xy[class_member_mask,0]) +
                            np.max(xy[class_member_mask,0])) / 2.
            xy_mean[i,1] = (np.min(xy[class_member_mask,1]) +
                            np.max(xy[class_member_mask,1])) / 2.

            xyi = xy[class_member_mask]
            kdt_xyi = KDT(xyi)
            ds_xyi,_ = kdt_xyi.query(xyi, k=2)
            nnd_min[i] = ds_xyi[:,1].min()
            nnd_max[i] = ds_xyi[:,1].max()

        # Exclude clusters with outlying (min_nnd, max_nnd) values
        iqr_nnd_min = np.subtract(*np.percentile(nnd_min, [75, 25]))
        iqr_nnd_max = np.subtract(*np.percentile(nnd_max, [75, 25]))
        m = ((nnd_min > np.percentile(nnd_min, 25) - 1.5 * iqr_nnd_min) &
             (nnd_min < np.percentile(nnd_min, 75) + 1.5 * iqr_nnd_min) &
             (nnd_max > np.percentile(nnd_max, 25) - 1.5 * iqr_nnd_max) &
             (nnd_max < np.percentile(nnd_max, 75) + 1.5 * iqr_nnd_max))

        pattern_angle = np.zeros(m.sum())
        pattern_scale = np.zeros(m.sum()) + 1.

        # Analyse scale and rotation only if number of clusters is
        # above threshold
        if num_accepted_clusters > 10 and numexp < 10:
            scale_rot = True

            # Find cluster nearest to image center
            im_center = np.array(((self.imwidth + 1.) / 2.,
                                  (self.imheight + 1.) / 2.))
            dist_center = np.linalg.norm(xy_mean[m] - im_center, axis=1)
            ind_nearest = dist_center.argmin()

            # Find scale and rotation relative to the cluster closest to
            # image center
            class_member_mask = (labels == labels_sel[m][ind_nearest])
            xy0 = xy[class_member_mask]
            xy00 = xy[class_member_mask] - xy_mean[m][ind_nearest]

            for i,k in enumerate(labels_sel[m]):
                class_member_mask = (labels == k)

                if i != ind_nearest:
                    xyi = xy[class_member_mask] - xy_mean[m][i]
                    res = self._get_scale_rotation(xyi, xy00)
                    pattern_scale[i],pattern_angle[i],_ = res

            # Fit 2D plane
            angle_mask = np.abs(pattern_angle) < 10
            A_angle = np.c_[xy_mean[m][angle_mask,0], xy_mean[m][angle_mask,1],
                            np.ones(angle_mask.sum())]
            C_angle,_,_,_ = lstsq(A_angle, pattern_angle[angle_mask])
            A_scale = np.c_[xy_mean[m][:,0], xy_mean[m][:,1],
                            np.ones(xy_mean[m].shape[0])]
            C_scale,_,_,_ = lstsq(A_scale, pattern_scale)
        else:
            scale_rot = False

        # Subtract cluster center coordinates and apply scale/rotation
        for i,k in enumerate(labels_sel):
            class_member_mask = (labels == k)
            xy[class_member_mask,0] -= xy_mean[i,0]
            xy[class_member_mask,1] -= xy_mean[i,1]

            if scale_rot:
                theta = np.radians(C_angle[0] * xy_mean[i,0] +
                                   C_angle[1] * xy_mean[i,1] + C_angle[2])
                rot = np.array(((np.cos(theta), -np.sin(theta)),
                                (np.sin(theta),  np.cos(theta))))
                scale = (C_scale[0] * xy_mean[i,0] +
                         C_scale[1] * xy_mean[i,1] + C_scale[2])
                xy[class_member_mask] = xy[class_member_mask].dot(scale*rot)

        label_mask = np.isin(labels, labels_sel[m])
        xy_local = xy[label_mask]

        # Analyse clustering on xy_local
        eps = 5.

        if num_accepted_clusters > 4:
            min_samples = 4
        else:
            min_samples = num_accepted_clusters

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_local)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        noise_mask = (db.labels_ == -1)
        exp_labels = db.labels_
        num_exp_clusters = len(set(exp_labels)) - (1 if -1 in exp_labels else 0)
        self.log.write('Number of clusters within source pattern: {:d}'
                       .format(num_exp_clusters), level=4, event=31,
                       double_newline=False)

        if num_exp_clusters == 0:
            self.log.write('Finding multi-exposure pattern failed!',
                           level=2, event=31)
            return None

        # Find cluster centers and sort them
        xy_mean_exp = np.zeros((num_exp_clusters, 2))

        for i,e in enumerate(set(exp_labels[~noise_mask])):
            xy_exp = xy_local[exp_labels == e].copy()
            xy_mean_exp[i,0] = np.mean(xy_exp[:,0])
            xy_mean_exp[i,1] = np.mean(xy_exp[:,1])

        # Check whether exposure pattern is horizontal or vertical
        exp_pattern_ratio = ((xy_mean_exp[:,1].max() - xy_mean_exp[:,1].min()) /
                             (xy_mean_exp[:,0].max() - xy_mean_exp[:,0].min()))

        # Sort cluster centers
        sort_axis = 1 if exp_pattern_ratio > 1 else 0
        xy_mean_exp = xy_mean_exp[np.argsort(xy_mean_exp[:,sort_axis])]
        xy_mean_exp[:,0] -= xy_mean_exp[0,0]
        xy_mean_exp[:,1] -= xy_mean_exp[0,1]

        # Find locations of pattern and assign sources to exposures
        exp_num = np.zeros(len(coords), dtype=np.int)
        kdt_coords = KDT(coords[:,:2])
        xy_found = np.empty((0, 2))

        for i in np.arange(len(coords)):
            # Take a source from list
            xy_eval = coords[i,:2]

            # Search for matches
            if scale_rot:
                theta = np.radians(C_angle[0] * xy_eval[0] + C_angle[2])
                rot = np.array(((np.cos(-theta), -np.sin(-theta)),
                                (np.sin(-theta),  np.cos(-theta))))
                scale = 1./(C_scale[1] * xy_eval[1] + C_scale[2])
                xy_transform = xy_eval + xy_mean_exp.dot(scale*rot)
                ds,ind = kdt_coords.query(xy_transform)
            else:
                ds,ind = kdt_coords.query(xy_eval+xy_mean_exp)

            if (ds < 10).sum() == num_exp_clusters:
                # Append found pattern to list of finds
                xy_found = np.vstack((xy_found, xy_eval))
                exp_num[ind] = np.arange(1, num_exp_clusters+1)

        num_found = len(xy_found)
        self.log.write('Found {:d} patterns'.format(num_found),
                       level=4, event=31)

        if num_found < 10:
            self.log.write('Not enough patterns found (<10)!',
                           level=2, event=31)
            exp_num = None

        return exp_num

    def solve_plate(self, plate_epoch=None, sip=None, skip_bright=None,
                    repeat_find=None):
        """
        Solve astrometry in a FITS file.

        Parameters
        ----------
        plate_epoch : float
            Epoch of plate in decimal years (default 1950.0)
        sip : int
            SIP distortion order (default 3)
        skip_bright : int
            Number of brightest stars to skip when solving with Astrometry.net
            (default 10).
        repeat_find : bool
            If True, repeat finding astrometric solutions until none is found.
            If False, stop after finding the expected number of solutions.

        """

        if plate_epoch is None:
            plate_epoch = self.plate_epoch
            plate_year = self.plate_year
        else:
            self.plate_epoch = plate_epoch

            try:
                plate_year = int(plate_epoch)
            except ValueError:
                plate_year = self.plate_year

        if repeat_find is None:
            repeat_find = self.repeat_find

        # Log Astrometry.net and SCAMP versions
        anet_ver = (sp.check_output([self.solve_field_path, '-h']).strip()
                    .decode('utf-8').split('\n')[4])
        scamp_ver = (sp.check_output([self.scamp_path, '-v']).strip()
                     .decode('utf-8'))
        self.log.write('Using Astrometry.net {}'.format(anet_ver),
                       level=4, event=30, double_newline=False)
        self.log.write('Using {}'.format(scamp_ver), level=4, event=30,
                       double_newline=False)

        # Log plate epoch
        self.log.write('Using plate epoch of {:.2f}'.format(plate_epoch),
                       level=4, event=30, double_newline=False)

        if sip is None:
            sip = self.sip

        if skip_bright is None:
            skip_bright = self.skip_bright

        # By default, use 1000 sources for solving
        num_keep = 1000

        try:
            numexp = self.platemeta['numexp']
        except Exception:
            numexp = 1

        # Calculate a factor that takes into account how much the longest
        # and shortest exposure times differ
        try:
            max_exp = np.max(self.platemeta['exptime'])
            min_exp = np.min(self.platemeta['exptime'])

            if min_exp > 0:
                exp_factor = np.log10(max_exp / min_exp) + 1.
            else:
                exp_factor = 1.

            exp_factor = min(max(1., exp_factor), 4.)
        except Exception:
            exp_factor = 1.

        num_keep = max([num_keep, int(self.num_sources_sixbins * numexp *
                                      exp_factor / 100)])

        # Limit number of stars with 100000/numexp
        num_keep = min([num_keep, int(100000/numexp)])

        self.log.write('Using max {:d} stars for astrometric solving'.format(num_keep),
                       level=4, event=30, double_newline=False)

        # Create another xy list for faster solving
        # Keep 1000 stars in brightness order, skip the brightest
        # Use only sources from annular bins 1-6

        # Check if artifacts have been classified.
        # If yes, then select sources that are classified as true sources.
        # If not, then rely only on flag_clean.
        #if np.isnan(self.sources['model_prediction']).sum() == 0 and numexp < 2:
        #    bclean = ((self.sources['flag_clean'] == 1) &
        #              (self.sources['model_prediction'] > 0.9))
        #else:
        #    bclean = self.sources['flag_clean'] == 1

        # Use only flag_clean, because model_prediction is not robust enough
        bclean = self.sources['flag_clean'] == 1

        indclean = np.where(bclean & (self.sources['annular_bin'] <= 8))[0]
        sb = skip_bright
        indsort = np.argsort(self.sources[indclean]['mag_auto'])[sb:sb+num_keep]
        indsel = indclean[indsort]
        nrows = len(indsel)

        self.log.write('Selected {:d} stars for astrometric solving'.format(nrows),
                       level=4, event=30)

        self.astrom_sources = self.sources[indsel]
        self.solutions = []
        self.duplicate_solutions = []
        self.num_solutions = 0
        self.num_duplicate_solutions = 0

        try:
            numexp = self.platemeta['numexp']
        except Exception:
            numexp = 1

        if numexp > 4 and have_sklearn:
            coords = np.empty((nrows, 3))
            coords[:,0] = self.astrom_sources['x_source']
            coords[:,1] = self.astrom_sources['y_source']
            coords[:,2] = 0.
            dist_peak = self.find_peak_separation(coords[:,:2])

            coords[:,2] = (self.astrom_sources['mag_auto'] * dist_peak * 10.
                           / exp_factor**2)
            self.exp_numbers = self.find_multiexp_pattern(coords, dist_peak,
                                                          numexp)

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
        fn_xy = os.path.join(self.scratch_dir, '{}.xy'.format(self.basefn))
        xycat.write(fn_xy, format='fits', overwrite=True)

        use_force = False

        # Repeat finding astrometric solutions until none is found
        while True:
            solution, astref_table = self.find_astrometric_solution(ref_year=plate_year, sip=sip,
                                                                    use_force=use_force)

            if solution is None:
                if (use_force or self.num_solutions > 0 or
                    not self.allow_force):
                    break
                else:
                    use_force = True
                    continue
            else:
                use_force = False

            unique_solution = True

            # If this is not the first solution, check if we got a unique
            # solution
            if self.num_solutions > 0:
                sol_ra = np.array([sol['ra_icrs'] for sol in self.solutions])
                sol_dec = np.array([sol['dec_icrs'] for sol in self.solutions])
                c_sol = SkyCoord(sol_ra * u.deg, sol_dec * u.deg, frame='icrs')
                c_cur = SkyCoord(solution['ra_icrs'] * u.deg,
                                 solution['dec_icrs'] * u.deg, frame='icrs')
                min_sep = c_cur.separation(c_sol).min()

                if min_sep > 1. * u.deg:
                    sep_str = '{:.2f} degrees'.format(min_sep.deg)
                elif min_sep > 1. * u.arcmin:
                    sep_str = '{:.2f} arcmin'.format(min_sep.arcmin)
                else:
                    sep_str = '{:.2f} arcsec'.format(min_sep.arcsec)

                self.log.write('Current solution is separated from the '
                               'nearest solution by {}'.format(sep_str),
                               level=4, event=32)

                if min_sep < (10. * solution['pixel_scale'] * u.pixel):
                    unique_solution = False
                    unique_num = c_cur.separation(c_sol).argmin() + 1

            if unique_solution:
                solution['solution_num'] = self.num_solutions + 1
                self.solutions.append(solution)
                self.astref_tables.append(astref_table)
                self.num_solutions = len(self.solutions)

                self.log.write('Current solution is unique; assigned number '
                               '{:d}'.format(solution['solution_num']),
                               level=4, event=32,
                               solution_num=solution['solution_num'])
            else:
                solution['solution_num'] = -self.num_duplicate_solutions - 1
                solution['unique_num'] = unique_num
                self.duplicate_solutions.append(solution)
                self.num_duplicate_solutions = len(self.duplicate_solutions)

                self.log.write('Current solution is a duplicate of '
                               'solution {:d}; assigned number {:d}'
                               .format(unique_num, solution['solution_num']),
                               level=4, event=32,
                               solution_num=solution['solution_num'])

            # Check the number of remaining solutions
            try:
                num_remain_exp = self.platemeta['numexp'] - self.num_solutions
            except Exception:
                num_remain_exp = 1 - self.num_solutions

            # If the expected number of solutions is found and repeat_find is
            # False, then stop finding.
            if repeat_find == False and num_remain_exp < 1:
                break

        # Improve astrometric solutions (two iterations)
        if self.plate_solved:
            self.improve_astrometric_solutions(distort=3)
            self.improve_astrometric_solutions()

            # Calculate mean pixel scale and FOV across all solutions
            pixscales = u.Quantity([sol['pixel_scale'] for sol in self.solutions])
            self.mean_pixel_scale = pixscales.mean()
            self.min_pixel_scale = pixscales.min()
            self.max_pixel_scale = pixscales.max()
            fov1 = u.Quantity([sol['fov1'] for sol in self.solutions])
            fov2 = u.Quantity([sol['fov2'] for sol in self.solutions])
            self.mean_fov1 = fov1.mean()
            self.mean_fov2 = fov2.mean()
            dens = u.Quantity([sol['source_density'] for sol in self.solutions])
            self.source_density = dens.mean()

        # Create PlateSolution instance
        plate_solution = PlateSolution()

        # Assign solutions and parameters to PlateSolution instance
        for attr in ['imwidth', 'imheight', 'plate_solved',
                     'num_solutions', 'solutions', 'num_iterations',
                     'num_duplicate_solutions', 'duplicate_solutions',
                     'pattern_x', 'pattern_y', 'pattern_ratio',
                     'pattern_table', 'mean_pixel_scale',
                     'min_pixel_scale', 'max_pixel_scale',
                     'mean_fov1', 'mean_fov2', 'source_density']:
            setattr(plate_solution, attr, getattr(self, attr))

        # Calculate solutions centroid
        if self.plate_solved:
            plate_solution.calculate_centroid()

        # Return
        return plate_solution

    def find_astrometric_solution(self, ref_year=None, sip=None,
                                  use_force=False):
        """
        Solve astrometry for a list of sources.

        Parameters
        ----------
        ref_year : int
            Year number that will be used for reference coordinates
        sip : int
            SIP distortion order (default 3)

        """

        self.log.write('Looking for an astrometric solution', level=3, event=32)

        if ref_year is None:
            ref_year = self.plate_year

        if sip is None:
            sip = self.sip

        num_astrom_sources = len(self.astrom_sources)

        try:
            num_remain_exp = self.platemeta['numexp'] - self.num_solutions
        except Exception:
            num_remain_exp = 1

        self.log.write('Number of remaining solutions: {:d}'
                       .format(num_remain_exp), level=4, event=32)

        if use_force:
            self.log.write('Using force to find solution', level=4, event=32)

        # Current solution sequence number
        solution_seq = self.num_solutions + self.num_duplicate_solutions + 1

        # If sources have been numbered according to exposures,
        # then select sources that match the current exposure number
        if (self.exp_numbers is not None
            and len(self.exp_numbers) > 4
            and self.exp_numbers.max() > self.num_solutions
            and use_force == False):
            self.log.write('Selecting sources that match the exposure number '
                           '{:d}'.format(solution_seq),
                           level=4, event=32)
            indmask = (self.exp_numbers == solution_seq)
            use_sources = self.astrom_sources[indmask]
            num_use_sources = indmask.sum()

        # If number of remaining exposures is larger than 2, then
        # select sources that have two nearest neighbours within 90-degree
        # angle
        elif num_remain_exp > 2 and use_force == False:
            self.log.write('Selecting sources that have two nearest neighbours '
                           'within 90-degree angle', level=4, event=32)
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

            # Find clusters of vectors (from the nearest neighbor to the
            # second nearest) and take the largest cluster
            if have_sklearn:
                dx = x2[indmask] - x1[indmask]
                dy = y2[indmask] - y1[indmask]
                dxdy = np.vstack((dx, dy)).T
                db = DBSCAN(eps=2.0, min_samples=10).fit(dxdy)
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

                        #self.log.write('i, count: {:d} {:d}'.format(i, cnt[i]))
                        #self.log.write('dx mean, stddev: {:.3f} {:.3f}'
                        #               .format(dxm, dxs), double_newline=False)
                        #self.log.write('dy mean, stddev: {:.3f} {:.3f}'
                        #               .format(dym, dys))

                        # Check if cluster is compact and away from center
                        # If yes, then interrupt the loop
                        if np.maximum(dxs, dys) < np.sqrt(dxm**2 + dym**2):
                            break

                    #self.log.write('i: {:d}'.format(i))
                    #self.log.write('dx mean, stddev: {:.3f} {:.3f}'
                    #               .format(dxm, dxs), double_newline=False)
                    #self.log.write('dy mean, stddev: {:.3f} {:.3f}'
                    #               .format(dym, dys))
                    use_sources = self.astrom_sources[indmask][clumpmask]
                    num_use_sources = clumpmask.sum()
                else:
                    use_sources = self.astrom_sources[indmask]
                    num_use_sources = indmask.sum()
            else:
                use_sources = self.astrom_sources[indmask]
                num_use_sources = indmask.sum()

            # Output for checking
            t = Table()
            t['x'] = x0[indmask]
            t['y'] = y0[indmask]
            t['dx1'] = x1[indmask] - x0[indmask]
            t['dy1'] = y1[indmask] - y0[indmask]
            t['dx2'] = x2[indmask] - x0[indmask]
            t['dy2'] = y2[indmask] - y0[indmask]
            t['label'] = labels
            basefn_solution = '{}-{:02d}'.format(self.basefn, solution_seq)
            fn_out = os.path.join(self.scratch_dir,
                                  '{}_dxy.fits'.format(basefn_solution))
            t.write(fn_out, format='fits', overwrite=True)

        # If use_force is True, then select stars that have been classified
        # as true sources, regardless of the number of exposures.
        # Also, narrow the selection to annular bins 1-6.
        elif use_force == True:
            if np.isnan(self.astrom_sources['model_prediction']).sum() == 0:
                btrue = self.astrom_sources['model_prediction'] > 0.9

                # If less than 5 sources are classified as true sources,
                # use all
                if btrue.sum() < 5:
                    btrue = np.full(len(self.astrom_sources), True)
            else:
                btrue = np.full(len(self.astrom_sources), True)

            bselect = btrue & (self.astrom_sources['annular_bin'] <= 6)

            # If less than 5 sources remain, select all
            if bselect.sum() < 5:
                bselect = np.full(len(self.astrom_sources), True)

            use_sources = self.astrom_sources[bselect]
            num_use_sources = bselect.sum()

        # In other cases, use all sources
        else:
            use_sources = self.astrom_sources
            num_use_sources = num_astrom_sources

        # Prepare filenames
        basefn_solution = '{}-{:02d}'.format(self.basefn, solution_seq)
        fn_xy = '{}.xy'.format(basefn_solution)
        fn_match = '{}.match'.format(basefn_solution)
        fn_corr = '{}.corr'.format(basefn_solution)

        # Prepare FITS file with a list of sources and write the file to disk
        xycat = Table()
        xycat['X_IMAGE'] = use_sources['x_source']
        xycat['Y_IMAGE'] = use_sources['y_source']
        xycat['MAG_AUTO'] = use_sources['mag_auto']
        xycat['FLUX'] = use_sources['flux_auto']
        xycat['X_IMAGE'].unit = 'pixel'
        xycat['Y_IMAGE'].unit = 'pixel'
        xycat['MAG_AUTO'].unit = 'mag'
        xycat['FLUX'].unit = 'count'

        fn_xy = os.path.join(self.scratch_dir, fn_xy)
        xycat.write(fn_xy, format='fits', overwrite=True)

        # Write backend config file
        fconf = open(os.path.join(self.scratch_dir,
                                  self.basefn + '_backend.cfg'), 'w')
        index_path = os.path.join(self.index_dir,
                                  'index_{:d}'.format(ref_year))
        fconf.write('add_path {}\n'.format(index_path))
        fconf.write('autoindex\n')
        fconf.write('inparallel\n')
        fconf.close()

        # Construct the solve-field call
        cmd = self.solve_field_path
        cmd += ' {}'.format(fn_xy)
        cmd += ' --width {:d}'.format(self.imwidth)
        cmd += ' --height {:d}'.format(self.imheight)
        cmd += ' --x-column X_IMAGE'
        cmd += ' --y-column Y_IMAGE'
        cmd += ' --sort-column MAG_AUTO'
        cmd += ' --sort-ascending'
        cmd += ' --backend-config {}_backend.cfg'.format(self.basefn)

        cmd += ' --crpix-center'
        cmd += ' --scamp {}.cat'.format(basefn_solution)
        cmd += ' --scamp-config {}_scamp.conf'.format(self.basefn)
        cmd += ' --no-plots'
        cmd += ' --out {}'.format(basefn_solution)
        cmd += ' --match {}'.format(fn_match)
        cmd += ' --rdls none'
        cmd += ' --corr {}'.format(fn_corr)
        cmd += ' --overwrite'
        cmd += ' --pixel-error 3'
        cmd += ' --uniformize 100'

        if self.num_solutions > 0:
            scale0 = self.solutions[0]['pixel_scale']
            scale_low = 0.95 * scale0
            scale_high = 1.05 * scale0
            cmd += ' --scale-units arcsecperpix'
            cmd += ' --scale-low {:.3f}'.format(scale_low.value)
            cmd += ' --scale-high {:.3f}'.format(scale_high.value)

        # If the number of solutions is larger than 4, then accept
        # solutions with lower odds
        numexp_condition = False

        try:
            if self.platemeta['numexp'] > 4:
                numexp_condition = True
        except Exception:
            pass

        if numexp_condition or self.num_solutions > 4:
            cmd += ' --odds-to-solve 1e8'

        if num_remain_exp > 0:
            # Higher cpu limit for the first solution
            if use_force:
                cmd += ' --cpulimit 600'
            elif self.num_solutions == 0:
                cmd += ' --cpulimit 300'
            else:
                cmd += ' --cpulimit 180'
        else:
            # Low limit for extra solutions (after numexp from metadata)
            cmd += ' --cpulimit 30'

        cmd_no_tweak = cmd + ' --no-tweak'

        if sip > 0:
            cmd += ' --tweak-order {:d}'.format(sip)
        else:
            cmd += ' --no-tweak'

        self.log.write('Subprocess: {}'.format(cmd), level=5, event=32)
        sp.call(cmd, shell=True, stdout=self.log.handle,
                stderr=self.log.handle, cwd=self.scratch_dir)
        self.log.write('', timestamp=False, double_newline=False)

        # Check the result of solve-field
        fn_solved = os.path.join(self.scratch_dir, '{}.solved'.format(basefn_solution))
        fn_wcs = os.path.join(self.scratch_dir, '{}.wcs'.format(basefn_solution))
        fn_match = os.path.join(self.scratch_dir, fn_match)
        fn_corr = os.path.join(self.scratch_dir, fn_corr)

        if os.path.exists(fn_solved) and os.path.exists(fn_wcs):
            wcshead = fits.getheader(fn_wcs)

            # If SIP distortions were not computed, then repeat solving
            # without tweaking (plane TAN projection)
            if sip > 0 and wcshead['CTYPE1'] == 'RA---TAN':
                self.log.write('Repeating astrometric solving without SIP distortions',
                               level=4, event=32)
                self.log.write('Subprocess: {}'.format(cmd_no_tweak),
                               level=5, event=32)
                sp.call(cmd_no_tweak, shell=True, stdout=self.log.handle,
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

        if os.path.exists(fn_solved) and os.path.exists(fn_wcs):
            self.plate_solved = True
            self.log.write('Astrometric solution found (sequence number {:d})'
                           .format(solution_seq), level=4, event=32)
            #self.db_update_process(solved=1)
        else:
            if self.num_solutions > 0:
                self.log.write('Could not find additional astrometric solutions',
                               level=4, event=32)
            else:
                self.log.write('Could not solve astrometry for the plate',
                               level=2, event=32)
                #self.db_update_process(solved=0)
            return None, None

        # Read Astrometry.net solution from file
        header_anet = fits.getheader(fn_wcs)
        header_wcs = header_anet.copy()
        header_wcs.set('NAXIS', 2)
        header_wcs.set('NAXIS1', self.imwidth, after='NAXIS')
        header_wcs.set('NAXIS2', self.imheight, after='NAXIS1')
        header_wcs_anet = header_wcs

        # Create AstrometricSolution instance and calculate parameters
        self.log.write('Calculating parameters for the initial solution '
                       '(sequence number {:d})'.format(solution_seq),
                       level=4, event=32)

        solution = AstrometricSolution()
        solution.assign_conf(self.conf)
        solution.populate()
        solution.imwidth = self.imwidth
        solution.imheight = self.imheight
        solution.num_sources_sixbins = self.num_sources_sixbins
        solution.rel_area_sixbins = self.rel_area_sixbins
        solution.log = self.log

        solution['header_anet'] = header_anet
        solution['header_wcs'] = header_wcs
        solution.calculate_parameters()

        #match_table = Table.read(fn_match)
        #half_diag = match_table['RADIUS'][0]

        # Get reference stars
        astref_table = self.get_reference_stars_for_solution(solution)

        # Improve solution with SCAMP
        self.log.write('Improving solution and recalculating parameters '
                       '(sequence number {:d})'.format(solution_seq),
                       level=4, event=32)

        # Create scampref file
        numref = len(astref_table)
        scampref = new_scampref()
        hduref = fits.BinTableHDU.from_columns(scampref[2].columns, nrows=numref)
        hduref.data.field('X_WORLD')[:] = astref_table['ra']
        hduref.data.field('Y_WORLD')[:] = astref_table['dec']
        hduref.data.field('ERRA_WORLD')[:] = np.zeros(numref) + 1./3600.
        hduref.data.field('ERRB_WORLD')[:] = np.zeros(numref) + 1./3600.
        hduref.data.field('MAG')[:] = astref_table['mag']
        hduref.data.field('MAGERR')[:] = np.zeros(numref) + 0.1
        hduref.data.field('OBSDATE')[:] = np.zeros(numref) + 2000.
        scampref[2].data = hduref.data
        scampref_file = os.path.join(self.scratch_dir,
                                     '{}_scampref.cat'.format(basefn_solution))
        scampref.writeto(scampref_file, overwrite=True)

        # Create SCAMP .ahead files
        fn_tan = '{}_tan.wcs'.format(basefn_solution)
        cmd = self.wcs_to_tan_path
        cmd += ' -w {}.wcs'.format(basefn_solution)
        cmd += ' -x 1'
        cmd += ' -y 1'
        cmd += ' -W {:f}'.format(self.imwidth)
        cmd += ' -H {:f}'.format(self.imheight)
        cmd += ' -N 20'
        cmd += ' -o {}'.format(fn_tan)
        self.log.write('Subprocess: {}'.format(cmd), level=5, event=32)
        sp.call(cmd, shell=True, stdout=self.log.handle,
                stderr=self.log.handle, cwd=self.scratch_dir)

        tanhead = fits.getheader(os.path.join(self.scratch_dir, fn_tan))
        tanhead.set('NAXIS', 2)
        ahead = fits.Header()
        ahead.set('NAXIS', 2)
        ahead.set('NAXIS1', self.imwidth)
        ahead.set('NAXIS2', self.imheight)
        ahead.set('IMAGEW', self.imwidth)
        ahead.set('IMAGEH', self.imheight)
        ahead.set('CTYPE1', 'RA---TAN')
        ahead.set('CTYPE2', 'DEC--TAN')
        ahead.set('CRPIX1', (self.imwidth + 1.) / 2.)
        ahead.set('CRPIX2', (self.imheight + 1.) / 2.)
        ahead.set('CRVAL1', tanhead['CRVAL1'])
        ahead.set('CRVAL2', tanhead['CRVAL2'])
        ahead.set('CD1_1', tanhead['CD1_1'])
        ahead.set('CD1_2', tanhead['CD1_2'])
        ahead.set('CD2_1', tanhead['CD2_1'])
        ahead.set('CD2_2', tanhead['CD2_2'])
        aheadfile = os.path.join(self.scratch_dir,
                                 '{}.ahead'.format(basefn_solution))
        ahead.totextfile(aheadfile, endcard=True, overwrite=True)

        # Use crossid radius of 5 pixels and transform it to arcsec scale
        crossid_radius = 5. * u.pixel * solution['pixel_scale']

        # Filename for XML output
        fn_xml = '{}_scamp.xml'.format(basefn_solution)

        # Run SCAMP
        cmd = self.scamp_path
        cmd += ' -c {}_scamp.conf {}.cat'.format(self.basefn, basefn_solution)
        cmd += ' -ASTREF_CATALOG FILE'
        cmd += ' -ASTREFCAT_NAME {}_scampref.cat'.format(basefn_solution)
        cmd += ' -ASTREFCENT_KEYS X_WORLD,Y_WORLD'
        cmd += ' -ASTREFERR_KEYS ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD'
        cmd += ' -ASTREFMAG_KEY MAG'
        cmd += ' -ASTRCLIP_NSIGMA 1.5'
        cmd += ' -FLAGS_MASK 0x00ff'
        cmd += ' -SN_THRESHOLDS 20.0,100.0'
        cmd += ' -MATCH Y'
        #cmd += ' -ASTREF_WEIGHT 100'
        cmd += ' -PIXSCALE_MAXERR 1.01'
        cmd += ' -POSANGLE_MAXERR 0.1'
        cmd += ' -POSITION_MAXERR 0.2'
        cmd += ' -CROSSID_RADIUS {:.2f}'.format(crossid_radius
                                                .to(u.arcsec).value)
        cmd += ' -DISTORT_DEGREES 3'
        cmd += ' -PROJECTION_TYPE TPV'
        cmd += ' -STABILITY_TYPE EXPOSURE'
        cmd += ' -SOLVE_PHOTOM N'
        cmd += ' -WRITE_XML Y'
        cmd += ' -XML_NAME {}'.format(fn_xml)
        cmd += ' -VERBOSE_TYPE LOG'
        cmd += ' -CHECKPLOT_TYPE NONE'
        self.log.write('Subprocess: {}'.format(cmd), level=5, event=32)
        sp.call(cmd, shell=True, stdout=self.log.handle,
                stderr=self.log.handle, cwd=self.scratch_dir)

        # Read SCAMP solution
        fn_scamphead = os.path.join(self.scratch_dir,
                                    '{}.head'.format(basefn_solution))

        try:
            with open(fn_scamphead, 'r') as f:
                scamphead_str = f.read()

            # Get rid of non-ascii characters
            scamphead_str = unidecode.unidecode(scamphead_str)

            # Create FITS header from string
            header_scamp = fits.Header.fromstring(scamphead_str, sep='\n')
        except FileNotFoundError:
            header_scamp = None

        if header_scamp is not None:
            header_wcs = fits.PrimaryHDU().header
            header_wcs.set('NAXIS', 2)
            header_wcs.set('NAXIS1', self.imwidth)
            header_wcs.set('NAXIS2', self.imheight)
            header_wcs.set('IMAGEW', self.imwidth)
            header_wcs.set('IMAGEH', self.imheight)
            header_wcs.extend(header_scamp)

            # Fix SCAMP header if TPV projection is not specified
            if 'PV1_1' in header_wcs and header_wcs['CTYPE1'] == 'RA---TAN':
                header_wcs.set('CTYPE1', 'RA---TPV')
                header_wcs.set('CTYPE2', 'DEC--TPV')

            # Read SCAMP XML output
            fn_xml = os.path.join(self.scratch_dir, fn_xml)
            warnings.filterwarnings('ignore', message='.*W42.*',
                                    category=votable.exceptions.VOTableSpecWarning)
            scamp_stats = votable.parse_single_table(fn_xml,
                                                     pedantic=False).to_table()
            scamp_ndeg = scamp_stats['NDeg_Reference'][0]

            if scamp_ndeg > 5:
                solution['scamp_dscale'] = scamp_stats['DPixelScale'][0]
                solution['scamp_dangle'] = scamp_stats['DPosAngle'].quantity[0]
                solution['scamp_dx'] = scamp_stats['DX'].quantity[0].to(u.arcsec)
                solution['scamp_dy'] = scamp_stats['DY'].quantity[0].to(u.arcsec)
                scamp_sigmas = scamp_stats['AstromSigma_Reference'][0,:].quantity
                solution['scamp_sigma_1'] = scamp_sigmas[0]
                solution['scamp_sigma_2'] = scamp_sigmas[1]
                scamp_sigma_mean = np.sqrt(scamp_sigmas[0]**2 +
                                           scamp_sigmas[1]**2)
                solution['scamp_sigma_mean'] = scamp_sigma_mean
                solution['scamp_chi2'] = scamp_stats['Chi2_Reference'][0]
                solution['scamp_ndeg'] = scamp_stats['NDeg_Reference'][0]
                solution['scamp_distort'] = 3
                solution['scamp_iteration'] = 0

                # Store SCAMP solution and recalculate parameters
                solution['header_scamp'] = header_scamp

                if valid_wcs_header(header_wcs, self.imwidth, self.imheight):
                    solution['header_wcs'] = header_wcs
                    solution.calculate_parameters()
                else:
                    self.log.write('SCAMP WCS not valid!', level=2, event=32)

        # Crossmatch sources with rerefence stars and throw out
        # stars that matched

        if solution['scamp_sigma_mean'] is not None:
            xmatch_radius = (20. * solution['scamp_sigma_mean']
                             / solution['pixel_scale']
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
        solution['x_centroid'] = matched_sources['x_source'].mean()
        solution['y_centroid'] = matched_sources['y_source'].mean()
        xcenter = (self.imwidth + 1.) / 2.
        ycenter = (self.imheight + 1.) / 2.
        solution['rel_x_centroid'] = ((solution['x_centroid'] - xcenter)
                                      / self.imwidth)
        solution['rel_y_centroid'] = ((solution['y_centroid'] - ycenter)
                                      / self.imheight)
        solution['num_xmatch'] = mask_xmatch.sum()
        self.log.write('Centroid of matched sources: {:.2f} {:.2f}, '
                       'relative centroid: {:.3f} {:.3f}, '
                       'number of matches: {:d}'
                       .format(solution['x_centroid'], solution['y_centroid'],
                               solution['rel_x_centroid'],
                               solution['rel_y_centroid'],
                               solution['num_xmatch']),
                       level=4, event=32)

        # Output reference stars for debugging
        #t = Table()
        #t['x_ref'] = xr
        #t['y_ref'] = yr
        #t['ra_ref'] = astref_table['ra']
        #t['dec_ref'] = astref_table['dec']
        #t['mag_ref'] = astref_table['mag']
        #fn_out = os.path.join(self.scratch_dir, '{}_ref.fits'.format(basefn_solution))
        #t.write(fn_out, format='fits', overwrite=True)

        # Output crossmatched stars for debugging
        #t = Table()
        #t['x_source'] = matched_sources['x_source']
        #t['y_source'] = matched_sources['y_source']
        #t['x_ref'] = xr[ind_ref[mask_xmatch]]
        #t['y_ref'] = yr[ind_ref[mask_xmatch]]
        #t['dist'] = ds[mask_xmatch]
        #fn_out = os.path.join(self.scratch_dir, '{}_xmatch.fits'.format(basefn_solution))
        #t.write(fn_out, format='fits', overwrite=True)

        # Keep only stars that were not crossmatched
        self.astrom_sources = self.astrom_sources[ind_plate[indmask]]
        num_astrom_sources = len(self.astrom_sources)

        if self.exp_numbers is not None:
            self.exp_numbers = self.exp_numbers[ind_plate[indmask]]

        # Also, throw out stars that appear in the Astrometry.net .corr file
        corr_tab = Table.read(fn_corr)
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

        # Convert x,y to RA/Dec with the global WCS solution
        #pixcrd = np.column_stack((self.sources['x_source'],
        #                          self.sources['y_source']))
        #worldcrd = w.all_pix2world(pixcrd, 1)
        #self.sources['raj2000_wcs'] = worldcrd[:,0]
        #self.sources['dej2000_wcs'] = worldcrd[:,1]

        #solution.min_dec = np.min((worldcrd[:,1].min(), corners[:,1].min()))
        #solution.max_dec = np.max((worldcrd[:,1].max(), corners[:,1].max()))
        #solution.min_ra = np.min((worldcrd[:,0].min(), corners[:,0].min()))
        #solution.max_ra = np.max((worldcrd[:,0].max(), corners[:,0].max()))

        #if solution.max_ra-solution.min_ra > 180:
        #    ra_all = np.append(worldcrd[:,0], corners[:,0])
        #    max_below180 = ra_all[np.where(ra_all<180)].max()
        #    min_above180 = ra_all[np.where(ra_all>180)].min()

        #    if min_above180-max_below180 > 10:
        #        solution.min_ra = min_above180
        #        solution.max_ra = max_below180

        return solution, astref_table

    def improve_astrometric_solutions(self, distort=None):
        """
        Improve astrometric solution based on a list of sources and
        reference sources.

        Parameters
        ----------
        distort : int
            Distortion order for improved solution (default: 3)

        """

        self.log.write('Improving astrometric solutions', level=3, event=33)

        if self.num_solutions < 1:
            self.log.write('No astrometric solutions to improve!',
                           level=2, event=33)
            return

        if distort is None:
            distort = self.distort

        # Create array for xy coordinates of reference stars
        coords_ref = np.zeros((0,2))

        # Create lists for xy coordinates of reference stars,
        # separate for each solution
        x_ref_list = []
        y_ref_list = []

        # Go through solutions and build a list of reference stars
        for i in np.arange(self.num_solutions):
            solution = self.solutions[i]
            astref_table = self.astref_tables[i]

            if len(astref_table) > 0:
                w = wcs.WCS(solution['header_wcs'])
                xr,yr = w.all_world2pix(astref_table['ra'],
                                        astref_table['dec'], 1, quiet=True)
                mask_inside = ((xr > 0.5) & (xr < self.imwidth) &
                               (yr > 0.5) & (yr < self.imheight))
                xy_ref = np.vstack((xr[mask_inside], yr[mask_inside])).T

                # Include only isolated reference stars
                kdt = KDT(xy_ref)
                ds,ind = kdt.query(xy_ref, k=2)
                mask_isolated = ds[:,1] > 5
                num_isolated = mask_isolated.sum()
                add_ref = xy_ref[mask_isolated]
                coords_ref = np.append(coords_ref, add_ref, axis=0)
                x_ref_list.append(add_ref[:,0])
                y_ref_list.append(add_ref[:,1])
            else:
                x_ref_list.append(np.array([]))
                y_ref_list.append(np.array([]))

        if len(coords_ref) <= 10:
            self.log.write('Too few astrometric reference stars: '
                           '{:d}'.format(len(coords_ref)), level=2, event=33)
            return

        # Exclude reference stars that have close neighbours either
        # naturally or due to other solution
        kdt = KDT(coords_ref)
        ds,ind = kdt.query(coords_ref, k=2)
        mask_isolated = ds[:,1] > 5
        num_isolated = mask_isolated.sum()

        num_excluded = (ds[:,1] <= 5).sum()
        self.log.write('Excluded {:d} reference stars due to close neighbours, '
                       '{:d} stars remained.'
                       ''.format(num_excluded, num_isolated), level=4, event=33)

        if num_isolated <= 10:
            self.log.write('Too few isolated astrometric reference stars: '
                           '{:d}'.format(num_isolated), level=2, event=33)
            return

        coords_ref = coords_ref[mask_isolated]

        # Output reference stars for debugging
        t = Table()
        t['x_ref'] = coords_ref[:,0]
        t['y_ref'] = coords_ref[:,1]
        fn_out = os.path.join(self.scratch_dir, '{}_ref_all.fits'.format(self.basefn))
        t.write(fn_out, format='fits', overwrite=True)

        # Take clean sources from annular bins 1-9

        # Check if artifacts have been classified.
        # If yes, then select sources that are classified as true sources.
        # If not, then rely only on flag_clean.
        #if np.isnan(self.sources['model_prediction']).sum() == 0:
        #    bclean = ((self.sources['flag_clean'] == 1) &
        #              (self.sources['model_prediction'] > 0.9))
        #else:
        #    bclean = self.sources['flag_clean'] == 1

        # Use only flag_clean, because model_prediction is not robust enough
        bclean = self.sources['flag_clean'] == 1
        mask_bins = bclean & (self.sources['annular_bin'] <= 9)

        if mask_bins.sum() <= 10:
            self.log.write('Too few clean sources in annular bins 1--9: '
                           '{:d}'.format(num_isolated), level=2, event=33)
            return

        coords_plate = np.vstack((self.sources[mask_bins]['x_source'],
                                  self.sources[mask_bins]['y_source'])).T
        ind_sources = np.arange(self.num_sources)[mask_bins]

        # Exclude sources that have close neighbours
        kdt = KDT(coords_plate)
        ds,ind = kdt.query(coords_plate, k=2)
        mask_isolated = ds[:,1] > 5
        num_isolated = mask_isolated.sum()

        if num_isolated <= 10:
            self.log.write('Too few isolated sources in annular bins 1--9: '
                           '{:d}'.format(num_isolated), level=2, event=33)
            return

        coords_plate = coords_plate[mask_isolated]
        ind_sources = ind_sources[mask_isolated]

        # Crossmatch sources and reference stars
        ind_plate, ind_ref, _ = crossmatch_cartesian(coords_plate, coords_ref)

        if len(ind_plate) < 10:
            self.log.write('Too few crossmatches between sources and reference '
                           'stars: {:d}'.format(len(ind_plate)),
                           level=2, event=33)
            return

        ind_sources = ind_sources[ind_plate]
        coords_wobble = coords_plate[ind_plate]

        # Find scanner pattern and get pattern-subtracted coordinates
        res = self.find_scanner_pattern(coords_plate[ind_plate],
                                        coords_ref[ind_ref])
        coords_dewobbled, pattern_x, pattern_y, pattern_ratio = res
        nsrc = len(coords_dewobbled)
        self.pattern_x = pattern_x
        self.pattern_y = pattern_y

        if np.isfinite(pattern_ratio):
            self.pattern_ratio = pattern_ratio

        self.log.write('Scanner pattern ratio (stdev_y/stdev_x): '
                       '{:.3f}'.format(pattern_ratio), level=4, event=33)

        # Calculate scanner pattern and output to a table
        xx = np.arange(0, self.imwidth + 1, 50)
        yy = np.arange(0, self.imheight + 1, 50)
        xx_extra = ((xx < coords_wobble[:,0].min()) |
                    (xx > coords_wobble[:,0].max()))
        yy_extra = ((yy < coords_wobble[:,1].min()) |
                    (yy > coords_wobble[:,1].max()))
        xx_pattern = pattern_x(xx)
        yy_pattern = pattern_y(yy)

        t = Table()
        t['axis'] = np.append(np.full(len(xx), 1), np.full(len(yy), 2))
        t['coord'] = np.append(xx, yy)
        t['shift'] = np.append(xx_pattern, yy_pattern)
        t['extrapolated'] = np.append(xx_extra, yy_extra).astype(np.int)
        self.pattern_table = t

        # Create new array for xy coordinates of reference stars
        coords_ref = np.zeros((0,2))

        # Go through individual solutions and improve them by
        # running SCAMP
        for i in np.arange(self.num_solutions):
            solution = self.solutions[i]
            astref_table = self.astref_tables[i]
            x_ref = x_ref_list[i]
            y_ref = y_ref_list[i]
            basefn_solution = '{}-{:02d}'.format(self.basefn, i+1)

            fn_scampcat = os.path.join(self.scratch_dir,
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

            fn_scampcat = os.path.join(self.scratch_dir,
                                       '{}_dewobbled.cat'.format(basefn_solution))
            cat.writeto(fn_scampcat, overwrite=True)

            # Write SCAMP header to .ahead file
            fn_ahead = os.path.join(self.scratch_dir,
                                    '{}_dewobbled.ahead'.format(basefn_solution))
            header = solution['header_wcs']
            header.totextfile(fn_ahead, endcard=True, overwrite=True)

            # Use crossid radius of 3 pixels and transform it to arcsec scale
            crossid_radius = 3. * u.pixel * solution['pixel_scale']

            # Limit distortion degree based on the number of sources
            use_distort = distort

            if solution['scamp_ndeg'] is None or solution['scamp_ndeg'] < 100:
                use_distort = min(distort, 3)
            elif solution['scamp_ndeg'] < 500:
                use_distort = min(distort, 5)

            if solution['scamp_ndeg'] is None:
                ndeg_str = 'unknown'
            else:
                ndeg_str = '{:d}'.format(solution['scamp_ndeg'])

            self.log.write('Using distortion degree {:d} (scamp_ndeg: {})'
                           .format(use_distort, ndeg_str),
                           level=3, event=33, solution_num=i+1)

            # Filename for XML output
            fn_xml = '{}_dewobbled_scamp.xml'.format(basefn_solution)

            # Run SCAMP again on dewobbled pixel coordinates
            cmd = self.scamp_path
            cmd += ' -c {}_scamp.conf {}_dewobbled.cat'.format(self.basefn, basefn_solution)
            cmd += ' -ASTREF_CATALOG FILE'
            cmd += ' -ASTREFCAT_NAME {}_scampref.cat'.format(basefn_solution)
            cmd += ' -ASTREFCENT_KEYS X_WORLD,Y_WORLD'
            cmd += ' -ASTREFERR_KEYS ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD'
            cmd += ' -ASTREFMAG_KEY MAG'
            cmd += ' -ASTRCLIP_NSIGMA 1.5'
            cmd += ' -FLAGS_MASK 0x00ff'
            cmd += ' -SN_THRESHOLDS 20.0,100.0'
            cmd += ' -MATCH N'
            cmd += ' -CROSSID_RADIUS {:.2f}'.format(crossid_radius
                                                    .to(u.arcsec).value)
            cmd += ' -DISTORT_DEGREES {:d}'.format(use_distort)
            cmd += ' -PROJECTION_TYPE TPV'
            cmd += ' -STABILITY_TYPE EXPOSURE'
            cmd += ' -SOLVE_PHOTOM N'
            cmd += ' -WRITE_XML Y'
            cmd += ' -XML_NAME {}'.format(fn_xml)
            cmd += ' -VERBOSE_TYPE LOG'
            cmd += ' -CHECKPLOT_TYPE NONE'
            self.log.write('Subprocess: {}'.format(cmd), level=5, event=33)
            sp.call(cmd, shell=True, stdout=self.log.handle,
                    stderr=self.log.handle, cwd=self.scratch_dir)

            # Read SCAMP solution
            fn_scamphead = '{}_dewobbled.head'.format(basefn_solution)
            fn_scamphead = os.path.join(self.scratch_dir, fn_scamphead)

            with open(fn_scamphead, 'r') as f:
                scamphead_str = f.read()

            # Get rid of non-ascii characters
            scamphead_str = unidecode.unidecode(scamphead_str)

            # Create FITS header from string
            header_scamp = fits.Header.fromstring(scamphead_str, sep='\n')

            header_wcs = fits.PrimaryHDU().header
            header_wcs.set('NAXIS', 2)
            header_wcs.set('NAXIS1', self.imwidth)
            header_wcs.set('NAXIS2', self.imheight)
            header_wcs.set('IMAGEW', self.imwidth)
            header_wcs.set('IMAGEH', self.imheight)
            header_wcs.extend(header_scamp)

            # Fix SCAMP header if TPV projection is not specified
            if 'PV1_1' in header_wcs and header_wcs['CTYPE1'] == 'RA---TAN':
                header_wcs.set('CTYPE1', 'RA---TPV')
                header_wcs.set('CTYPE2', 'DEC--TPV')

            # Read SCAMP XML output
            fn_xml = os.path.join(self.scratch_dir, fn_xml)
            warnings.filterwarnings('ignore', message='.*W42.*',
                                    category=votable.exceptions.VOTableSpecWarning)
            scamp_stats = votable.parse_single_table(fn_xml, pedantic=False).to_table()
            scamp_ndeg = scamp_stats['NDeg_Reference'][0]

            if scamp_ndeg > 5:
                scamp_sigmas = scamp_stats['AstromSigma_Reference'][0,:].quantity
                self.solutions[i]['scamp_sigma_1'] = scamp_sigmas[0]
                self.solutions[i]['scamp_sigma_2'] = scamp_sigmas[1]
                scamp_sigma_mean = np.sqrt(scamp_sigmas[0]**2 +
                                           scamp_sigmas[1]**2)
                self.solutions[i]['scamp_sigma_mean'] = scamp_sigma_mean
                self.solutions[i]['scamp_chi2'] = scamp_stats['Chi2_Reference'][0]
                self.solutions[i]['scamp_ndeg'] = scamp_stats['NDeg_Reference'][0]
                self.solutions[i]['scamp_distort'] = use_distort
                self.solutions[i]['scamp_iteration'] = self.num_iterations + 1

                # Store improved solution
                self.solutions[i]['header_scamp'] = header_scamp

                if valid_wcs_header(header_wcs, self.imwidth, self.imheight):
                    self.solutions[i]['header_wcs'] = header_wcs
                    self.solutions[i].calculate_parameters()
                else:
                    self.log.write('SCAMP WCS for solution {:d} not valid!'
                                   .format(i+1),
                                   level=2, event=33, solution_num=i+1)

            # Crossmatch sources with rerefence stars
            w = wcs.WCS(header_wcs)
            xr,yr = w.wcs_world2pix(astref_table['ra'],
                                    astref_table['dec'], 1)

            coords_ref_sol = np.vstack((xr, yr)).T
            coords_ref = np.append(coords_ref, coords_ref_sol, axis=0)

            if scamp_ndeg > 5:
                tolerance = ((5. * scamp_sigma_mean / solution['pixel_scale'])
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
                self.solutions[i]['x_centroid'] = x_centroid
                self.solutions[i]['y_centroid'] = y_centroid
                xcenter = (self.imwidth + 1.) / 2.
                ycenter = (self.imheight + 1.) / 2.
                self.solutions[i]['rel_x_centroid'] = ((x_centroid - xcenter)
                                                       / self.imwidth)
                self.solutions[i]['rel_y_centroid'] = ((y_centroid - ycenter)
                                                       / self.imheight)
                self.solutions[i]['num_xmatch'] = mask_xmatch.sum()
                self.log.write('Centroid of matched sources: {:.2f} {:.2f}, '
                               'relative centroid: {:.3f} {:.3f}, '
                               'number of matches: {:d}'
                               .format(x_centroid, y_centroid,
                                       self.solutions[i]['rel_x_centroid'],
                                       self.solutions[i]['rel_y_centroid'],
                                       self.solutions[i]['num_xmatch']),
                               level=4, event=33, solution_num=i+1)
            else:
                self.log.write('Cannot calculate centroid of solution '
                               'due to no matched sources!',
                               level=2, event=33, solution_num=i+1)

            # Output crossmatched stars for debugging
            #t = Table()
            #t['x_source'] = coords_wobble[ind_plate[mask_xmatch]][:,0]
            #t['y_source'] = coords_wobble[ind_plate[mask_xmatch]][:,1]
            #t['x_dewobbled'] = coords_dewobbled[ind_plate[mask_xmatch]][:,0]
            #t['y_dewobbled'] = coords_dewobbled[ind_plate[mask_xmatch]][:,1]
            #t['x_ref'] = xr[ind_ref[mask_xmatch]]
            #t['y_ref'] = yr[ind_ref[mask_xmatch]]
            #t['dist'] = ds[mask_xmatch]
            #fn_out = os.path.join(self.scratch_dir,
            #                      '{}_xmatch2_{:d}.fits'.format(basefn_solution,
            #                                                    self.num_iterations+1))
            #t.write(fn_out, format='fits', overwrite=True)

        # Increase iteration count
        self.num_iterations += 1

    def get_reference_stars_for_solution(self, solution):
        """
        Get astrometric reference stars based on the WCS solution.

        """

        #self.log.write('Getting reference catalogs', level=3, event=40)

        # Read the Gaia EDR3 catalogue
        if self.use_gaia_fits:
            fn_gaia = os.path.join(self.gaia_dir, 'gaiaedr3_pyplate.fits')

            tab = Table.read(fn_gaia)

            # Calculate RA and Dec for the plate epoch
            ra_ref = (tab['ra'] + (self.plate_epoch - 2016.0) * tab['pmra']
                      / np.cos(tab['dec'] * np.pi / 180.) / 3600000.)
            dec_ref = (tab['dec'] + (self.plate_epoch - 2016.0)
                       * tab['pmdec'] / 3600000.)
            catalog = SkyCoord(ra_ref, dec_ref, frame='icrs')

            # Query stars around the plate center
            c = SkyCoord(solution['ra_icrs'] * u.deg,
                         solution['dec_icrs'] * u.deg, frame='icrs')
            dist = catalog.separation(c)
            mask_dist = dist < solution['half_diag']
            ind = np.arange(len(tab))[mask_dist]

            # Check which stars fall inside image area
            w = wcs.WCS(solution['header_wcs'])
            xr,yr = w.all_world2pix(ra_ref[mask_dist], dec_ref[mask_dist], 1,
                                    quiet=True)
            mask_inside = ((xr > 0) & (xr < self.imwidth) &
                           (yr > 0) & (yr < self.imheight))
            ind_ref = ind[mask_inside]

            numref = mask_inside.sum()
            self.log.write('Fetched {:d} entries from Gaia EDR3'
                           ''.format(numref))

            # Construct table for return
            astref_table = Table()
            astref_table['ra'] = ra_ref[ind_ref]
            astref_table['dec'] = dec_ref[ind_ref]
            astref_table['mag'] = tab[ind_ref]['phot_g_mean_mag']

            return astref_table

        # Read the Tycho-2 catalogue
        if self.use_tycho2_fits:
            #self.log.write('Reading the Tycho-2 catalogue', level=3, event=41)
            fn_tycho2 = os.path.join(self.tycho2_dir, 'tycho2_pyplate.fits')

            tycho2 = Table.read(fn_tycho2)

            #except IOError:
                #self.log.write('Missing Tycho-2 data', level=2, event=41)

            ra_tyc = tycho2['_RAJ2000']
            dec_tyc = tycho2['_DEJ2000']

            # For stars that have proper motion data, calculate RA, Dec
            # for the plate epoch
            pm_mask = np.isfinite(tycho2['pmRA']) & np.isfinite(tycho2['pmDE'])
            ra_tyc[pm_mask] = (ra_tyc[pm_mask]
                               + (self.plate_epoch - 2000.) * tycho2['pmRA'][pm_mask]
                               / np.cos(dec_tyc[pm_mask] * np.pi / 180.)
                               / 3600000.)
            dec_tyc[pm_mask] = (dec_tyc[pm_mask]
                                + (self.plate_epoch - 2000.)
                                * tycho2['pmDE'][pm_mask] / 3600000.)
            catalog = SkyCoord(ra_tyc, dec_tyc, frame='icrs')

            # Query stars around the plate center
            c = SkyCoord(solution['ra_icrs'] * u.deg,
                         solution['dec_icrs'] * u.deg, frame='icrs')
            dist = catalog.separation(c)
            mask_dist = dist < solution['half_diag']
            ind = np.arange(len(tycho2))[mask_dist]

            # Check which stars fall inside image area
            w = wcs.WCS(solution['header_wcs'])
            xr,yr = w.all_world2pix(ra_tyc[mask_dist], dec_tyc[mask_dist], 1,
                                    quiet=True)
            mask_inside = ((xr > 0) & (xr < self.imwidth) &
                           (yr > 0) & (yr < self.imheight))
            ind_ref = ind[mask_inside]

            numtyc = mask_inside.sum()
            self.log.write('Fetched {:d} entries from Tycho-2'
                           ''.format(numtyc))

            # Construct table for return
            astref_table = Table()
            astref_table['ra'] = ra_tyc[ind_ref]
            astref_table['dec'] = dec_tyc[ind_ref]
            astref_table['mag'] = tycho2[ind_ref]['BTmag']

            return astref_table

        return None
