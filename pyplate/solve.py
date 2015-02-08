import os
import glob
import shutil
import sys
import math
import datetime as dt
import subprocess as sp
import numpy as np
import ConfigParser
import warnings
from astropy import wcs
from astropy.io import fits
from astropy.io import votable
from astropy.coordinates import Angle
from astropy import units
from collections import OrderedDict
from .database import PlateDB
from .conf import read_conf
from ._version import __version__

try:
    from astropy.coordinates import ICRS
    use_newangsep = True
except ImportError:
    from astropy.coordinates import ICRSCoordinates as ICRS
    use_newangsep = False

try:
    from astropy.coordinates import match_coordinates_sky
    have_match_coord = True
except ImportError:
    have_match_coord = False

try:
    from pyspherematch import spherematch
    have_pyspherematch = True
except ImportError:
    have_pyspherematch = False

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

try:
    import MySQLdb
except ImportError:
    pass

try:
    from esutil import wcsutil
    have_esutil = True
except ImportError:
    have_esutil = False

try:
    import healpy
    have_healpy = True
except ImportError:
    have_healpy = False
    

class AstrometryNetIndex:
    """
    Astrometry.net index class

    """

    def __init__(self, *args):
        self.vizquery_path = 'vizquery'
        self.build_index_path = 'build-index'
        self.index_dir = './'
        
        if len(args) == 1:
            self.index_dir = args[0]

    def download_tycho2(self, site=None):
        """
        Download full Tycho-2 catalogue with vizquery.

        Parameters
        ----------
        site : str
            A site name that vizquery recognizes

        """

        fn_tyc = os.path.join(self.index_dir, 'tycho2_pyplate.fits')

        if not os.path.exists(fn_tyc):
            try:
                os.makedirs(self.index_dir)
            except OSError:
                if not os.path.isdir(self.index_dir):
                    raise

            cmd = self.vizquery_path
            cmd += (' -mime=binfits'
                    ' -source=I/259/tyc2'
                    ' -out="_RA _DE pmRA pmDE BTmag VTmag e_BTmag e_VTmag '
                    'TYC1 TYC2 TYC3 HIP"'
                    ' -out.max=unlimited')

            if site:
                cmd += ' -site={}'.format(site)

            # Download Tycho-2 catalogue to a temporary FITS file
            fn_vizout = os.path.join(self.index_dir, 'vizout.fits')

            with open(fn_vizout, 'wb') as vizout:
                sp.call(cmd, shell=True, stdout=vizout, cwd=self.index_dir)

            # Copy FITS file and remove first 24 bytes if file begins with "#"
            with open(fn_tyc, 'wb') as tycout:
                with open(fn_vizout, 'rb') as viz:
                    if viz.read(1) == '#':
                        viz.seek(24)
                    else:
                        viz.seek(0)

                    tycout.write(viz.read())

            os.remove(fn_vizout)

    def create_index_year(self, year, max_scale=None, min_scale=None,
                          sort_by='BTmag'):
        """
        Create Astrometry.net index for a given epoch.

        """

        if not max_scale:
            max_scale = 16
        elif max_scale > 19:
            max_scale = 19
        elif max_scale < 1:
            max_scale = 1

        if not min_scale:
            min_scale = 7
        elif min_scale > 19:
            min_scale = 19
        elif min_scale < 1:
            min_scale = 1

        if sort_by != 'BTmag' and sort_by != 'VTmag':
            sort_by = 'BTmag'

        tyc = fits.open(os.path.join(self.index_dir, 'tycho2_pyplate.fits'))
        data = tyc[1].data

        cols = tyc[1].columns[0:2] + tyc[1].columns[4:6]
        cols[0].name = 'RA'
        cols[1].name = 'Dec'
        hdu = fits.new_table(cols)
        tyc.close()

        hdu.data.field(0)[:] = data.field(0) + (year - 2000. + 0.5) * \
                data.field(2) / np.cos(data.field(1) * math.pi / 180.) / 3600000.
        hdu.data.field(1)[:] = data.field(1) + (year - 2000. + 0.5) * \
                data.field(3) / 3600000.
        hdu.data.field(2)[:] = data.field(4)
        hdu.data.field(3)[:] = data.field(5)

        # Leave out rows with missing proper motion and magnitudes
        mask1 = np.isfinite(hdu.data.field(0))
        mask2 = np.isfinite(hdu.data.field(2))
        mask3 = np.isfinite(hdu.data.field(3))
        hdu.data = hdu.data[mask1 & mask2 & mask3]

        # Sort rows
        if sort_by == 'VTmag':
            indsort = np.argsort(hdu.data.field(3))
        else:
            indsort = np.argsort(hdu.data.field(2))

        hdu.data = hdu.data[indsort]

        fn_tyc_year = os.path.join(self.index_dir, 
                                   'tycho2_{:d}.fits'.format(year))
        hdu.writeto(fn_tyc_year, clobber=True)

        tycho2_index_dir = os.path.join(self.index_dir, 
                                        'index_{:d}'.format(year))

        try:
            os.makedirs(tycho2_index_dir)
        except OSError:
            if not os.path.isdir(tycho2_index_dir):
                raise

        for scale_num in np.arange(max_scale, min_scale-1, -1):
            cmd = self.build_index_path
            cmd += ' -i {}'.format(fn_tyc_year)
            cmd += ' -S {}'.format(sort_by)
            cmd += ' -P {:d}'.format(scale_num)
            cmd += ' -I {:d}{:02d}'.format(year, scale_num)
            fn_index = 'index_{:d}_{:02d}.fits'.format(year, scale_num)
            cmd += ' -o {}'.format(os.path.join(tycho2_index_dir, fn_index))

            sp.call(cmd, shell=True, cwd=self.index_dir)

        #os.remove(fn_tyc_year)

    def create_index_loop(self, start_year, end_year, step, max_scale=None, 
                          min_scale=None, sort_by='BTmag'):
        """
        Create Astrometry.net indexes for a set of epochs.

        """

        for year in np.arange(start_year, end_year+1, step):
            self.create_index_year(year, max_scale=max_scale, 
                                   min_scale=min_scale, sort_by=sort_by)


class SolveProcessLog:
    """
    Plate solve process log class

    """

    def __init__(self, file_path):
        if file_path:
            self.path = file_path
            self.enable = True
        else:
            self.path = None
            self.enable = False
            
        self.handle = None

        self.platedb = None
        self.archive_id = None
        self.plate_id = None
        self.scan_id = None
        self.process_id = None

    def open(self):
        """
        Open log file.

        """

        if self.enable:
            try:
                self.handle = open(self.path, 'w', 1)
            except IOError:
                print 'Cannot open log file {}'.format(self.path)
                self.handle = sys.stdout
        else:
            self.handle = sys.stdout

    def write(self, message, timestamp=True, double_newline=True, 
              level=None, event=None):
        """
        Write a message to the log file and optionally to the database.

        Parameters
        ----------
        message : str
            Message to be written to the log file
        timestamp : bool
            Write timestamp with the message (default True)
        double_newline : bool
            Add two newline characters after the message (default True)

        """

        log_message = '{}'.format(message)

        if timestamp:
            log_message = '***** {} ***** {}'.format(str(dt.datetime.now()), 
                                                     log_message)

        if double_newline:
            log_message += '\n'

        self.handle.write('{}\n'.format(log_message))

        if level is not None:
            self.to_db(level, message, event=event)

    def to_db(self, level, message, event=None):
        """
        Write a log message to the database.

        Parameters
        ----------
        level : int
            Log level (1 = error, 2 = warning, 3 = info, 4 = debug, 5 = trace)
        message : str
            Message to be written to the log file
        event : int
            Event code (default None)

        """

        if self.platedb is not None and self.process_id is not None:
            self.platedb.write_processlog(level, message, event=event,
                                          process_id=self.process_id,
                                          scan_id=self.scan_id, 
                                          plate_id=self.plate_id, 
                                          archive_id=self.archive_id)

    def close(self):
        """
        Close log file.

        """

        if self.enable and self.handle is not sys.stdout:
            self.handle.close()


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
    tbl = fits.new_table([col])
    tbl.header.set('EXTNAME', 'LDAC_IMHEAD', 'table name', after='TFIELDS')
    tbl.header.set('TDIM1', '(80, 36)', after='TFORM1')
    hdulist.append(tbl)

    col1 = fits.Column(name='X_WORLD', format='1D', unit='deg', disp='E15')
    col2 = fits.Column(name='Y_WORLD', format='1D', unit='deg', disp='E15')
    col3 = fits.Column(name='ERRA_WORLD', format='1E', unit='deg', 
                       disp='E12')
    col4 = fits.Column(name='ERRB_WORLD', format='1E', unit='deg', 
                       disp='E12')
    col5 = fits.Column(name='MAG', format='1E', unit='mag', disp='F8.4')
    col6 = fits.Column(name='MAGERR', format='1E', unit='mag', disp='F8.4')
    col7 = fits.Column(name='OBSDATE', format='1D', unit='yr', 
                       disp='F13.8')
    tbl = fits.new_table([col1, col2, col3, col4, col5, col6, col7])
    tbl.header.set('EXTNAME', 'LDAC_OBJECTS', 'table name', 
                   after='TFIELDS')
    hdulist.append(tbl)

    return hdulist


def run_solve_process(filename, conf_file=None, **kwargs):
    """
    Run the source extraction and plate solving process.

    Parameters
    ----------
    filename : str
        Filename of the digitised plate
        
    """

    proc = SolveProcess(filename)

    if conf_file:
        conf = ConfigParser.ConfigParser()
        conf.read(conf_file)
        proc.assign_conf(conf)

    for key in ('threshold_sigma', 'use_psf', 'psf_threshold_sigma', 
                'psf_model_sigma', 'plate_epoch', 'sip', 'skip_bright',
                'max_recursion_depth', 'force_recursion_depth',
                'circular_film'):
        if key in kwargs:
            setattr(proc, key, kwargs[key])

    proc.setup()
    proc.invert_plate()
    proc.extract_sources()
    proc.solve_plate()
    proc.output_wcs_header()
    proc.solve_recursive()
    proc.output_sources_db()
    proc.output_sources_csv()
    proc.finish()


_source_meta = OrderedDict([
    ('source_num',          ('i4', '%7d', 'NUMBER')),
    ('x_source',            ('f8', '%11.4f', 'X_IMAGE')),
    ('y_source',            ('f8', '%11.4f', 'Y_IMAGE')),
    ('erra_source',         ('f4', '%9.5f', 'ERRA_IMAGE')),
    ('errb_source',         ('f4', '%9.5f', 'ERRB_IMAGE')),
    ('errtheta_source',     ('f4', '%6.2f', 'ERRTHETA_IMAGE')),
    ('a_source',            ('f4', '%9.3f', 'A_IMAGE')),
    ('b_source',            ('f4', '%9.3f', 'B_IMAGE')),
    ('theta_source',        ('f4', '%6.2f', 'THETA_IMAGE')),
    ('elongation',          ('f4', '%8.3f', 'ELONGATION')),
    ('x_peak',              ('i4', '%6d', 'XPEAK_IMAGE')),
    ('y_peak',              ('i4', '%6d', 'YPEAK_IMAGE')),
    ('flag_usepsf',         ('i1', '%1d', '')),
    ('x_image',             ('f8', '%11.4f', 'X_IMAGE')),
    ('y_image',             ('f8', '%11.4f', 'Y_IMAGE')),
    ('erra_image',          ('f4', '%9.5f', 'ERRA_IMAGE')),
    ('errb_image',          ('f4', '%9.5f', 'ERRB_IMAGE')),
    ('errtheta_image',      ('f4', '%6.2f', 'ERRTHETA_IMAGE')),
    ('x_psf',               ('f8', '%11.4f', '')),
    ('y_psf',               ('f8', '%11.4f', '')),
    ('erra_psf',            ('f4', '%9.5f', '')),
    ('errb_psf',            ('f4', '%9.5f', '')),
    ('errtheta_psf',        ('f4', '%6.2f', '')),
    ('mag_auto',            ('f4', '%7.4f', 'MAG_AUTO')),
    ('magerr_auto',         ('f4', '%7.4f', 'MAGERR_AUTO')),
    ('flux_auto',           ('f4', '%12.5e', 'FLUX_AUTO')),
    ('fluxerr_auto',        ('f4', '%12.5e', 'FLUXERR_AUTO')),
    ('mag_iso',             ('f4', '%7.4f', 'MAG_ISO')),
    ('magerr_iso',          ('f4', '%7.4f', 'MAGERR_ISO')),
    ('flux_iso',            ('f4', '%12.5e', 'FLUX_ISO')),
    ('fluxerr_iso',         ('f4', '%12.5e', 'FLUXERR_ISO')),
    ('flux_max',            ('f4', '%12.5e', 'FLUX_MAX')),
    ('flux_radius',         ('f4', '%12.5e', 'FLUX_RADIUS')),
    ('isoarea',             ('i4', '%6d', 'ISOAREA_IMAGE')),
    ('sqrt_isoarea',        ('f4', '%12.5e', '')),
    ('background',          ('f4', '%12.5e', 'BACKGROUND')),
    ('sextractor_flags',    ('i2', '%3d', 'FLAGS')),
    ('dist_center',         ('f4', '%9.3f', '')),
    ('dist_edge',           ('f4', '%9.3f', '')),
    ('annular_bin',         ('i1', '%1d', '')),
    ('flag_negradius',      ('i1', '%1d', '')),
    ('flag_rim',            ('i1', '%1d', '')),
    ('flag_clean',          ('i1', '%1d', '')),
    ('raj2000',             ('f8', '%11.7f', '')),
    ('dej2000',             ('f8', '%11.7f', '')),
    ('x_sphere',            ('f8', '%10.7f', '')),
    ('y_sphere',            ('f8', '%10.7f', '')),
    ('z_sphere',            ('f8', '%10.7f', '')),
    ('healpix256',          ('i4', '%6d', '')),
    ('raj2000_wcs',         ('f8', '%11.7f', '')),
    ('dej2000_wcs',         ('f8', '%11.7f', '')),
    ('raj2000_sub',         ('f8', '%11.7f', '')),
    ('dej2000_sub',         ('f8', '%11.7f', '')),
    ('raerr_sub',           ('f4', '%7.4f', '')),
    ('decerr_sub',          ('f4', '%7.4f', '')),
    ('gridsize_sub',        ('i2', '%3d', '')),
    ('ucac4_id',            ('a10', '%s', '')),
    ('ucac4_ra',            ('f8', '%11.7f', '')),
    ('ucac4_dec',           ('f8', '%11.7f', '')),
    ('ucac4_bmag',          ('f8', '%7.4f', '')),
    ('ucac4_vmag',          ('f8', '%7.4f', '')),
    ('ucac4_dist',          ('f4', '%6.3f', '')),
    ('tycho2_id',           ('a12', '%s', '')),
    ('tycho2_ra',           ('f8', '%11.7f', '')),
    ('tycho2_dec',          ('f8', '%11.7f', '')),
    ('tycho2_btmag',        ('f8', '%7.4f', '')),
    ('tycho2_vtmag',        ('f8', '%7.4f', '')),
    ('tycho2_hip',          ('i4', '%6d', '')),
    ('tycho2_dist',         ('f4', '%6.3f', ''))
])


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
        self.tycho2_dir = ''
        self.work_dir = ''
        self.write_source_dir = ''
        self.write_wcs_dir = ''
        self.write_log_dir = ''

        self.use_tycho2_fits = False
        self.use_ucac4_db = False
        self.ucac4_db_host = 'localhost'
        self.ucac4_db_user = ''
        self.ucac4_db_name = ''
        self.ucac4_db_passwd = ''

        self.output_db_host = 'localhost'
        self.output_db_user = ''
        self.output_db_name = ''
        self.output_db_passwd = ''

        self.sextractor_path = 'sex'
        self.scamp_path = 'scamp'
        self.psfex_path = 'psfex'
        self.solve_field_path = 'solve-field'
        self.wcs_to_tan_path = 'wcs-to-tan'
        self.xy2sky_path = 'xy2sky'

        self.timestamp = dt.datetime.now()
        self.timestamp_str = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.scratch_dir = None
        self.enable_log = True
        self.log = None
        self.enable_db_log = False
    
        self.plate_epoch = 1950
        self.plate_year = int(self.plate_epoch)
        self.threshold_sigma = 4.
        self.use_psf = False
        self.psf_threshold_sigma = 20.
        self.psf_model_sigma = 20.
        self.sip = 3
        self.skip_bright = 10
        self.max_recursion_depth = 5
        self.force_recursion_depth = 0
        self.circular_film = False
        self.crossmatch_radius = 5.

        self.plate_header = None
        self.imwidth = None
        self.imheight = None
        self.plate_solved = False
        self.mean_pixscale = None
        self.num_sources = None
        self.num_sources_sixbins = None
        self.rel_area_sixbins = None
        self.stars_sqdeg = None
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
        self.wcshead = None
        self.wcs_plate = None
        self.solution = None

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)
            
        self.conf = conf

        try:
            self.archive_id = conf.get('Archive', 'archive_id')
        except ConfigParser.Error:
            pass

        for attr in ['sextractor_path', 'scamp_path', 'psfex_path',
                     'solve_field_path', 'wcs_to_tan_path', 'xy2sky_path']:
            try:
                setattr(self, attr, conf.get('Programs', attr))
            except ConfigParser.Error:
                pass

        for attr in ['fits_dir', 'tycho2_dir', 
                     'work_dir', 'write_log_dir',
                     'write_source_dir', 'write_wcs_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['use_tycho2_fits', 'use_ucac4_db', 
                     'ucac4_db_host', 'ucac4_db_user', 'ucac4_db_name', 
                     'ucac4_db_passwd', 'output_db_host', 'output_db_user',
                     'output_db_name', 'output_db_passwd', 'enable_db_log']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except ConfigParser.Error:
                pass

        for attr in ['plate_epoch', 'threshold_sigma', 'use_psf', 
                     'psf_threshold_sigma', 'psf_model_sigma', 
                     'sip', 'skip_bright', 'max_recursion_depth', 
                     'force_recursion_depth', 'circular_film',
                     'crossmatch_radius']:
            try:
                setattr(self, attr, conf.get('Solve', attr))
            except ConfigParser.Error:
                pass

    def assign_header(self, header):
        """
        Assign FITS header with metadata.

        """

        self.plate_header = header

    def setup(self):
        """
        Set up plate solve process.

        """

        # Set up filename attributes
        fn, ext = os.path.splitext(self.filename)
        self.basefn = fn

        if ext == '':
            self.fn_fits = os.path.join(self.fits_dir, fn + '.fits')
        else:
            self.fn_fits = os.path.join(self.fits_dir, self.filename)

        # Open log file
        if self.enable_log:
            fn_log = '{}_{}.log'.format(self.basefn, self.timestamp_str)
            log_path = os.path.join(self.write_log_dir, fn_log)
            self.log = SolveProcessLog(log_path)
            self.log.open()
        else:
            self.log = SolveProcessLog(None)

        # Get process_id from the database
        self.db_process_start()

        # Open database connection for logs
        if self.enable_db_log:
            platedb = PlateDB()
            platedb.open_connection(host=self.output_db_host,
                                    user=self.output_db_user,
                                    dbname=self.output_db_name,
                                    passwd=self.output_db_passwd)
            self.log.platedb = platedb
            self.log.archive_id = self.archive_id
            self.log.plate_id = self.plate_id
            self.log.scan_id = self.scan_id
            self.log.process_id = self.process_id
            self.log.to_db(3, 'Setting up plate solve process', event=10)

        self.log.write('Using PyPlate v{}'.format(__version__), 
                       level=4, event=10)

        # Check if FITS file exists
        if not os.path.exists(self.fn_fits):
            self.log.write('FITS file does not exist: {}'.format(self.fn_fits), 
                           level=1, event=11)
            return

        # Read FITS header
        if not self.plate_header:
            try:
                self.plate_header = fits.getheader(self.fn_fits)
            except IOError:
                self.log.write('Cannot read FITS file {}'.format(self.fn_fits), 
                               level=1, event=11)
                return

        self.imwidth = self.plate_header['NAXIS1']
        self.imheight = self.plate_header['NAXIS2']

        # Look for observation date in the FITS header.
        if ('YEAR' in self.plate_header and 
            isinstance(self.plate_header['YEAR'], float)):
            self.plate_epoch = self.plate_header['YEAR']
        elif 'DATEORIG' in self.plate_header:
            self.plate_year = int(self.plate_header['DATEORIG'].split('-')[0])
            self.plate_epoch = float(self.plate_year) + 0.5
        elif 'DATE-OBS' in self.plate_header:
            self.plate_year = int(self.plate_header['DATE-OBS'].split('-')[0])
            self.plate_epoch = float(self.plate_year) + 0.5

        self.plate_year = int(self.plate_epoch)

        # Create scratch directory
        self.scratch_dir = os.path.join(self.work_dir, 
                                        '{}_{}'.format(self.basefn,
                                                       self.timestamp_str))
        os.makedirs(self.scratch_dir)

    def finish(self):
        """
        Finish plate process.

        """

        # Close open FITS files
        #if isinstance(self.xyclean, fits.HDUList):
        #    self.xyclean.close()

        if isinstance(self.scampref, fits.HDUList):
            self.scampref.close()

        if isinstance(self.scampcat, fits.HDUList):
            self.scampcat.close()

        # Remove scratch directory and its contents
        if self.scratch_dir:
            shutil.rmtree(self.scratch_dir)

        # Write process end to the database
        self.db_process_end(completed=1)

        # Close database connection used for logging
        if self.log.platedb is not None:
            self.log.to_db(3, 'Finish plate solve process', event=99)
            self.log.platedb.close_connection()

        # Close log file
        self.log.close()

    def db_process_start(self):
        """
        Write process start to the database.

        """

        platedb = PlateDB()
        platedb.open_connection(host=self.output_db_host,
                                user=self.output_db_user,
                                dbname=self.output_db_name,
                                passwd=self.output_db_passwd)
        self.scan_id, self.plate_id = platedb.get_scan_id(self.filename, 
                                                          self.archive_id)
        pid = platedb.write_process_start(scan_id=self.scan_id,
                                          plate_id=self.plate_id,
                                          archive_id=self.archive_id,
                                          filename=self.filename, 
                                          use_psf=self.use_psf)
        self.process_id = pid

        plate_epoch = platedb.get_plate_epoch(self.plate_id)

        if plate_epoch:
            self.plate_epoch = plate_epoch
            self.plate_year = int(plate_epoch)

        platedb.close_connection()

    def db_update_process(self, num_sources=None, num_ucac4=None, 
                          num_tycho2=None, solved=None):
        """
        Update process in the database.

        Parameters
        ----------
        num_sources : int
            Number of extracted sources
        solved : int
            A boolean value specifying whether plate was solved successfully
            with Astrometry.net

        """

        if self.process_id is not None:
            platedb = PlateDB()
            platedb.open_connection(host=self.output_db_host,
                                    user=self.output_db_user,
                                    dbname=self.output_db_name,
                                    passwd=self.output_db_passwd)
            platedb.update_process(self.process_id, num_sources=num_sources, 
                                   num_ucac4=num_ucac4, num_tycho2=num_tycho2,
                                   solved=solved)
            platedb.close_connection()

    def db_process_end(self, completed=None):
        """
        Write process end to the database.

        Parameters
        ----------
        completed : int
            A boolean value specifying whether the process was completed

        """

        if self.process_id is not None:
            platedb = PlateDB()
            platedb.open_connection(host=self.output_db_host,
                                    user=self.output_db_user,
                                    dbname=self.output_db_name,
                                    passwd=self.output_db_passwd)
            duration = (dt.datetime.now()-self.timestamp).seconds
            platedb.write_process_end(self.process_id, 
                                      completed=completed, 
                                      duration=duration)
            platedb.close_connection()

    def invert_plate(self):
        """
        Invert FITS image and save the result (*_inverted.fits) in the scratch 
        or work directory.
        """

        fn_inverted = '{}_inverted.fits'.format(self.basefn)

        if self.scratch_dir:
            fn_inverted = os.path.join(self.scratch_dir, fn_inverted)
        else:
            fn_inverted = os.path.join(self.work_dir, fn_inverted)
        
        if not os.path.exists(fn_inverted):
            self.log.write('Inverting image', level=3, event=20)

            fitsfile = fits.open(self.fn_fits, do_not_scale_image_data=True)

            invfits = fits.PrimaryHDU(-fitsfile[0].data)
            invfits.header = fitsfile[0].header.copy()
            invfits.header.set('BZERO', 32768)
            invfits.header.set('BSCALE', 1.0)
            self.log.write('Writing inverted image: {}'.format(fn_inverted), 
                           level=4, event=21)
            invfits.writeto(fn_inverted)

            fitsfile.close()
            del fitsfile
            del invfits
        else:
            self.log.write('Inverted file exists: {}'.format(fn_inverted), 
                           level=4, event=20)

    def extract_sources(self, threshold_sigma=None, use_psf=None, 
                        psf_threshold_sigma=None, psf_model_sigma=None, 
                        circular_film=None):
        """
        Extract sources from a FITS file.

        Parameters
        ----------
        threshold_sigma : float
            SExtractor threshold in sigmas (default 4.0)
        use_psf : bool
            Use PSF for bright stars (default False)
        psf_threshold_sigma : float
            SExtractor threshold in sigmas for using PSF (default 20.0)
        psf_model_sigma : float
            SExtractor threshold in sigmas for PSF model stars (default 20.0)
        circular_film : bool
            Assume circular film (default False)
        """

        if threshold_sigma is None:
            threshold_sigma = self.threshold_sigma

        if use_psf is None:
            use_psf = self.use_psf

        if psf_threshold_sigma is None:
            psf_threshold_sigma = self.psf_threshold_sigma

        if psf_model_sigma is None:
            psf_model_sigma = self.psf_model_sigma

        if circular_film is None:
            circular_film = self.circular_film

        self.log.write('Extracting sources from image', level=3, event=30)

        if use_psf:
            # If PSFEx input file does not exist then run SExtractor
            if not os.path.exists(os.path.join(self.scratch_dir, 
                                               self.basefn + '_psfex.cat')):
                self.log.write('Running SExtractor to get sources for PSFEx', 
                               level=3, event=31)
                self.log.write('Using threshold {:.1f}'
                               .format(psf_model_sigma), level=4, event=31)

                # Create parameter file
                fn_sex_param = self.basefn + '_sextractor.param'
                fconf = open(os.path.join(self.scratch_dir, fn_sex_param), 'w')
                fconf.write('VIGNET(120,120)\n')
                fconf.write('X_IMAGE\n')
                fconf.write('Y_IMAGE\n')
                #fconf.write('FLUX_APER(1)\n')
                #fconf.write('FLUXERR_APER(1)\n')
                fconf.write('MAG_AUTO\n')
                fconf.write('FLUX_AUTO\n')
                fconf.write('FLUXERR_AUTO\n')
                fconf.write('SNR_WIN\n')
                fconf.write('FLUX_RADIUS\n')
                fconf.write('ELONGATION\n')
                fconf.write('FLAGS')
                fconf.close()

                # Create configuration file
                cnf = 'DETECT_THRESH    {:f}\n'.format(psf_model_sigma)
                cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_model_sigma)
                cnf += 'FILTER           N\n'
                #cnf += 'PHOT_APERTURES   10\n'
                #cnf += 'WEIGHT_IMAGE     %s_wmap.fits\n' % self.basefn
                #cnf += 'WEIGHT_TYPE      MAP_WEIGHT\n'
                #cnf += 'WEIGHT_THRESH    1\n'
                cnf += 'SATUR_LEVEL      65000.0\n'
                cnf += 'BACKPHOTO_TYPE   LOCAL\n'
                #cnf += 'BACKPHOTO_THICK  96\n'
                cnf += 'MAG_ZEROPOINT    25.0\n'
                cnf += 'PARAMETERS_NAME  {}\n'.format(fn_sex_param)
                cnf += 'CATALOG_TYPE     FITS_LDAC\n'
                cnf += 'CATALOG_NAME     {}_psfex.cat\n'.format(self.basefn)

                fn_sex_conf = self.basefn + '_sextractor.conf'
                self.log.write('Writing SExtractor configuration file {}'
                               .format(fn_sex_conf), level=4, event=31)
                self.log.write('SExtractor configuration file:\n{}'
                               .format(cnf), level=5, event=31)
                fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.sextractor_path
                cmd += ' %s_inverted.fits' % self.basefn
                cmd += ' -c %s' % fn_sex_conf
                self.log.write('Subprocess: {}'.format(cmd), level=4, event=31)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

            # Run PSFEx
            if not os.path.exists(os.path.join(self.scratch_dir, 
                                               self.basefn + '_psfex.psf')):
                self.log.write('Running PSFEx', level=3, event=32)

                #cnf = 'PHOTFLUX_KEY       FLUX_APER(1)\n'
                #cnf += 'PHOTFLUXERR_KEY    FLUXERR_APER(1)\n'
                cnf = 'PHOTFLUX_KEY       FLUX_AUTO\n'
                cnf += 'PHOTFLUXERR_KEY    FLUXERR_AUTO\n'
                cnf += 'PSFVAR_KEYS        X_IMAGE,Y_IMAGE\n'
                cnf += 'PSFVAR_GROUPS      1,1\n'
                cnf += 'PSFVAR_DEGREES     3\n'
                cnf += 'SAMPLE_FWHMRANGE   3.0,50.0\n'
                cnf += 'SAMPLE_VARIABILITY 3.0\n'
                #cnf += 'PSF_SIZE           25,25\n'
                cnf += 'PSF_SIZE           50,50\n'
                cnf += 'CHECKPLOT_TYPE     ellipticity\n'
                cnf += 'CHECKPLOT_NAME     ellipticity\n'
                cnf += 'CHECKIMAGE_TYPE    SNAPSHOTS\n'
                cnf += 'CHECKIMAGE_NAME    snap.fits\n'
                #cnf += 'CHECKIMAGE_NAME    %s_psfex_snap.fits\n' % self.basefn
                #cnf += 'CHECKIMAGE_TYPE    NONE\n'
                cnf += 'XML_NAME           {}_psfex.xml\n'.format(self.basefn)
                cnf += 'VERBOSE_TYPE       LOG\n'

                fn_psfex_conf = self.basefn + '_psfex.conf'
                self.log.write('Writing PSFEx configuration file {}'
                               .format(fn_psfex_conf), level=4, event=32)
                self.log.write('PSFEx configuration file:\n{}'
                               .format(cnf), level=5, event=32)
                fconf = open(os.path.join(self.scratch_dir, fn_psfex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.psfex_path
                cmd += ' %s_psfex.cat' % self.basefn
                cmd += ' -c %s' % fn_psfex_conf
                self.log.write('Subprocess: {}'.format(cmd), level=4, event=32)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

            # Run SExtractor with PSF
            if not os.path.exists(os.path.join(self.scratch_dir, 
                                               self.basefn + '.cat-psf')):
                self.log.write('Running SExtractor with PSF model',
                               level=3, event=33)
                self.log.write('Using threshold {:.1f}'
                               .format(psf_threshold_sigma), level=4, event=33)

                fn_sex_param = self.basefn + '_sextractor.param'
                fconf = open(os.path.join(self.scratch_dir, fn_sex_param), 'w')
                fconf.write('XPEAK_IMAGE\n')
                fconf.write('YPEAK_IMAGE\n')
                fconf.write('XPSF_IMAGE\n')
                fconf.write('YPSF_IMAGE\n')
                fconf.write('ERRAPSF_IMAGE\n')
                fconf.write('ERRBPSF_IMAGE\n')
                fconf.write('ERRTHETAPSF_IMAGE\n')
                fconf.close()

                cnf = 'DETECT_THRESH    {:f}\n'.format(psf_threshold_sigma)
                cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_threshold_sigma)
                cnf += 'FILTER           N\n'
                cnf += 'SATUR_LEVEL      65000.0\n'
                cnf += 'BACKPHOTO_TYPE   LOCAL\n'
                #cnf += 'BACKPHOTO_THICK  96\n'
                cnf += 'MAG_ZEROPOINT    25.0\n'
                cnf += 'PARAMETERS_NAME  {}\n'.format(fn_sex_param)
                cnf += 'CATALOG_TYPE     FITS_1.0\n'
                cnf += 'CATALOG_NAME     {}.cat-psf\n'.format(self.basefn)
                cnf += 'PSF_NAME         {}_psfex.psf\n'.format(self.basefn)
                cnf += 'NTHREADS         0\n'

                fn_sex_conf = self.basefn + '_sextractor.conf'
                self.log.write('Writing SExtractor configuration file {}'
                               .format(fn_sex_conf), level=4, event=33)
                self.log.write('SExtractor configuration file:\n{}'
                               .format(cnf), level=5, event=33)
                fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.sextractor_path
                cmd += ' %s_inverted.fits' % self.basefn
                cmd += ' -c %s' % fn_sex_conf
                self.log.write('Subprocess: {}'.format(cmd), level=4, event=33)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

        # If SExtractor catalog does not exist then run SExtractor
        if not os.path.exists(os.path.join(self.scratch_dir, 
                                           self.basefn + '.cat')):
            self.log.write('Running SExtractor without PSF model',
                           level=3, event=34)
            self.log.write('Using threshold {:.1f}'.format(threshold_sigma),
                           level=4, event=34)

            fn_sex_param = self.basefn + '_sextractor.param'
            fconf = open(os.path.join(self.scratch_dir, fn_sex_param), 'w')
            fconf.write('NUMBER\n')
            fconf.write('X_IMAGE\n')
            fconf.write('Y_IMAGE\n')
            fconf.write('A_IMAGE\n')
            fconf.write('B_IMAGE\n')
            fconf.write('THETA_IMAGE\n')
            fconf.write('ERRA_IMAGE\n')
            fconf.write('ERRB_IMAGE\n')
            fconf.write('ERRTHETA_IMAGE\n')
            fconf.write('ELONGATION\n')
            fconf.write('XPEAK_IMAGE\n')
            fconf.write('YPEAK_IMAGE\n')
            fconf.write('MAG_AUTO\n')
            fconf.write('MAGERR_AUTO\n')
            fconf.write('FLUX_AUTO\n')
            fconf.write('FLUXERR_AUTO\n')
            fconf.write('MAG_ISO\n')
            fconf.write('MAGERR_ISO\n')
            fconf.write('FLUX_ISO\n')
            fconf.write('FLUXERR_ISO\n')
            fconf.write('FLUX_MAX\n')
            fconf.write('FLUX_RADIUS\n')
            fconf.write('ISOAREA_IMAGE\n')
            fconf.write('BACKGROUND\n')
            fconf.write('FLAGS')
            fconf.close()

            cnf = 'DETECT_THRESH    {:f}\n'.format(threshold_sigma)
            cnf += 'ANALYSIS_THRESH  {:f}\n'.format(threshold_sigma)
            cnf += 'FILTER           N\n'
            cnf += 'SATUR_LEVEL      65000.0\n'
            cnf += 'BACKPHOTO_TYPE   LOCAL\n'
            cnf += 'CLEAN            N\n'
            #cnf += 'CLEAN_PARAM      0.2\n'
            #cnf += 'BACKPHOTO_THICK  96\n'
            cnf += 'MAG_ZEROPOINT    25.0\n'
            cnf += 'PARAMETERS_NAME  {}\n'.format(fn_sex_param)
            cnf += 'CATALOG_TYPE     FITS_1.0\n'
            cnf += 'CATALOG_NAME     {}.cat\n'.format(self.basefn)
            cnf += 'NTHREADS         0\n'
            #cnf += 'DETECT_TYPE      PHOTO\n'
            #cnf += 'MAG_GAMMA        1.0\n'
            #cnf += 'MEMORY_OBJSTACK  8000\n'
            #cnf += 'MEMORY_PIXSTACK  800000\n'
            #cnf += 'MEMORY_BUFSIZE   256\n'

            fn_sex_conf = self.basefn + '_sextractor.conf'
            self.log.write('Writing SExtractor configuration file {}'
                           .format(fn_sex_conf), level=4, event=34)
            self.log.write('SExtractor configuration file:\n{}'.format(cnf), 
                           level=5, event=34)
            fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 'w')
            fconf.write(cnf)
            fconf.close()

            cmd = self.sextractor_path
            cmd += ' %s_inverted.fits' % self.basefn
            cmd += ' -c %s' % fn_sex_conf
            self.log.write('Subprocess: {}'.format(cmd), level=4, event=34)
            sp.call(cmd, shell=True, stdout=self.log.handle, 
                    stderr=self.log.handle, cwd=self.scratch_dir)
            self.log.write('', timestamp=False, double_newline=False)

        # Read the SExtractor output catalog
        xycat = fits.open(os.path.join(self.scratch_dir, self.basefn + '.cat'))
        self.num_sources = len(xycat[1].data)
        self.db_update_process(num_sources=self.num_sources)

        self.sources = np.zeros(self.num_sources,
                                dtype=[(k,_source_meta[k][0]) 
                                       for k in _source_meta])

        self.sources['raj2000'] = np.nan
        self.sources['dej2000'] = np.nan
        self.sources['raj2000_wcs'] = np.nan
        self.sources['dej2000_wcs'] = np.nan
        self.sources['raj2000_sub'] = np.nan
        self.sources['dej2000_sub'] = np.nan
        self.sources['raerr_sub'] = np.nan
        self.sources['decerr_sub'] = np.nan
        self.sources['x_sphere'] = np.nan
        self.sources['y_sphere'] = np.nan
        self.sources['z_sphere'] = np.nan
        self.sources['healpix256'] = -1
        self.sources['ucac4_ra'] = np.nan
        self.sources['ucac4_dec'] = np.nan
        self.sources['ucac4_bmag'] = np.nan
        self.sources['ucac4_vmag'] = np.nan
        self.sources['ucac4_dist'] = np.nan
        self.sources['tycho2_ra'] = np.nan
        self.sources['tycho2_dec'] = np.nan
        self.sources['tycho2_btmag'] = np.nan
        self.sources['tycho2_vtmag'] = np.nan
        self.sources['tycho2_dist'] = np.nan
        
        # Copy values from the SExtractor catalog, xycat
        for k,v in [(n,_source_meta[n][2]) for n in _source_meta 
                    if _source_meta[n][2]]:
            self.sources[k] = xycat[1].data.field(v)

        self.sources['sqrt_isoarea'] = np.sqrt(self.sources['isoarea'])

        # For brevity, define following variables
        xim = self.sources['x_image']
        yim = self.sources['y_image']
        bg = self.sources['background']

        # Calculate mean background in an annulus around the centre of the plate
        self.sources['dist_center'] = np.sqrt((xim - self.imwidth / 2.)**2 +
                                              (yim - self.imheight / 2.)**2)
        min_halfwidth = min([self.imwidth, self.imheight]) / 2.
        max_halfwidth = max([self.imwidth, self.imheight]) / 2.
        bannulus = ((self.sources['dist_center'] > 0.1*min_halfwidth) &
                    (self.sources['dist_center'] < 0.9*min_halfwidth))

        if bannulus.sum() > 3:
            mean_bg = bg[np.where(bannulus)].mean()
            std_bg = bg[np.where(bannulus)].std()
        else:
            mean_bg = bg.mean()
            bg.std()

        # Calculate distance from edge
        distarr = np.column_stack((xim, self.imwidth-xim, 
                                   yim, self.imheight-yim))
        self.sources['dist_edge'] = np.amin(distarr, 1)
        
        # Define 8 concentric annular bins + bin9 for edges
        self.log.write('Flagging sources', level=3, event=35)
        sampling = 100
        imwidth_s = int(self.imwidth / sampling)
        imheight_s = int(self.imheight / sampling)
        dist_s = np.zeros((imheight_s, imwidth_s))
        area_all = dist_s.size

        for x in np.arange(imwidth_s):
            for y in np.arange(imheight_s):
                dist_s[y,x] = np.sqrt((x*sampling - self.imwidth/2.)**2 +
                                 (y*sampling - self.imheight/2.)**2)

        # Define bin9 as 1/10 of min_halfwidth at all edges
        bin9_width_s = 0.1 * min_halfwidth / sampling
        dist_s[:bin9_width_s,:] = -100.
        dist_s[imheight_s-bin9_width_s:,:] = -100.
        dist_s[:,:bin9_width_s] = -100.
        dist_s[:,imwidth_s-bin9_width_s:] = -100.

        # Additionally, include in bin9 corner pixels that are closer
        # to corners than 1/6 of half-diagonal
        bin9_corner_dist = (np.sqrt(self.imwidth**2 + self.imheight**2) * 
                            5. / 12.)
        dist_s[np.where(dist_s > bin9_corner_dist)] = -100.

        # Exclude bin9 from dist_s
        dist_s = dist_s[np.where(dist_s >= 0)]
        self.rel_area_sixbins = 0.75 * dist_s.size / area_all

        # Divide the rest of pixels between 8 equal-area bins 1-8
        bin_dist = np.array([np.percentile(dist_s, perc)
                             for perc in (np.arange(8)+1.)*100./8.])
        bin_dist[7] = bin9_corner_dist
        bin_dist = np.insert(bin_dist, 0, 0.)

        for b in np.arange(8)+1:
            bbin = ((self.sources['dist_center'] >= bin_dist[b-1]) &
                    (self.sources['dist_center'] < bin_dist[b]))
            nbin = bbin.sum()
            self.log.write('Annular bin {:d} (radius {:8.2f} pixels): '
                           '{:6d} sources'.format(b, bin_dist[b], nbin), 
                           double_newline=False, level=4, event=35)

            if nbin > 0:
                indbin = np.where(bbin)
                self.sources['annular_bin'][indbin] = b

        bbin = ((self.sources['dist_edge'] < 0.1*min_halfwidth) |
                (self.sources['dist_center'] >= bin9_corner_dist))
        nbin = bbin.sum()
        self.log.write('Annular bin 9 (radius {:8.2f} pixels): '
                       '{:6d} sources'.format(bin9_corner_dist, nbin), 
                       level=4, event=35)

        if nbin > 0:
            indbin = np.where(bbin)
            self.sources['annular_bin'][indbin] = 9

        indbin6 = (self.sources['annular_bin'] <= 6)
        self.num_sources_sixbins = indbin6.sum()

        # Find and flag dubious stars at the edges
        if circular_film:
            rim_dist = max_halfwidth - self.sources['dist_center']
            self.sources['dist_edge'] = rim_dist

            # Consider sources that are further away from centre than 
            # half-width, to be at the rim.
            bneg = (rim_dist <= 0)

            if bneg.sum() > 0:
                rim_dist[np.where(bneg)] = 0.1

            borderbg = max_halfwidth / rim_dist * (np.abs(bg - mean_bg) / 
                                                   std_bg)
        else:
            min_xedgedist = np.minimum(xim, self.imwidth-xim)
            min_yedgedist = np.minimum(yim, self.imheight-yim)
            min_reldist = np.minimum(min_xedgedist/self.imwidth,
                                     min_yedgedist/self.imheight)

            # Avoid dividing by zero
            min_reldist[np.where(min_reldist < 1e-5)] = 1e-5
            
            # Combine distance from edge with deviation from mean background
            borderbg = (1. / min_reldist) * (np.abs(bg - mean_bg) / std_bg)

        bclean = ((self.sources['flux_radius'] > 0) & 
                  (self.sources['elongation'] < 5) & 
                  (borderbg < 100))
        indclean = np.where(bclean)[0]
        self.sources['flag_clean'][indclean] = 1
        self.log.write('Flagged {:d} clean sources'.format(bclean.sum()),
                       double_newline=False, level=4, event=35)

        indrim = np.where(borderbg >= 100)[0]
        self.sources['flag_rim'][indrim] = 1
        self.log.write('Flagged {:d} sources at the plate rim'
                       ''.format(len(indrim)), double_newline=False, 
                       level=4, event=35)

        indnegrad = np.where(self.sources['flux_radius'] <= 0)[0]
        self.sources['flag_negradius'][indnegrad] = 1
        self.log.write('Flagged {:d} sources with negative FLUX_RADIUS'
                       ''.format(len(indnegrad)), level=4, event=35)

        # For bright stars, update coordinates with PSF coordinates
        if use_psf:
            self.log.write('Updating coordinates with PSF coordinates '
                           'for bright sources', level=3, event=36)

            fn_psfcat = os.path.join(self.scratch_dir, self.basefn + '.cat-psf')

            if os.path.exists(fn_psfcat):
                try:
                    psfcat = fits.open(fn_psfcat)
                except IOError:
                    self.log.write('Cannot read PSF coordinates, file {} '
                                   'is corrupt'.format(fn_psfcat), 
                                   level=2, event=36)
                    psfcat = None

                if psfcat is not None:
                    xpeakpsf = psfcat[1].data.field('XPEAK_IMAGE')
                    ypeakpsf = psfcat[1].data.field('YPEAK_IMAGE')

                    # Match sources in two lists (distance < 1 px)
                    coords1 = np.empty((self.num_sources, 2))
                    coords1[:,0] = self.sources['x_peak']
                    coords1[:,1] = self.sources['y_peak']
                    coords2 = np.empty((xpeakpsf.size, 2))
                    coords2[:,0] = xpeakpsf
                    coords2[:,1] = ypeakpsf
                    kdt = KDT(coords2)
                    ds,ind2 = kdt.query(coords1)
                    ind1 = np.arange(self.num_sources)
                    indmask = ds < 1.
                    ind1 = ind1[indmask]
                    ind2 = ind2[indmask]

                    #ind1,ind2,ds = pyspherematch.xymatch(self.sources['x_peak'],
                    #                                     self.sources['y_peak'],
                    #                                     xpeakpsf,
                    #                                     ypeakpsf,
                    #                                     tol=1.)

                    self.log.write('Replacing x,y values from PSF photometry '
                                   'for {:d} sources'.format(len(ind1)), 
                                   level=3, event=36)
                    self.sources[ind1]['x_psf'] = \
                            psfcat[1].data.field('XPSF_IMAGE')[ind2]
                    self.sources[ind1]['y_psf'] = \
                            psfcat[1].data.field('YPSF_IMAGE')[ind2]
                    self.sources[ind1]['erra_psf'] = \
                            psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
                    self.sources[ind1]['errb_psf'] = \
                            psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
                    self.sources[ind1]['errtheta_psf'] = \
                            psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
                    self.sources[ind1]['x_source'] = \
                            psfcat[1].data.field('XPSF_IMAGE')[ind2]
                    self.sources[ind1]['y_source'] = \
                            psfcat[1].data.field('YPSF_IMAGE')[ind2]
                    self.sources[ind1]['erra_source'] = \
                            psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
                    self.sources[ind1]['errb_source'] = \
                            psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
                    self.sources[ind1]['errtheta_source'] = \
                            psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
            else:
                self.log.write('Cannot read PSF coordinates, file {} does not '
                               'exist!'.format(fn_psfcat), level=2, event=36)

        # Keep clean xy data for later use
        #self.xyclean = xycat[1].copy()
        #self.xyclean.data = self.xyclean.data[indclean]

        #xycat[1].data = xycat[1].data[indclean]

        # Output clean xy data to .xy file
        #fnxy = os.path.join(self.scratch_dir, self.basefn + '.xy')

        #if os.path.exists(fnxy):
        #    os.remove(fnxy)

        #xycat.writeto(fnxy)

    def solve_plate(self, plate_epoch=None, sip=None, skip_bright=None):
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

        """

        self.log.write('Solving astrometry', level=3, event=40)

        if plate_epoch is None:
            plate_epoch = self.plate_epoch
            plate_year = self.plate_year
        else:
            try:
                plate_year = int(plate_epoch)
            except ValueError:
                plate_year = self.plate_year

        self.log.write('Using plate epoch of {:.2f}'.format(plate_epoch), 
                       level=4, event=40)

        if sip is None:
            sip = self.sip

        if skip_bright is None:
            skip_bright = self.skip_bright

        # Create another xy list for faster solving
        # Keep 1000 stars in brightness order, skip the brightest

        xycat = fits.HDUList()
        hdu = fits.PrimaryHDU()
        xycat.append(hdu)

        indclean = np.where(self.sources['flag_clean'] == 1)[0]
        indsort = np.argsort(self.sources[indclean]['mag_auto'])[skip_bright:skip_bright+1000]
        indsel = indclean[indsort]
        nrows = len(indsel)

        col1 = fits.Column(name='X_IMAGE', format='1E', unit='pixel', 
                           disp='F11.4')
        col2 = fits.Column(name='Y_IMAGE', format='1E', unit='pixel', 
                           disp='F11.4')
        col3 = fits.Column(name='MAG_AUTO', format='1E', unit='mag', 
                           disp='F8.4')
        col4 = fits.Column(name='FLUX', format='1E', unit='count', 
                           disp='F12.7')
        tbl = fits.new_table([col1, col2, col3, col4], nrows=nrows)
        tbl.data.field('X_IMAGE')[:] = self.sources[indsel]['x_image']
        tbl.data.field('Y_IMAGE')[:] = self.sources[indsel]['y_image']
        tbl.data.field('MAG_AUTO')[:] = self.sources[indsel]['mag_auto']
        tbl.data.field('FLUX')[:] = self.sources[indsel]['flux_auto']
        xycat.append(tbl)

        fnxy_short = os.path.join(self.scratch_dir, self.basefn + '.xy-short')

        if os.path.exists(fnxy_short):
            os.remove(fnxy_short)

        xycat.writeto(fnxy_short)

        # Write backend config file
        fconf = open(os.path.join(self.scratch_dir, 
                                  self.basefn + '_backend.cfg'), 'w')
        index_path = os.path.join(self.tycho2_dir, 
                                  'index_{:d}'.format(plate_year))
        fconf.write('add_path {}\n'.format(index_path))
        fconf.write('autoindex\n')
        fconf.write('inparallel\n')
        fconf.close()

        # Construct the solve-field call
        cmd = self.solve_field_path
        cmd += ' {}'.format(fnxy_short)
        cmd += ' --no-fits2fits'
        cmd += ' --width {:d}'.format(self.imwidth)
        cmd += ' --height {:d}'.format(self.imheight)
        cmd += ' --x-column X_IMAGE'
        cmd += ' --y-column Y_IMAGE'
        cmd += ' --sort-column MAG_AUTO'
        cmd += ' --sort-ascending'
        cmd += ' --backend-config {}_backend.cfg'.format(self.basefn)

        if sip > 0:
            cmd += ' --tweak-order %d' % sip
        else:
            cmd += ' --no-tweak'
            
        cmd += ' --crpix-center'
        #cmd += ' --pixel-error 3'
        cmd += ' --scamp {}_scamp.cat'.format(self.basefn)
        cmd += ' --scamp-config {}_scamp.conf'.format(self.basefn)
        cmd += ' --no-plots'
        cmd += ' --out {}'.format(self.basefn)
        #cmd += ' --solved none'
        cmd += ' --match none'
        cmd += ' --rdls none'
        cmd += ' --corr none'
        cmd += ' --overwrite'
        #cmd += ' --timestamp'
        #cmd += ' --verbose'
        cmd += ' --cpulimit 120'
        self.log.write('Subprocess: {}'.format(cmd), level=4, event=40)
        sp.call(cmd, shell=True, stdout=self.log.handle, 
                stderr=self.log.handle, cwd=self.scratch_dir)
        self.log.write('', timestamp=False, double_newline=False)

        # Check the result of solve-field
        fn_solved = os.path.join(self.scratch_dir, self.basefn + '.solved')
        fn_wcs = os.path.join(self.scratch_dir, self.basefn + '.wcs')

        if os.path.exists(fn_solved) and os.path.exists(fn_wcs):
            self.plate_solved = True
            self.log.write('Astrometry solved', level=3, event=41)
            self.db_update_process(solved=1)
        else:
            self.log.write('Could not solve astrometry for the plate', 
                           level=2, event=40)
            self.db_update_process(solved=0)
            return

        self.log.write('Calculating plate-solution related parameters', 
                       level=3, event=42)

        # Read the .wcs file and calculate star density
        self.wcshead = fits.getheader(fn_wcs)
        self.wcshead.set('NAXIS', 2)
        self.wcshead.set('NAXIS1', self.imwidth, after='NAXIS')
        self.wcshead.set('NAXIS2', self.imheight, after='NAXIS1')
        self.wcs_plate = wcs.WCS(self.wcshead)
        ra_deg = self.wcshead['CRVAL1']
        dec_deg = self.wcshead['CRVAL2']

        pix_edge_midpoints = np.array([[1., (self.imheight+1.)/2.],
                                       [self.imwidth, (self.imheight+1.)/2.],
                                       [(self.imwidth + 1.)/2., 1.],
                                       [(self.imwidth + 1.)/2., self.imheight]])
        edge_midpoints = self.wcs_plate.all_pix2world(pix_edge_midpoints, 1)

        c1 = ICRS(ra=edge_midpoints[0,0], dec=edge_midpoints[0,1], 
                  unit=(units.degree, units.degree))
        c2 = ICRS(ra=edge_midpoints[1,0], dec=edge_midpoints[1,1],
                  unit=(units.degree, units.degree))

        if use_newangsep:
            imwidth_deg = c1.separation(c2).degree
        else:
            imwidth_deg = c1.separation(c2).degrees

        c3 = ICRS(ra=edge_midpoints[2,0], dec=edge_midpoints[2,1],
                  unit=(units.degree, units.degree))
        c4 = ICRS(ra=edge_midpoints[3,0], dec=edge_midpoints[3,1],
                  unit=(units.degree, units.degree))

        if use_newangsep:
            imheight_deg = c3.separation(c4).degree
        else:
            imheight_deg = c3.separation(c4).degrees

        #self.stars_sqdeg = self.num_sources / (imwidth_deg * imheight_deg)
        self.stars_sqdeg = (self.num_sources_sixbins / 
                            (imwidth_deg*imheight_deg*self.rel_area_sixbins))
        pixscale1 = imwidth_deg / self.imwidth * 3600.
        pixscale2 = imheight_deg / self.imheight * 3600.
        self.mean_pixscale = np.mean([pixscale1, pixscale2])

        # Check if a celestial pole is nearby or on the plate
        half_diag = math.sqrt(imwidth_deg**2 + imheight_deg**2) / 2.
        self.ncp_close = 90. - dec_deg <= half_diag
        self.scp_close = 90. + dec_deg <= half_diag
        self.ncp_on_plate = False
        self.scp_on_plate = False

        if self.ncp_close:
            ncp_pix = self.wcs_plate.wcs_world2pix([[ra_deg,90.]], 1)

            if (ncp_pix[0,0] > 0 and ncp_pix[0,0] < self.imwidth 
                and ncp_pix[0,1] > 0 and ncp_pix[0,1] < self.imheight):
                self.ncp_on_plate = True

        if self.scp_close:
            scp_pix = self.wcs_plate.wcs_world2pix([[ra_deg,-90.]], 1)

            if (scp_pix[0,0] > 0 and scp_pix[0,0] < self.imwidth 
                and scp_pix[0,1] > 0 and scp_pix[0,1] < self.imheight):
                self.scp_on_plate = True

        # Construct coordinate strings
        ra_angle = Angle(ra_deg, units.deg)
        dec_angle = Angle(dec_deg, units.deg)

        try:
            ra_str = ra_angle.to_string(unit=units.hour, sep=':', precision=1, 
                                        pad=True)
            dec_str = dec_angle.to_string(unit=units.deg, sep=':', precision=1,
                                          pad=True)
        except AttributeError:
            ra_str = ra_angle.format(unit='hour', sep=':', precision=1, 
                                     pad=True)
            dec_str = dec_angle.format(sep=':', precision=1, pad=True)

        stc_box = ('Box ICRS {:.5f} {:.5f} {:.5f} {:.5f}'
                   .format(self.wcshead['CRVAL1'], self.wcshead['CRVAL2'], 
                           imwidth_deg, imheight_deg))

        pix_corners = np.array([[1., 1.], [self.imwidth, 1.],
                                [self.imwidth, self.imheight], 
                                [1., self.imheight]])
        corners = self.wcs_plate.all_pix2world(pix_corners, 1)
        stc_polygon = ('Polygon ICRS {:.5f} {:.5f} {:.5f} {:.5f} '
                       '{:.5f} {:.5f} {:.5f} {:.5f}'
                       .format(corners[0,0], corners[0,1], 
                               corners[1,0], corners[1,1],
                               corners[2,0], corners[2,1],
                               corners[3,0], corners[3,1]))

        # Calculate plate rotation angle
        try:
            cp = np.array([self.wcshead['CRPIX1'], self.wcshead['CRPIX2']])

            if dec_angle.deg > 89.:
                cn = self.wcs_plate.wcs_world2pix([[ra_deg,90.]], 1)
            else:
                cn = self.wcs_plate.wcs_world2pix([[ra_deg,dec_deg+1.]], 1)

            if ra_angle.deg > 359.:
                ce = self.wcs_plate.wcs_world2pix([[ra_deg-359.,dec_deg]], 1)
            else:
                ce = self.wcs_plate.wcs_world2pix([[ra_deg+1.,dec_deg]], 1)

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
                           level=2, event=42)

        # Prepare WCS header for output
        wcshead_strip = fits.Header()

        for c in self.wcshead.cards:
            if c[0] != 'COMMENT':
                wcshead_strip.append(c, bottom=True)

        self.solution = OrderedDict([
            ('raj2000', ra_deg),
            ('dej2000', dec_deg),
            ('raj2000_hms', ra_str),
            ('dej2000_dms', dec_str),
            ('fov1', imwidth_deg),
            ('fov2', imheight_deg),
            ('pixel_scale', self.mean_pixscale),
            ('source_density', self.stars_sqdeg),
            ('cd1_1', self.wcshead['CD1_1']),
            ('cd1_2', self.wcshead['CD1_2']),
            ('cd2_1', self.wcshead['CD2_1']),
            ('cd2_2', self.wcshead['CD2_2']),
            ('rotation_angle', rotation_angle),
            ('plate_mirrored', plate_mirrored),
            ('ncp_on_plate', self.ncp_on_plate),
            ('scp_on_plate', self.scp_on_plate),
            ('stc_box', stc_box),
            ('stc_polygon', stc_polygon),
            ('wcs', wcshead_strip)
        ])

        self.log.write('Image dimensions: {:.2f} x {:.2f} degrees'
                       ''.format(imwidth_deg, imheight_deg),
                       double_newline=False)
        self.log.write('Mean pixel scale: {:.3f} arcsec'
                       ''.format(self.mean_pixscale),
                       double_newline=False)
        self.log.write('The image has {:.0f} stars per square degree'
                       ''.format(self.stars_sqdeg))
        self.log.write('Plate rotation angle: {}'.format(rotation_angle),
                       double_newline=False)
        self.log.write('Plate is mirrored: {}'.format(plate_mirrored),
                       double_newline=False)
        self.log.write('North Celestial Pole is on the plate: {}'
                       ''.format(self.ncp_on_plate),
                       double_newline=False)
        self.log.write('South Celestial Pole is on the plate: {}'
                       ''.format(self.scp_on_plate))

        # Convert x,y to RA/Dec with the global WCS solution
        pixcrd = np.column_stack((self.sources['x_source'], 
                                  self.sources['y_source']))
        worldcrd = self.wcs_plate.all_pix2world(pixcrd, 1)
        self.sources['raj2000_wcs'] = worldcrd[:,0]
        self.sources['dej2000_wcs'] = worldcrd[:,1]

        self.min_ra = np.min((worldcrd[:,0].min(), corners[:,0].min()))
        self.max_ra = np.max((worldcrd[:,0].max(), corners[:,0].max()))
        self.min_dec = np.min((worldcrd[:,1].min(), corners[:,1].min()))
        self.max_dec = np.max((worldcrd[:,1].max(), corners[:,1].max()))

    def output_wcs_header(self):
        """
        Write WCS header to an ASCII file.

        """

        if self.plate_solved:
            self.log.write('Writing WCS header to a file', level=3, event=48)

            # Create output directory, if missing
            if self.write_wcs_dir and not os.path.isdir(self.write_wcs_dir):
                self.log.write('Creating WCS output directory {}'
                               ''.format(self.write_wcs_dir), level=4, event=48)
                os.makedirs(self.write_wcs_dir)

            fn_wcshead = os.path.join(self.write_wcs_dir, self.basefn + '.wcs')
            self.log.write('Writing WCS output file {}'.format(fn_wcshead), 
                           level=4, event=48)
            self.wcshead.tofile(fn_wcshead, clobber=True)

    def output_solution_db(self):
        """
        Write plate solution to the database.

        """

        self.log.to_db(3, 'Writing astrometric solution to the database', 
                       event=49)

        if self.solution is None:
            self.log.write('No plate solution to write to the database', 
                           level=2, event=49)
            return

        self.log.write('Open database connection for writing to the '
                       'solution table')
        platedb = PlateDB()
        platedb.open_connection(host=self.output_db_host,
                                user=self.output_db_user,
                                dbname=self.output_db_name,
                                passwd=self.output_db_passwd)

        if (self.scan_id is not None and self.plate_id is not None and 
            self.archive_id is not None and self.process_id is not None):
            platedb.write_solution(self.solution, process_id=self.process_id,
                                   scan_id=self.scan_id,
                                   plate_id=self.plate_id,
                                   archive_id=self.archive_id)
            
        platedb.close_connection()
        self.log.write('Closed database connection')

    def solve_recursive(self, plate_epoch=None, sip=None, skip_bright=None, 
                        max_recursion_depth=None, force_recursion_depth=None):
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
        max_recursion_depth : int
            Maximum recursion depth (default 5)
        force_recursion_depth : int 
            Force recursion depth if enough stars (default 0)

        """

        self.log.write('Recursive solving of astrometry', level=3, event=50)

        if not self.plate_solved:
            self.log.write('Missing initial solution, '
                           'recursive solving not possible!', 
                           level=2, event=50)
            return

        if plate_epoch is None:
            plate_epoch = self.plate_epoch
            plate_year = self.plate_year
        else:
            try:
                plate_year = int(plate_epoch)
            except ValueError:
                plate_year = self.plate_year

        self.log.write('Using plate epoch of {:.2f}'.format(plate_epoch), 
                       level=4, event=50)

        if sip is None:
            sip = self.sip

        if skip_bright is None:
            skip_bright = self.skip_bright

        if max_recursion_depth is None:
            max_recursion_depth = self.max_recursion_depth

        if force_recursion_depth is None:
            force_recursion_depth = self.force_recursion_depth

        try:
            skip_bright = int(skip_bright)
        except ValueError:
            skip_bright = 10

        # Check UCAC4 database name
        if self.use_ucac4_db and (self.ucac4_db_name == ''):
            self.use_ucac4_db = False
            self.log.write('UCAC-4 database name missing!', level=2, event=50)

        # Read the SCAMP input catalog
        self.scampcat = fits.open(os.path.join(self.scratch_dir,
                                               self.basefn + '_scamp.cat'))

        # Create or download the SCAMP reference catalog
        if not os.path.exists(os.path.join(self.scratch_dir, 
                                            self.basefn + '_scampref.cat')):
            if self.stars_sqdeg > 1000:
                astref_catalog = 'UCAC-4'
            else:
                astref_catalog = 'PPMX'
                #astref_catalog = 'Tycho-2'

            if self.use_ucac4_db:
                astref_catalog = 'UCAC-4'
            elif self.use_tycho2_fits:
                astref_catalog = 'Tycho-2'

            if astref_catalog == 'Tycho-2':
                # Build custom SCAMP reference catalog from Tycho-2 FITS file
                fn_tycho = os.path.join(self.tycho2_dir, 'tycho2_{:d}.fits'
                                        .format(plate_year))
                tycho = fits.open(fn_tycho)
                ra_tyc = tycho[1].data.field(0)
                dec_tyc = tycho[1].data.field(1)
                mag_tyc = tycho[1].data.field(2)

                if self.ncp_close:
                    btyc = (dec_tyc > self.min_dec)
                elif self.scp_close:
                    btyc = (dec_tyc < self.max_dec)
                elif self.max_ra-self.min_ra > 180:
                    btyc = (((ra_tyc < self.min_ra) |
                            (ra_tyc > self.max_ra)) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))
                else:
                    btyc = ((ra_tyc > self.min_ra) & 
                            (ra_tyc < self.max_ra) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))

                indtyc = np.where(btyc)
                numtyc = btyc.sum()

                self.log.write('Fetched {:d} entries from Tycho-2'
                               ''.format(numtyc))

                self.scampref = new_scampref()
                hduref = fits.new_table(scampref[2].columns, nrows=numtyc)
                hduref.data.field('X_WORLD')[:] = ra_tyc[indtyc]
                hduref.data.field('Y_WORLD')[:] = dec_tyc[indtyc]
                hduref.data.field('ERRA_WORLD')[:] = np.zeros(numtyc) + 1./3600.
                hduref.data.field('ERRB_WORLD')[:] = np.zeros(numtyc) + 1./3600.
                hduref.data.field('MAG')[:] = mag_tyc[indtyc]
                hduref.data.field('MAGERR')[:] = np.zeros(numtyc) + 0.1
                hduref.data.field('OBSDATE')[:] = np.zeros(numtyc) + 2000.
                self.scampref[2].data = hduref.data

                scampref_file = os.path.join(self.scratch_dir, 
                                             self.basefn + '_scampref.cat')

                if os.path.exists(scampref_file):
                    os.remove(scampref_file)

                self.scampref.writeto(scampref_file)
                tycho.close()
            elif (astref_catalog == 'UCAC-4') and self.use_ucac4_db:
                # Query MySQL database
                db = MySQLdb.connect(host=self.ucac4_db_host, 
                                     user=self.ucac4_db_user, 
                                     passwd=self.ucac4_db_passwd,
                                     db=self.ucac4_db_name)
                cur = db.cursor()

                sql1 = 'SELECT RAJ2000,DEJ2000,e_RAJ2000,e_DEJ2000,amag,e_amag,'
                sql1 += 'pmRA,pmDE,e_pmRA,e_pmDE,UCAC4,Bmag,Vmag'
                sql2 = ' FROM ucac4'
                sql2 += ' FORCE INDEX (idx_radecmag)'

                if self.ncp_close:
                    sql2 += ' WHERE DEJ2000 > {}'.format(self.min_dec)
                elif self.scp_close:
                    sql2 += ' WHERE DEJ2000 < {}'.format(self.max_dec)
                elif self.max_ra-self.min_ra > 180:
                    sql2 += (' WHERE (RAJ2000 < {} OR RAJ2000 > {})'
                             ' AND DEJ2000 BETWEEN {} AND {}'
                             ''.format(self.min_ra, self.max_ra,
                                       self.min_dec, self.max_dec))
                else:
                    sql2 += (' WHERE RAJ2000 BETWEEN {} AND {}'
                             ' AND DEJ2000 BETWEEN {} AND {}'
                             ''.format(self.min_ra, self.max_ra, 
                                       self.min_dec, self.max_dec))

                sql3 = ''

                if self.stars_sqdeg < 200:
                    sql3 += ' AND amag < 13'
                elif self.stars_sqdeg < 1000:
                    sql3 += ' AND amag < 15'

                sql = sql1 + sql2 + sql3 + ';'
                self.log.write(sql)
                numrows = cur.execute(sql)
                self.log.write('Fetched {:d} rows'.format(numrows))

                res = np.fromiter(cur.fetchall(), 
                                  dtype='f8,f8,i,i,f8,f8,f8,f8,f8,f8,a10,f8,f8')

                cur.close()
                db.commit()
                db.close()

                ra_ucac = (res['f0'] + (plate_epoch - 2000.) * res['f6'] / 
                           np.cos(res['f1'] * np.pi / 180.) / 3600000.)
                dec_ucac = (res['f1'] + (plate_epoch - 2000.) * res['f7'] / 
                            3600000.)

                id_ucac = res['f10']
                bmag_ucac = res['f11']
                vmag_ucac = res['f12']

                self.scampref = new_scampref()
                hduref = fits.new_table(self.scampref[2].columns, 
                                        nrows=numrows)
                hduref.data.field('X_WORLD')[:] = ra_ucac
                hduref.data.field('Y_WORLD')[:] = dec_ucac
                hduref.data.field('ERRA_WORLD')[:] = res['f2']
                hduref.data.field('ERRB_WORLD')[:] = res['f3']
                hduref.data.field('MAG')[:] = res['f4']
                hduref.data.field('MAGERR')[:] = res['f5']
                hduref.data.field('OBSDATE')[:] = np.zeros(numrows) + 2000.
                self.scampref[2].data = hduref.data

                scampref_file = os.path.join(self.scratch_dir, 
                                             self.basefn + '_scampref.cat')

                if os.path.exists(scampref_file):
                    os.remove(scampref_file)

                self.scampref.writeto(scampref_file)
            else:
                # Let SCAMP download a reference catalog
                cmd = self.scamp_path
                cmd += ' -c %s_scamp.conf %s_scamp.cat' % (self.basefn, 
                                                           self.basefn)
                cmd += ' -ASTREF_CATALOG %s' % astref_catalog
                cmd += ' -SAVE_REFCATALOG Y'
                cmd += ' -CROSSID_RADIUS 20.0'
                cmd += ' -DISTORT_DEGREES 3'
                cmd += ' -SOLVE_PHOTOM N'
                cmd += ' -VERBOSE_TYPE LOG'
                cmd += ' -CHECKPLOT_TYPE NONE'
                self.log.write('Subprocess: {}'.format(cmd), level=4)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)

                # Rename the saved reference catalog
                fn_scamprefs = glob.glob(os.path.join(self.scratch_dir, 
                                                      astref_catalog + '*'))

                if fn_scamprefs:
                    latest_scampref = max(fn_scamprefs, key=os.path.getctime)
                    os.rename(latest_scampref, 
                              os.path.join(self.scratch_dir,
                                           self.basefn + '_scampref.cat'))
                    self.scampref = fits.open(os.path.join(self.scratch_dir,
                                                           self.basefn + 
                                                           '_scampref.cat'))

        # Improve astrometry in sub-fields (recursively)
        self.wcshead.set('XMIN', 0)
        self.wcshead.set('XMAX', self.imwidth)
        self.wcshead.set('YMIN', 0)
        self.wcshead.set('YMAX', self.imheight)

        radec,gridsize = \
                self._solverec(self.wcshead, np.array([99.,99.]), distort=3,
                               max_recursion_depth=max_recursion_depth,
                               force_recursion_depth=force_recursion_depth)
        self.sources['raj2000_sub'] = radec[:,0]
        self.sources['dej2000_sub'] = radec[:,1]
        self.sources['raerr_sub'] = radec[:,2]
        self.sources['decerr_sub'] = radec[:,3]
        self.sources['gridsize_sub'] = gridsize

        bool_finite = (np.isfinite(self.sources['raj2000_sub']) &
                       np.isfinite(self.sources['dej2000_sub']))
        num_finite = bool_finite.sum()

        self.log.write('Cross-matching sources with the UCAC-4 catalogue', 
                       level=3, event=53)

        if num_finite == 0:
            self.log.write('No sources with improved coordinates', 
                           level=2, event=53)
        else:
            ind_finite = np.where(bool_finite)[0]
            ra_finite = self.sources['raj2000_sub'][ind_finite]
            dec_finite = self.sources['dej2000_sub'][ind_finite]

            # Match sources with the UCAC4 catalogue
            if have_match_coord:
                coords = ICRS(ra_finite, dec_finite, 
                              unit=(units.degree, units.degree))
                catalog = ICRS(ra_ucac, dec_ucac, 
                              unit=(units.degree, units.degree))
                ind_ucac, ds2d, ds3d = match_coordinates_sky(coords, catalog, 
                                                             nthneighbor=1)
                ind_plate = np.arange(ind_ucac.size)
                indmask = ds2d < float(self.crossmatch_radius)*units.arcsec
                ind_plate = ind_plate[indmask]
                ind_ucac = ind_ucac[indmask]
                matchdist = ds2d[indmask].to(units.arcsec).value
            elif have_pyspherematch:
                ind_plate,ind_ucac,ds = \
                        spherematch(ra_finite, dec_finite, ra_ucac, dec_ucac,
                                    tol=float(self.crossmatch_radius)/3600., 
                                    nnearest=1)
                matchdist = ds * 3600.

            if have_match_coord or have_pyspherematch:
                num_match = len(ind_plate)
                self.db_update_process(num_ucac4=num_match)

                if num_match > 0:
                    ind = ind_finite[ind_plate]
                    self.sources['ucac4_id'][ind] = id_ucac[ind_ucac]
                    self.sources['ucac4_ra'][ind] = ra_ucac[ind_ucac]
                    self.sources['ucac4_dec'][ind] = dec_ucac[ind_ucac]
                    self.sources['ucac4_bmag'][ind] = bmag_ucac[ind_ucac]
                    self.sources['ucac4_vmag'][ind] = vmag_ucac[ind_ucac]
                    self.sources['ucac4_dist'][ind] = matchdist

            # Match sources with the Tycho-2 catalogue
            self.log.write('Cross-matching sources with the Tycho-2 catalogue', 
                           level=3, event=54)
            fn_tycho2 = os.path.join(self.tycho2_dir, 'tycho2_pyplate.fits')

            try:
                tycho2 = fits.open(fn_tycho2)
                tycho2_available = True
            except IOError:
                tycho2_available = False

            if tycho2_available:
                ra_tyc = tycho2[1].data.field(0)
                dec_tyc = tycho2[1].data.field(1)
                pmra_tyc = tycho2[1].data.field(2)
                pmdec_tyc = tycho2[1].data.field(3)
                btmag_tyc = tycho2[1].data.field(4)
                vtmag_tyc = tycho2[1].data.field(5)
                ebtmag_tyc = tycho2[1].data.field(6)
                evtmag_tyc = tycho2[1].data.field(7)
                tyc1 = tycho2[1].data.field(8)
                tyc2 = tycho2[1].data.field(9)
                tyc3 = tycho2[1].data.field(10)
                hip_tyc = tycho2[1].data.field(11)

                # For stars that have proper motion data, calculate RA, Dec
                # for the plate epoch
                indpm = np.where(np.isfinite(pmra_tyc) & 
                                  np.isfinite(pmdec_tyc))[0]
                ra_tyc[indpm] = (ra_tyc[indpm] 
                                 + (plate_epoch - 2000.) * pmra_tyc[indpm]
                                 / np.cos(dec_tyc[indpm] * np.pi / 180.) 
                                 / 3600000.)
                dec_tyc[indpm] = (dec_tyc[indpm] 
                                  + (plate_epoch - 2000.) * pmdec_tyc[indpm]
                                  / 3600000.)

                if self.ncp_close:
                    btyc = (dec_tyc > self.min_dec)
                elif self.scp_close:
                    btyc = (dec_tyc < self.max_dec)
                elif self.max_ra-self.min_ra > 180:
                    btyc = (((ra_tyc < self.min_ra) |
                            (ra_tyc > self.max_ra)) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))
                else:
                    btyc = ((ra_tyc > self.min_ra) & 
                            (ra_tyc < self.max_ra) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))

                indtyc = np.where(btyc)[0]
                numtyc = btyc.sum()

                ra_tyc = ra_tyc[indtyc] 
                dec_tyc = dec_tyc[indtyc] 
                btmag_tyc = btmag_tyc[indtyc]
                vtmag_tyc = vtmag_tyc[indtyc]
                ebtmag_tyc = ebtmag_tyc[indtyc]
                evtmag_tyc = evtmag_tyc[indtyc]
                id_tyc = np.array(['{:04d}-{:05d}-{:1d}'
                                   .format(tyc1[i], tyc2[i], tyc3[i]) 
                                   for i in indtyc])
                hip_tyc = hip_tyc[indtyc]

                self.log.write('Fetched {:d} entries from Tycho-2'
                               ''.format(numtyc))

                if have_match_coord:
                    coords = ICRS(ra_finite, dec_finite, 
                                  unit=(units.degree, units.degree))
                    catalog = ICRS(ra_tyc, dec_tyc, 
                                  unit=(units.degree, units.degree))
                    ind_tyc, ds2d, ds3d = match_coordinates_sky(coords, catalog,
                                                                nthneighbor=1)
                    ind_plate = np.arange(ind_tyc.size)
                    indmask = ds2d < float(self.crossmatch_radius)*units.arcsec
                    ind_plate = ind_plate[indmask]
                    ind_tyc = ind_tyc[indmask]
                    matchdist = ds2d[indmask].to(units.arcsec).value
                elif have_pyspherematch:
                    ind_plate,ind_tyc,ds = \
                        spherematch(ra_finite, dec_finite, ra_tyc, dec_tyc,
                                    tol=float(self.crossmatch_radius)/3600., 
                                    nnearest=1)
                    matchdist = ds * 3600.

                if have_match_coord or have_pyspherematch:
                    num_match = len(ind_plate)
                    self.db_update_process(num_tycho2=num_match)

                    if num_match > 0:
                        ind = ind_finite[ind_plate]
                        self.sources['tycho2_id'][ind] = id_tyc[ind_tyc]
                        self.sources['tycho2_ra'][ind] = ra_tyc[ind_tyc]
                        self.sources['tycho2_dec'][ind] = dec_tyc[ind_tyc]
                        self.sources['tycho2_btmag'][ind] = btmag_tyc[ind_tyc]
                        self.sources['tycho2_vtmag'][ind] = vtmag_tyc[ind_tyc]
                        self.sources['tycho2_hip'][ind] = hip_tyc[ind_tyc]
                        self.sources['tycho2_dist'][ind] = matchdist

    def _solverec(self, in_head, in_astromsigma, distort=3, 
                  max_recursion_depth=None, force_recursion_depth=None):
        """
        Improve astrometry of a FITS file recursively in sub-fields.

        """

        if max_recursion_depth is None:
            max_recursion_depth = self.max_recursion_depth

        if force_recursion_depth is None:
            force_recursion_depth = self.force_recursion_depth

        fnsub = self.basefn + '_sub'
        scampxml_file = fnsub + '_scamp.xml'
        aheadfile = os.path.join(self.scratch_dir, fnsub + '_scamp.ahead')

        if 'DEPTH' in in_head:
            recdepth = in_head['DEPTH'] + 1
        else:
            recdepth = 1

        # Read SCAMP reference catalog for this plate
        #ref = fits.open(os.path.join(self.scratch_dir, 
        #                             self.basefn + '_scampref.cat'))
        ra_ref = self.scampref[2].data.field(0)
        dec_ref = self.scampref[2].data.field(1)
        reftmp = fits.HDUList()
        reftmp.append(self.scampref[0].copy())
        reftmp.append(self.scampref[1].copy())
        reftmp.append(self.scampref[2].copy())
        #reftmp = fits.open(os.path.join(self.scratch_dir, 
        #                                self.basefn + '_scampref.cat'))

        # Get the list of stars in the scan
        #xyfits = fits.open(os.path.join(self.scratch_dir,
        #                                self.basefn + '.xy'))
        #x = xyfits[1].data.field(0)
        #y = xyfits[1].data.field(1)
        #erra_arcsec = xyfits[1].data.field('ERRA_IMAGE') * self.mean_pixscale
        x = self.sources['x_source']
        y = self.sources['y_source']
        erra_arcsec = (self.sources['erra_source'] * self.mean_pixscale)
        ra = np.zeros(len(x)) * np.nan
        dec = np.zeros(len(x)) * np.nan
        sigma_ra = np.zeros(len(x)) * np.nan
        sigma_dec = np.zeros(len(x)) * np.nan
        gridsize = np.zeros(len(x))

        xsize = (in_head['XMAX'] - in_head['XMIN']) / 2.
        ysize = (in_head['YMAX'] - in_head['YMIN']) / 2.

        subrange = np.arange(4)
        xoffset = np.array([0., 1., 0., 1.])
        yoffset = np.array([0., 0., 1., 1.])

        for sub in subrange:
            xmin = in_head['XMIN'] + xoffset[sub] * xsize
            ymin = in_head['YMIN'] + yoffset[sub] * ysize
            xmax = xmin + xsize
            ymax = ymin + ysize

            width = xmax - xmin
            height = ymax - ymin

            self.log.write('Sub-field ({:d}x{:d}) {:.2f} : {:.2f}, '
                           '{:.2f} : {:.2f}'.format(2**recdepth, 2**recdepth, 
                                                    xmin, xmax, ymin, ymax))
            
            xmin_ext = xmin - 0.1 * xsize
            xmax_ext = xmax + 0.1 * xsize
            ymin_ext = ymin - 0.1 * ysize
            ymax_ext = ymax + 0.1 * ysize

            width_ext = xmax_ext - xmin_ext
            height_ext = ymax_ext - ymin_ext

            bsub = ((x >= xmin_ext + 0.5) & (x < xmax_ext + 0.5) &
                    (y >= ymin_ext + 0.5) & (y < ymax_ext + 0.5) &
                    (self.sources['flag_clean'] == 1))

            db_log_msg = ('Sub-field: {:d}x{:d}, '
                          'X: {:.2f} {:.2f}, Y: {:.2f} {:.2f}, '
                          'X_ext: {:.2f} {:.2f}, Y_ext: {:.2f} {:.2f}, '
                          '#stars: {:d}'
                          .format(2**recdepth, 2**recdepth, 
                                  xmin, xmax, ymin, ymax, 
                                  xmin_ext, xmax_ext, ymin_ext, ymax_ext, 
                                  bsub.sum()))

            self.log.write('Found {:d} stars in the sub-field'
                           .format(bsub.sum()), double_newline=False)

            if bsub.sum() < 50:
                self.log.write('Fewer stars than the threshold (50)')
                db_log_msg = '{} (<50)'.format(db_log_msg)
                self.log.to_db(4, db_log_msg, event=51)
                continue

            indsub = np.where(bsub)

            # Create a SCAMP catalog for the sub-field
            scampdata = fits.new_table(self.scampcat[2].columns, 
                                       nrows=bsub.sum()).data
            scampdata.field('X_IMAGE')[:] = x[indsub] - xmin_ext
            scampdata.field('Y_IMAGE')[:] = y[indsub] - ymin_ext
            scampdata.field('ERR_A')[:] = self.sources[indsub]['erra_source']
            scampdata.field('ERR_B')[:] = self.sources[indsub]['errb_source']
            scampdata.field('FLUX')[:] = self.sources[indsub]['flux_auto']
            scampdata.field('FLUX_ERR')[:] = self.sources[indsub]['fluxerr_auto']
            scampdata.field('FLAGS')[:] = self.sources[indsub]['sextractor_flags']
            self.scampcat[2].data = scampdata

            subscampfile = os.path.join(self.scratch_dir, fnsub + '_scamp.cat')

            if os.path.exists(subscampfile):
                os.remove(subscampfile)

            self.scampcat.writeto(subscampfile)

            # Build custom SCAMP reference catalog for the sub-field
            pixcorners = np.array([[xmin_ext,ymin_ext], 
                                   [xmax_ext,ymin_ext],
                                   [xmin_ext,ymax_ext],
                                   [xmax_ext,ymax_ext]])
            corners = self.wcs_plate.all_pix2world(pixcorners, 1)
            bref = ((ra_ref > corners[:,0].min()) & 
                    (ra_ref < corners[:,0].max()) &
                    (dec_ref > corners[:,1].min()) & 
                    (dec_ref < corners[:,1].max()))

            self.log.write('Found {:d} reference stars in the sub-field'
                           .format(bref.sum()), double_newline=False)
            db_log_msg = '{}, #reference: {:d}'.format(db_log_msg, bref.sum())

            if bref.sum() < 50:
                self.log.write('Fewer reference stars than the threshold (50)')
                db_log_msg = '{} (<50)'.format(db_log_msg)
                self.log.to_db(4, db_log_msg, event=51)
                continue

            self.log.to_db(4, db_log_msg, event=51)
            self.log.write('', timestamp=False, double_newline=False)

            indref = np.where(bref)
            reftmp[2].data = self.scampref[2].data[indref]

            scampref_file = os.path.join(self.scratch_dir, 
                                         fnsub + '_scampref.cat')

            if os.path.exists(scampref_file):
                os.remove(scampref_file)

            #reftmp[1].header.set('NAXIS1', 2880)
            #reftmp[1].header.set('TFORM1', '2880A')
            #reftmp[1].data.field(0)[0] = reftmp[1].data.field(0)[0].ljust(2880)
            reftmp[1].header.set('TDIM1', '(80, 21)')
            reftmp.writeto(scampref_file, output_verify='ignore')

            # Find a TAN solution for the sub-scan
            cmd = self.wcs_to_tan_path
            cmd += ' -w %s.wcs' % self.basefn

            # If .wcs file does not exist for the sub-scan, use global .wcs
            #if os.path.exists(os.path.join(self.scratch_dir, fnsub + '.wcs')):
            #    cmd += ' -w %s.wcs' % fnsub
            #else:
            #    cmd += ' -w %s.wcs' % self.basefn

            cmd += ' -x %f' % xmin_ext
            cmd += ' -y %f' % ymin_ext
            cmd += ' -W %f' % xmax_ext
            cmd += ' -H %f' % ymax_ext
            cmd += ' -N 10'
            cmd += ' -o %s_tan.wcs' % fnsub
            self.log.write('Subprocess: {}'.format(cmd))
            sp.call(cmd, shell=True, stdout=self.log.handle, 
                    stderr=self.log.handle, cwd=self.scratch_dir)

            tanhead = fits.getheader(os.path.join(self.scratch_dir, 
                                                  fnsub + '_tan.wcs'))
            tanhead.set('NAXIS', 2)

            ahead = fits.Header()
            ahead.set('NAXIS', 2)
            ahead.set('NAXIS1', int(width_ext+1.))
            ahead.set('NAXIS2', int(height_ext+1.))
            ahead.set('IMAGEW', int(width_ext+1.))
            ahead.set('IMAGEH', int(height_ext+1.))
            ahead.set('CTYPE1', 'RA---TAN')
            ahead.set('CTYPE2', 'DEC--TAN')
            ahead.set('CRPIX1', (width_ext + 1.) / 2.)
            ahead.set('CRPIX2', (height_ext + 1.) / 2.)
            ahead.set('CRVAL1', tanhead['CRVAL1'])
            ahead.set('CRVAL2', tanhead['CRVAL2'])
            ahead.set('CD1_1', tanhead['CD1_1'])
            ahead.set('CD1_2', tanhead['CD1_2'])
            ahead.set('CD2_1', tanhead['CD2_1'])
            ahead.set('CD2_2', tanhead['CD2_2'])

            # Output .ahead file
            if os.path.exists(aheadfile):
                os.remove(aheadfile)

            ahead.totextfile(aheadfile, endcard=True, clobber=True)

            crossid_radius = 20.
            
            # Run SCAMP 
            cmd = self.scamp_path
            cmd += ' -c %s_scamp.conf %s_scamp.cat' % (self.basefn, fnsub)
            cmd += ' -ASTREF_CATALOG FILE'
            cmd += ' -ASTREFCAT_NAME %s_scampref.cat' % fnsub
            cmd += ' -ASTREFCENT_KEYS X_WORLD,Y_WORLD'
            cmd += ' -ASTREFERR_KEYS ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD'
            cmd += ' -ASTREFMAG_KEY MAG'
            cmd += ' -ASTRCLIP_NSIGMA 1.5'
            cmd += ' -FLAGS_MASK 0x00ff'
            cmd += ' -SN_THRESHOLDS 20.0,100.0'
            cmd += ' -CROSSID_RADIUS %.2f' % crossid_radius
            cmd += ' -DISTORT_DEGREES %i' % distort
            cmd += ' -STABILITY_TYPE EXPOSURE'
            cmd += ' -SOLVE_PHOTOM N'
            cmd += ' -WRITE_XML Y'
            cmd += ' -XML_NAME %s' % scampxml_file
            cmd += ' -VERBOSE_TYPE LOG'
            cmd += ' -CHECKPLOT_TYPE NONE'
            #cmd += ' -CHECKPLOT_DEV PNG'
            #cmd += ' -CHECKPLOT_TYPE FGROUPS,DISTORTION,ASTR_REFERROR2D,ASTR_REFERROR1D'
            #cmd += ' -CHECKPLOT_NAME %s_fgroups,%s_distort,%s_astr_referror2d,%s_astr_referror1d' % \
            #    (fnsub, fnsub, fnsub, fnsub)
            self.log.write('CROSSID_RADIUS: {:.2f}'.format(crossid_radius))
            db_log_msg = 'SCAMP: CROSSID_RADIUS: {:.2f}'.format(crossid_radius)
            self.log.write('Subprocess: {}'.format(cmd))
            sp.call(cmd, shell=True, stdout=self.log.handle, 
                    stderr=self.log.handle, cwd=self.scratch_dir)

            # Read statistics from SCAMP XML file
            warnings.simplefilter('ignore')
            scampxml = votable.parse(os.path.join(self.scratch_dir, 
                                                  scampxml_file))
            warnings.resetwarnings()
            xmltab = scampxml.get_first_table()
            ndetect = xmltab.array['NDeg_Reference'].data[0]
            astromsigma = xmltab.array['AstromSigma_Reference'].data[0]

            # Read SCAMP .head file and update it
            head = fits.PrimaryHDU().header
            head.set('NAXIS', 2)
            head.set('NAXIS1', int(width_ext+1.))
            head.set('NAXIS2', int(height_ext+1.))
            head.set('IMAGEW', int(width_ext+1.))
            head.set('IMAGEH', int(height_ext+1.))
            head.set('WCSAXES', 2)
            #head.set('NAXIS1', in_head['IMAGEW'])
            #head.set('NAXIS2', in_head['IMAGEH'])
            #head.set('IMAGEW', in_head['IMAGEW'])
            #head.set('IMAGEH', in_head['IMAGEH'])
            fn_scamphead = os.path.join(self.scratch_dir, fnsub + '_scamp.head')
            head.extend(fits.Header.fromfile(fn_scamphead, sep='\n', 
                                             endcard=False, padding=False))

            if os.path.exists(aheadfile):
                os.remove(aheadfile)

            head.totextfile(aheadfile, endcard=True, clobber=True)

            #prev_crossid_radius = crossid_radius
            crossid_radius = 3. * in_astromsigma.max()

            if crossid_radius > 20:
                crossid_radius = 20.

            #if crossid_radius < 0.5 * prev_crossid_radius:
            #    crossid_radius = 0.5 * prev_crossid_radius

            if crossid_radius < 5:
                crossid_radius = 5.

            # Run SCAMP 
            cmd = self.scamp_path
            cmd += ' -c %s_scamp.conf %s_scamp.cat' % (self.basefn, fnsub)
            cmd += ' -ASTREF_CATALOG FILE'
            cmd += ' -ASTREFCAT_NAME %s_scampref.cat' % fnsub
            cmd += ' -ASTREFCENT_KEYS X_WORLD,Y_WORLD'
            cmd += ' -ASTREFERR_KEYS ERRA_WORLD,ERRB_WORLD,ERRTHETA_WORLD'
            cmd += ' -ASTREFMAG_KEY MAG'
            cmd += ' -ASTRCLIP_NSIGMA 1.5'
            cmd += ' -FLAGS_MASK 0x00ff'
            cmd += ' -SN_THRESHOLDS 20.0,100.0'
            cmd += ' -CROSSID_RADIUS %.2f' % crossid_radius
            cmd += ' -DISTORT_DEGREES %i' % distort
            cmd += ' -STABILITY_TYPE EXPOSURE'
            cmd += ' -SOLVE_PHOTOM N'
            cmd += ' -WRITE_XML Y'
            cmd += ' -XML_NAME %s' % scampxml_file
            cmd += ' -VERBOSE_TYPE LOG'
            cmd += ' -CHECKPLOT_TYPE NONE'
            #cmd += ' -CHECKPLOT_DEV PNG'
            #cmd += ' -CHECKPLOT_TYPE FGROUPS,DISTORTION,ASTR_REFERROR2D,ASTR_REFERROR1D'
            #cmd += ' -CHECKPLOT_NAME %s_fgroups,%s_distort,%s_astr_referror2d,%s_astr_referror1d' % \
            #    (fnsub, fnsub, fnsub, fnsub)
            self.log.write('CROSSID_RADIUS: {:.2f}'.format(crossid_radius))
            db_log_msg = '{} {:.2f}'.format(db_log_msg, crossid_radius)
            self.log.write('Subprocess: {}'.format(cmd))
            sp.call(cmd, shell=True, stdout=self.log.handle, 
                    stderr=self.log.handle, cwd=self.scratch_dir)

            # Read statistics from SCAMP XML file
            warnings.simplefilter('ignore')
            scampxml = votable.parse(os.path.join(self.scratch_dir, 
                                                  scampxml_file))
            warnings.resetwarnings()
            xmltab = scampxml.get_first_table()
            ndetect = xmltab.array['NDeg_Reference'].data[0]
            astromsigma = xmltab.array['AstromSigma_Reference'].data[0]

            self.log.write('SCAMP reported {:d} detections'.format(ndetect), 
                           double_newline=False)
            self.log.write('Astrometric sigmas: {:.3f} {:.3f}, '
                           'previous: {:.3f} {:.3f}'
                           ''.format(astromsigma[0], astromsigma[1],
                                     in_astromsigma[0], in_astromsigma[1]),
                           double_newline=False)
            mean_diff = (astromsigma-in_astromsigma).mean()
            mean_diff_ratio = mean_diff / in_astromsigma.mean()
            self.log.write('Mean astrometric sigma difference: '
                           '{:.3f} ({:+.1f}%)'
                           ''.format(mean_diff, mean_diff_ratio*100.))
            db_log_msg = ('{}, #detections: {:d}, sigmas: {:.3f} {:.3f}, '
                          'previous: {:.3f} {:.3f}, '
                          'mean difference: {:.3f} ({:+.1f}%)'
                          .format(db_log_msg, ndetect, 
                                  astromsigma[0], astromsigma[1], 
                                  in_astromsigma[0], in_astromsigma[1], 
                                  mean_diff, mean_diff_ratio*100.))
            self.log.to_db(5, db_log_msg, event=52)

            # Use decreasing threshold for astrometric sigmas
            astrom_threshold = 2. - (recdepth - 1) * 0.2

            if astrom_threshold < 1.:
                astrom_threshold = 1.

            if ((ndetect >= 5.*crossid_radius) and 
                (((astromsigma-in_astromsigma).min() < 0) or
                (astromsigma.mean() < astrom_threshold * in_astromsigma.mean()))):
                # Read SCAMP .head file and update it
                head = fits.PrimaryHDU().header
                head.set('NAXIS', 2)
                head.set('NAXIS1', int(width_ext+1.))
                head.set('NAXIS2', int(height_ext+1.))
                head.set('IMAGEW', int(width_ext+1.))
                head.set('IMAGEH', int(height_ext+1.))
                head.set('WCSAXES', 2)
                #head.set('NAXIS1', in_head['IMAGEW'])
                #head.set('NAXIS2', in_head['IMAGEH'])
                #head.set('IMAGEW', in_head['IMAGEW'])
                #head.set('IMAGEH', in_head['IMAGEH'])
                fn_scamphead = os.path.join(self.scratch_dir, 
                                            fnsub + '_scamp.head')
                head.extend(fits.Header.fromfile(fn_scamphead, sep='\n', 
                                                 endcard=False, padding=False))

                # Select stars for coordinate conversion
                bout = ((x >= xmin + 0.5) & (x < xmax + 0.5) & 
                        (y >= ymin + 0.5) & (y < ymax + 0.5))

                if bout.sum() == 0:
                    continue
                
                indout = np.where(bout)

                if have_esutil:
                    subwcs = wcsutil.WCS(head)
                    ra_out,dec_out = subwcs.image2sky(x[indout]-xmin_ext, y[indout]-ymin_ext)
                    ra[indout] = ra_out
                    dec[indout] = dec_out
                else:
                    # Save header file without line-breaks
                    hdrfile = os.path.join(self.scratch_dir, 
                                           fnsub + '_scamp.hdr')

                    if os.path.exists(hdrfile):
                        os.remove(hdrfile)

                    head.tofile(hdrfile, sep='', endcard=True, padding=False)

                    # Output x,y in ASCII format
                    xyout = np.column_stack((x[indout]-xmin_ext, y[indout]-ymin_ext))
                    np.savetxt(os.path.join(self.scratch_dir, 
                                            fnsub + '_xy.txt'), 
                               xyout, fmt='%9.3f\t%9.3f')

                    # Convert x,y to RA,Dec
                    cmd = self.xy2sky_path
                    cmd += (' -d -o rd {} @{}_xy.txt > {}_world.txt'
                            ''.format(hdrfile, fnsub, fnsub))
                    self.log.write('Subprocess: {}'.format(cmd))
                    sp.call(cmd, shell=True, stdout=self.log.handle, 
                            stderr=self.log.handle, cwd=self.scratch_dir)

                    # Read RA,Dec from a file
                    world = np.loadtxt(os.path.join(self.scratch_dir, 
                                                    fnsub + '_world.txt'), 
                                       usecols=(0,1))
                    ra[indout] = world[:,0]
                    dec[indout] = world[:,1]

                sigma_ra[indout] = np.sqrt(erra_arcsec[indout]**2 + astromsigma[0]**2)
                sigma_dec[indout] = np.sqrt(erra_arcsec[indout]**2 + astromsigma[1]**2)
                gridsize[indout] = 2**recdepth

                # Solve sub-fields recursively if recursion depth is less
                # than maximum.
                if recdepth < max_recursion_depth:
                    head.set('XMIN', xmin)
                    head.set('XMAX', xmax)
                    head.set('YMIN', ymin)
                    head.set('YMAX', ymax)
                    head.set('DEPTH', recdepth)

                    new_radec,new_gridsize = \
                            self._solverec(head, astromsigma, 
                                           distort=distort,
                                           max_recursion_depth=max_recursion_depth,
                                           force_recursion_depth=force_recursion_depth)
                    bnew = (np.isfinite(new_radec[:,0]) & 
                            np.isfinite(new_radec[:,1]))
                    #bnew = (new_radec[:,0] < 999) & (new_radec[:,1] < 999)

                    if bnew.sum() > 0:
                        indnew = np.where(bnew)
                        ra[indnew] = new_radec[indnew,0]
                        dec[indnew] = new_radec[indnew,1]
                        sigma_ra[indnew] = new_radec[indnew,2]
                        sigma_dec[indnew] = new_radec[indnew,3]
                        gridsize[indnew] = new_gridsize[indnew]
            elif recdepth < force_recursion_depth:
                # Solve sub-fields recursively if recursion depth is less
                # than the required minimum.
                head.set('XMIN', xmin)
                head.set('XMAX', xmax)
                head.set('YMIN', ymin)
                head.set('YMAX', ymax)
                head.set('DEPTH', recdepth)

                new_radec,new_gridsize = \
                        self._solverec(head, astromsigma,
                                       distort=distort,
                                       max_recursion_depth=max_recursion_depth,
                                       force_recursion_depth=force_recursion_depth)
                bnew = (np.isfinite(new_radec[:,0]) & 
                        np.isfinite(new_radec[:,1]))
                #bnew = (new_radec[:,0] < 999) & (new_radec[:,1] < 999)

                if bnew.sum() > 0:
                    indnew = np.where(bnew)
                    ra[indnew] = new_radec[indnew,0]
                    dec[indnew] = new_radec[indnew,1]
                    sigma_ra[indnew] = new_radec[indnew,2]
                    sigma_dec[indnew] = new_radec[indnew,3]
                    gridsize[indnew] = new_gridsize[indnew]

        #ref.close()
        #reftmp.close()

        return (np.column_stack((ra, dec, sigma_ra, sigma_dec)), 
                gridsize)

    def process_source_coordinates(self):
        """
        Combine coordinates from the global and recursive solutions.
        Calculate X, Y, and Z on the unit sphere.

        """

        self.log.to_db(3, 'Processing source coordinates', event=60)

        self.sources['raj2000'] = self.sources['raj2000_wcs']
        self.sources['dej2000'] = self.sources['dej2000_wcs']

        ind = np.where(np.isfinite(self.sources['raj2000_sub']) &
                       np.isfinite(self.sources['dej2000_sub']))

        if len(ind[0]) > 0:
            self.sources['raj2000'][ind] = self.sources['raj2000_sub'][ind]
            self.sources['dej2000'][ind] = self.sources['dej2000_sub'][ind]

        # Calculate X, Y, and Z on the unit sphere
        # http://www.sdss3.org/svn/repo/idlutils/tags/v5_5_5/pro/coord/angles_to_xyz.pro
        phi_rad = np.radians(self.sources['raj2000'])
        theta_rad = np.radians(90. - self.sources['dej2000'])
        self.sources['x_sphere'] = np.cos(phi_rad) * np.sin(theta_rad)
        self.sources['y_sphere'] = np.sin(phi_rad) * np.sin(theta_rad)
        self.sources['z_sphere'] = np.cos(theta_rad)

        if have_healpy:
            ind = np.where(np.isfinite(self.sources['raj2000']) &
                           np.isfinite(self.sources['dej2000']))

            if len(ind[0]) > 0:
                phi_rad = np.radians(self.sources['raj2000'][ind])
                theta_rad = np.radians(90. - self.sources['dej2000'][ind])
                hp256 = healpy.ang2pix(256, theta_rad, phi_rad, nest=True)
                self.sources['healpix256'][ind] = hp256

    def output_sources_csv(self, filename=None):
        """
        Write source list with calibrated RA and Dec to an ASCII file.

        """

        self.log.to_db(3, 'Writing sources to a file', event=62)

        # Create output directory, if missing
        if self.write_source_dir and not os.path.isdir(self.write_source_dir):
            self.log.write('Creating output directory {}'
                           .format(self.write_source_dir), level=4, event=62)
            os.makedirs(self.write_source_dir)

        if filename:
            fn_world = os.path.join(self.write_source_dir, 
                                    os.path.basename(filename))
        else:
            fn_world = os.path.join(self.write_source_dir, 
                                    '{}_sources.csv'.format(self.basefn))

        outfields = ['source_num', 'x_source', 'y_source', 
                     'erra_source', 'errb_source', 'errtheta_source',
                     'a_source', 'b_source', 'theta_source',
                     'elongation',
                     'raj2000_wcs', 'dej2000_wcs',
                     'raj2000_sub', 'dej2000_sub', 
                     'raerr_sub', 'decerr_sub',
                     'gridsize_sub',
                     'mag_auto', 'magerr_auto', 
                     'flux_auto', 'fluxerr_auto',
                     'mag_iso', 'magerr_iso', 
                     'flux_iso', 'fluxerr_iso',
                     'flux_max', 'flux_radius',
                     'isoarea', 'sqrt_isoarea', 'background',
                     'sextractor_flags', 
                     'dist_center', 'dist_edge', 'annular_bin',
                     'flag_rim', 'flag_negradius', 'flag_clean',
                     'ucac4_id', 'ucac4_ra', 'ucac4_dec',
                     'ucac4_bmag', 'ucac4_vmag', 'ucac4_dist',
                     'tycho2_id', 'tycho2_ra', 'tycho2_dec',
                     'tycho2_btmag', 'tycho2_vtmag', 'tycho2_dist']
        outfmt = [_source_meta[f][1] for f in outfields]
        outhdr = ','.join(outfields)
        #outhdr = ','.join(['"{}"'.format(f) for f in outfields])
        delimiter = ','

        # Output ascii file with refined coordinates
        self.log.write('Writing output file {}'.format(fn_world), level=4, 
                       event=62)
        np.savetxt(fn_world, self.sources[outfields], fmt=outfmt, 
                   delimiter=delimiter, header=outhdr, comments='')

    def output_sources_db(self):
        """
        Write source list with calibrated RA and Dec to the database.

        """

        self.log.to_db(3, 'Writing sources to the database', event=61)
        self.log.write('Open database connection for writing to the '
                       'source and source_calib tables.')
        platedb = PlateDB()
        platedb.open_connection(host=self.output_db_host,
                                user=self.output_db_user,
                                dbname=self.output_db_name,
                                passwd=self.output_db_passwd)

        if (self.scan_id is not None and self.plate_id is not None and 
            self.archive_id is not None and self.process_id is not None):
            platedb.write_sources(self.sources, process_id=self.process_id,
                                  scan_id=self.scan_id, plate_id=self.plate_id,
                                  archive_id=self.archive_id)
            
        platedb.close_connection()
        self.log.write('Closed database connection.')

