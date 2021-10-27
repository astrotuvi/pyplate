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
from astropy import __version__ as astropy_version
from astropy import wcs
from astropy.io import fits, votable
from astropy.table import Table, vstack
from astropy.coordinates import Angle, EarthLocation, SkyCoord, ICRS, AltAz
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy import __version__ as scipy_version
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.ndimage.filters import generic_filter
from scipy.linalg import lstsq
from collections import OrderedDict
from ..database.database import PlateDB
from ..conf import read_conf
from .._version import __version__
from .sources import SourceTable
from .catalog import StarCatalog
from .solve import SolveProcess
from .photometry import PhotometryProcess

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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import load_model
    have_keras = True
except ImportError:
    have_keras = False

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
    from statsmodels import __version__ as statsmodels_version
    import statsmodels.api as sm
    have_statsmodels = True
except ImportError:
    have_statsmodels = False


class ProcessLog:
    """
    Plate process log class

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
            log_dir = os.path.dirname(self.path)

            try:
                os.makedirs(log_dir)
            except OSError:
                if not os.path.isdir(log_dir):
                    print('Could not create directory {}'.format(log_dir))

            try:
                self.handle = open(self.path, 'w', 1)
            except IOError:
                print('Could not open log file {}'.format(self.path))
                self.handle = sys.stdout
        else:
            self.handle = sys.stdout

    def write(self, message, timestamp=True, double_newline=True, 
              level=None, event=None, solution_num=None):
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
        solution_num : int
            Astrometric solution number

        """

        log_message = '{}'.format(message)

        if solution_num:
            log_message = '{{Solution {:d}}} {}'.format(solution_num,
                                                        log_message)

        if timestamp:
            log_message = '[{}] {}'.format(str(dt.datetime.now()), log_message)

        if double_newline:
            log_message += '\n'

        self.handle.write('{}\n'.format(log_message))

        if level is not None:
            self.to_db(level, message, event=event, solution_num=solution_num)

    def to_db(self, level, message, event=None, solution_num=None):
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
        solution_num : int
            Astrometric solution number

        """

        if self.platedb is not None and self.process_id is not None:
            self.platedb.write_processlog(level, message, event=event,
                                          solution_num=solution_num,
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


class Process:
    """
    Plate process class

    """

    def __init__(self, filename, archive_id=None):
        self.filename = os.path.basename(filename)
        self.archive_id = archive_id
        self.basefn = ''
        self.fn_fits = ''
        self.process_id = None
        self.scan_id = None
        self.plate_id = None
        self.conf = None
        self.active = False

        self.fits_dir = ''
        self.index_dir = ''
        self.gaia_dir = ''
        self.tycho2_dir = ''
        self.work_dir = ''
        self.write_source_dir = ''
        self.write_db_source_dir = ''
        self.write_db_source_calib_dir = ''
        self.write_db_source_xmatch_dir = ''
        self.write_db_solution_healpix_dir = ''
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

        self.output_db = None
        self.write_sources_csv = False
        self.write_solution_healpix_csv = False

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
        self.solveproc = None
        self.star_catalog = None

        self.plate_solution = None
        self.scampref = None
        self.scampcat = None
        self.gaia_files = None
        self.phot_cterm_list = []
        self.phot_calib_list = []
        self.phot_calibrated = False
        self.phot_calib_curves = None
        self.num_iterations = 0

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
                     'write_db_source_dir', 'write_db_source_calib_dir',
                     'write_db_source_xmatch_dir',
                     'write_db_solution_healpix_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except configparser.Error:
                pass

        if self.write_log_dir:
            self.enable_log = True

        for attr in ['use_gaia_fits', 'use_tycho2_fits', 
                     'use_ucac4_db', 'use_apass_db',
                     'enable_db_log', 'write_sources_csv',
                     'write_solution_healpix_csv']:
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
                     'output_db']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except configparser.Error:
                pass

        for attr in ['use_filter', 'use_psf', 'circular_film']:
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
            self.log = ProcessLog(log_path)
            self.log.open()
        else:
            self.log = ProcessLog(None)
            self.log.open()

        # Get process_id from the database
        if self.output_db:
            self.db_process_start()

        # Open database connection for logs
        if self.enable_db_log:
            platedb = PlateDB()
            platedb.assign_conf(self.conf)
            platedb.open_connection()

            self.log.platedb = platedb
            self.log.archive_id = self.archive_id
            self.log.plate_id = self.plate_id
            self.log.scan_id = self.scan_id
            self.log.process_id = self.process_id
            self.log.to_db(3, 'Setting up plate solve process', event=10)

        self.log.write('Using PyPlate version {}'.format(__version__), 
                       level=4, event=10)
        self.log.write('Using Astropy version {}'.format(astropy_version), 
                       level=4, event=10)
        self.log.write('Using NumPy version {}'.format(np.__version__), 
                       level=4, event=10)
        self.log.write('Using SciPy version {}'.format(scipy_version),
                       level=4, event=10)

        if have_statsmodels:
            self.log.write('Using statsmodels version {}'
                           .format(statsmodels_version), level=4, event=10)

        # Check if FITS file exists
        if not os.path.exists(self.fn_fits):
            self.log.write('FITS file does not exist: {}'.format(self.fn_fits), 
                           level=1, event=10)
            self.finish(completed=0)
            return

        # Read FITS header
        if (not self.plate_header or 
            isinstance(self.plate_header['NAXIS1'], fits.card.Undefined) or
            isinstance(self.plate_header['NAXIS2'], fits.card.Undefined)):
            try:
                self.plate_header = fits.getheader(self.fn_fits, 
                                                   ignore_missing_end=True)
            except IOError:
                self.log.write('Could not read FITS file {}'
                               .format(self.fn_fits), 
                               level=1, event=10)
                self.finish(completed=0)
                return

        if self.plate_header['NAXIS'] != 2:
            self.log.write('Incompatible FITS file: NAXIS != 2',
                           level=1, event=10)
            self.finish(completed=0)
            return

        self.imwidth = self.plate_header['NAXIS1']
        self.imheight = self.plate_header['NAXIS2']

        # Look for observation date in the FITS header.
        if ('YEAR' in self.plate_header and 
            isinstance(self.plate_header['YEAR'], float) and
            self.plate_header['YEAR']>1800):
            self.plate_epoch = self.plate_header['YEAR']
        elif 'DATEORIG' in self.plate_header:
            try:
                self.plate_year = int(self.plate_header['DATEORIG']
                                      .split('-')[0])
                self.plate_epoch = float(self.plate_year) + 0.5
            except ValueError:
                pass
        elif 'DATE-OBS' in self.plate_header:
            try:
                self.plate_year = int(self.plate_header['DATE-OBS']
                                      .split('-')[0])
                self.plate_epoch = float(self.plate_year) + 0.5
            except ValueError:
                pass

        self.plate_year = int(self.plate_epoch)

        if self.platemeta is not None and 'numexp' in self.platemeta:
            num_exposures = self.platemeta['numexp']
        else:
            num_exposures = None

        self.db_update_process(num_exposures=num_exposures,
                               plate_epoch=self.plate_epoch)

        # Create scratch directory
        self.scratch_dir = os.path.join(self.work_dir, 
                                        '{}_{}'.format(self.basefn,
                                                       self.timestamp_str))

        try:
            os.makedirs(self.scratch_dir)
        except OSError:
            if not os.path.isdir(self.scratch_dir):
                raise

        # Declare process active
        self.active = True

    def finish(self, completed=1):
        """
        Finish plate process.

        """

        # Close open FITS files
        if isinstance(self.scampref, fits.HDUList):
            self.scampref.close()

        if isinstance(self.scampcat, fits.HDUList):
            self.scampcat.close()

        # Remove scratch directory and its contents
        if self.scratch_dir:
            shutil.rmtree(self.scratch_dir)

        # Write process end to the database
        self.db_process_end(completed=completed)

        # Close database connection used for logging
        if self.log.platedb is not None:
            self.log.to_db(3, 'Finish plate solve process', event=99)
            self.log.platedb.close_connection()

        # Close log file
        self.log.close()

        # Deactivate process
        self.active = False

    def db_process_start(self):
        """
        Write process start to the database.

        """

        platedb = PlateDB()
        platedb.assign_conf(self.conf)
        platedb.open_connection()
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

    def db_update_process(self, **kwargs):
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
            platedb.assign_conf(self.conf)
            platedb.open_connection()
            platedb.update_process(self.process_id, **kwargs)
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
            platedb.assign_conf(self.conf)
            platedb.open_connection()
            duration = (dt.datetime.now()-self.timestamp).seconds
            platedb.write_process_end(self.process_id, 
                                      completed=completed, 
                                      duration=duration)
            platedb.close_connection()

    def invert_plate(self):
        """
        Invert FITS image and save the result (\*_inverted.fits) in the scratch 
        or work directory.

        """

        fn_inverted = '{}_inverted.fits'.format(self.basefn)

        if self.scratch_dir:
            fn_inverted = os.path.join(self.scratch_dir, fn_inverted)
        else:
            fn_inverted = os.path.join(self.work_dir, fn_inverted)
        
        if not os.path.exists(fn_inverted):
            self.log.write('Inverting image', level=3, event=11)

            fitsfile = fits.open(self.fn_fits, do_not_scale_image_data=True, 
                                 ignore_missing_end=True)

            invfits = fits.PrimaryHDU(-fitsfile[0].data)
            invfits.header = fitsfile[0].header.copy()
            invfits.header.set('BZERO', 32768)
            invfits.header.set('BSCALE', 1.0)
            self.log.write('Writing inverted image: {}'.format(fn_inverted), 
                           level=4, event=11)
            invfits.writeto(fn_inverted)

            fitsfile.close()
            del fitsfile
            del invfits
        else:
            self.log.write('Inverted file exists: {}'.format(fn_inverted), 
                           level=4, event=11)

    def extract_sources(self, threshold_sigma=None, use_filter=None,
                        filter_path=None, use_psf=None, 
                        psf_threshold_sigma=None, psf_model_sigma=None, 
                        circular_film=None):
        """
        Extract sources from a FITS file.

        Parameters
        ----------
        threshold_sigma : float
            SExtractor threshold in sigmas (default 4.0)
        use_filter : bool
            Use SExtractor filter for source detection (default False)
        filter_path : string
            Path to SExtractor filter file
        use_psf : bool
            Use PSF for bright stars (default False)
        psf_threshold_sigma : float
            SExtractor threshold in sigmas for using PSF (default 20.0)
        psf_model_sigma : float
            SExtractor threshold in sigmas for PSF model stars (default 20.0)
        circular_film : bool
            Assume circular film (default False)
        """

        self.log.write('Extracting sources from image', level=3, event=20)
        sex_ver = (sp.check_output([self.sextractor_path, '-v']).strip()
                   .decode('utf-8'))
        self.log.write('Using {}'.format(sex_ver), level=4, event=20)

        if threshold_sigma is None:
            threshold_sigma = self.threshold_sigma

        if use_filter is None:
            use_filter = self.use_filter

        if filter_path is None:
            filter_path = self.filter_path

        if use_filter and filter_path:
            filter_exists = os.path.exists(filter_path)
        else:
            filter_exists = False

        if use_filter and not filter_exists:
            use_filter = False
            self.log.write('Filter file does not exist: {}'
                           ''.format(filter_path), 
                           level=2, event=20)

        if use_psf is None:
            use_psf = self.use_psf

        if psf_threshold_sigma is None:
            psf_threshold_sigma = self.psf_threshold_sigma

        if psf_model_sigma is None:
            psf_model_sigma = self.psf_model_sigma

        if circular_film is None:
            circular_film = self.circular_film

        self.log.write('Running SExtractor to get sky value', level=3, 
                       event=21)

        # Create parameter file
        fn_sex_param = self.basefn + '_sextractor.param'
        fconf = open(os.path.join(self.scratch_dir, fn_sex_param), 
                     'w')
        fconf.write('NUMBER')
        fconf.close()

        # Create configuration file
        cnf = 'DETECT_THRESH    {:f}\n'.format(60000)
        cnf += 'ANALYSIS_THRESH  {:f}\n'.format(60000)
        cnf += 'THRESH_TYPE      ABSOLUTE\n'
        cnf += 'FILTER           N\n'
        cnf += 'SATUR_LEVEL      65000.0\n'
        cnf += 'BACKPHOTO_TYPE   LOCAL\n'
        cnf += 'MAG_ZEROPOINT    25.0\n'
        cnf += 'PARAMETERS_NAME  {}\n'.format(fn_sex_param)
        cnf += 'CATALOG_TYPE     FITS_1.0\n'
        cnf += 'CATALOG_NAME     {}_sky.cat\n'.format(self.basefn)
        cnf += 'WRITE_XML        Y\n'
        cnf += 'XML_NAME         sex.xml\n'

        fn_sex_conf = self.basefn + '_sextractor.conf'
        self.log.write('Writing SExtractor configuration file {}'
                       .format(fn_sex_conf), level=4, event=21)
        self.log.write('SExtractor configuration file:\n{}'
                       .format(cnf), level=5, event=21)
        fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 
                     'w')
        fconf.write(cnf)
        fconf.close()

        cmd = self.sextractor_path
        cmd += ' %s_inverted.fits' % self.basefn
        cmd += ' -c %s' % fn_sex_conf
        self.log.write('Subprocess: {}'.format(cmd), 
                       level=5, event=21)
        sp.call(cmd, shell=True, stdout=self.log.handle, 
                stderr=self.log.handle, cwd=self.scratch_dir)
        self.log.write('', timestamp=False, double_newline=False)

        tree = ET.parse(os.path.join(self.scratch_dir, 'sex.xml'))
        root = tree.getroot()
        use_fix_threshold = False

        if root[1][4][15][11].attrib['name'] == 'Background_Mean':
            sky = float(root[1][4][15][19][0][0][8].text)
            sky_sigma = float(root[1][4][15][19][0][0][9].text)
            self.log.write('Sky: {:.3f}, sigma: {:.3f}'.format(sky, sky_sigma), 
                       level=4, event=21)
            self.db_update_process(sky=sky, sky_sigma=sky_sigma)

            if sky < 2*sky_sigma or sky < 100:
                use_fix_threshold = True
                psf_model_threshold = 20000
                psf_threshold_adu = 20000
                threshold_adu = 5000
                self.log.write('Sky value too low, using fixed thresholds', 
                               level=4, event=21)

        # If SExtractor catalog does not exist then run SExtractor
        if not os.path.exists(os.path.join(self.scratch_dir, 
                                           self.basefn + '.cat')):
            self.log.write('Running SExtractor without PSF model',
                           level=3, event=22)

            if use_filter:
                self.log.write('Using filter {}'.format(filter_path), 
                               level=4, event=22)

            if use_fix_threshold:
                self.log.write('Using threshold {:f} ADU'.format(threshold_adu), 
                               level=4, event=22)
                self.db_update_process(threshold=threshold_adu)
            else:
                threshold_adu = sky_sigma * threshold_sigma
                self.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                               .format(threshold_sigma, threshold_adu),
                               level=4, event=22)
                self.db_update_process(threshold=threshold_adu)

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
            fconf.write('FWHM_IMAGE\n')
            fconf.write('ISOAREA_IMAGE\n')
            fconf.write('BACKGROUND\n')
            fconf.write('FLAGS')
            fconf.close()

            if use_fix_threshold:
                cnf = 'DETECT_THRESH    {:f}\n'.format(threshold_adu)
                cnf += 'ANALYSIS_THRESH  {:f}\n'.format(threshold_adu)
                cnf += 'THRESH_TYPE      ABSOLUTE\n'
            else:
                cnf = 'DETECT_THRESH    {:f}\n'.format(threshold_sigma)
                cnf += 'ANALYSIS_THRESH  {:f}\n'.format(threshold_sigma)

            if use_filter:
                cnf += 'FILTER           Y\n'
                cnf += 'FILTER_NAME      {}\n'.format(filter_path)
            else:
                cnf += 'FILTER           N\n'

            cnf += 'DEBLEND_NTHRESH  64\n'
            cnf += 'DEBLEND_MINCONT  0.0001\n'
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
                           .format(fn_sex_conf), level=4, event=22)
            self.log.write('SExtractor configuration file:\n{}'.format(cnf), 
                           level=5, event=22)
            fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 'w')
            fconf.write(cnf)
            fconf.close()

            cmd = self.sextractor_path
            cmd += ' %s_inverted.fits' % self.basefn
            cmd += ' -c %s' % fn_sex_conf
            self.log.write('Subprocess: {}'.format(cmd), level=5, event=22)
            sp.call(cmd, shell=True, stdout=self.log.handle, 
                    stderr=self.log.handle, cwd=self.scratch_dir)
            self.log.write('', timestamp=False, double_newline=False)

        if use_psf:
            enough_psf_sources = None

            # If PSFEx input file does not exist then run SExtractor
            fn_psfex_cat = os.path.join(self.scratch_dir, 
                                        self.basefn + '_psfex.cat')

            if not os.path.exists(fn_psfex_cat):
                self.log.write('Running SExtractor to get sources for PSFEx', 
                               level=3, event=23)

                while True:
                    if use_filter:
                        self.log.write('Using filter {}'.format(filter_path), 
                                       level=4, event=23)

                    # Calculate threshold for PSF sources based on 
                    # extracted sources without PSF model
                    fn_cat = os.path.join(self.scratch_dir, 
                                          self.basefn + '.cat')
                    t = Table.read(fn_cat, hdu=2)
                    num_sources = len(t)
                    flux_peak = np.sort(t['FLUX_MAX'])[::-1]

                    if self.min_model_sources < num_sources:
                        th_max = flux_peak[self.min_model_sources]
                        th_max_sigma = th_max / sky_sigma
                    else:
                        th_max = None
                        th_max_sigma = None

                    if self.max_model_sources < num_sources:
                        th_min = flux_peak[self.max_model_sources]
                        th_min_sigma = th_min / sky_sigma
                    else:
                        th_min = None
                        th_min_sigma = None

                    if use_fix_threshold:
                        if (th_min is not None and th_min > 0 and
                            psf_model_threshold < th_min):
                            psf_model_threshold = th_min
                        elif (th_max is not None and th_max > 0 and
                              psf_model_threshold > th_max):
                            psf_model_threshold = th_max

                        self.log.write('Using threshold {:f} ADU'
                                       .format(psf_model_threshold), 
                                       level=4, event=23)
                    else:
                        if (th_min_sigma is not None and th_min_sigma > 0 and
                            psf_model_sigma < th_min_sigma):
                            psf_model_sigma = th_min_sigma
                        elif (th_max_sigma is not None and th_max_sigma > 0 and
                              psf_model_sigma > th_max_sigma):
                            psf_model_sigma = th_max_sigma

                        threshold_adu = sky_sigma * psf_model_sigma
                        self.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                       .format(psf_model_sigma, threshold_adu), 
                                       level=4, event=23)

                    # Create parameter file
                    fn_sex_param = self.basefn + '_sextractor.param'
                    fconf = open(os.path.join(self.scratch_dir, fn_sex_param), 
                                 'w')
                    fconf.write('VIGNET(120,120)\n')
                    fconf.write('X_IMAGE\n')
                    fconf.write('Y_IMAGE\n')
                    fconf.write('MAG_AUTO\n')
                    fconf.write('FLUX_AUTO\n')
                    fconf.write('FLUXERR_AUTO\n')
                    fconf.write('SNR_WIN\n')
                    fconf.write('FLUX_RADIUS\n')
                    fconf.write('ELONGATION\n')
                    fconf.write('FLAGS')
                    fconf.close()

                    # Create configuration file
                    if use_fix_threshold:
                        cnf = 'DETECT_THRESH    {:f}\n'.format(psf_model_threshold)
                        cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_model_threshold)
                        cnf += 'THRESH_TYPE      ABSOLUTE\n'
                    else:
                        cnf = 'DETECT_THRESH    {:f}\n'.format(psf_model_sigma)
                        cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_model_sigma)

                    if use_filter:
                        cnf += 'FILTER           Y\n'
                        cnf += 'FILTER_NAME      {}\n'.format(filter_path)
                    else:
                        cnf += 'FILTER           N\n'

                    cnf += 'SATUR_LEVEL      65000.0\n'
                    cnf += 'BACKPHOTO_TYPE   LOCAL\n'
                    cnf += 'MAG_ZEROPOINT    25.0\n'
                    cnf += 'PARAMETERS_NAME  {}\n'.format(fn_sex_param)
                    cnf += 'CATALOG_TYPE     FITS_LDAC\n'
                    cnf += 'CATALOG_NAME     {}_psfex.cat\n'.format(self.basefn)

                    fn_sex_conf = self.basefn + '_sextractor.conf'
                    self.log.write('Writing SExtractor configuration file {}'
                                   .format(fn_sex_conf), level=4, event=23)
                    self.log.write('SExtractor configuration file:\n{}'
                                   .format(cnf), level=5, event=23)
                    fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 
                                 'w')
                    fconf.write(cnf)
                    fconf.close()

                    cmd = self.sextractor_path
                    cmd += ' %s_inverted.fits' % self.basefn
                    cmd += ' -c %s' % fn_sex_conf
                    self.log.write('Subprocess: {}'.format(cmd), 
                                   level=5, event=23)
                    sp.call(cmd, shell=True, stdout=self.log.handle, 
                            stderr=self.log.handle, cwd=self.scratch_dir)
                    self.log.write('', timestamp=False, double_newline=False)

                    hcat = fits.getheader(fn_psfex_cat, 2)
                    num_psf_sources = hcat['NAXIS2']
                    self.log.write('Extracted {:d} PSF-model sources'
                                   .format(num_psf_sources), level=4, event=23)
                    enough_psf_sources = False

                    if num_psf_sources >= self.min_model_sources: 
                        enough_psf_sources = True
                        break
                    else:
                        # Repeat with lower threshold to get more sources
                        if use_fix_threshold:
                            psf_model_threshold *= 0.9
                        else:
                            psf_model_sigma *= 0.9

                        self.log.write('Too few PSF-model sources (min {:d}), '
                                       'repeating extraction with lower '
                                       'threshold'
                                       .format(self.min_model_sources), 
                                       level=4, event=23)

                    #if num_psf_sources > self.max_model_sources:
                    #    # Repeat with higher threshold to get less sources
                    #    if use_fix_threshold:
                    #        psf_model_threshold *= 1.2
                    #    else:
                    #        psf_model_sigma *= 1.2

                    #    self.log.write('Too many PSF-model sources (max {:d}), '
                    #                   'repeating extraction with higher '
                    #                   'threshold'
                    #                   .format(self.max_model_sources), 
                    #                   level=4, event=23)

            # Run PSFEx
            if (enough_psf_sources and
                not os.path.exists(os.path.join(self.scratch_dir, 
                                                self.basefn + '_psfex.psf'))):
                self.log.write('Running PSFEx', level=3, event=24)
                psfex_ver = (sp.check_output([self.psfex_path, '-v']).strip()
                             .decode('utf-8'))
                self.log.write('Using {}'.format(psfex_ver), level=4, event=24)

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
                cnf += 'NTHREADS           2\n'

                fn_psfex_conf = self.basefn + '_psfex.conf'
                self.log.write('Writing PSFEx configuration file {}'
                               .format(fn_psfex_conf), level=4, event=24)
                self.log.write('PSFEx configuration file:\n{}'
                               .format(cnf), level=5, event=24)
                fconf = open(os.path.join(self.scratch_dir, fn_psfex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.psfex_path
                cmd += ' %s_psfex.cat' % self.basefn
                cmd += ' -c %s' % fn_psfex_conf
                self.log.write('Subprocess: {}'.format(cmd), level=5, event=24)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

            # Run SExtractor with PSF
            if (enough_psf_sources and
                not os.path.exists(os.path.join(self.scratch_dir, 
                                                self.basefn + '.cat-psf'))):
                self.log.write('Running SExtractor with PSF model',
                               level=3, event=25)

                if use_filter:
                    self.log.write('Using filter {}'.format(filter_path), 
                                   level=4, event=25)

                if use_fix_threshold:
                    self.log.write('Using threshold {:f} ADU'
                                   .format(psf_threshold_adu), 
                                   level=4, event=25)
                else:
                    threshold_adu = sky_sigma * psf_threshold_sigma
                    self.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                   .format(psf_threshold_sigma, threshold_adu), 
                                   level=4, event=25)

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

                if use_fix_threshold:
                    cnf = 'DETECT_THRESH    {:f}\n'.format(psf_threshold_adu)
                    cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_threshold_adu)
                    cnf += 'THRESH_TYPE      ABSOLUTE\n'
                else:
                    cnf = 'DETECT_THRESH    {:f}\n'.format(psf_threshold_sigma)
                    cnf += 'ANALYSIS_THRESH  {:f}\n'.format(psf_threshold_sigma)

                if use_filter:
                    cnf += 'FILTER           Y\n'
                    cnf += 'FILTER_NAME      {}\n'.format(filter_path)
                else:
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
                               .format(fn_sex_conf), level=4, event=25)
                self.log.write('SExtractor configuration file:\n{}'
                               .format(cnf), level=5, event=25)
                fconf = open(os.path.join(self.scratch_dir, fn_sex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.sextractor_path
                cmd += ' %s_inverted.fits' % self.basefn
                cmd += ' -c %s' % fn_sex_conf
                self.log.write('Subprocess: {}'.format(cmd), level=5, event=25)
                sp.call(cmd, shell=True, stdout=self.log.handle, 
                        stderr=self.log.handle, cwd=self.scratch_dir)
                self.log.write('', timestamp=False, double_newline=False)

        # Read the SExtractor output catalog
        self.log.write('Reading sources from the SExtractor output file', 
                       level=3, event=26)
        xycat = fits.open(os.path.join(self.scratch_dir, self.basefn + '.cat'))
        self.num_sources = len(xycat[1].data)
        self.db_update_process(num_sources=self.num_sources)
        self.log.write('Extracted {:d} sources'
                       .format(self.num_sources), level=4, event=26)

        # Initialise source table
        self.sources = SourceTable()
        self.sources.populate(num_sources=self.num_sources)
        self.sources.platemeta = self.platemeta
        self.sources.log = self.log

        # Copy values from the SExtractor catalog, xycat
        self.sources.copy_from_sextractor(xycat)

        # Calculate square root of isoarea
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
            std_bg = bg.std()

        # Calculate distance from edge
        distarr = np.column_stack((xim, self.imwidth-xim, 
                                   yim, self.imheight-yim))
        self.sources['dist_edge'] = np.amin(distarr, 1)
        
        # Define 8 concentric annular bins + bin9 for edges
        self.log.write('Flagging sources', level=3, event=27)
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
        bin9_width_s = int(0.1 * min_halfwidth / sampling)
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
                           double_newline=False, level=4, event=27)

            if nbin > 0:
                indbin = np.where(bbin)
                self.sources['annular_bin'][indbin] = b

        bbin = ((self.sources['dist_edge'] < 0.1*min_halfwidth) |
                (self.sources['dist_center'] >= bin9_corner_dist))
        nbin = bbin.sum()
        self.log.write('Annular bin 9 (radius {:8.2f} pixels): '
                       '{:6d} sources'.format(bin9_corner_dist, nbin), 
                       level=4, event=27)

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
                  (self.sources['magerr_auto'] < 5) &
                  (borderbg < 100))
        indclean = np.where(bclean)[0]
        self.sources['flag_clean'][indclean] = 1
        self.log.write('Flagged {:d} clean sources'.format(bclean.sum()),
                       double_newline=False, level=4, event=27)

        indrim = np.where(borderbg >= 100)[0]
        self.sources['flag_rim'][indrim] = 1
        self.log.write('Flagged {:d} sources at the plate rim'
                       ''.format(len(indrim)), double_newline=False, 
                       level=4, event=27)

        indnegrad = np.where(self.sources['flux_radius'] <= 0)[0]
        self.sources['flag_negradius'][indnegrad] = 1
        self.log.write('Flagged {:d} sources with negative FLUX_RADIUS'
                       ''.format(len(indnegrad)), level=4, event=27)

        # For bright stars, update coordinates with PSF coordinates
        if use_psf and enough_psf_sources:
            self.log.write('Updating coordinates with PSF coordinates '
                           'for bright sources', level=3, event=28)

            fn_psfcat = os.path.join(self.scratch_dir, self.basefn + '.cat-psf')

            if os.path.exists(fn_psfcat):
                try:
                    psfcat = fits.open(fn_psfcat)
                except IOError:
                    self.log.write('Could not read PSF coordinates, file {} '
                                   'is corrupt'.format(fn_psfcat), 
                                   level=2, event=28)
                    psfcat = None

                if psfcat is not None and psfcat[1].header['NAXIS2'] > 0:
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

                    num_psf_sources = len(ind1)
                    self.log.write('Replacing x,y values from PSF photometry '
                                   'for {:d} sources'.format(num_psf_sources),
                                   level=4, event=28)
                    self.sources['x_psf'][ind1] = \
                            psfcat[1].data.field('XPSF_IMAGE')[ind2]
                    self.sources['y_psf'][ind1] = \
                            psfcat[1].data.field('YPSF_IMAGE')[ind2]
                    self.sources['erra_psf'][ind1] = \
                            psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
                    self.sources['errb_psf'][ind1] = \
                            psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
                    self.sources['errtheta_psf'][ind1] = \
                            psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
                    self.sources['x_source'][ind1] = \
                            psfcat[1].data.field('XPSF_IMAGE')[ind2]
                    self.sources['y_source'][ind1] = \
                            psfcat[1].data.field('YPSF_IMAGE')[ind2]
                    self.sources['erra_source'][ind1] = \
                            psfcat[1].data.field('ERRAPSF_IMAGE')[ind2]
                    self.sources['errb_source'][ind1] = \
                            psfcat[1].data.field('ERRBPSF_IMAGE')[ind2]
                    self.sources['errtheta_source'][ind1] = \
                            psfcat[1].data.field('ERRTHETAPSF_IMAGE')[ind2]
                    self.sources['flag_usepsf'][ind1] = 1
                    self.db_update_process(num_psf_sources=num_psf_sources)
                elif psfcat is not None and psfcat[1].header['NAXIS2'] == 0:
                    self.log.write('There are no sources with PSF coordinates!',
                                   level=2, event=28)
                    self.db_update_process(num_psf_sources=0)
            else:
                self.log.write('Could not read PSF coordinates, '
                               'file {} does not exist!'.format(fn_psfcat), 
                               level=2, event=28)
                self.db_update_process(num_psf_sources=0)

    def classify_artifacts(self):
        """
        Classify extracted sources as celestial or artifacts.

        Algorithm and model by Gal Matijevic

        """

        self.log.write('Classifying artifacts', level=3, event=29)

        if not have_keras:
            self.log.write('Missing dependency (keras) for artifact '
                           'classification!', level=2, event=29)
            return

        # Read model
        fn_model = os.path.join(os.path.dirname(__file__), 'artifact_model.h5')
        model = load_model(fn_model)

        # Read inverted FITS file
        fn_image = os.path.join(self.scratch_dir,
                                self.basefn + '_inverted.fits')
        plate_fits = fits.open(fn_image)
        plate_data = plate_fits[0].data
        plate_fits.close()

        # 1/2 of the size of the thumbnail
        thumb_size = 10

        # number of thumbnails
        noft = len(self.sources)

        # initialize thumbnails array
        thumbs = np.zeros((noft, 1, thumb_size * 2, thumb_size * 2))

        for i, (x, y) in enumerate(zip(self.sources['x_image'],
                                       self.sources['y_image'])):
            # get the centers
            xx = min(max(int(np.floor(x)), thumb_size),
                     plate_data.shape[1] - thumb_size)
            yy = min(max(int(np.floor(y)), thumb_size),
                     plate_data.shape[0] - thumb_size)
            z = plate_data[yy - thumb_size:yy + thumb_size,
                           xx - thumb_size:xx + thumb_size]

            za, zs = np.mean(z), np.std(z)

            # if there is no variance, every pixel in the thumb is 0 after mean subtraction
            # so no need to divide
            if zs < 1e-12:
                z = z - za
            else:
                # Invert to get negative from positive image
                z = 1. - (z - za) / zs

            thumbs[i][0] = z

        prediction = model.predict(thumbs).flatten()
        self.sources['model_prediction'] = prediction

        # Count probable artifacts and write the number to the database
        num_artifacts = (prediction < 0.1).sum()
        num_true_sources = (prediction > 0.9).sum()
        self.db_update_process(num_artifacts=num_artifacts,
                               num_true_sources=num_true_sources)
        self.log.write('Classified {:d} sources as true sources '
                       '(prediction > 0.9)'
                       .format(num_true_sources), level=4, event=29,
                       double_newline=False)
        self.log.write('Classified {:d} sources as artifacts (prediction < 0.1)'
                       .format(num_artifacts), level=4, event=29)

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

        self.log.write('Solving astrometry', level=3, event=30)

        # Initialise solve process
        solveproc = SolveProcess(self.filename, archive_id=self.archive_id)

        # Assign configuration
        solveproc.assign_conf(self.conf)

        # Assign process attributes
        solveproc.__dict__.update(self.__dict__)

        # Transform source table to numpy array
        solveproc.sources = self.sources.as_array()

        # Do plate solving
        plate_solution = solveproc.solve_plate(plate_epoch=plate_epoch, sip=sip,
                                               skip_bright=skip_bright,
                                               repeat_find=repeat_find)

        # Create WCS header
        if plate_solution.plate_solved:
            plate_solution.create_wcs_header()

        # Retrieve solutions
        self.plate_solution = plate_solution
        self.plate_solved = plate_solution.plate_solved

        # Apply scanner pattern to source coordinates
        if self.plate_solved:
            if self.plate_solution.pattern_ratio is not None:
                self.sources.apply_scanner_pattern(self.plate_solution)

            self.db_update_process(solved=1,
                                   num_solutions=plate_solution.num_solutions,
                                   pattern_ratio=plate_solution.pattern_ratio)
        else:
            self.db_update_process(solved=0, num_solutions=0)

    def output_solution_db(self, write_csv=None):
        """
        Write plate solution to the database.

        """

        if write_csv is None:
            write_csv = self.write_solution_healpix_csv

        self.log.to_db(3, 'Writing astrometric solution to the database', 
                       event=77)

        if self.plate_solution.num_solutions == 0:
            self.log.write('No plate solution to write to the database', 
                           level=2, event=77)
            return

        if write_csv:
            # Create output directory, if missing
            if (self.write_db_solution_healpix_dir and
                not os.path.isdir(self.write_db_solution_healpix_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_solution_healpix_dir),
                               level=4, event=77)
                os.makedirs(self.write_db_solution_healpix_dir)

        self.log.write('Open database connection for writing to the '
                       'solution, solution_set, solution_healpix and '
                       'scanner_pattern tables')
        platedb = PlateDB()
        platedb.assign_conf(self.conf)
        platedb.open_connection()

        if (self.scan_id is not None and self.plate_id is not None and 
            self.archive_id is not None and self.process_id is not None):
            kw = {'process_id': self.process_id,
                  'scan_id': self.scan_id,
                  'plate_id': self.plate_id,
                  'archive_id': self.archive_id}

            set_id = platedb.write_platesolution(self.plate_solution, **kw)
            kw['solutionset_id'] = set_id

            for solution in self.plate_solution.solutions:
                sol_id = platedb.write_solution(solution, **kw)

                if solution['healpix_table'] is not None:
                    kw['solution_id'] = sol_id
                    kw['solution_num'] = solution['solution_num']
                    kw['write_csv'] = write_csv
                    platedb.write_solution_healpix(solution['healpix_table'],
                                                   **kw)
                    del kw['solution_id']
                    del kw['solution_num']
                    del kw['write_csv']

            for solution in self.plate_solution.duplicate_solutions:
                platedb.write_solution(solution, **kw)

            if self.plate_solution.pattern_table is not None:
                for row in self.plate_solution.pattern_table:
                    platedb.write_scanner_pattern(row, **kw)
            
        platedb.close_connection()
        self.log.write('Closed database connection')

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

            fn_header = '{}.wcs'.format(self.basefn)
            fn_header = os.path.join(self.write_wcs_dir, fn_header)
            self.log.write('Writing WCS output file {}'.format(fn_header),
                           level=4, event=36)
            header = self.plate_solution.header_wcs
            header.tofile(fn_header, overwrite=True)

    def query_star_catalog(self, mag_range=[0,15], color_term=None):
        """
        Query external star catalog for astrometric and photometric reference
        stars.

        Parameters:
        -----------
        mag_range : list
            A two-element list specifying bright and faint magnitude limits
            for external catalog query.

        """

        self.log.write('Getting reference catalogs', level=3, event=40)

        if not self.plate_solved:
            self.log.write('Cannot query external catalog due to missing '
                           'astrometric solutions!', level=2, event=40)
            return

        # Initialise star_catalog
        if self.star_catalog is None:
            self.star_catalog = StarCatalog()
            self.star_catalog.assign_conf(self.conf)
            self.star_catalog.gaia_dir = self.gaia_dir
            self.star_catalog.scratch_dir = self.scratch_dir
            self.star_catalog.log = self.log

        # Query Gaia catalog
        self.star_catalog.query_gaia(self.plate_solution, mag_range=mag_range,
                                     color_term=color_term)

    def get_reference_catalogs(self):
        """
        Get reference catalogs for astrometric and photometric calibration.

        """

        self.log.write('Getting reference catalogs', level=3, event=40)

        if not self.plate_solved:
            self.log.write('Missing initial solution, '
                           'cannot get reference catalogs!', 
                           level=2, event=40)
            return

        # Read the Tycho-2 catalogue
        if self.use_tycho2_fits:
            self.log.write('Reading the Tycho-2 catalogue', level=3, event=41)
            fn_tycho2 = os.path.join(self.tycho2_dir, 'tycho2_pyplate.fits')

            try:
                tycho2 = fits.open(fn_tycho2)
                tycho2_available = True
            except IOError:
                self.log.write('Missing Tycho-2 data', level=2, event=41)
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
                ind_nullhip = np.where(hip_tyc == -2147483648)[0]

                if len(ind_nullhip) > 0:
                    hip_tyc[ind_nullhip] = 0

                # For stars that have proper motion data, calculate RA, Dec
                # for the plate epoch
                indpm = np.where(np.isfinite(pmra_tyc) & 
                                  np.isfinite(pmdec_tyc))[0]
                ra_tyc[indpm] = (ra_tyc[indpm] 
                                 + (self.plate_epoch - 2000.) * pmra_tyc[indpm]
                                 / np.cos(dec_tyc[indpm] * np.pi / 180.) 
                                 / 3600000.)
                dec_tyc[indpm] = (dec_tyc[indpm] 
                                  + (self.plate_epoch - 2000.) 
                                  * pmdec_tyc[indpm] / 3600000.)

                if self.ncp_close:
                    btyc = (dec_tyc > self.min_dec)
                elif self.scp_close:
                    btyc = (dec_tyc < self.max_dec)
                elif self.max_ra < self.min_ra:
                    btyc = (((ra_tyc < self.max_ra) |
                            (ra_tyc > self.min_ra)) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))
                else:
                    btyc = ((ra_tyc > self.min_ra) & 
                            (ra_tyc < self.max_ra) &
                            (dec_tyc > self.min_dec) & 
                            (dec_tyc < self.max_dec))

                indtyc = np.where(btyc)[0]
                numtyc = btyc.sum()

                self.ra_tyc = ra_tyc[indtyc] 
                self.dec_tyc = dec_tyc[indtyc] 
                self.btmag_tyc = btmag_tyc[indtyc]
                self.vtmag_tyc = vtmag_tyc[indtyc]
                self.btmagerr_tyc = ebtmag_tyc[indtyc]
                self.vtmagerr_tyc = evtmag_tyc[indtyc]
                self.id_tyc = np.array(['{:d}-{:d}-{:d}'
                                        .format(tyc1[i], tyc2[i], tyc3[i])
                                        for i in indtyc])
                self.id_tyc_pad = np.array(['{:04d}-{:05d}-{:1d}'
                                            .format(tyc1[i], tyc2[i], tyc3[i])
                                            for i in indtyc])
                self.hip_tyc = hip_tyc[indtyc]
                self.num_tyc = numtyc

                self.log.write('Fetched {:d} entries from Tycho-2'
                               ''.format(numtyc), level=4, event=41)

        query_combined = False
        query_healpix = False

        # Query the UCAC4 catalog
        if self.use_ucac4_db:
            self.log.write('Querying the UCAC4 catalogue', level=3, event=42)
            query_ucac4 = True

            # Check UCAC4 database name and table name
            if self.ucac4_db_name == '':
                self.log.write('UCAC4 database name missing!', 
                               level=2, event=42)
                query_ucac4 = False

            if self.ucac4_db_table == '':
                self.log.write('UCAC4 database table name missing!', 
                               level=2, event=42)
                query_ucac4 = False

            ucac4_cols = [col.strip() 
                          for col,typ in self.ucac4_columns.values()]
            ucac4_types = [typ.strip() 
                           for col,typ in self.ucac4_columns.values()]

            # Check if all columns are specified
            if '' in ucac4_cols:
                self.log.write('One ore more UCAC4 database column '
                               'names missing!', level=2, event=42)
                query_ucac4 = False

            # Check if APASS and UCAC4 data are in the same table
            if (query_ucac4 and self.use_apass_db and 
                    (self.ucac4_db_name == self.apass_db_name) and 
                    (self.ucac4_db_table == self.apass_db_table)):
                apass_cols = [col.strip() 
                              for col,typ in self.apass_columns.values()]
                apass_types = [typ.strip() 
                               for col,typ in self.apass_columns.values()]

                # Check if all columns are specified
                if '' in apass_cols:
                    self.log.write('One ore more APASS database column '
                                   'names missing!', level=2, event=42)
                    query_combined = False
                else:
                    ucac4_cols.extend(apass_cols)
                    ucac4_types.extend(apass_types)
                    query_combined = True

                    if self.conf.has_section(self.ucac4_db_table):
                        try:
                            colstr = self.conf.get(self.ucac4_db_table, 
                                                   'healpix')

                            if colstr != '':
                                ucac4_cols.append(colstr)
                                ucac4_types.append('i')
                                query_healpix = True
                        except configparser.Error:
                            pass

            if query_ucac4:
                ucac4_ra_col = self.ucac4_columns['ucac4_ra'][0]
                ucac4_dec_col = self.ucac4_columns['ucac4_dec'][0]

                # Query MySQL database
                db = MySQLdb.connect(host=self.ucac4_db_host, 
                                     user=self.ucac4_db_user, 
                                     passwd=self.ucac4_db_passwd,
                                     db=self.ucac4_db_name)
                cur = db.cursor()

                sql = 'SELECT {} FROM {} '.format(','.join(ucac4_cols),
                                                  self.ucac4_db_table)

                if self.ncp_close:
                    sql += 'WHERE {} > {}'.format(ucac4_dec_col, self.min_dec)
                elif self.scp_close:
                    sql += 'WHERE {} < {}'.format(ucac4_dec_col, self.max_dec)
                elif self.max_ra < self.min_ra:
                    sql += ('WHERE ({} < {} OR {} > {}) AND {} BETWEEN {} AND {}'
                            ''.format(ucac4_ra_col, self.max_ra, 
                                      ucac4_ra_col, self.min_ra,
                                      ucac4_dec_col, self.min_dec, self.max_dec))
                else:
                    sql += ('WHERE {} BETWEEN {} AND {} AND {} BETWEEN {} AND {}'
                            ''.format(ucac4_ra_col, self.min_ra, self.max_ra, 
                                      ucac4_dec_col, self.min_dec, self.max_dec))

                sql += ';'
                self.log.write('Query: {}'.format(sql), level=5, event=43)
                numrows = cur.execute(sql)
                self.log.write('Fetched {:d} rows from UCAC4'.format(numrows), 
                               level=4, event=42)

                res = np.fromiter(cur.fetchall(), dtype=','.join(ucac4_types))

                cur.close()
                db.commit()
                db.close()

                ra_ucac = res['f0']
                dec_ucac = res['f1']
                pmra_ucac = res['f6']
                pmdec_ucac = res['f7']

                self.raerr_ucac = res['f2']
                self.decerr_ucac = res['f3']
                self.mag_ucac = res['f4']
                self.magerr_ucac = res['f5']
                self.id_ucac = res['f8']
                self.bmag_ucac = res['f9']
                self.vmag_ucac = res['f10']
                self.bmagerr_ucac = res['f11']
                self.vmagerr_ucac = res['f12']
                self.num_ucac = numrows

                # Use proper motions to calculate RA/Dec for the plate epoch
                bpm = ((pmra_ucac != 0) & (pmdec_ucac != 0))
                num_pm = bpm.sum()

                if num_pm > 0:
                    ind_pm = np.where(bpm)
                    ra_ucac[ind_pm] = (ra_ucac[ind_pm] + (self.plate_epoch - 2000.)
                                       * pmra_ucac[ind_pm]
                                       / np.cos(dec_ucac[ind_pm] * np.pi / 180.) 
                                       / 3600000.)
                    dec_ucac[ind_pm] = (dec_ucac[ind_pm] + 
                                        (self.plate_epoch - 2000.)
                                        * pmdec_ucac[ind_pm] / 3600000.)

                self.ra_ucac = ra_ucac
                self.dec_ucac = dec_ucac

                if query_combined:
                    self.ra_apass = res['f13']
                    self.dec_apass = res['f14']
                    self.bmag_apass = res['f15']
                    self.vmag_apass = res['f16']
                    self.berr_apass = res['f17']
                    self.verr_apass = res['f18']
                    self.num_apass = numrows
                    self.combined_ucac_apass = True

                    # Use UCAC4 magnitudes to fill gaps in APASS
                    if query_healpix:
                        self.log.write('Filling gaps in the APASS data', 
                                       level=3, event=43)
                        healpix_ucac = res['f19']
                        uhp = np.unique(healpix_ucac)
                        indsort = np.argsort(healpix_ucac)
                        healpix_ucac_sort = healpix_ucac[indsort]
                        bmag_ucac_sort = self.bmag_ucac[indsort]
                        vmag_ucac_sort = self.vmag_ucac[indsort]
                        bmag_apass_sort = self.bmag_apass[indsort]
                        vmag_apass_sort = self.vmag_apass[indsort]

                        bgoodapass = (np.isfinite(bmag_apass_sort) &
                                      np.isfinite(vmag_apass_sort))

                        if bgoodapass.sum() > 0:
                            indgood = np.where(bgoodapass)[0]
                            bbright = ((bmag_apass_sort[indgood] <= 10) &
                                       (vmag_apass_sort[indgood] <= 10))

                            if bbright.sum() > 0:
                                bgoodapass[indgood[np.where(bbright)]] = False
                            
                        nfill = 0

                        # Go through all unique HEALPix in the plate area
                        for hp in uhp:
                            bapass = ((healpix_ucac_sort == hp) & bgoodapass)

                            # If there are more than 10 APASS stars in HEALPix,
                            # then do not do anything
                            if bapass.sum() > 10:
                                continue

                            bucac = ((healpix_ucac_sort == hp) &
                                     (bmag_ucac_sort > 10) &
                                     (vmag_ucac_sort > 10))

                            # There is a HEALPix with 10 or less valid 
                            # APASS magnitudes, but available magnitudes 
                            # in UCAC4
                            if (bucac.sum() - bapass.sum() > 0):
                                ind1 = indsort[np.where(bucac)]
                                ind2 = indsort[np.where(bapass)]
                                ind = np.setdiff1d(ind1, ind2)
                                self.bmag_apass[ind] = self.bmag_ucac[ind]
                                self.vmag_apass[ind] = self.vmag_ucac[ind]
                                self.berr_apass[ind] = self.bmagerr_ucac[ind]
                                self.verr_apass[ind] = self.vmagerr_ucac[ind]
                                nfill += ind.size

                        self.log.write('Added UCAC4 magnitudes to {:d} stars '
                                       'to fill gaps in the APASS data'
                                       ''.format(nfill), level=4, event=43)

        # Query the APASS catalog
        if self.use_apass_db and not query_combined:
            self.log.write('Querying the APASS catalogue', level=3, event=44)
            query_apass = True

            # Check UCAC4 database name
            if self.apass_db_name == '':
                self.log.write('APASS database name missing!', 
                               level=2, event=44)
                query_apass = False

            apass_cols = [col.strip() 
                          for col,typ in self.apass_columns.values()]
            apass_types = [typ.strip() 
                           for col,typ in self.apass_columns.values()]

            # Check if all columns are specified
            if '' in apass_cols:
                self.log.write('One ore more APASS database column '
                               'names missing!', level=2, event=44)
                query_apass = False

            if query_apass:
                apass_ra_col = self.apass_columns['apass_ra'][0]
                apass_dec_col = self.apass_columns['apass_dec'][0]

                # Query MySQL database
                db = MySQLdb.connect(host=self.apass_db_host, 
                                     user=self.apass_db_user, 
                                     passwd=self.apass_db_passwd,
                                     db=self.apass_db_name)
                cur = db.cursor()

                sql = 'SELECT {} FROM {} '.format(','.join(apass_cols),
                                                  self.apass_db_table)

                if self.ncp_close:
                    sql += 'WHERE {} > {}'.format(apass_dec_col, self.min_dec)
                elif self.scp_close:
                    sql += 'WHERE {} < {}'.format(apass_dec_col, self.max_dec)
                elif self.max_ra < self.min_ra:
                    sql += ('WHERE ({} < {} OR {} > {}) AND {} BETWEEN {} AND {}'
                            ''.format(apass_ra_col, self.max_ra, 
                                      apass_ra_col, self.min_ra,
                                      apass_dec_col, self.min_dec, self.max_dec))
                else:
                    sql += ('WHERE {} BETWEEN {} AND {} AND {} BETWEEN {} AND {}'
                            ''.format(apass_ra_col, self.min_ra, self.max_ra, 
                                      apass_dec_col, self.min_dec, self.max_dec))

                sql += ';'
                self.log.write('Query: {}'.format(sql), level=5, event=44)
                num_apass = cur.execute(sql)
                self.log.write('Fetched {:d} rows from APASS'.format(num_apass),
                               level=4, event=44)
                res = np.fromiter(cur.fetchall(), dtype='f8,f8,f8,f8,f8,f8')
                cur.close()
                db.commit()
                db.close()

                self.ra_apass = res['f0']
                self.dec_apass = res['f1']
                self.bmag_apass = res['f2']
                self.vmag_apass = res['f3']
                self.berr_apass = res['f4']
                self.verr_apass = res['f5']
                self.num_apass = num_apass

    def calibrate_photometry(self):
        """
        Calibrate extracted magnitudes.

        """

        self.log.write('Calibrating photometry', event=70, level=3)

        # Initialise photometry process
        photproc = PhotometryProcess()

        # Assign process attributes
        photproc.__dict__.update(self.__dict__)

        # Pass source table to the photometry process
        photproc.sources = self.sources

        # Get plate limiting magnitude from the distribution of magnitudes
        # of all clean sources (all solutions)
        mask_clean = ((self.sources['mag_auto'] > 0)
                      & (self.sources['mag_auto'] < 90)
                      & (self.sources['flag_clean'] == 1))
        mag_all = self.sources['mag_auto'][mask_clean].astype(np.double)
        kde_all = sm.nonparametric.KDEUnivariate(mag_all)
        kde_all.fit()
        ind_dense = kde_all.density > 0.2 * kde_all.density.max()
        plate_mag_lim = kde_all.support[ind_dense][-1]

        # Default values
        mean_cur_color_term = None
        weighted_cur_color_term = None
        max_cur_faint_limit = None
        min_cur_bright_limit = None
        num_calib = None
        self.num_iterations = 0

        # Variables to store calibration results
        phot_calib_curves = []
        prelim_color_term = []
        prelim_cterm_nstars = []
        cur_faint_limit = []
        est_faint_limit = []

        # Carry out photometric calibration for all solutions
        for i in np.arange(1, self.plate_solution.num_solutions+1):
            photproc.calibrate_photometry_gaia(solution_num=i, iteration=1)
            self.num_iterations = 1

            if not photproc.phot_calibrated:
                continue

            # Retrieve calibration curve
            phot_calib_curves.append(photproc.calib_curve)

            # Second iteration
            photproc.calibrate_photometry_gaia(solution_num=i, iteration=2)
            self.num_iterations = 2

            if not photproc.phot_calibrated:
                continue

            # Retrieve calibration curve
            phot_calib_curves.append(photproc.calib_curve)

            # Collect color term and faint limit from the second iteration
            prelim_color_term.append(photproc.phot_calib['color_term'])
            prelim_cterm_nstars.append(photproc.phot_calib['cterm_num_stars'])
            cur_faint_limit.append(photproc.phot_calib['faint_limit'])

            # Estimate faint limit with the current calibration curve
            estlim = photproc.calib_curve(plate_mag_lim).item()
            est_faint_limit.append(estlim)

        # If photometric calibration failed for all solutions, then give up
        if len(phot_calib_curves) == 0 or len(cur_faint_limit) == 0:
            num_gaia_edr3 = self.sources.num_crossmatch_gaia
            self.db_update_process(num_gaia_edr3=num_gaia_edr3,
                                   num_iterations=self.num_iterations,
                                   calibrated=0)
            self.log.write('Photometric calibration failed for all solutions',
                           event=74, level=2)
            return

        # Calculate mean color term and max faint limit over all solutions
        mean_color_term = np.array(prelim_color_term).mean()
        weighted_color_term = ((np.array(prelim_color_term) *
                                np.array(prelim_cterm_nstars)).sum() /
                               np.array(prelim_cterm_nstars).sum())
        max_cur_faint_limit = np.array(cur_faint_limit).max()
        est_faint_limit = np.array(est_faint_limit).max()

        self.log.write('Mean color term {:.3f}, '
                       'weighted color term {:.3f}'
                       .format(mean_color_term, weighted_color_term),
                       event=74, level=4, double_newline=False)
        self.log.write('Current faint limit {:.3f}, '
                       'estimated faint limit {:.3f}'
                       .format(max_cur_faint_limit, est_faint_limit),
                       event=74, level=4)

        # Magnitude range for catalog query
        if est_faint_limit < max_cur_faint_limit + 2.:
            new_max_mag = est_faint_limit + 1.
        else:
            new_max_mag = (max_cur_faint_limit + est_faint_limit) / 2. + 1.

        # Set magnitude ceiling at 15
        new_max_mag = min([new_max_mag, 15])

        # Remove current star catalog to get a fresh set of stars
        self.star_catalog = None
        cur_catalog_limit = 0
        new_mag_range = [0, new_max_mag]

        # Set max catalog magnitude dependent on color term
        max_catalog_mag = 19. + weighted_color_term

        iteration = 3
        num_calib_solutions = 0

        # Erase natmag and corrections of the preliminary calibration
        self.sources['natmag'] = np.nan
        self.sources['natmag_correction'] = np.nan

        # Get fainter stars to star_catalog until star_catalog is approximately
        # 1 mag deeper than the plate faint limit, or max_catalog_mag is reached
        while (cur_catalog_limit < max_cur_faint_limit + 0.7
               and cur_catalog_limit < max_catalog_mag):
            self.query_star_catalog(mag_range=new_mag_range,
                                    color_term=weighted_color_term)
            cur_catalog_limit = new_mag_range[1]
            self.sources.crossmatch_gaia(self.plate_solution, self.star_catalog)

            num_calib_solutions = 0
            cur_calib_stars = []
            cur_color_term = []
            cur_cterm_nstars = []
            cur_bright_limit = []
            cur_faint_limit = []
            est_faint_limit = []

            for i in np.arange(1, self.plate_solution.num_solutions+1):
                photproc.calibrate_photometry_gaia(solution_num=i,
                                                   iteration=iteration)

                if not photproc.phot_calibrated:
                    continue

                num_calib_solutions += 1
                phot_calib_curves.append(photproc.calib_curve)

                # Get current color term, bright and faint limits
                last_calib = photproc.phot_calib
                cur_color_term.append(last_calib['color_term'])
                cur_cterm_nstars.append(last_calib['cterm_num_stars'])
                cur_bright_limit.append(last_calib['bright_limit'])
                cur_faint_limit.append(last_calib['faint_limit'])
                cur_calib_stars.append(last_calib['num_calib_stars'])

                # Estimate faint limit with the current calibration curve
                estlim = photproc.calib_curve(plate_mag_lim).item()
                est_faint_limit.append(estlim)

            if num_calib_solutions == 0:
                break

            # Update number of iterations
            self.num_iterations = iteration

            # Calculate total number of calibration stars in all solutions
            num_calib = np.sum(cur_calib_stars)

            # Calculate mean value in color-term list
            mean_cur_color_term = np.array(cur_color_term).mean()
            weighted_cur_color_term = ((np.array(cur_color_term) *
                                        np.array(cur_cterm_nstars)).sum() /
                                       np.array(cur_cterm_nstars).sum())

            # Find min value in bright-limit list
            min_cur_bright_limit = np.array(cur_bright_limit).min()

            # Calculate max values in faint-limit lists
            max_cur_faint_limit = np.array(cur_faint_limit).max()
            est_faint_limit = np.array(est_faint_limit).max()

            self.log.write('Mean color term {:.3f}, '
                           'weighted color term {:.3f}'
                           .format(mean_cur_color_term,
                                   weighted_cur_color_term),
                           event=74, level=4, double_newline=False)
            self.log.write('Current faint limit {:.3f}, '
                           'estimated faint limit {:.3f}, '
                           'catalog limit {:.3f}'
                           .format(max_cur_faint_limit, est_faint_limit,
                                   cur_catalog_limit),
                           event=75, level=4)

            # New magnitude range for catalog query
            if est_faint_limit < max_cur_faint_limit + 2.:
                new_max_mag = est_faint_limit + 1.
            else:
                new_max_mag = ((max_cur_faint_limit
                                + min([est_faint_limit, max_catalog_mag])) / 2.
                               + 1.)
                # Do not go fainter by more than 2 mag
                new_max_mag = min(new_max_mag, cur_catalog_limit + 2.)

            # If new max mag is brighter than current catalog limit, then
            # stop iterations
            if new_max_mag <= cur_catalog_limit:
                break

            new_mag_range = [self.star_catalog.mag_range[1],
                             min(new_max_mag, max_catalog_mag)]

            iteration += 1

        num_gaia_edr3 = self.sources.num_crossmatch_gaia

        if num_calib_solutions == 0:
            self.db_update_process(num_gaia_edr3=num_gaia_edr3,
                                   num_iterations=self.num_iterations,
                                   calibrated=0)
            self.log.write('Photometric calibration failed for all solutions',
                           event=75, level=2)
        else:
            self.phot_calib_list = photproc.phot_calib_list
            self.phot_calib_curves = phot_calib_curves
            mag_range = max_cur_faint_limit - min_cur_bright_limit

            self.db_update_process(num_gaia_edr3=num_gaia_edr3,
                                   color_term=weighted_cur_color_term,
                                   bright_limit=min_cur_bright_limit,
                                   faint_limit=max_cur_faint_limit,
                                   mag_range=mag_range, num_calib=num_calib,
                                   num_iterations=self.num_iterations,
                                   calibrated=1)

    def output_calibration_db(self):
        """
        Write photometric calibration to the database.

        """

        self.log.to_db(3, 'Writing photometric calibration to the database',
                       event=78)

        if len(self.phot_calib_list) == 0:
            self.log.write('No photometric calibration to write to the database',
                           level=2, event=78)
            return

        if (self.scan_id is None or self.plate_id is None or
            self.archive_id is None or self.process_id is None):
            self.log.write('Cannot output calibration data. Missing process_id,'
                           'scan_id, plate_id or archive_id.',
                           level=2, event=78)
            return

        kwargs = {'process_id': self.process_id,
                  'scan_id': self.scan_id,
                  'plate_id': self.plate_id,
                  'archive_id': self.archive_id}

        self.log.write('Open database connection for writing to the '
                       'phot_cterm and phot_calib tables')
        platedb = PlateDB()
        platedb.assign_conf(self.conf)
        platedb.open_connection()

        for i, calib in enumerate(self.phot_calib_list):
            calib_id = platedb.write_phot_calib(calib, **kwargs)

            # For the last iteration, also write calib curves to the database
            if calib['iteration'] == self.num_iterations:
                if self.phot_calib_curves is None:
                    self.log.write('Cannot write photometric calibration '
                                   'curves to the database', level=2, event=78)
                else:
                    calib_curve = self.phot_calib_curves[i]

                    min_mag = np.floor(calib['plate_mag_brightest'] * 10) / 10.
                    max_mag = np.ceil(calib['plate_mag_lim'] * 10) / 10.
                    plate_mags = np.arange(min_mag, max_mag + 0.01, 0.1)

                    for m in plate_mags:
                        curve_element = {'calib_id': calib_id,
                                         'solution_num': calib['solution_num'],
                                         'iteration': calib['iteration'],
                                         'plate_mag': m,
                                         'natmag': calib_curve(m).item()}
                        platedb.write_calib_curve(curve_element, **kwargs)

        platedb.close_connection()
        self.log.write('Closed database connection')

    def output_sources_db(self, write_csv=None):
        """
        Write extracted sources to the database.

        """

        if write_csv is None:
            write_csv = self.write_sources_csv

        if write_csv:
            self.log.to_db(3, 'Writing sources to database files', event=80)
        else:
            self.log.to_db(3, 'Writing sources to the database', event=80)

        platedb = PlateDB()
        platedb.assign_conf(self.conf)

        if write_csv:
            # Create output directories, if missing
            if (self.write_db_source_dir and
                not os.path.isdir(self.write_db_source_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_source_dir),
                               level=4, event=80)
                os.makedirs(self.write_db_source_dir)

            if (self.write_db_source_calib_dir and
                not os.path.isdir(self.write_db_source_calib_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_source_calib_dir),
                               level=4, event=80)
                os.makedirs(self.write_db_source_calib_dir)

            if (self.write_db_source_xmatch_dir and
                not os.path.isdir(self.write_db_source_xmatch_dir)):
                self.log.write('Creating output directory {}'
                               .format(self.write_db_source_xmatch_dir),
                               level=4, event=80)
                os.makedirs(self.write_db_source_xmatch_dir)
        else:
            # Open database connection
            self.log.write('Open database connection for writing to the '
                           'source and source_calib tables.')
            platedb.open_connection()

        # Check for identification numbers and write data
        if (self.scan_id is not None and self.plate_id is not None and
            self.archive_id is not None and self.process_id is not None):
            kwargs = {'process_id': self.process_id,
                      'scan_id': self.scan_id,
                      'plate_id': self.plate_id,
                      'archive_id': self.archive_id,
                      'write_csv': write_csv}
            platedb.write_sources(self.sources, **kwargs)

            if self.sources.neighbors_gaia is not None:
                platedb.write_source_xmatches(self.sources.neighbors_gaia,
                                              **kwargs)
        else:
            self.log.write('Cannot write source data due to missing '
                           'plate identification number(s).',
                           level=2, event=80)

        # Close database connection
        if not write_csv:
            platedb.close_connection()
            self.log.write('Closed database connection.')

    def output_sources_csv(self, filename=None):
        """
        Write extracted sources to a CSV file.

        """

        self.log.to_db(3, 'Writing sources to a file', event=81)

        # Create output directory, if missing
        if self.write_source_dir and not os.path.isdir(self.write_source_dir):
            self.log.write('Creating output directory {}'
                           .format(self.write_source_dir), level=4, event=81)
            os.makedirs(self.write_source_dir)

        if filename:
            fn_world = os.path.join(self.write_source_dir,
                                    os.path.basename(filename))
        else:
            fn_world = os.path.join(self.write_source_dir,
                                    '{}_sources.csv'.format(self.basefn))

        # Output CSV file with extracted sources
        self.log.write('Writing output file {}'.format(fn_world), level=4,
                       event=81)
        self.sources.output_csv(fn_world)
