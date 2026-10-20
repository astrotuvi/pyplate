"""PlateProcess: the steps of processing one plate image, and Process: the
``process`` table row."""
import os
import shutil
import datetime as dt
import subprocess as sp
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import __version__ as astropy_version
from astropy.io import fits
from astropy.table import Table
from scipy import __version__ as scipy_version

from ..config import Config
from ..database.database import PlateDB
from ..metadata import Plate, Scan
from ..metadata.base import DbRecord
from .._version import __version__
from .catalog import StarCatalog
from .context import ProcessContext
from .log import ProcessLog
from .photometry import PhotCalib, PhotometryProcess
from .solve import SolutionSet, SolveProcess
from .sources import SourceTable

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import load_model
    have_keras = True
except ImportError:
    have_keras = False

try:
    from statsmodels import __version__ as statsmodels_version
    import statsmodels.api as sm
    have_statsmodels = True
except ImportError:
    have_statsmodels = False


@dataclass
class Process(DbRecord):
    """A plate process (``process`` table).

    The columns are described in ``applause_dr5.yaml``.
    """

    table = 'process'

    process_id: int | None = None
    scan_id: int | None = None
    plate_id: int | None = None
    archive_id: int | None = None
    filename: str | None = None
    num_exposures: int | None = None
    plate_epoch: float | None = None
    hostname: str | None = None
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    duration: int | None = None
    sky: float | None = None
    sky_sigma: float | None = None
    use_psf: bool | None = None
    threshold: float | None = None
    num_sources: int | None = None
    num_psf_sources: int | None = None
    num_true_sources: int | None = None
    num_artifacts: int | None = None
    solved: bool | None = None
    num_solutions: int | None = None
    pattern_ratio: float | None = None
    num_gaia_dr4: int | None = None
    color_term: float | None = None
    bright_limit: float | None = None
    faint_limit: float | None = None
    mag_range: float | None = None
    num_calib: int | None = None
    num_iterations: int | None = None
    calibrated: bool | None = None
    completed: bool | None = None
    pyplate_version: str | None = None


class PlateProcess:
    """Process of one plate image (DESIGN.md 6.2).

    The steps are called in the order of :meth:`PlatePipeline.single_image`:
    :meth:`setup`, :meth:`invert_plate`, :meth:`extract_sources`,
    :meth:`classify_artifacts`, :meth:`solve_plate`, :meth:`crossmatch`,
    :meth:`calibrate_photometry`, :meth:`finish`. The writers in
    :mod:`pyplate.process.output` write the results.

    Parameters
    ----------
    cfg
        The configuration.
    filename
        File name of the scan, read from ``[Files] fits_dir``.
    plate, scan
        Metadata of the plate and the scan.
    header
        Plate header made from the metadata.

    Attributes
    ----------
    ctx
        State shared by the steps.
    sources
        Extracted sources.
    solution
        Astrometric solutions.
    phot_calib
        Photometric calibrations.
    """

    def __init__(self, cfg: Config, filename: str, plate: Plate | None = None,
                 scan: Scan | None = None, header: fits.Header | None = None):
        filename = os.path.basename(filename)
        archive_id = cfg.archive.archive_id

        if plate is not None and plate.archive_id:
            archive_id = plate.archive_id

        self.ctx = ProcessContext(cfg=cfg, filename=filename,
                                  record=Process(filename=filename,
                                                 archive_id=archive_id),
                                  plate=plate, scan=scan, header=header,
                                  plate_epoch=cfg.solve.plate_epoch)
        self.sources: SourceTable | None = None
        self.solution: SolutionSet | None = None
        self.phot_calib: list[PhotCalib] | None = None
        self.timestamp = dt.datetime.now()
        self.active = False

    @property
    def fn_fits(self) -> str:
        """Path of the FITS file."""
        fn, ext = os.path.splitext(self.ctx.filename)
        fits_dir = self.ctx.cfg.files.fits_dir

        if ext == '':
            return os.path.join(fits_dir, fn + '.fits')

        return os.path.join(fits_dir, self.ctx.filename)

    @property
    def plate_solved(self) -> bool:
        """Whether an astrometric solution was found."""
        return self.solution is not None and self.solution.plate_solved

    def setup(self) -> None:
        """Open the log, write the process start to the database, read the
        image size and the plate epoch from the header, and create the
        scratch directory. If the FITS file cannot be read, the process
        finishes."""
        ctx = self.ctx
        cfg = ctx.cfg
        timestamp_str = self.timestamp.strftime('%Y%m%dT%H%M%S')

        # Open log file
        if cfg.files.write_log_dir:
            fn_log = '{}_{}.log'.format(ctx.basefn, timestamp_str)
            log_path = os.path.join(cfg.files.write_log_dir, fn_log)
            ctx.log = ProcessLog(log_path)
        else:
            ctx.log = ProcessLog(None)

        ctx.log.open()

        # Get process_id from the database
        if cfg.database.connection is not None:
            self.db_process_start()

        # Open database connection for logs
        if cfg.database.enable_db_log:
            platedb = PlateDB(cfg.database, cfg.files)
            platedb.open_connection()

            ctx.log.platedb = platedb
            ctx.log.archive_id = ctx.record.archive_id
            ctx.log.plate_id = ctx.record.plate_id
            ctx.log.scan_id = ctx.record.scan_id
            ctx.log.process_id = ctx.record.process_id
            ctx.log.to_db(3, 'Setting up plate solve process', event=10)

        ctx.log.write('Using PyPlate version {}'.format(__version__),
                      level=4, event=10)
        ctx.log.write('Using Astropy version {}'.format(astropy_version),
                      level=4, event=10)
        ctx.log.write('Using NumPy version {}'.format(np.__version__),
                      level=4, event=10)
        ctx.log.write('Using SciPy version {}'.format(scipy_version),
                      level=4, event=10)

        if have_statsmodels:
            ctx.log.write('Using statsmodels version {}'
                          .format(statsmodels_version), level=4, event=10)

        # Check if FITS file exists
        if not os.path.exists(self.fn_fits):
            ctx.log.write('FITS file does not exist: {}'.format(self.fn_fits),
                          level=1, event=10)
            self.finish(completed=0)
            return

        # Read FITS header
        if (not ctx.header or
            isinstance(ctx.header['NAXIS1'], fits.card.Undefined) or
            isinstance(ctx.header['NAXIS2'], fits.card.Undefined)):
            try:
                ctx.header = fits.getheader(self.fn_fits,
                                            ignore_missing_end=True)
            except IOError:
                ctx.log.write('Could not read FITS file {}'
                              .format(self.fn_fits),
                              level=1, event=10)
                self.finish(completed=0)
                return

        if ctx.header['NAXIS'] != 2:
            ctx.log.write('Incompatible FITS file: NAXIS != 2',
                          level=1, event=10)
            self.finish(completed=0)
            return

        ctx.imwidth = ctx.header['NAXIS1']
        ctx.imheight = ctx.header['NAXIS2']

        # Look for observation date in the FITS header.
        if ('YEAR' in ctx.header and
            isinstance(ctx.header['YEAR'], float) and
            ctx.header['YEAR']>1800):
            ctx.plate_epoch = ctx.header['YEAR']
        elif 'DATEORIG' in ctx.header:
            try:
                plate_year = int(ctx.header['DATEORIG'].split('-')[0])
                ctx.plate_epoch = float(plate_year) + 0.5
            except ValueError:
                pass
        elif 'DATE-OBS' in ctx.header:
            try:
                plate_year = int(ctx.header['DATE-OBS'].split('-')[0])
                ctx.plate_epoch = float(plate_year) + 0.5
            except ValueError:
                pass

        if ctx.plate is not None:
            num_exposures = ctx.plate.numexp
        else:
            num_exposures = None

        self.update_process(num_exposures=num_exposures,
                            plate_epoch=ctx.plate_epoch)

        # Create scratch directory
        ctx.work_dir = Path(cfg.files.work_dir,
                            '{}_{}'.format(ctx.basefn, timestamp_str))

        try:
            os.makedirs(ctx.work_dir)
        except OSError:
            if not os.path.isdir(ctx.work_dir):
                raise

        # Declare process active
        self.active = True

    def finish(self, completed: int = 1) -> None:
        """Remove the scratch directory, write the process end to the
        database and close the log."""
        ctx = self.ctx

        # Remove scratch directory and its contents
        if ctx.work_dir:
            shutil.rmtree(ctx.work_dir)

        # Write process end to the database
        self.db_process_end(completed=completed)

        # Close database connection used for logging
        if ctx.log.platedb is not None:
            ctx.log.to_db(3, 'Finish plate solve process', event=99)
            ctx.log.platedb.close_connection()

        # Close log file
        ctx.log.close()

        # Deactivate process
        self.active = False

    def update_process(self, **kwargs) -> None:
        """Set columns of the process record and write them to the
        database; None values are skipped."""
        record = self.ctx.record

        for k, v in kwargs.items():
            if v is not None:
                setattr(record, k, v)

        if record.process_id is not None:
            platedb = PlateDB(self.ctx.cfg.database, self.ctx.cfg.files)
            platedb.open_connection()
            platedb.update_process(record.process_id, **kwargs)
            platedb.close_connection()

    def db_process_start(self) -> None:
        """Write the process start to the database. The scan and plate IDs
        are looked up by file name, and the plate epoch from the exposures
        of the plate, if there are any."""
        ctx = self.ctx
        record = ctx.record
        platedb = PlateDB(ctx.cfg.database, ctx.cfg.files)
        platedb.open_connection()
        record.scan_id, record.plate_id = platedb.get_scan_id(
            ctx.filename, record.archive_id)
        record.use_psf = ctx.cfg.extraction.use_psf
        record.process_id = platedb.write_process_start(
            scan_id=record.scan_id, plate_id=record.plate_id,
            archive_id=record.archive_id, filename=ctx.filename,
            use_psf=record.use_psf)
        plate_epoch = platedb.get_plate_epoch(record.plate_id)

        if plate_epoch:
            ctx.plate_epoch = plate_epoch

        platedb.close_connection()

    def db_process_end(self, completed: int | None = None) -> None:
        """Write the process end to the database."""
        record = self.ctx.record

        if record.process_id is not None:
            record.completed = completed
            record.duration = (dt.datetime.now() - self.timestamp).seconds
            platedb = PlateDB(self.ctx.cfg.database, self.ctx.cfg.files)
            platedb.open_connection()
            platedb.write_process_end(record.process_id, completed=completed,
                                      duration=record.duration)
            platedb.close_connection()

    def invert_plate(self):
        """
        Invert FITS image and save the result (\*_inverted.fits) in the scratch 
        or work directory.

        """

        fn_inverted = '{}_inverted.fits'.format(self.ctx.basefn)

        if self.ctx.work_dir:
            fn_inverted = os.path.join(self.ctx.work_dir, fn_inverted)
        else:
            fn_inverted = os.path.join(self.ctx.cfg.files.work_dir, fn_inverted)
        
        if not os.path.exists(fn_inverted):
            self.ctx.log.write('Inverting image', level=3, event=11)

            fitsfile = fits.open(self.fn_fits, do_not_scale_image_data=True, 
                                 ignore_missing_end=True)

            invfits = fits.PrimaryHDU(-fitsfile[0].data)
            invfits.header = fitsfile[0].header.copy()
            invfits.header.set('BZERO', 32768)
            invfits.header.set('BSCALE', 1.0)
            self.ctx.log.write('Writing inverted image: {}'.format(fn_inverted), 
                               level=4, event=11)
            invfits.writeto(fn_inverted)

            fitsfile.close()
            del fitsfile
            del invfits
        else:
            self.ctx.log.write('Inverted file exists: {}'.format(fn_inverted), 
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

        self.ctx.log.write('Extracting sources from image', level=3, event=20)
        sex_ver = (sp.check_output([self.ctx.cfg.programs.sextractor_path, '--version']).strip()
                   .decode('utf-8'))
        self.ctx.log.write('Using {}'.format(sex_ver), level=4, event=20)

        if threshold_sigma is None:
            threshold_sigma = self.ctx.cfg.extraction.threshold_sigma

        if use_filter is None:
            use_filter = self.ctx.cfg.extraction.use_filter

        if filter_path is None:
            filter_path = self.ctx.cfg.extraction.filter_path

        if use_filter and filter_path:
            filter_exists = os.path.exists(filter_path)
        else:
            filter_exists = False

        if use_filter and not filter_exists:
            use_filter = False
            self.ctx.log.write('Filter file does not exist: {}'
                               ''.format(filter_path), 
                               level=2, event=20)

        if use_psf is None:
            use_psf = self.ctx.cfg.extraction.use_psf

        if psf_threshold_sigma is None:
            psf_threshold_sigma = self.ctx.cfg.extraction.psf_threshold_sigma

        if psf_model_sigma is None:
            psf_model_sigma = self.ctx.cfg.extraction.psf_model_sigma

        if circular_film is None:
            circular_film = self.ctx.cfg.extraction.circular_film

        self.ctx.log.write('Running SExtractor to get sky value', level=3, 
                           event=21)

        # Create parameter file
        fn_sex_param = self.ctx.basefn + '_sextractor.param'
        fconf = open(os.path.join(self.ctx.work_dir, fn_sex_param), 
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
        cnf += 'CATALOG_NAME     {}_sky.cat\n'.format(self.ctx.basefn)
        cnf += 'WRITE_XML        Y\n'
        cnf += 'XML_NAME         sex.xml\n'

        fn_sex_conf = self.ctx.basefn + '_sextractor.conf'
        self.ctx.log.write('Writing SExtractor configuration file {}'
                           .format(fn_sex_conf), level=4, event=21)
        self.ctx.log.write('SExtractor configuration file:\n{}'
                           .format(cnf), level=5, event=21)
        fconf = open(os.path.join(self.ctx.work_dir, fn_sex_conf), 
                     'w')
        fconf.write(cnf)
        fconf.close()

        cmd = self.ctx.cfg.programs.sextractor_path
        cmd += ' %s_inverted.fits' % self.ctx.basefn
        cmd += ' -c %s' % fn_sex_conf
        self.ctx.log.write('Subprocess: {}'.format(cmd), 
                           level=5, event=21)
        sp.call(cmd, shell=True, stdout=self.ctx.log.handle, 
                stderr=self.ctx.log.handle, cwd=self.ctx.work_dir)
        self.ctx.log.write('', timestamp=False, double_newline=False)

        tree = ET.parse(os.path.join(self.ctx.work_dir, 'sex.xml'))
        root = tree.getroot()
        use_fix_threshold = False

        if root[1][4][15][11].attrib['name'] == 'Background_Mean':
            sky = float(root[1][4][15][19][0][0][8].text)
            sky_sigma = float(root[1][4][15][19][0][0][9].text)
            self.ctx.log.write('Sky: {:.3f}, sigma: {:.3f}'.format(sky, sky_sigma), 
                       level=4, event=21)
            self.update_process(sky=sky, sky_sigma=sky_sigma)

            if sky < 2*sky_sigma or sky < 100:
                use_fix_threshold = True
                psf_model_threshold = 20000
                psf_threshold_adu = 20000
                threshold_adu = 5000
                self.ctx.log.write('Sky value too low, using fixed thresholds', 
                                   level=4, event=21)

        # If SExtractor catalog does not exist then run SExtractor
        if not os.path.exists(os.path.join(self.ctx.work_dir, 
                                           self.ctx.basefn + '.cat')):
            self.ctx.log.write('Running SExtractor without PSF model',
                               level=3, event=22)

            if use_filter:
                self.ctx.log.write('Using filter {}'.format(filter_path), 
                                   level=4, event=22)

            if use_fix_threshold:
                self.ctx.log.write('Using threshold {:f} ADU'.format(threshold_adu), 
                                   level=4, event=22)
                self.update_process(threshold=threshold_adu)
            else:
                threshold_adu = sky_sigma * threshold_sigma
                self.ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                   .format(threshold_sigma, threshold_adu),
                                   level=4, event=22)
                self.update_process(threshold=threshold_adu)

            fn_sex_param = self.ctx.basefn + '_sextractor.param'
            fconf = open(os.path.join(self.ctx.work_dir, fn_sex_param), 'w')
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
            cnf += 'CATALOG_NAME     {}.cat\n'.format(self.ctx.basefn)
            cnf += 'NTHREADS         0\n'
            #cnf += 'DETECT_TYPE      PHOTO\n'
            #cnf += 'MAG_GAMMA        1.0\n'
            #cnf += 'MEMORY_OBJSTACK  8000\n'
            #cnf += 'MEMORY_PIXSTACK  800000\n'
            #cnf += 'MEMORY_BUFSIZE   256\n'

            fn_sex_conf = self.ctx.basefn + '_sextractor.conf'
            self.ctx.log.write('Writing SExtractor configuration file {}'
                               .format(fn_sex_conf), level=4, event=22)
            self.ctx.log.write('SExtractor configuration file:\n{}'.format(cnf), 
                               level=5, event=22)
            fconf = open(os.path.join(self.ctx.work_dir, fn_sex_conf), 'w')
            fconf.write(cnf)
            fconf.close()

            cmd = self.ctx.cfg.programs.sextractor_path
            cmd += ' %s_inverted.fits' % self.ctx.basefn
            cmd += ' -c %s' % fn_sex_conf
            self.ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=22)
            sp.call(cmd, shell=True, stdout=self.ctx.log.handle, 
                    stderr=self.ctx.log.handle, cwd=self.ctx.work_dir)
            self.ctx.log.write('', timestamp=False, double_newline=False)

        if use_psf:
            enough_psf_sources = None

            # If PSFEx input file does not exist then run SExtractor
            fn_psfex_cat = os.path.join(self.ctx.work_dir, 
                                        self.ctx.basefn + '_psfex.cat')

            if not os.path.exists(fn_psfex_cat):
                self.ctx.log.write('Running SExtractor to get sources for PSFEx', 
                                   level=3, event=23)

                while True:
                    if use_filter:
                        self.ctx.log.write('Using filter {}'.format(filter_path), 
                                           level=4, event=23)

                    # Calculate threshold for PSF sources based on 
                    # extracted sources without PSF model
                    fn_cat = os.path.join(self.ctx.work_dir, 
                                          self.ctx.basefn + '.cat')
                    t = Table.read(fn_cat)
                    num_sources = len(t)
                    flux_peak = np.sort(t['FLUX_MAX'])[::-1]

                    if self.ctx.cfg.extraction.min_model_sources < num_sources:
                        th_max = flux_peak[self.ctx.cfg.extraction.min_model_sources]
                        th_max_sigma = th_max / sky_sigma
                    else:
                        th_max = None
                        th_max_sigma = None

                    if self.ctx.cfg.extraction.max_model_sources < num_sources:
                        th_min = flux_peak[self.ctx.cfg.extraction.max_model_sources]
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

                        self.ctx.log.write('Using threshold {:f} ADU'
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
                        self.ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                           .format(psf_model_sigma, threshold_adu), 
                                           level=4, event=23)

                    # Create parameter file
                    fn_sex_param = self.ctx.basefn + '_sextractor.param'
                    fconf = open(os.path.join(self.ctx.work_dir, fn_sex_param), 
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
                    cnf += 'CATALOG_NAME     {}_psfex.cat\n'.format(self.ctx.basefn)

                    fn_sex_conf = self.ctx.basefn + '_sextractor.conf'
                    self.ctx.log.write('Writing SExtractor configuration file {}'
                                       .format(fn_sex_conf), level=4, event=23)
                    self.ctx.log.write('SExtractor configuration file:\n{}'
                                       .format(cnf), level=5, event=23)
                    fconf = open(os.path.join(self.ctx.work_dir, fn_sex_conf), 
                                 'w')
                    fconf.write(cnf)
                    fconf.close()

                    cmd = self.ctx.cfg.programs.sextractor_path
                    cmd += ' %s_inverted.fits' % self.ctx.basefn
                    cmd += ' -c %s' % fn_sex_conf
                    self.ctx.log.write('Subprocess: {}'.format(cmd), 
                                       level=5, event=23)
                    sp.call(cmd, shell=True, stdout=self.ctx.log.handle, 
                            stderr=self.ctx.log.handle, cwd=self.ctx.work_dir)
                    self.ctx.log.write('', timestamp=False, double_newline=False)

                    hcat = fits.getheader(fn_psfex_cat, 2)
                    num_psf_sources = hcat['NAXIS2']
                    self.ctx.log.write('Extracted {:d} PSF-model sources'
                                       .format(num_psf_sources), level=4, event=23)
                    enough_psf_sources = False

                    if num_psf_sources >= self.ctx.cfg.extraction.min_model_sources: 
                        enough_psf_sources = True
                        break
                    else:
                        # Repeat with lower threshold to get more sources
                        if use_fix_threshold:
                            psf_model_threshold *= 0.9
                        else:
                            psf_model_sigma *= 0.9

                        self.ctx.log.write('Too few PSF-model sources (min {:d}), '
                                           'repeating extraction with lower '
                                           'threshold'
                                           .format(self.ctx.cfg.extraction.min_model_sources), 
                                           level=4, event=23)

                    #if num_psf_sources > self.ctx.cfg.extraction.max_model_sources:
                    #    # Repeat with higher threshold to get less sources
                    #    if use_fix_threshold:
                    #        psf_model_threshold *= 1.2
                    #    else:
                    #        psf_model_sigma *= 1.2

                    #    self.ctx.log.write('Too many PSF-model sources (max {:d}), '
                    #                   'repeating extraction with higher '
                    #                   'threshold'
                    #                   .format(self.ctx.cfg.extraction.max_model_sources), 
                    #                   level=4, event=23)

            # Run PSFEx
            if (enough_psf_sources and
                not os.path.exists(os.path.join(self.ctx.work_dir, 
                                                self.ctx.basefn + '_psfex.psf'))):
                self.ctx.log.write('Running PSFEx', level=3, event=24)
                psfex_ver = (sp.check_output([self.ctx.cfg.programs.psfex_path, '-v']).strip()
                             .decode('utf-8'))
                self.ctx.log.write('Using {}'.format(psfex_ver), level=4, event=24)

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
                #cnf += 'CHECKIMAGE_NAME    %s_psfex_snap.fits\n' % self.ctx.basefn
                #cnf += 'CHECKIMAGE_TYPE    NONE\n'
                cnf += 'XML_NAME           {}_psfex.xml\n'.format(self.ctx.basefn)
                cnf += 'VERBOSE_TYPE       LOG\n'
                cnf += 'NTHREADS           2\n'

                fn_psfex_conf = self.ctx.basefn + '_psfex.conf'
                self.ctx.log.write('Writing PSFEx configuration file {}'
                                   .format(fn_psfex_conf), level=4, event=24)
                self.ctx.log.write('PSFEx configuration file:\n{}'
                                   .format(cnf), level=5, event=24)
                fconf = open(os.path.join(self.ctx.work_dir, fn_psfex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.ctx.cfg.programs.psfex_path
                cmd += ' %s_psfex.cat' % self.ctx.basefn
                cmd += ' -c %s' % fn_psfex_conf
                self.ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=24)
                sp.call(cmd, shell=True, stdout=self.ctx.log.handle, 
                        stderr=self.ctx.log.handle, cwd=self.ctx.work_dir)
                self.ctx.log.write('', timestamp=False, double_newline=False)

            # Run SExtractor with PSF
            if (enough_psf_sources and
                not os.path.exists(os.path.join(self.ctx.work_dir, 
                                                self.ctx.basefn + '.cat-psf'))):
                self.ctx.log.write('Running SExtractor with PSF model',
                                   level=3, event=25)

                if use_filter:
                    self.ctx.log.write('Using filter {}'.format(filter_path), 
                                       level=4, event=25)

                if use_fix_threshold:
                    self.ctx.log.write('Using threshold {:f} ADU'
                                       .format(psf_threshold_adu), 
                                       level=4, event=25)
                else:
                    threshold_adu = sky_sigma * psf_threshold_sigma
                    self.ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                       .format(psf_threshold_sigma, threshold_adu), 
                                       level=4, event=25)

                fn_sex_param = self.ctx.basefn + '_sextractor.param'
                fconf = open(os.path.join(self.ctx.work_dir, fn_sex_param), 'w')
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
                cnf += 'CATALOG_NAME     {}.cat-psf\n'.format(self.ctx.basefn)
                cnf += 'PSF_NAME         {}_psfex.psf\n'.format(self.ctx.basefn)
                cnf += 'NTHREADS         0\n'

                fn_sex_conf = self.ctx.basefn + '_sextractor.conf'
                self.ctx.log.write('Writing SExtractor configuration file {}'
                                   .format(fn_sex_conf), level=4, event=25)
                self.ctx.log.write('SExtractor configuration file:\n{}'
                                   .format(cnf), level=5, event=25)
                fconf = open(os.path.join(self.ctx.work_dir, fn_sex_conf), 'w')
                fconf.write(cnf)
                fconf.close()

                cmd = self.ctx.cfg.programs.sextractor_path
                cmd += ' %s_inverted.fits' % self.ctx.basefn
                cmd += ' -c %s' % fn_sex_conf
                self.ctx.log.write('Subprocess: {}'.format(cmd), level=5, event=25)
                sp.call(cmd, shell=True, stdout=self.ctx.log.handle, 
                        stderr=self.ctx.log.handle, cwd=self.ctx.work_dir)
                self.ctx.log.write('', timestamp=False, double_newline=False)

        # Read the SExtractor output catalog
        self.ctx.log.write('Reading sources from the SExtractor output file', 
                           level=3, event=26)
        xycat = fits.open(os.path.join(self.ctx.work_dir, self.ctx.basefn + '.cat'))
        num_sources = len(xycat[1].data)
        self.update_process(num_sources=num_sources)
        self.ctx.log.write('Extracted {:d} sources'
                           .format(num_sources), level=4, event=26)

        # Initialise source table
        self.sources = SourceTable()
        self.sources.populate(num_sources=num_sources)
        self.sources.plate = self.ctx.plate
        self.sources.log = self.ctx.log

        # Copy values from the SExtractor catalog, xycat
        self.sources.copy_from_sextractor(xycat)

        # Calculate square root of isoarea
        self.sources['sqrt_isoarea'] = np.sqrt(self.sources['isoarea'])

        # For brevity, define following variables
        xim = self.sources['x_image']
        yim = self.sources['y_image']
        bg = self.sources['background']

        # Calculate mean background in an annulus around the centre of the plate
        self.sources['dist_center'] = np.sqrt((xim - self.ctx.imwidth / 2.)**2 +
                                              (yim - self.ctx.imheight / 2.)**2)
        min_halfwidth = min([self.ctx.imwidth, self.ctx.imheight]) / 2.
        max_halfwidth = max([self.ctx.imwidth, self.ctx.imheight]) / 2.
        bannulus = ((self.sources['dist_center'] > 0.1*min_halfwidth) &
                    (self.sources['dist_center'] < 0.9*min_halfwidth))

        if bannulus.sum() > 3:
            mean_bg = bg[np.where(bannulus)].mean()
            std_bg = bg[np.where(bannulus)].std()
        else:
            mean_bg = bg.mean()
            std_bg = bg.std()

        # Calculate distance from edge
        distarr = np.column_stack((xim, self.ctx.imwidth-xim, 
                                   yim, self.ctx.imheight-yim))
        self.sources['dist_edge'] = np.amin(distarr, 1)
        
        # Define 8 concentric annular bins + bin9 for edges
        self.ctx.log.write('Flagging sources', level=3, event=27)
        sampling = 100
        imwidth_s = int(self.ctx.imwidth / sampling)
        imheight_s = int(self.ctx.imheight / sampling)
        dist_s = np.zeros((imheight_s, imwidth_s))
        area_all = dist_s.size

        for x in np.arange(imwidth_s):
            for y in np.arange(imheight_s):
                dist_s[y,x] = np.sqrt((x*sampling - self.ctx.imwidth/2.)**2 +
                                 (y*sampling - self.ctx.imheight/2.)**2)

        # Define bin9 as 1/10 of min_halfwidth at all edges
        bin9_width_s = int(0.1 * min_halfwidth / sampling)
        dist_s[:bin9_width_s,:] = -100.
        dist_s[imheight_s-bin9_width_s:,:] = -100.
        dist_s[:,:bin9_width_s] = -100.
        dist_s[:,imwidth_s-bin9_width_s:] = -100.

        # Additionally, include in bin9 corner pixels that are closer
        # to corners than 1/6 of half-diagonal
        bin9_corner_dist = (np.sqrt(self.ctx.imwidth**2 + self.ctx.imheight**2) * 
                            5. / 12.)
        dist_s[np.where(dist_s > bin9_corner_dist)] = -100.

        # Exclude bin9 from dist_s
        dist_s = dist_s[np.where(dist_s >= 0)]
        self.sources.rel_area_sixbins = 0.75 * dist_s.size / area_all

        # Divide the rest of pixels between 8 equal-area bins 1-8
        bin_dist = np.array([np.percentile(dist_s, perc)
                             for perc in (np.arange(8)+1.)*100./8.])
        bin_dist[7] = bin9_corner_dist
        bin_dist = np.insert(bin_dist, 0, 0.)

        for b in np.arange(8)+1:
            bbin = ((self.sources['dist_center'] >= bin_dist[b-1]) &
                    (self.sources['dist_center'] < bin_dist[b]))
            nbin = bbin.sum()
            self.ctx.log.write('Annular bin {:d} (radius {:8.2f} pixels): '
                               '{:6d} sources'.format(b, bin_dist[b], nbin), 
                               double_newline=False, level=4, event=27)

            if nbin > 0:
                indbin = np.where(bbin)
                self.sources['annular_bin'][indbin] = b

        bbin = ((self.sources['dist_edge'] < 0.1*min_halfwidth) |
                (self.sources['dist_center'] >= bin9_corner_dist))
        nbin = bbin.sum()
        self.ctx.log.write('Annular bin 9 (radius {:8.2f} pixels): '
                           '{:6d} sources'.format(bin9_corner_dist, nbin), 
                           level=4, event=27)

        if nbin > 0:
            indbin = np.where(bbin)
            self.sources['annular_bin'][indbin] = 9

        indbin6 = (self.sources['annular_bin'] <= 6)
        self.sources.num_sources_sixbins = indbin6.sum()

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
            min_xedgedist = np.minimum(xim, self.ctx.imwidth-xim)
            min_yedgedist = np.minimum(yim, self.ctx.imheight-yim)
            min_reldist = np.minimum(min_xedgedist/self.ctx.imwidth,
                                     min_yedgedist/self.ctx.imheight)

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
        self.ctx.log.write('Flagged {:d} clean sources'.format(bclean.sum()),
                           double_newline=False, level=4, event=27)

        indrim = np.where(borderbg >= 100)[0]
        self.sources['flag_rim'][indrim] = 1
        self.ctx.log.write('Flagged {:d} sources at the plate rim'
                           ''.format(len(indrim)), double_newline=False, 
                           level=4, event=27)

        indnegrad = np.where(self.sources['flux_radius'] <= 0)[0]
        self.sources['flag_negradius'][indnegrad] = 1
        self.ctx.log.write('Flagged {:d} sources with negative FLUX_RADIUS'
                           ''.format(len(indnegrad)), level=4, event=27)

        # For bright stars, update coordinates with PSF coordinates
        if use_psf and enough_psf_sources:
            self.ctx.log.write('Updating coordinates with PSF coordinates '
                               'for bright sources', level=3, event=28)

            fn_psfcat = os.path.join(self.ctx.work_dir, self.ctx.basefn + '.cat-psf')

            if os.path.exists(fn_psfcat):
                try:
                    psfcat = fits.open(fn_psfcat)
                except IOError:
                    self.ctx.log.write('Could not read PSF coordinates, file {} '
                                       'is corrupt'.format(fn_psfcat), 
                                       level=2, event=28)
                    psfcat = None

                if psfcat is not None and psfcat[1].header['NAXIS2'] > 0:
                    xpeakpsf = psfcat[1].data.field('XPEAK_IMAGE')
                    ypeakpsf = psfcat[1].data.field('YPEAK_IMAGE')

                    # Match sources in two lists (distance < 1 px)
                    coords1 = np.empty((num_sources, 2))
                    coords1[:,0] = self.sources['x_peak']
                    coords1[:,1] = self.sources['y_peak']
                    coords2 = np.empty((xpeakpsf.size, 2))
                    coords2[:,0] = xpeakpsf
                    coords2[:,1] = ypeakpsf
                    kdt = KDT(coords2)
                    ds,ind2 = kdt.query(coords1)
                    ind1 = np.arange(num_sources)
                    indmask = ds < 1.
                    ind1 = ind1[indmask]
                    ind2 = ind2[indmask]

                    num_psf_sources = len(ind1)
                    self.ctx.log.write('Replacing x,y values from PSF photometry '
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
                    self.update_process(num_psf_sources=num_psf_sources)
                elif psfcat is not None and psfcat[1].header['NAXIS2'] == 0:
                    self.ctx.log.write('There are no sources with PSF coordinates!',
                                       level=2, event=28)
                    self.update_process(num_psf_sources=0)
            else:
                self.ctx.log.write('Could not read PSF coordinates, '
                                   'file {} does not exist!'.format(fn_psfcat), 
                                   level=2, event=28)
                self.update_process(num_psf_sources=0)

    def classify_artifacts(self):
        """
        Classify extracted sources as celestial or artifacts.

        Algorithm and model by Gal Matijevic

        """

        self.ctx.log.write('Classifying artifacts', level=3, event=29)

        if not have_keras:
            self.ctx.log.write('Missing dependency (keras) for artifact '
                               'classification!', level=2, event=29)
            return

        # Read model
        fn_model = os.path.join(os.path.dirname(__file__), 'artifact_model.keras')
        model = load_model(fn_model)

        # Read inverted FITS file
        fn_image = os.path.join(self.ctx.work_dir,
                                self.ctx.basefn + '_inverted.fits')
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
        self.update_process(num_artifacts=num_artifacts,
                            num_true_sources=num_true_sources)
        self.ctx.log.write('Classified {:d} sources as true sources '
                           '(prediction > 0.9)'
                           .format(num_true_sources), level=4, event=29,
                           double_newline=False)
        self.ctx.log.write('Classified {:d} sources as artifacts (prediction < 0.1)'
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

        self.ctx.log.write('Solving astrometry', level=3, event=30)

        # Do plate solving
        solveproc = SolveProcess(self.ctx, self.sources)
        solution = solveproc.solve_plate(plate_epoch=plate_epoch, sip=sip,
                                         skip_bright=skip_bright,
                                         repeat_find=repeat_find)

        # Create WCS header
        if solution.plate_solved:
            solution.create_wcs_header(self.ctx.log)

        self.solution = solution

        # Apply scanner pattern to source coordinates
        if self.plate_solved:
            if solution.pattern_ratio is not None:
                self.sources.apply_scanner_pattern(solution)

            self.update_process(solved=1,
                                num_solutions=solution.num_solutions,
                                pattern_ratio=solution.pattern_ratio)
        else:
            self.update_process(solved=0, num_solutions=0)

    def query_star_catalog(self, star_catalog=None, mag_range=[0,15],
                           color_term=None):
        """
        Query external star catalog for astrometric and photometric reference
        stars.

        Parameters:
        -----------
        star_catalog : StarCatalog
            Catalog to add the stars to; a new one if None.
        mag_range : list
            A two-element list specifying bright and faint magnitude limits
            for external catalog query.

        Returns
        -------
        StarCatalog or None
            The catalog, or None without astrometric solutions.

        """

        self.ctx.log.write('Getting reference catalogs', level=3, event=40)

        if not self.plate_solved:
            self.ctx.log.write('Cannot query external catalog due to missing '
                               'astrometric solutions!', level=2, event=40)
            return None

        # Initialise star_catalog
        if star_catalog is None:
            star_catalog = StarCatalog()
            star_catalog.assign_conf(self.ctx.cfg)
            star_catalog.gaia_dir = self.ctx.cfg.files.gaia_dir
            star_catalog.scratch_dir = self.ctx.work_dir
            star_catalog.log = self.ctx.log

        # Query Gaia catalog
        star_catalog.query_gaia(self.solution, mag_range=mag_range,
                                color_term=color_term)
        return star_catalog

    def crossmatch(self):
        """
        Crossmatch the sources with Gaia stars brighter than G = 12.

        """

        star_catalog = self.query_star_catalog(mag_range=[0,12])
        self.sources.crossmatch_gaia(self.solution, star_catalog)

    def calibrate_photometry(self):
        """
        Calibrate extracted magnitudes.

        """

        self.ctx.log.write('Calibrating photometry', event=70, level=3)

        # Initialise photometry process
        photproc = PhotometryProcess(self.ctx, self.sources, self.solution)

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
        num_iterations = 0

        # Variables to store calibration results
        phot_calib_curves = []
        prelim_color_term = []
        prelim_cterm_nstars = []
        cur_faint_limit = []
        est_faint_limit = []

        # Carry out photometric calibration for all solutions
        for i in np.arange(1, self.solution.num_solutions+1):
            photproc.calibrate_photometry_gaia(solution_num=i, iteration=1)
            num_iterations = 1

            if not photproc.phot_calibrated:
                continue

            # Retrieve calibration curve
            phot_calib_curves.append(photproc.calib_curve)

            # Second iteration
            photproc.calibrate_photometry_gaia(solution_num=i, iteration=2)
            num_iterations = 2

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

        # If photometric calibration failed for all solutions, then give up.
        # The calibrations of the finished iterations are still written to
        # the database, without curves.
        if len(phot_calib_curves) == 0 or len(cur_faint_limit) == 0:
            self.phot_calib = [PhotCalib.from_row(c)
                               for c in photproc.phot_calib_list]
            num_gaia_dr4 = self.sources.num_crossmatch_gaia
            self.update_process(num_gaia_dr4=num_gaia_dr4,
                                num_iterations=num_iterations,
                                calibrated=0)
            self.ctx.log.write('Photometric calibration failed for all '
                               'solutions', event=74, level=2)
            return

        # Calculate mean color term and max faint limit over all solutions
        mean_color_term = np.array(prelim_color_term).mean()
        weighted_color_term = ((np.array(prelim_color_term) *
                                np.array(prelim_cterm_nstars)).sum() /
                               np.array(prelim_cterm_nstars).sum())
        max_cur_faint_limit = np.array(cur_faint_limit).max()
        est_faint_limit = np.array(est_faint_limit).max()

        self.ctx.log.write('Mean color term {:.3f}, '
                           'weighted color term {:.3f}'
                           .format(mean_color_term, weighted_color_term),
                           event=74, level=4, double_newline=False)
        self.ctx.log.write('Current faint limit {:.3f}, '
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

        # Start from a fresh set of stars
        star_catalog = None
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
            star_catalog = self.query_star_catalog(
                star_catalog, mag_range=new_mag_range,
                color_term=weighted_color_term)
            cur_catalog_limit = new_mag_range[1]
            self.sources.crossmatch_gaia(self.solution, star_catalog)

            num_calib_solutions = 0
            cur_calib_stars = []
            cur_color_term = []
            cur_cterm_nstars = []
            cur_bright_limit = []
            cur_faint_limit = []
            est_faint_limit = []

            for i in np.arange(1, self.solution.num_solutions+1):
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
            num_iterations = iteration

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

            self.ctx.log.write('Mean color term {:.3f}, '
                               'weighted color term {:.3f}'
                               .format(mean_cur_color_term,
                                       weighted_cur_color_term),
                               event=74, level=4, double_newline=False)
            self.ctx.log.write('Current faint limit {:.3f}, '
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

            new_mag_range = [star_catalog.mag_range[1],
                             min(new_max_mag, max_catalog_mag)]

            iteration += 1

        num_gaia_dr4 = self.sources.num_crossmatch_gaia

        if num_calib_solutions == 0:
            # As above, the calibrations are written without curves
            self.phot_calib = [PhotCalib.from_row(c)
                               for c in photproc.phot_calib_list]
            self.update_process(num_gaia_dr4=num_gaia_dr4,
                                num_iterations=num_iterations,
                                calibrated=0)
            self.ctx.log.write('Photometric calibration failed for all '
                               'solutions', event=75, level=2)
        else:
            self.phot_calib = []

            for c, curve in zip(photproc.phot_calib_list, phot_calib_curves):
                calib = PhotCalib.from_row(c)
                calib.curve = curve
                self.phot_calib.append(calib)

            mag_range = max_cur_faint_limit - min_cur_bright_limit

            self.update_process(num_gaia_dr4=num_gaia_dr4,
                                color_term=weighted_cur_color_term,
                                bright_limit=min_cur_bright_limit,
                                faint_limit=max_cur_faint_limit,
                                mag_range=mag_range, num_calib=num_calib,
                                num_iterations=num_iterations,
                                calibrated=1)
