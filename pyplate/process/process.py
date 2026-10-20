"""PlateProcess: the steps of processing one plate image, and Process: the
``process`` table row."""
import os
import shutil
import datetime as dt
import subprocess as sp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import __version__ as astropy_version
from astropy.io import fits
from scipy import __version__ as scipy_version

from ..config import Config
from ..database.database import PlateDB
from ..metadata import Plate, Scan
from ..metadata.base import DbRecord
from .._version import __version__
from . import artifacts, extraction
from .catalog import StarCatalog
from .context import ProcessContext
from .crossmatch import (Association, NearestMatcher, ReferenceStars,
                         find_pairs, tolerance)
from .log import ProcessLog
from .photometry import PhotCalib, PhotometryProcess
from .solve import SolutionSet, SolveProcess
from .sources import SourceTable

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
    association
        Crossmatch of the sources with Gaia stars (the last one, made with
        the deepest catalogue).
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
        self.association: Association | None = None
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

    def extract_sources(self, threshold_sigma: float | None = None,
                        use_filter: bool | None = None,
                        filter_path: str | None = None,
                        use_psf: bool | None = None,
                        psf_threshold_sigma: float | None = None,
                        psf_model_sigma: float | None = None,
                        circular_film: bool | None = None) -> None:
        """
        Extract sources from a FITS file.

        SExtractor first measures the sky, then extracts the sources;
        with use_psf, PSFEx builds a PSF model from bright sources and the
        PSF coordinates replace the coordinates of the bright sources. The
        sources are put into annular bins and flagged. Existing catalogues
        in the scratch directory are reused.

        The arguments default to the values in ``[Extraction]``.

        Parameters
        ----------
        threshold_sigma
            SExtractor threshold in sigmas
        use_filter
            Use SExtractor filter for source detection
        filter_path
            Path to SExtractor filter file
        use_psf
            Use PSF for bright stars
        psf_threshold_sigma
            SExtractor threshold in sigmas for using PSF
        psf_model_sigma
            SExtractor threshold in sigmas for PSF model stars
        circular_film
            Assume circular film
        """

        ctx = self.ctx
        settings = ctx.cfg.extraction
        ctx.log.write('Extracting sources from image', level=3, event=20)
        sex_ver = (sp.check_output([ctx.cfg.programs.sextractor_path, '--version']).strip()
                   .decode('utf-8'))
        ctx.log.write('Using {}'.format(sex_ver), level=4, event=20)

        if threshold_sigma is None:
            threshold_sigma = settings.threshold_sigma

        if use_filter is None:
            use_filter = settings.use_filter

        if filter_path is None:
            filter_path = settings.filter_path

        if use_filter and filter_path:
            filter_exists = os.path.exists(filter_path)
        else:
            filter_exists = False

        if use_filter and not filter_exists:
            use_filter = False
            ctx.log.write('Filter file does not exist: {}'
                          ''.format(filter_path),
                          level=2, event=20)

        if not use_filter:
            filter_path = None

        if use_psf is None:
            use_psf = settings.use_psf

        if psf_threshold_sigma is None:
            psf_threshold_sigma = settings.psf_threshold_sigma

        if psf_model_sigma is None:
            psf_model_sigma = settings.psf_model_sigma

        if circular_film is None:
            circular_film = settings.circular_film

        ctx.log.write('Running SExtractor to get sky value', level=3,
                      event=21)
        sky_values = extraction.measure_sky(ctx)
        use_fix_threshold = False
        sky, sky_sigma = None, None

        if sky_values is not None:
            sky, sky_sigma = sky_values
            ctx.log.write('Sky: {:.3f}, sigma: {:.3f}'.format(sky, sky_sigma),
                          level=4, event=21)
            self.update_process(sky=sky, sky_sigma=sky_sigma)

            if sky < 2*sky_sigma or sky < 100:
                use_fix_threshold = True
                psf_model_threshold = 20000
                psf_threshold_adu = 20000
                threshold_adu = 5000
                ctx.log.write('Sky value too low, using fixed thresholds',
                              level=4, event=21)

        # If SExtractor catalog does not exist then run SExtractor
        if not os.path.exists(os.path.join(ctx.work_dir,
                                           ctx.basefn + '.cat')):
            ctx.log.write('Running SExtractor without PSF model',
                          level=3, event=22)

            if filter_path:
                ctx.log.write('Using filter {}'.format(filter_path),
                              level=4, event=22)

            if use_fix_threshold:
                ctx.log.write('Using threshold {:f} ADU'.format(threshold_adu),
                              level=4, event=22)
                self.update_process(threshold=threshold_adu)
                extraction.extract_catalog(ctx, threshold_adu, True,
                                           filter_path)
            else:
                threshold_adu = sky_sigma * threshold_sigma
                ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                              .format(threshold_sigma, threshold_adu),
                              level=4, event=22)
                self.update_process(threshold=threshold_adu)
                extraction.extract_catalog(ctx, threshold_sigma, False,
                                           filter_path)

        enough_psf_sources = None

        if use_psf:
            # If PSFEx input file does not exist then run SExtractor
            fn_psfex_cat = os.path.join(ctx.work_dir,
                                        ctx.basefn + '_psfex.cat')

            if not os.path.exists(fn_psfex_cat):
                ctx.log.write('Running SExtractor to get sources for PSFEx',
                              level=3, event=23)

                if use_fix_threshold:
                    enough_psf_sources = self._extract_psf_model_sources(
                        psf_model_threshold, True, sky_sigma, filter_path)
                else:
                    enough_psf_sources = self._extract_psf_model_sources(
                        psf_model_sigma, False, sky_sigma, filter_path)

            # Run PSFEx
            if (enough_psf_sources and
                not os.path.exists(os.path.join(ctx.work_dir,
                                                ctx.basefn + '_psfex.psf'))):
                extraction.run_psfex(ctx)

            # Run SExtractor with PSF
            if (enough_psf_sources and
                not os.path.exists(os.path.join(ctx.work_dir,
                                                ctx.basefn + '.cat-psf'))):
                ctx.log.write('Running SExtractor with PSF model',
                              level=3, event=25)

                if filter_path:
                    ctx.log.write('Using filter {}'.format(filter_path),
                                  level=4, event=25)

                if use_fix_threshold:
                    ctx.log.write('Using threshold {:f} ADU'
                                  .format(psf_threshold_adu),
                                  level=4, event=25)
                    extraction.extract_psf_catalog(ctx, psf_threshold_adu,
                                                   True, filter_path)
                else:
                    threshold_adu = sky_sigma * psf_threshold_sigma
                    ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                                  .format(psf_threshold_sigma, threshold_adu),
                                  level=4, event=25)
                    extraction.extract_psf_catalog(ctx, psf_threshold_sigma,
                                                   False, filter_path)

        # Read the SExtractor output catalog
        ctx.log.write('Reading sources from the SExtractor output file',
                      level=3, event=26)
        self.sources = extraction.build_source_table(ctx)
        num_sources = len(self.sources)
        self.update_process(num_sources=num_sources)
        ctx.log.write('Extracted {:d} sources'
                      .format(num_sources), level=4, event=26)

        ctx.log.write('Flagging sources', level=3, event=27)
        self.sources.compute_annular_bins(ctx.imwidth, ctx.imheight, ctx.log)
        self.sources.flag_edge_sources(ctx.imwidth, ctx.imheight,
                                       circular_film, ctx.log)

        # For bright stars, update coordinates with PSF coordinates
        if use_psf and enough_psf_sources:
            ctx.log.write('Updating coordinates with PSF coordinates '
                          'for bright sources', level=3, event=28)
            fn_psfcat = os.path.join(ctx.work_dir, ctx.basefn + '.cat-psf')
            num_psf_sources = self.sources.apply_psf_coordinates(fn_psfcat,
                                                                 ctx.log)

            if num_psf_sources is not None:
                self.update_process(num_psf_sources=num_psf_sources)

    def _extract_psf_model_sources(self, threshold, absolute, sky_sigma,
                                   filter_path):
        """Extract the sources for the PSF model, lowering the threshold
        (in ADU if absolute, otherwise in sigmas) until there are at least
        ``[Extraction] min_model_sources``. The threshold is first kept
        between the peak fluxes of the ``max_model_sources``-th and
        ``min_model_sources``-th brightest sources. Returns True when there
        are enough sources."""
        ctx = self.ctx
        min_model_sources = ctx.cfg.extraction.min_model_sources

        while True:
            if filter_path:
                ctx.log.write('Using filter {}'.format(filter_path),
                              level=4, event=23)

            # Calculate threshold for PSF sources based on
            # extracted sources without PSF model
            th_min, th_max = extraction.psf_model_thresholds(ctx)

            if not absolute:
                th_min = th_min / sky_sigma if th_min is not None else None
                th_max = th_max / sky_sigma if th_max is not None else None

            if (th_min is not None and th_min > 0 and
                threshold < th_min):
                threshold = th_min
            elif (th_max is not None and th_max > 0 and
                  threshold > th_max):
                threshold = th_max

            if absolute:
                ctx.log.write('Using threshold {:f} ADU'
                              .format(threshold),
                              level=4, event=23)
            else:
                threshold_adu = sky_sigma * threshold
                ctx.log.write('Using threshold {:.1f} sigma ({:.2f} ADU)'
                              .format(threshold, threshold_adu),
                              level=4, event=23)

            num_psf_sources = extraction.extract_psf_model_sources(
                ctx, threshold, absolute, filter_path)
            ctx.log.write('Extracted {:d} PSF-model sources'
                          .format(num_psf_sources), level=4, event=23)

            if num_psf_sources >= min_model_sources:
                return True

            # Repeat with lower threshold to get more sources
            threshold *= 0.9
            ctx.log.write('Too few PSF-model sources (min {:d}), '
                          'repeating extraction with lower '
                          'threshold'
                          .format(min_model_sources),
                          level=4, event=23)

    def classify_artifacts(self) -> None:
        """
        Classify extracted sources as celestial or artifacts
        (``model_prediction``, see :mod:`pyplate.process.artifacts`).

        """

        self.ctx.log.write('Classifying artifacts', level=3, event=29)

        if not artifacts.have_keras:
            self.ctx.log.write('Missing dependency (keras) for artifact '
                               'classification!', level=2, event=29)
            return

        prediction = artifacts.predict(self.ctx, self.sources)
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

    def crossmatch(self) -> None:
        """
        Crossmatch the sources with Gaia stars brighter than G = 12.

        """

        star_catalog = self.query_star_catalog(mag_range=[0,12])
        self.match_catalog(star_catalog)

    def match_catalog(self, star_catalog: StarCatalog | None) -> None:
        """
        Crossmatch the sources with the stars of a catalogue, moved to the
        plate epoch (DESIGN.md 6.5).

        The matches and the catalogue values of the matched stars go into
        the source table and the number of matches of each solution into
        the solution; the source coordinates are then processed again,
        because the matches may change the solution of a source. The
        result is :attr:`association`.

        """

        log = self.ctx.log
        solution_set = self.solution
        log.write('Crossmatching sources with Gaia objects',
                  level=3, event=44)

        if solution_set is None or solution_set.num_solutions == 0:
            log.write('Cannot crossmatch sources with Gaia objects '
                      'due to missing astrometric solutions!',
                      level=2, event=44)
            return

        if star_catalog is None:
            log.write('Cannot crossmatch sources with Gaia objects '
                      'due to missing Gaia catalog data!',
                      level=2, event=44)
            return

        log.write('Number of Gaia stars: {:d}'.format(len(star_catalog)),
                  level=4, event=44, double_newline=False)
        refs = ReferenceStars.from_catalog(star_catalog, solution_set,
                                           self.ctx.plate_epoch, log)
        radii = tolerance.radii(solution_set)
        log.write('Crossmatch tolerance: {:.2f} arcsec ({:.2f} pixels)'
                  .format(radii.match_arcsec, radii.match), level=4,
                  event=44, double_newline=False)
        log.write('Finding all reference stars around sources within '
                  'the radius of {:.2f} arcsec ({:.2f} pixels)'
                  .format(radii.search_arcsec, radii.search),
                  level=4, event=44)
        pairs = find_pairs(self.sources, refs, radii.search)
        association = NearestMatcher(radii).associate(self.sources, refs,
                                                      pairs)
        self.sources.add_matches(association, refs)

        # Store number of crossmatched sources to each solution. They are
        # counted in the source table, which keeps the matches of earlier
        # calls (DESIGN.md 9).
        for solution in solution_set.solutions:
            m = self.sources['solution_num'] == solution.solution_num
            solution.num_gaia_dr4 = int(m.sum())

        self.association = association

        # Process coordinates again, because solution_num assignments may
        # have changed
        self.sources.process_coordinates(solution_set=solution_set)

    @property
    def num_gaia_matches(self) -> int | None:
        """Number of sources matched with Gaia stars."""
        if self.association is None:
            return None

        return len(self.association.matches)

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
            num_gaia_dr4 = self.num_gaia_matches
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
            self.match_catalog(star_catalog)

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

        num_gaia_dr4 = self.num_gaia_matches

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
