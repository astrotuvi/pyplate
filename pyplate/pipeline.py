import os
import multiprocessing as mp
import time
import ConfigParser
from .metadata import ArchiveMeta, PlateMeta, PlateHeader, read_conf
from .solve import SolveProcess
from .database import PlateDB
from .image import PlateConverter


class PlateImagePipeline:
    """
    Plate processing pipeline class

    """

    def __init__(self, plate_converter=None):
        self.conf = None
        self.work_dir = ''
        self.write_log_dir = ''
        self.input_queue = None
        self.done_queue = None
        self.renew_worker_queue = None
        self.plate_converter = plate_converter
        self.plate_epoch = None
        self.processes = 1
        self.process_max_tasks = 0
        self.wait_start = 1.0

        self.read_wfpdb = False
        self.read_csv = False
        self.read_fits = False
        self.output_header_file = False
        self.output_header_fits = False
        self.invert_image = False
        self.extract_sources = False
        self.solve_plate = False
        self.output_solution_db = False
        self.output_wcs_file = False
        self.get_reference_catalogs = False
        self.solve_recursive = False
        self.calibrate_photometry = False
        self.improve_photometry = False
        self.output_calibration_db = False
        self.output_sources_db = False
        self.output_sources_csv = False

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['work_dir', 'write_log_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['read_wfpdb', 'read_csv', 'read_fits', 
                     'output_header_file', 'output_header_fits', 
                     'invert_image', 'extract_sources', 'solve_plate', 
                     'output_solution_db', 'output_wcs_file',
                     'get_reference_catalogs',
                     'solve_recursive', 'calibrate_photometry', 
                     'improve_photometry',
                     'output_calibration_db', 
                     'output_sources_db', 'output_sources_csv']:
            try:
                setattr(self, attr, conf.getboolean('Pipeline', attr))
            except ValueError:
                print ('Error in configuration file: not a boolean value '
                       '([{}], {})'.format('Pipeline', attr))
            except ConfigParser.Error:
                pass

        for attr in ['processes', 'process_max_tasks']:
            try:
                setattr(self, attr, conf.getint('Pipeline', attr))
            except ValueError:
                print ('Error in configuration file: not an integer value '
                       '([{}], {})'.format('Pipeline', attr))
            except ConfigParser.Error:
                pass

        for attr in ['wait_start']:
            try:
                setattr(self, attr, conf.getfloat('Pipeline', attr))
            except ValueError:
                print ('Error in configuration file: not a float value '
                       '([{}], {})'.format('Pipeline', attr))
            except ConfigParser.Error:
                pass

    def single_image(self, filename, plate_epoch=None):
        """
        Process single plate image.

        Parameters
        ----------
        filename : str
            Filename of the FITS image to be processed
        plate_epoch : float
            Plate epoch (decimal year)
            
        """

        ameta = ArchiveMeta()
        ameta.assign_conf(self.conf)

        if self.read_wfpdb:
            ameta.read_wfpdb()

        if self.read_csv:
            ameta.read_csv()

        fn = os.path.basename(filename)
        pmeta = ameta.get_platemeta(filename=fn)
        pmeta.compute_values()

        h = PlateHeader()
        h.assign_conf(pmeta.conf)
        h.assign_platemeta(pmeta)
        h.update_from_platemeta()
        h.assign_values()
        h.update_comments()
        h.rewrite()
        h.reorder()

        if self.output_header_file:
            fn_header = os.path.splitext(fn)[0] + '.hdr'
            h.output_to_file(fn_header)

        proc = SolveProcess(fn)
        proc.assign_conf(pmeta.conf)
        proc.assign_header(h)
        proc.assign_metadata(pmeta)

        if self.plate_epoch is not None:
            proc.plate_epoch = self.plate_epoch

        if plate_epoch is not None:
            proc.plate_epoch = plate_epoch

        proc.setup()

        if self.invert_image:
            proc.invert_plate()

        if self.extract_sources:
            proc.extract_sources()

            if self.solve_plate:
                proc.solve_plate()

            if self.output_solution_db:
                proc.output_solution_db()

            if self.output_wcs_file:
                proc.output_wcs_header()

            if proc.solution is not None:
                proc.log.write('Updating FITS header with the WCS', 
                               level=3, event=37)
                h.insert_wcs(proc.solution['wcs'])

            if self.output_header_file:
                proc.log.write('Writing FITS header to a file', 
                               level=3, event=38)
                h.output_to_file(fn_header)

            if self.output_header_fits:
                proc.log.write('Writing FITS header to the FITS file', 
                               level=3, event=39)
                h.output_to_fits(fn)

                # Get metadata for the updated FITS file
                pmeta['fits_datetime'] = h.fits_datetime
                pmeta['fits_size'] = h.fits_size
                pmeta['fits_checksum'] = h.fits_checksum
                pmeta['fits_datasum'] = h.fits_datasum

                # Updating scan metadata in the scan table
                platedb = PlateDB()
                platedb.assign_conf(self.conf)
                platedb.open_connection()
                platedb.update_scan(pmeta, filecols=True)
                platedb.close_connection()
        
            if self.get_reference_catalogs:
                proc.get_reference_catalogs()

            if self.solve_recursive:
                proc.solve_recursive()

                if self.output_solution_db:
                    proc.output_astrom_sub_db()

            proc.process_source_coordinates()

            if self.calibrate_photometry:
                proc.calibrate_photometry()

            if self.improve_photometry:
                proc.improve_photometry_recursive()

            if self.output_calibration_db:
                proc.output_cterm_db()
                proc.output_color_db()
                proc.output_calibration_db()

            if self.output_sources_db:
                proc.output_sources_db()

            if self.output_sources_csv:
                proc.output_sources_csv()

        proc.finish()

    def worker(self):
        """
        Take a filename from the queue and process the file.

        """

        if self.input_queue is None:
            return

        task_count = 0

        while True:
            if os.path.exists(os.path.join(self.work_dir, 'pyplate.stop')):
                break
            
            fn = self.input_queue.get()

            if fn == 'DONE':
                break

            if self.plate_converter:
                plateconv = PlateConverter()
                plateconv.assign_conf(self.conf)
                plateconv.tiff2fits(fn)
            else:
                self.single_image(fn)

            self.done_queue.put(fn)
            task_count += 1

            if (self.process_max_tasks > 0 and 
                task_count >= self.process_max_tasks):
                self.renew_worker_queue.put(True)
                break

    def parallel_run(self, filenames, processes=None, process_max_tasks=None,
                     wait_start=None):
        """
        Run plate image processes in parallel.

        Parameters
        ----------
        filenames : list
            List of filenames to process
        processes : int
            Number of parallel processes
        process_max_tasks : int
            Number of images processed after which the worker process is renewed
        wait_start : float
            Number of seconds to wait before starting another worker process 
            at the beginning

        """

        if processes is None:
            processes = self.processes

        if not isinstance(processes, int) or processes < 1:
            processes = 1

        if wait_start is None:
            wait_start = self.wait_start

        if not isinstance(wait_start, float):
            try:
                wait_start = float(wait_start)
            except ValueError:
                wait_start = 1.0

        if wait_start < 0:
            wait_start = 1.0

        if process_max_tasks is not None:
            try:
                self.process_max_tasks = int(process_max_tasks)
            except ValueError:
                pass

        self.input_queue = mp.Queue()
        self.done_queue = mp.Queue()
        self.renew_worker_queue = mp.Queue()
        jobs = []
        queue_list = []

        try:
            with open(os.path.join(self.work_dir, 'pyplate.queue'), 
                      'rb') as f:
                queue_list = [fn.strip() for fn in f.readlines()]
        except IOError:
            pass

        if not queue_list:
            queue_list = filenames

        for fn in queue_list:
            self.input_queue.put(fn)

        for i in range(processes):
            self.input_queue.put('DONE')
            job = mp.Process(target=self.worker)
            job.start()
            jobs.append(job)
            # Wait before starting another process
            time.sleep(wait_start)

        # Write unfinished and finished file lists to disk every 10 seconds
        while True:
            time.sleep(10)
            self.done_queue.put('STOP')
            done_list = [fn for fn in iter(self.done_queue.get, 'STOP')]
            queue_list = [fn for fn in queue_list if fn not in done_list]

            try:
                with open(os.path.join(self.work_dir, 'pyplate.done'), 
                          'ab') as f:
                    for fn in done_list:
                        f.write('{}\n'.format(fn))
            except IOError:
                pass

            try:
                with open(os.path.join(self.work_dir, 'pyplate.queue'), 
                          'wb') as f:
                    for fn in queue_list:
                        f.write('{}\n'.format(fn))
            except IOError:
                pass

            if os.path.exists(os.path.join(self.work_dir, 'pyplate.stop')):
                jobs_finished = True

                for job in jobs:
                    if job.is_alive():
                        jobs_finished = False

                if jobs_finished:
                    break

            if queue_list == []:
                break

            if not self.renew_worker_queue.empty():
                # Clean job list
                for i,job in enumerate(jobs):
                    if not job.is_alive():
                        del jobs[i]

                # Start new worker process
                self.renew_worker_queue.get()
                job = mp.Process(target=self.worker)
                job.start()
                jobs.append(job)

        for job in jobs:
            job.join()

        self.done_queue.put('STOP')
        done_list = [fn for fn in iter(self.done_queue.get, 'STOP')]
        queue_list = [fn for fn in queue_list if fn not in done_list]

        try:
            with open(os.path.join(self.work_dir, 'pyplate.done'), 
                      'ab') as f:
                for fn in done_list:
                    f.write('{}\n'.format(fn))
        except IOError:
            pass

        try:
            with open(os.path.join(self.work_dir, 'pyplate.queue'), 
                      'wb') as f:
                for fn in queue_list:
                    f.write('{}\n'.format(fn))
        except IOError:
            pass

        # Empty the input queue
        while not self.input_queue.empty():
            self.input_queue.get()

