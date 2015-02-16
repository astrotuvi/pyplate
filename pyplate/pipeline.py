import os
from metadata import ArchiveMeta, PlateMeta, PlateHeader, read_conf
from solve import SolveProcess
from database import PlateDB
import multiprocessing as mp
import time


class PlateImagePipeline():
    """
    Plate processing pipeline class

    """

    def __init__(self):
        self.conf = None
        self.work_dir = ''
        self.write_log_dir = ''
        self.input_queue = None
        self.done_queue = None

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
        self.solve_recursive = False
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
                     'solve_recursive', 
                     'output_sources_db', 'output_sources_csv']:
            try:
                setattr(self, attr, conf.getboolean('Pipeline', attr))
            except ValueError:
                print ('Error in configuration file: not a boolean value '
                       '([{}], {})'.format('Pipeline', attr))
            except ConfigParser.Error:
                pass

    def single_image(self, filename):
        """
        Process single plate image.

        Parameters
        ----------
        filename : str
            Filename of the FITS image to be processed
            
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

        #platedb = PlateDB()
        #platedb.assign_conf(self.conf)
        #platedb.open_connection()
        #platedb.write_plate(pmeta)
        #platedb.write_scan(pmeta)
        #platedb.close_connection()
        
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
                               level=3, event=47)
                h.insert_wcs(proc.solution['wcs'])

            if self.output_header_file:
                proc.log.write('Writing FITS header to a file', 
                               level=3, event=48)
                h.output_to_file(fn_header)

            if self.output_header_fits:
                proc.log.write('Writing FITS header to the FITS file', 
                               level=3, event=49)
                h.output_to_fits(fn)

            if self.solve_recursive:
                proc.solve_recursive()

            proc.process_source_coordinates()

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

        while True:
            if os.path.exists(os.path.join(self.work_dir, 'pyplate.stop')):
                break
            
            fn = self.input_queue.get()

            if fn == 'DONE':
                break

            self.single_image(fn)
            self.done_queue.put(fn)

    def parallel_run(self, filenames, processes=1):
        """
        Run plate image processes in parallel.

        Parameters
        ----------
        filenames : list
            List of filenames to process
        processes : int
            Number of parallel processes

        """

        self.input_queue = mp.Queue()
        self.done_queue = mp.Queue()
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
                break

            if queue_list == []:
                break

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

def run_pipeline(filenames, fn_conf):
    """
    Run metadata processing and plate solving pipeline.

    Parameters
    ----------
    filenames : list
        List of FITS files to be processed
    conf_file : str
        Full path to the configuration file
        
    """

    ameta = ArchiveMeta()
    ameta.assign_conf(fn_conf)
    ameta.read_wfpdb()
    ameta.read_csv()

    for fn in filenames:
        fn = os.path.basename(fn)
        pmeta = ameta.get_platemeta(filename=fn)
        pmeta['archive_id'] = 0
        pmeta.compute_values()

        platedb = PlateDB()
        platedb.open_connection(host=pmeta.output_db_host,
                                user=pmeta.output_db_user,
                                dbname=pmeta.output_db_name,
                                passwd=pmeta.output_db_passwd)
        platedb.write_plate(pmeta)
        platedb.write_scan(pmeta)
        platedb.close_connection()
        
        h = PlateHeader()
        h.assign_conf(pmeta.conf)
        h.assign_platemeta(pmeta)
        h.update_from_platemeta()
        h.assign_values()
        h.update_comments()
        h.rewrite()
        h.reorder()
        fn_header = os.path.splitext(fn)[0] + '.hdr'
        h.output_to_file(fn_header)

        proc = SolveProcess(fn)
        proc.assign_conf(pmeta.conf)
        proc.assign_header(h)
        proc.setup()
        proc.invert_plate()
        proc.extract_sources()
        proc.solve_plate()
        proc.output_wcs_header()
        proc.solve_recursive()
        proc.output_sources_db()
        proc.output_sources_csv()
        proc.finish()


