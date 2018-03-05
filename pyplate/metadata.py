# coding: iso-8859-15

import os
import glob
import shutil
import copy
import re
import textwrap
import ConfigParser
import unicodecsv as csv
import unidecode
import math
import datetime as dt
import numpy as np
import ephem
import pytimeparse
from astropy import wcs
from astropy.io import fits
from astropy.io import votable
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle, SkyCoord, EarthLocation
from astropy import units
from collections import OrderedDict
from .database import PlateDB
from .conf import read_conf
from ._version import __version__

try:
    from PIL import Image
except ImportError:
    import Image

gexiv_available = True

try:
    from gi.repository import GExiv2
except ImportError:
    gexiv_available = False

pytz_available = True

try:
    import pytz
except ImportError:
    pytz_available = False

_keyword_meta = OrderedDict([
    ('plate_id', (int, False, None, None, None)),
    ('db_plate_id', (int, False, None, None, None)),
    ('scan_id', (int, False, None, None, None)),
    ('archive_id', (int, False, None, None, None)),
    ('fits_simple', (bool, False, True, 'SIMPLE', None)),
    ('fits_bitpix', (int, False, None, 'BITPIX', None)),
    ('fits_naxis', (int, False, None, 'NAXIS', None)),
    ('fits_naxis1', (int, False, None, 'NAXIS1', None)),
    ('fits_naxis2', (int, False, None, 'NAXIS2', None)),
    ('fits_bscale', (float, False, None, 'BSCALE', None)),
    ('fits_bzero', (int, False, None, 'BZERO', None)),
    ('fits_minval', (float, False, None, 'MINVAL', None)),
    ('fits_maxval', (float, False, None, 'MAXVAL', None)),
    ('fits_extend', (bool, False, True, 'EXTEND', None)),
    ('fits_datetime', (str, False, None, None, None)),
    ('fits_size', (int, False, None, None, None)),
    ('fits_checksum', (str, False, None, 'CHECKSUM', None)),
    ('fits_datasum', (str, False, None, 'DATASUM', None)),
    ('date_orig', (str, True, [], 'DATEORIG', 'DATEORn')),
    ('date_orig_end', (str, True, [], None, None)),
    ('tms_orig', (str, True, [], 'TMS-ORIG', 'TMS-ORn')),
    ('tme_orig', (str, True, [], 'TME-ORIG', 'TME-ORn')),
    ('tz_orig', (str, False, None, None, None)),
    ('ut_start_orig', (str, True, [], None, None)),
    ('ut_end_orig', (str, True, [], None, None)),
    ('st_start_orig', (str, True, [], None, None)),
    ('st_end_orig', (str, True, [], None, None)),
    ('jd_avg_orig', (float, True, [], 'JDA-ORIG', 'JDA-ORn')),
    ('time_flag', (str, False, None, 'TIMEFLAG', None)),
    ('ra_orig', (str, True, [], 'RA-ORIG', 'RA-ORn')),
    ('dec_orig', (str, True, [], 'DEC-ORIG', 'DEC-ORn')),
    ('coord_flag', (str, False, None, 'COORFLAG', None)),
    ('object_name', (str, True, [], 'OBJECT', 'OBJECTn')),
    ('object_type_code', (str, True, [], None, None)),
    ('object_type', (str, True, [], 'OBJTYPE', 'OBJTYPn')),
    ('numexp', (int, False, 1, 'NUMEXP', None)),
    ('exptime', (float, True, [], 'EXPTIME', 'EXPTIMn')),
    ('numsub', (int, True, [], None, None)),
    ('observatory', (str, False, None, 'OBSERVAT', None)),
    ('site_name', (str, False, None, 'SITENAME', None)),
    ('site_latitude', (float, False, None, 'SITELAT', None)),
    ('site_longitude', (float, False, None, 'SITELONG', None)),
    ('site_elevation', (float, False, None, 'SITEELEV', None)),
    ('telescope', (str, False, None, 'TELESCOP', None)),
    ('ota_name', (str, False, None, 'OTA-NAME', None)),
    ('ota_diameter', (float, False, None, 'OTA-DIAM', None)),
    ('ota_aperture', (float, False, None, 'OTA-APER', None)),
    ('ota_foclen', (float, False, None, 'FOCLEN', None)),
    ('ota_scale', (float, False, None, 'PLTSCALE', None)),
    ('instrument', (str, False, None, 'INSTRUME', None)),
    ('detector', (str, False, None, 'DETNAM', None)),
    ('method_code', (int, False, None, None, None)),
    ('method', (str, False, None, 'METHOD', None)),
    ('filter', (str, False, None, 'FILTER', None)),
    ('prism', (str, False, None, 'PRISM', None)),
    ('prism_angle', (str, False, None, 'PRISMANG', None)),
    ('dispersion', (float, False, None, 'DISPERS', None)),
    ('grating', (str, False, None, 'GRATING', None)),
    ('focus', (float, True, [], 'FOCUS', 'FOCUSn')),
    ('air_temperature', (float, False, None, 'TEMPERAT', None)),
    ('sky_calmness', (str, False, None, 'CALMNESS', None)),
    ('sky_sharpness', (str, False, None, 'SHARPNES', None)),
    ('sky_transparency', (str, False, None, 'TRANSPAR', None)),
    ('sky_conditions', (str, False, None, 'SKYCOND', None)),
    ('observer', (str, False, None, 'OBSERVER', None)),
    ('observer_notes', (str, False, None, 'OBSNOTES', None)),
    ('notes', (str, False, None, 'NOTES', None)),
    ('bibcode', (str, False, None, 'BIBCODE', None)),
    ('plate_num', (str, False, None, 'PLATENUM', None)),
    ('plate_num_orig', (str, False, None, 'PNUMORIG', None)),
    ('wfpdb_id', (str, False, None, 'WFPDB-ID', None)),
    ('series', (str, False, None, 'SERIES', None)),
    ('plate_format', (str, False, None, 'PLATEFMT', None)),
    ('plate_size1', (float, False, None, 'PLATESZ1', None)),
    ('plate_size2', (float, False, None, 'PLATESZ2', None)),
    ('emulsion', (str, False, None, 'EMULSION', None)),
    ('spectral_band', (str, False, None, None, None)),
    ('development', (str, False, None, 'DEVELOP', None)),
    ('plate_quality', (str, False, None, 'PQUALITY', None)),
    ('plate_notes', (str, False, None, 'PLATNOTE', None)),
    ('date_obs', (str, True, [], 'DATE-OBS', 'DT-OBSn')),
    ('date_avg', (str, True, [], 'DATE-AVG', 'DT-AVGn')),
    ('date_weighted', (str, True, [], None, None)),
    ('date_end', (str, True, [], 'DATE-END', 'DT-ENDn')),
    ('jd', (float, True, [], 'JD', 'JDn')),
    ('jd_avg', (float, True, [], 'JD-AVG', 'JD-AVGn')),
    ('jd_weighted', (float, True, [], None, None)),
    ('jd_end', (float, True, [], 'JD-END', 'JD-ENDn')),
    ('hjd_avg', (float, True, [], 'HJD-AVG', 'HJD-AVn')),
    ('hjd_weighted', (float, True, [], None, None)),
    ('year', (float, True, [], 'YEAR', 'YEARn')),
    ('year_avg', (float, True, [], 'YEAR-AVG', 'YR-AVGn')),
    ('year_weighted', (float, True, [], None, None)),
    ('year_end', (float, True, [], 'YEAR-END', 'YR-ENDn')),
    ('ra', (str, True, [], 'RA', 'RAn')),
    ('dec', (str, True, [], 'DEC', 'DECn')),
    ('ra_deg', (str, True, [], 'RA_DEG', 'RA_DEGn')),
    ('dec_deg', (str, True, [], 'DEC_DEG', 'DEC_DEn')),
    ('scanner', (str, False, None, 'SCANNER', None)),
    ('scan_res1', (int, False, None, 'SCANRES1', None)),
    ('scan_res2', (int, False, None, 'SCANRES2', None)),
    ('pix_size1', (float, False, None, 'PIXSIZE1', None)),
    ('pix_size2', (float, False, None, 'PIXSIZE2', None)),
    ('scan_software', (str, False, None, 'SCANSOFT', None)),
    ('scan_gamma', (str, False, None, 'SCANGAM', None)),
    ('scan_focus', (str, False, None, 'SCANFOC', None)),
    ('wedge', (str, False, None, 'WEDGE', None)),
    ('datescan', (str, False, None, 'DATESCAN', None)),
    ('scan_author', (str, False, None, 'SCANAUTH', None)),
    ('scan_notes', (str, False, None, 'SCANNOTE', None)),
    ('filename', (str, False, None, 'FILENAME', None)),
    ('fn_scan', (str, True, [], None, 'FN-SCNn')),
    ('fn_wedge', (str, False, None, 'FN-WEDGE', None)),
    ('fn_pre', (str, False, None, 'FN-PRE', None)),
    ('fn_cover', (str, False, None, 'FN-COVER', None)),
    ('fn_log', (str, True, [], None, 'FN-LOGn')),
    ('origin', (str, False, None, 'ORIGIN', None)),
    ('licence', (str, False, None, 'LICENCE', None)),
    ('date', (str, False, None, 'DATE', None)),
    ('fits_acknowledgements', (str, False, '', None, None)),
    ('fits_history', (str, True, [], 'HISTORY', None))
    ])

_logbook_meta = OrderedDict([
    ('logbook_id', (int, None)),
    ('archive_id', (int, None)),
    ('logbook_num', (str, None)),
    ('logbook_title', (str, None)),
    ('logbook_type', (int, None)),
    ('logbook_notes', (str, None))
    ])

_logpage_meta = OrderedDict([
    ('filename', (str, None)),
    ('logpage_id', (int, None)),
    ('archive_id', (int, None)),
    ('logbook_num', (str, None)),
    ('logbook_id', (int, None)),
    ('logpage_type', (int, None)),
    ('page_num', (int, None)),
    ('logpage_order', (int, None)),
    ('file_format', (str, None)),
    ('image_width', (int, None)),
    ('image_height', (int, None)),
    ('image_datetime', (str, None)),
    ('file_datetime', (str, None)),
    ('file_size', (int, None))
    ])

_preview_meta = OrderedDict([
    ('filename', (str, None)),
    ('preview_id', (int, None)),
    ('plate_id', (int, None)),
    ('db_plate_id', (int, None)),
    ('archive_id', (int, None)),
    ('plate_num', (str, None)),
    ('wfpdb_id', (str, None)),
    ('preview_type', (int, None)),
    ('file_format', (str, None)),
    ('image_width', (int, None)),
    ('image_height', (int, None)),
    ('image_datetime', (str, None)),
    ('file_datetime', (str, None)),
    ('file_size', (int, None))
    ])

def str_to_num(s):
    """
    Convert string to int or float, if possible. Otherwise return 
    the original string.
    
    """

    if s == '':
        return None

    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def _combine_orig_times(orig_time, ut_time, st_time):
    """
    Combine original observation times into a list of strings.

    """

    ntimes = max([len(orig_time), len(ut_time), len(st_time)])
    tms = []

    for i in np.arange(ntimes):
        tm = []

        try:
            tm_str = orig_time[i]
        except IndexError:
            tm_str = None

        if tm_str:
            tm.append(tm_str)

        try:
            tm_str = 'UT {}'.format(ut_time[i])
        except IndexError:
            tm_str = None

        if tm_str:
            if tm and tm[0][:2] == 'UT':
                tm[0] = tm_str
            else:
                tm.append(tm_str)

        try:
            tm_str = 'ST {}'.format(st_time[i])
        except IndexError:
            tm_str = None

        if tm_str:
            if tm and tm[0][:2] == 'ST':
                tm[0] = tm_str
            else:
                tm.append(tm_str)

        tms.append(', '.join(tm))

    return tms

def _split_orig_times(combined_str):
    """
    Split string of combined original observation times.

    """

    if isinstance(combined_str, unicode):
        combined_str = unidecode.unidecode(combined_str)

    if not isinstance(combined_str, str):
        return None

    orig_time = None
    ut_time = None
    st_time = None

    tm = [s.strip() for s in combined_str.split(',')]

    for tm_str in tm:
        if tm_str[:2] == 'ST':
            st_time = tm_str[3:].strip()
        elif tm_str[:2] == 'UT':
            ut_time = tm_str[3:].strip()
        else:
            orig_time = tm_str.strip()

    return (orig_time, ut_time, st_time)


class CSV_Dict(OrderedDict):
    """
    Metadata CSV dictionary class.

    """

    def __init__(self, *args, **kwargs):
        super(CSV_Dict, self).__init__(*args, **kwargs)
        self.filename = None


class ArchiveMeta:
    """
    Plate archive metadata class.

    """

    def __init__(self):
        self.archive_id = None
        self.archive_name = None
        self.logbooks = None
        self.logbookmeta = OrderedDict()
        self.conf = ConfigParser.ConfigParser()
        self.maindata_dict = OrderedDict()
        self.notes_dict = OrderedDict()
        self.observer_dict = OrderedDict()
        self.quality_dict = OrderedDict()
        self.plate_csv_dict = [CSV_Dict()]
        self.scan_csv_dict = CSV_Dict()
        self.preview_csv_dict = CSV_Dict()
        self.logbook_csv_dict = CSV_Dict()
        self.logpage_csv_dict = CSV_Dict()
        self.fits_dir = ''

        self.output_db_host = 'localhost'
        self.output_db_user = ''
        self.output_db_name = ''
        self.output_db_passwd = ''

    def assign_conf(self, conf):
        """
        Assign and parse configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['archive_id']:
            try:
                setattr(self, attr, conf.getint('Archive', attr))
            except ValueError:
                print ('Error: Value in configuration file must be '
                       'integer ([{}], {})'.format('Archive', attr))
            except ConfigParser.Error:
                pass

        for attr in ['archive_name', 'logbooks']:
            try:
                setattr(self, attr, conf.get('Archive', attr))
            except ConfigParser.Error:
                pass

        for attr in ['fits_dir', 'write_log_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['output_db_host', 'output_db_user',
                     'output_db_name', 'output_db_passwd']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except ConfigParser.Error:
                pass

        if self.logbooks:
            self.logbooks = [lb.strip() for lb in self.logbooks.split(',')]

            for lb in self.logbooks:
                lbmeta = {'archive_id': self.archive_id}

                # Assign integer values
                for k in ['logbook_id', 'logbook_type']:
                    try:
                        lbmeta[k] = conf.getint(lb, k)
                    except ValueError:
                        print ('Error: Value in configuration file must be '
                               'integer ([{}], {})'.format(lb, k))
                        lbmeta[k] = None
                    except ConfigParser.Error:
                        lbmeta[k] = None

                # Assign string values
                for k in ['logbook_num', 'logbook_title', 'logbook_notes']:
                    try:
                        lbmeta[k] = conf.get(lb, k)
                    except ConfigParser.Error:
                        lbmeta[k] = None

                # Require non-zero logbook_id
                if lbmeta['logbook_id'] == 0:
                    lbmeta['logbook_id'] = None

                self.logbookmeta[lb] = lbmeta

    def read_wfpdb(self, wfpdb_dir=None, fn_maindata=None, fn_quality=None, 
                   fn_notes=None, fn_observer=None):
        """
        Read WFPDB data files.

        Parameters
        ----------
        wfpdb_dir : str
            Path to the directory with WFPDB files.
        fn_maindata : str
            Name of the WFPDB maindata file.
        fn_quality : str
            Name of the WFPDB quality file.
        fn_notes : str
            Name of the WFPDB notes file.
        fn_observer : str
            Name of the WFPDB observer file.

        """

        if wfpdb_dir is None:
            try:
                wfpdb_dir = self.conf.get('Files', 'wfpdb_dir')
            except ConfigParser.Error:
                wfpdb_dir = ''

        if fn_maindata is None:
            try:
                fn_maindata = self.conf.get('Files', 'wfpdb_maindata')
            except ConfigParser.Error:
                pass

        if fn_quality is None:
            try:
                fn_quality = self.conf.get('Files', 'wfpdb_quality')
            except ConfigParser.Error:
                pass

        if fn_notes is None:
            try:
                fn_notes = self.conf.get('Files', 'wfpdb_notes')
            except ConfigParser.Error:
                pass

        if fn_observer is None:
            try:
                fn_observer = self.conf.get('Files', 'wfpdb_observer')
            except ConfigParser.Error:
                pass

        if fn_maindata:
            try:
                with open(os.path.join(wfpdb_dir, fn_maindata), 'r') as f:
                    lst = [(line[:14].rstrip().replace(' ','_'),
                            line.rstrip('\n')) for line in f.readlines()]
                    self.maindata_dict = OrderedDict(lst)
            except IOError:
                print 'Could not read the WFPDB maindata file!'

        if fn_quality:
            try:
                with open(os.path.join(wfpdb_dir, fn_quality), 'r') as f:
                    lst = [(line[:14].rstrip().replace(' ','_'),
                            line.rstrip('\n')) for line in f.readlines()]
                    self.quality_dict = OrderedDict(lst)
            except IOError:
                print 'Could not read the WFPDB quality file!'

        if fn_notes:
            try:
                with open(os.path.join(wfpdb_dir, fn_notes), 'r') as f:
                    lst = [(line[:14].rstrip().replace(' ','_'),
                            line.rstrip('\n')) for line in f.readlines()]
                    self.notes_dict = OrderedDict(lst)
            except IOError:
                print 'Could not read the WFPDB notes file!'

        if fn_observer:
            try:
                with open(os.path.join(wfpdb_dir, fn_observer), 'r') as f:
                    lst = [(line[:14].rstrip().replace(' ','_'),
                            line.rstrip('\n')) for line in f.readlines()]
                    self.observer_dict = OrderedDict(lst)
            except IOError:
                print 'Could not read the WFPDB observer file!'

    def read_csv(self, csv_dir=None, fn_plate_csv=None, fn_scan_csv=None, 
                 fn_preview_csv=None, fn_logbook_csv=None, fn_logpage_csv=None):
        """
        Read CSV files.

        Parameters
        ----------
        csv_dir : str
            Path to the directory with CSV files.
        fn_plate_csv : str
            Name of the plate metadata CSV file.
        fn_scan_csv : str
            Name of the scan metadata CSV file.
        fn_preview_csv : str
            Name of the preview metadata CSV file.
        fn_logbook_csv : str
            Name of the logbook CSV file.
        fn_logpage_csv : str
            Name of the logpage CSV file.

        """

        if csv_dir is None:
            try:
                csv_dir = self.conf.get('Files', 'csv_dir')
            except ConfigParser.Error:
                csv_dir = ''
                
        if self.conf.has_section('Files'):
            if self.conf.has_option('Files', 'plate_csv'):
                fn_str = self.conf.get('Files', 'plate_csv')

                if fn_str:
                    fn_plate_csv = [os.path.join(csv_dir, fn.strip()) 
                                    for fn in fn_str.split(',')]

            if self.conf.has_option('Files', 'scan_csv'):
                fn_str = self.conf.get('Files', 'scan_csv')

                if fn_str:
                    fn_scan_csv = os.path.join(csv_dir, fn_str)

            if self.conf.has_option('Files', 'preview_csv'):
                fn_str = self.conf.get('Files', 'preview_csv')

                if fn_str:
                    fn_preview_csv = os.path.join(csv_dir, fn_str)

            if self.conf.has_option('Files', 'logbook_csv'):
                fn_str = self.conf.get('Files', 'logbook_csv')

                if fn_str:
                    fn_logbook_csv = os.path.join(csv_dir, fn_str)

            if self.conf.has_option('Files', 'logpage_csv'):
                fn_str = self.conf.get('Files', 'logpage_csv')

                if fn_str:
                    fn_logpage_csv = os.path.join(csv_dir, fn_str)

        if fn_plate_csv:
            if isinstance(fn_plate_csv, str):
                fn_plate_csv = [fn_plate_csv]
                
            csv_delimiter = ','
            csv_quotechar = '"'

            for ind,fn_path in enumerate(fn_plate_csv):
                fn_base = os.path.basename(fn_path)

                if self.conf.has_section(fn_base):
                    if self.conf.has_option(fn_base, 'csv_delimiter'):
                        csv_delimiter = self.conf.get(fn_base, 'csv_delimiter')

                    if self.conf.has_option(fn_base, 'csv_quotechar'):
                        csv_quotechar = self.conf.get(fn_base, 'csv_quotechar')

                with open(fn_path, 'rb') as f:
                    reader = csv.reader(f, delimiter=csv_delimiter, 
                                        quotechar=csv_quotechar)
                    csvdict = CSV_Dict(((row[0],row) for row in reader))
                    csvdict.filename = fn_base

                    if ind == 0:
                        self.plate_csv_dict[0] = csvdict
                    else:
                        self.plate_csv_dict.append(csvdict)

        if fn_scan_csv:
            fn_base = os.path.basename(fn_scan_csv)
            csv_delimiter = ','
            csv_quotechar = '"'

            if self.conf.has_section(fn_base):
                if self.conf.has_option(fn_base, 'csv_delimiter'):
                    csv_delimiter = self.conf.get(fn_base, 'csv_delimiter')

                if self.conf.has_option(fn_base, 'csv_quotechar'):
                    csv_quotechar = self.conf.get(fn_base, 'csv_quotechar')

            with open(fn_scan_csv, 'rb') as f:
                reader = csv.reader(f, delimiter=csv_delimiter,
                                    quotechar=csv_quotechar)
                self.scan_csv_dict = CSV_Dict(((row[0],row)
                                               for row in reader))
                self.scan_csv_dict.filename = fn_base

        if fn_preview_csv:
            fn_base = os.path.basename(fn_preview_csv)
            csv_delimiter = ','
            csv_quotechar = '"'

            if self.conf.has_section(fn_base):
                if self.conf.has_option(fn_base, 'csv_delimiter'):
                    csv_delimiter = self.conf.get(fn_base, 'csv_delimiter')

                if self.conf.has_option(fn_base, 'csv_quotechar'):
                    csv_quotechar = self.conf.get(fn_base, 'csv_quotechar')

            with open(fn_preview_csv, 'rb') as f:
                reader = csv.reader(f, delimiter=csv_delimiter,
                                    quotechar=csv_quotechar)
                self.preview_csv_dict = CSV_Dict(((row[0],row)
                                                   for row in reader))
                self.preview_csv_dict.filename = fn_base

        if fn_logbook_csv:
            fn_base = os.path.basename(fn_logbook_csv)
            csv_delimiter = ','
            csv_quotechar = '"'

            if self.conf.has_section(fn_base):
                if self.conf.has_option(fn_base, 'csv_delimiter'):
                    csv_delimiter = self.conf.get(fn_base, 'csv_delimiter')

                if self.conf.has_option(fn_base, 'csv_quotechar'):
                    csv_quotechar = self.conf.get(fn_base, 'csv_quotechar')

            with open(fn_logbook_csv, 'rb') as f:
                reader = csv.reader(f, delimiter=csv_delimiter,
                                    quotechar=csv_quotechar)
                self.logbook_csv_dict = CSV_Dict(((row[0],row)
                                                  for row in reader))
                self.logbook_csv_dict.filename = fn_base

        if fn_logpage_csv:
            fn_base = os.path.basename(fn_logpage_csv)
            csv_delimiter = ','
            csv_quotechar = '"'

            if self.conf.has_section(fn_base):
                if self.conf.has_option(fn_base, 'csv_delimiter'):
                    csv_delimiter = self.conf.get(fn_base, 'csv_delimiter')

                if self.conf.has_option(fn_base, 'csv_quotechar'):
                    csv_quotechar = self.conf.get(fn_base, 'csv_quotechar')

            with open(fn_logpage_csv, 'rb') as f:
                reader = csv.reader(f, delimiter=csv_delimiter,
                                    quotechar=csv_quotechar)
                self.logpage_csv_dict = CSV_Dict(((row[0],row)
                                                  for row in reader))
                self.logpage_csv_dict.filename = fn_base

    def get_platelist(self):
        """
        Get list of plate IDs based on WFPDB and CSV files

        """

        if self.maindata_dict:
            return self.maindata_dict.keys()
        elif self.plate_csv_dict[0]:
            return self.plate_csv_dict[0].keys()
        else:
            return []

    def get_scanlist(self):
        """
        Get list of filenames from CSV files

        """

        if self.scan_csv_dict:
            return self.scan_csv_dict.keys()
        else:
            return []

    def get_previewlist(self):
        """
        Get list of preview files

        """

        if self.preview_csv_dict:
            return self.preview_csv_dict.keys()
        else:
            return []

    def get_logbooklist(self):
        """
        Get list of logbook identifications

        """

        if self.logbooks:
            return self.logbooks
        if self.logbook_csv_dict:
            return self.logbook_csv_dict.keys()
        else:
            return []

    def get_logpagelist(self):
        """
        Get list of logpage filenames

        """

        if self.logpage_csv_dict:
            return self.logpage_csv_dict.keys()
        else:
            return []

    def get_logbookmeta(self, num=None):
        """
        Get metadata for the specific logbook.

        """

        logbookmeta = LogbookMeta(num=num)
        logbookmeta['archive_id'] = self.archive_id

        if self.conf is not None:
            logbookmeta.assign_conf(self.conf)

        if num:
            if self.logbooks and (num in self.logbooks):
                logbookmeta = self.logbookmeta[num]
            elif num in self.logbook_csv_dict:
                logbookmeta.parse_csv(self.logbook_csv_dict[num], 
                                      csv_filename=self.logbook_csv_dict
                                      .filename)

        return logbookmeta
        
    def get_logpagemeta(self, filename=None):
        """
        Get metadata for the specific logpage.

        """

        logpagemeta = LogpageMeta(filename=filename)
        logpagemeta['archive_id'] = self.archive_id

        if self.conf is not None:
            logpagemeta.assign_conf(self.conf)

        if filename:
            if filename in self.logpage_csv_dict:
                logpagemeta.parse_csv(self.logpage_csv_dict[filename], 
                                      csv_filename=self.logpage_csv_dict
                                      .filename)

            logpagemeta.parse_exif()

        return logpagemeta
        
    def get_previewmeta(self, filename=None):
        """
        Get metadata for the specific preview image.

        """

        previewmeta = PreviewMeta(filename=filename)
        previewmeta['archive_id'] = self.archive_id

        if self.conf is not None:
            previewmeta.assign_conf(self.conf)

        if filename:
            if filename in self.preview_csv_dict:
                previewmeta.parse_csv(self.preview_csv_dict[filename], 
                                      csv_filename=self.preview_csv_dict
                                      .filename)

            previewmeta.parse_exif()

        if previewmeta['plate_id']:
            platemeta = self.get_platemeta(previewmeta['plate_id'])

            if not previewmeta['db_plate_id'] and platemeta['db_plate_id']:
                previewmeta['db_plate_id'] = platemeta['db_plate_id']

            if not previewmeta['plate_num'] and platemeta['plate_num']:
                previewmeta['plate_num'] = platemeta['plate_num']

            if not previewmeta['wfpdb_id'] and platemeta['wfpdb_id']:
                previewmeta['wfpdb_id'] = platemeta['wfpdb_id']

            if not previewmeta['archive_id'] and platemeta['archive_id']:
                previewmeta['archive_id'] = platemeta['archive_id']

        return previewmeta
        
    def output_plates_db(self):
        """
        Write plates to the database.

        """

        platedb = PlateDB()
        platedb.assign_conf(self.conf)
        platedb.open_connection()

        for plate_id in self.get_platelist():
            platemeta = self.get_platemeta(plate_id=plate_id)
            platemeta.compute_values()
            platedb.write_plate(platemeta)

        platedb.close_connection()

    def output_scans_db(self):
        """
        Write scans to the database.

        """

        if self.scan_csv_dict:
            platedb = PlateDB()
            platedb.assign_conf(self.conf)
            platedb.open_connection()

            for filename in self.scan_csv_dict.keys():
                platemeta = self.get_platemeta(filename=filename)
                platemeta.compute_values()
                platedb.write_scan(platemeta)

            platedb.close_connection()

    def output_previews_db(self):
        """
        Write previews to the database.

        """

        if self.preview_csv_dict:
            platedb = PlateDB()
            platedb.assign_conf(self.conf)
            platedb.open_connection()

            for filename in self.get_previewlist():
                previewmeta = self.get_previewmeta(filename=filename)
                platedb.write_preview(previewmeta)

            platedb.close_connection()

    def output_logpages_db(self):
        """
        Write logpages to the database.

        """

        platedb = PlateDB()
        platedb.assign_conf(self.conf)
        platedb.open_connection()

        for num in self.get_logbooklist():
            logbookmeta = self.get_logbookmeta(num=num)
            platedb.write_logbook(logbookmeta)

        for filename in self.get_logpagelist():
            logpagemeta = self.get_logpagemeta(filename=filename)
            platedb.write_logpage(logpagemeta)

        for plate_id in self.get_platelist():
            platemeta = self.get_platemeta(plate_id=plate_id)
            platedb.write_plate_logpage(platemeta)

        platedb.close_connection()

    def get_platemeta(self, plate_id=None, wfpdb_id=None, filename=None):
        """
        Get metadata for the specific plate (or scan).
        
        Source files are parsed in the following order: WFPDB maindata, 
        WFPDB quality, WFPDB observer, WFPDB notes, plate CSV, scan CSV, 
        configuration file.

        Parameters
        ----------
        plate_id : str
            Plate ID used in metadata files.
        filename : str
            Name of the plate scan file.

        Returns
        -------
        platemeta : a :class:`PlateMeta` object
            :class:`PlateMeta` object that contains the plate metadata.

        """

        platemeta = PlateMeta(plate_id=plate_id)
        platemeta['archive_id'] = self.archive_id

        if self.conf is not None:
            platemeta.assign_conf(self.conf)

        if filename:
            if self.fits_dir:
                filename = os.path.join(self.fits_dir, filename)

            # Read FITS image and header
            try:
                fitsdata,header = fits.getdata(filename, header=True,
                                               do_not_scale_image_data=True,
                                               ignore_missing_end=True)
            except Exception:
                print 'Error reading {}'.format(filename)
                fitsdata = None
                header = None

            if header:
                platemeta.parse_header(header)
                mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(filename))
                mtime_str = mtime.strftime('%Y-%m-%dT%H:%M:%S')
                platemeta['fits_datetime'] = mtime_str
                platemeta['fits_size'] = os.path.getsize(filename)

            # Use image data to determine min and max values
            if fitsdata is not None:
                platemeta['fits_minval'] = fitsdata.min()
                platemeta['fits_maxval'] = fitsdata.max()

                try:
                    platemeta['fits_minval'] = (platemeta['fits_bzero'] 
                                                + platemeta['fits_bscale'] 
                                                * platemeta['fits_minval'])
                    platemeta['fits_maxval'] = (platemeta['fits_bzero'] 
                                                + platemeta['fits_bscale'] 
                                                * platemeta['fits_maxval'])
                except TypeError:
                    pass

        if filename:
            fn_base = os.path.basename(filename)
        else:
            fn_base = None

        # Parse scan CSV to get plate_id
        if fn_base in self.scan_csv_dict:
            platemeta.parse_csv(self.scan_csv_dict[fn_base],
                                csv_filename=self.scan_csv_dict.filename)

            if not plate_id and platemeta['plate_id']:
                plate_id = platemeta['plate_id']

            # Parse plate CSV to get wfpdb_id
            if plate_id in self.plate_csv_dict[0]:
                platemeta.parse_csv(self.plate_csv_dict[0][plate_id], 
                                    csv_filename=self.plate_csv_dict[0].filename)

            if not wfpdb_id and platemeta['wfpdb_id']:
                wfpdb_id = platemeta['wfpdb_id']

        # Check if plate_id matches any WFPDB ID
        if plate_id and not wfpdb_id:
            if plate_id in self.maindata_dict:
                platemeta['wfpdb_id'] = plate_id
                wfpdb_id = plate_id

        # Check if filename without extension matches any WFPDB ID or 
        # plate ID in the CSV file
        if not plate_id and not wfpdb_id:
            fn_part = os.path.splitext(fn_base)[0]

            if fn_part in self.maindata_dict:
                plate_id = fn_part
                wfpdb_id = plate_id
                platemeta['wfpdb_id'] = wfpdb_id

            if fn_part in self.plate_csv_dict[0]:
                plate_id = fn_part

        # Check if WFPDB ID exists and plate_id does not
        if wfpdb_id and not plate_id:
            platemeta['plate_id'] = wfpdb_id
            plate_id = wfpdb_id

        if wfpdb_id in self.maindata_dict:
            platemeta.parse_maindata(self.maindata_dict[wfpdb_id])
        elif plate_id in self.maindata_dict:
            platemeta.parse_maindata(self.maindata_dict[plate_id])

        if wfpdb_id in self.quality_dict:
            platemeta.parse_quality(self.quality_dict[wfpdb_id])
        elif plate_id in self.quality_dict:
            platemeta.parse_quality(self.quality_dict[plate_id])

        if wfpdb_id in self.observer_dict:
            platemeta.parse_observer(self.observer_dict[wfpdb_id])
        elif plate_id in self.observer_dict:
            platemeta.parse_observer(self.observer_dict[plate_id])

        if wfpdb_id in self.notes_dict:
            platemeta.parse_notes(self.notes_dict[wfpdb_id])
        elif plate_id in self.notes_dict:
            platemeta.parse_notes(self.notes_dict[plate_id])

        # Parse plate CSV and scan CSV (and possibly override values from 
        # other sources)
        for csvdict in self.plate_csv_dict:
            if plate_id in csvdict:
                platemeta.parse_csv(csvdict[plate_id], 
                                    csv_filename=csvdict.filename)
            elif platemeta['plate_id'] in csvdict:
                platemeta.parse_csv(csvdict[platemeta['plate_id']], 
                                    csv_filename=csvdict.filename)

        if fn_base in self.scan_csv_dict:
            platemeta.parse_csv(self.scan_csv_dict[fn_base], 
                                csv_filename=self.scan_csv_dict.filename)

        if fn_base and not platemeta['filename']:
            platemeta['filename'] = fn_base

        platemeta.parse_conf()

        return platemeta


class LogbookMeta(OrderedDict):
    """
    Logbook metadata class.

    """

    def __init__(self, *args, **kwargs):
        num = kwargs.pop('num', None)
        super(LogbookMeta, self).__init__(*args, **kwargs)

        for k,v in _logbook_meta.items():
            self[k] = copy.deepcopy(v[1])

        self['logbook_num'] = num
        self.conf = ConfigParser.ConfigParser()

    def assign_conf(self, conf):
        """
        Assign configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

    def parse_csv(self, val_list, csv_filename=None):
        """
        Extract data from a CSV row.

        """

        if self.conf.has_section(csv_filename):
            for (key, pos) in self.conf.items(csv_filename):
                if key in self:
                    try:
                        pos = int(pos)
                    except ValueError:
                        pos = 0

                    if (pos > 0 and pos <= len(val_list)):
                        if (_logbook_meta[key][0] is int or 
                            _logbook_meta[key][0] is float):
                            val = str_to_num(val_list[pos-1])
                        else:
                            val = val_list[pos-1]

                        try:
                            self[key].append(val)
                        except AttributeError:
                            self[key] = val

            # Require non-zero logbook_id
            if self['logbook_id'] == 0:
                self['logbook_id'] = None


class LogpageMeta(OrderedDict):
    """
    Logpage metadata class.

    """

    def __init__(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        super(LogpageMeta, self).__init__(*args, **kwargs)

        for k,v in _logpage_meta.items():
            self[k] = copy.deepcopy(v[1])

        self.conf = ConfigParser.ConfigParser()
        self.logpage_dir = ''
        self.cover_dir = ''
        self.logpage_exif_timezone = None
        self['filename'] = filename
        fmt = filename.split('.')[-1].upper()

        if fmt == 'JPG':
            fmt = 'JPEG'
        elif fmt == 'TIF':
            fmt = 'TIFF'

        if fmt in ['JPEG', 'PNG', 'TIFF']:
            self['file_format'] = fmt
        
    def assign_conf(self, conf):
        """
        Assign configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['logpage_dir', 'cover_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['logpage_exif_timezone']:
            try:
                setattr(self, attr, conf.get('Image', attr))
            except ConfigParser.Error:
                pass

    def parse_csv(self, val_list, csv_filename=None):
        """
        Extract data from a CSV row.

        """

        if self.conf.has_section(csv_filename):
            for (key, pos) in self.conf.items(csv_filename):
                if (key in self) and (key != 'filename'):
                    try:
                        pos = int(pos)
                    except ValueError:
                        pos = 0

                    if (pos > 0 and pos <= len(val_list)):
                        if (_logpage_meta[key][0] is int or 
                            _logpage_meta[key][0] is float):
                            val = str_to_num(val_list[pos-1])
                        else:
                            val = val_list[pos-1]

                        try:
                            self[key].append(val)
                        except AttributeError:
                            self[key] = val

            # Require non-zero logpage_id
            if self['logpage_id'] == 0:
                self['logpage_id'] = None

    def parse_exif(self):
        """
        Extract data from image EXIF.

        """

        fn_path = os.path.join(self.logpage_dir, self['filename'])

        if not os.path.exists(fn_path):
            fn_path = os.path.join(self.cover_dir, self['filename'])

            if not os.path.exists(fn_path):
                return

        mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(fn_path))
        self['file_datetime'] = mtime.strftime('%Y-%m-%dT%H:%M:%S')
        self['file_size'] = os.path.getsize(fn_path)

        if gexiv_available:
            try:
                exif = GExiv2.Metadata()
                exif.open_path(fn_path)
            except Exception:
                return

            self['image_width'] = exif.get_pixel_width()
            self['image_height'] = exif.get_pixel_height()

            try:
                self['image_datetime'] = exif.get_date_time().isoformat()
            except (KeyError, TypeError, AttributeError):
                pass

        if (not self['image_width'] or not self['image_height'] 
            or not self['image_datetime']):
            try:
                im_pil = Image.open(fn_path)
            except Exception:
                return
            
            self['image_width'], self['image_height'] = im_pil.size
            self['file_format'] = im_pil.format
            exif_datetime = None

            if self['file_format'] == 'JPEG':
                exif = im_pil._getexif()

                try:
                    exif_datetime = exif[306]
                except (KeyError, TypeError):
                    pass
            elif self['file_format'] == 'TIFF':
                try:
                    exif_datetime = im_pil.tag[306]
                except (KeyError, TypeError):
                    pass

            if exif_datetime:
                if exif_datetime[4] == ':':
                    exif_datetime = '{} {}'.format(exif_datetime[:10]
                                                   .replace(':', '-'),
                                                   exif_datetime[11:])

                if pytz_available and self.logpage_exif_timezone:
                    dt_exif = dt.datetime.strptime(exif_datetime, 
                                                   '%Y-%m-%d %H:%M:%S')

                    try:
                        dt_local = (pytz.timezone(self.logpage_exif_timezone)
                                    .localize(dt_exif))
                        exif_datetime = (dt_local.astimezone(pytz.utc)
                                         .strftime('%Y-%m-%dT%H:%M:%S'))
                    except pytz.exceptions.UnknownTimeZoneError:
                        pass

                self['image_datetime'] = exif_datetime


class PreviewMeta(OrderedDict):
    """
    Preview image metadata class.

    """

    def __init__(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        super(PreviewMeta, self).__init__(*args, **kwargs)

        for k,v in _preview_meta.items():
            self[k] = copy.deepcopy(v[1])

        self.conf = ConfigParser.ConfigParser()
        self.preview_dir = ''
        self.preview_exif_timezone = None
        self['filename'] = filename
        fmt = filename.split('.')[-1].upper()

        if fmt == 'JPG':
            fmt = 'JPEG'
        elif fmt == 'TIF':
            fmt = 'TIFF'

        if fmt in ['JPEG', 'PNG', 'TIFF']:
            self['file_format'] = fmt
        
    def assign_conf(self, conf):
        """
        Assign configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['preview_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['preview_exif_timezone']:
            try:
                setattr(self, attr, conf.get('Image', attr))
            except ConfigParser.Error:
                pass

    def parse_csv(self, val_list, csv_filename=None):
        """
        Extract data from a CSV row.

        """

        if self.conf.has_section(csv_filename):
            for (key, pos) in self.conf.items(csv_filename):
                if (key in self) and (key != 'filename'):
                    try:
                        pos = int(pos)
                    except ValueError:
                        pos = 0

                    if (pos > 0 and pos <= len(val_list)):
                        if (_preview_meta[key][0] is int or 
                            _preview_meta[key][0] is float):
                            val = str_to_num(val_list[pos-1])
                        else:
                            val = val_list[pos-1]

                        try:
                            self[key].append(val)
                        except AttributeError:
                            self[key] = val

            # Require non-zero preview_id
            if self['preview_id'] == 0:
                self['preview_id'] = None

    def parse_exif(self):
        """
        Extract data from image EXIF.

        """

        fn_path = os.path.join(self.preview_dir, self['filename'])

        if not os.path.exists(fn_path):
            return

        mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(fn_path))
        self['file_datetime'] = mtime.strftime('%Y-%m-%dT%H:%M:%S')
        self['file_size'] = os.path.getsize(fn_path)

        if gexiv_available:
            try:
                exif = GExiv2.Metadata()
                exif.open_path(fn_path)
            except Exception:
                return

            self['image_width'] = exif.get_pixel_width()
            self['image_height'] = exif.get_pixel_height()

            try:
                self['image_datetime'] = exif.get_date_time().isoformat()
            except (KeyError, TypeError, AttributeError):
                pass

        if (not self['image_width'] or not self['image_height'] 
            or not self['image_datetime']):
            try:
                im_pil = Image.open(fn_path)
            except Exception:
                return
            
            self['image_width'], self['image_height'] = im_pil.size
            self['file_format'] = im_pil.format
            exif_datetime = None

            if self['file_format'] == 'JPEG':
                exif = im_pil._getexif()

                try:
                    exif_datetime = exif[306]
                except (KeyError, TypeError):
                    pass
            elif self['file_format'] == 'TIFF':
                try:
                    exif_datetime = im_pil.tag[306]
                except (KeyError, TypeError):
                    pass

            if exif_datetime:
                if exif_datetime[4] == ':':
                    exif_datetime = '{} {}'.format(exif_datetime[:10]
                                                   .replace(':', '-'),
                                                   exif_datetime[11:])

                if pytz_available and self.preview_exif_timezone:
                    dt_exif = dt.datetime.strptime(exif_datetime, 
                                                   '%Y-%m-%d %H:%M:%S')

                    try:
                        dt_local = (pytz.timezone(self.preview_exif_timezone)
                                    .localize(dt_exif))
                        exif_datetime = (dt_local.astimezone(pytz.utc)
                                         .strftime('%Y-%m-%dT%H:%M:%S'))
                    except pytz.exceptions.UnknownTimeZoneError:
                        pass

                self['image_datetime'] = exif_datetime


class PlateMeta(OrderedDict):
    """
    Plate metadata class.

    """

    def __init__(self, *args, **kwargs):
        plate_id = kwargs.pop('plate_id', None)
        super(PlateMeta, self).__init__(*args, **kwargs)
        
        for k,v in _keyword_meta.items():
            self[k] = copy.deepcopy(v[2])

        self['plate_id'] = plate_id
        self.exposures = None

        self.conf = ConfigParser.ConfigParser()
        self.output_db_host = 'localhost'
        self.output_db_user = ''
        self.output_db_name = ''
        self.output_db_passwd = ''

    def copy(self):
        pmeta = PlateMeta()

        for k,v in self.items():
            pmeta[k] = copy.deepcopy(v)

        return pmeta

    _object_types = {'A1': 'planet',
        'A2': 'Moon',
        'A3': 'Sun',
        'A4': 'asteroid',
        'A5': 'comet',
        'A6': 'meteor',
        'A7': 'artificial satellite',
        'S1': 'star',
        'S2': 'double star or multiple star',
        'S3': 'variable star',
        'S4': 'star cluster',
        'S5': 'HII region',
        'S6': 'nebula',
        'S7': 'planetary nebula',
        'S8': 'supernova + SN remnants',
        'S9': 'fundamental star',
        'SR': 'reference star around a radio source',
        'SA': 'stellar association',
        'SD': 'dark nebula',
        'SH': 'Herbig-Haro object',
        'SM': 'molecular cloud',
        'SP': 'pulsar',
        'G1': 'galaxy',
        'G2': 'QSO',
        'G3': 'group of galaxies',
        'G4': 'cluster of galaxies',
        'G5': 'supercluster',
        'G6': 'void',
        'F':  'field',
        'G7': 'radio galaxy',
        'GR': 'gamma-ray source',
        'XR': 'X-ray source',
        'RS': 'radio source',
        'IR': 'infrared source',
        'U':  'object of unknown nature'}

    _method_codes = {1: 'direct photograph',
        2:  'direct photograph, multi-exposure',
        3:  'stellar tracks',
        4:  'objective prism',
        5:  'objective prism, multi-exposure',
        6:  'Metcalf\'s method',
        7:  'proper motions',
        8:  'no guiding',
        9:  'out of focus',
        10: 'test plate',
        11: 'Hartmann test',
        12: 'with mask',
        14: 'sub-beam (Pickering) prism',
        13: 'focusing',
        15: 'raster scan/trail',
        24: 'objective grating',
        25: 'objective grating, multi-exposure',
        31: 'objective prism together with a grating'}

    def assign_conf(self, conf):
        """
        Assign configuration.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['output_db_host', 'output_db_user',
                     'output_db_name', 'output_db_passwd']:
            try:
                setattr(self, attr, conf.get('Database', attr))
            except ConfigParser.Error:
                pass

    def get_value(self, key, exp=0):
        """
        Get PlateMeta value for SQL query

        Parameters
        ----------
        key : str
            PlateMeta keyword to be fetched

        Returns
        -------
        val
            PlateMeta value corresponding to the keyword

        """

        kwmeta = _keyword_meta[key]

        if kwmeta[1] and self[key]:
            if isinstance(self[key], str):
                val = self[key]
            else:
                if len(self[key]) > exp:
                    val = self[key][exp]
                elif len(self[key]) == 1:
                    val = self[key][0]
                else:
                    val = None
        else:
            val = self[key]

        if val == []:
            val = None

        return val

    def printdata(self):
        """
        Print keywords and their values.

        """
        
        for k,v in self.iteritems():
            if isinstance(v, unicode):
                v = v.encode('utf-8')

            print '{:15}: {}'.format(k, v)

    def parse_maindata(self, maindata_entry):
        """
        Extract data from the WFPDB maindata entry.

        """

        plate_num = maindata_entry[7:14]
        plate_num_suffix = plate_num[6].strip()
        self['plate_num'] = '{:d}{}'.format(int(plate_num[:6]), 
                                            plate_num_suffix)
        
        radec = maindata_entry[14:27]

        if radec[0:2].strip():
            self['ra_orig'] = ['{:0>2}:{:0>2}'.format(radec[0:2].strip(), 
                                                      radec[2:4].strip())]

            if radec[4:6].strip():
                self['ra_orig'][0] += ':{:0>2}'.format(radec[4:6].strip())

        if radec[7:8].strip():
            self['dec_orig'] = ['{}:{:0>2}'.format(radec[6:9], 
                                                   radec[9:11].strip())]

            if radec[11:13].strip():
                self['dec_orig'][0] += ':{:0>2}'.format(radec[11:13].strip())

        self['coord_flag'] = maindata_entry[27]

        datetime = maindata_entry[28:42]

        if datetime[0:4].strip():
            self['date_orig'] = ['{}-{}-{}'.format(datetime[0:4], datetime[4:6],
                                                   datetime[6:8])]

        if datetime[8:10].strip():
            self['tms_orig'] = ['{}'.format(datetime[8:10])]

            if datetime[10:12].strip():
                self['tms_orig'][0] += ':{}'.format(datetime[10:12])

                if datetime[12:14].strip():
                    self['tms_orig'][0] += ':{}'.format(datetime[12:14])

            if self.conf.has_section('WFPDB'):
                if self.conf.has_option('WFPDB', 'maindata_timezone'):
                    maindata_tz = self.conf.get('WFPDB', 'maindata_timezone')

                    if (maindata_tz.upper() == 'ST' or 
                        maindata_tz.upper() == 'UT'):
                        self['tz_orig'] = maindata_tz.upper()
                    else:
                        self['tz_orig'] = maindata_tz

                    if self['tz_orig'] == 'ST':
                        self['st_start_orig'] = self['tms_orig']
                    elif self['tz_orig'] == 'UT':
                        self['ut_start_orig'] = self['tms_orig']

                    self['tms_orig'] = ['{} {}'.format(self['tz_orig'],
                                                       x) for x in self['tms_orig']]

        self['time_flag'] = maindata_entry[42].strip()
        self['object_name'] = [maindata_entry[43:63].strip()]
        object_type_code = maindata_entry[63:65].strip()

        if object_type_code:
            self['object_type_code'] = [object_type_code]

            if object_type_code in self._object_types:
                self['object_type'] = [self._object_types[object_type_code]]
            else:
                self['object_type'] = [object_type_code]

        try:
            self['method_code'] = int(maindata_entry[65:67])
        except ValueError:
            self['method_code'] = None

        if self['method_code'] and self['method_code'] in self._method_codes:
            self['method'] = self._method_codes[self['method_code']]
        else:
            self['method'] = None

        try:
            self['numexp'] = int(maindata_entry[67:69])
        except ValueError:
            self['numexp'] = 1

        exptime_multiply = 1.

        if self.conf.has_section('WFPDB'):
            if self.conf.has_option('WFPDB', 'exptime_minutes'):
                if self.conf.getboolean('WFPDB', 'exptime_minutes'):
                    exptime_multiply = 60.

        try:
            self['exptime'] = [float(maindata_entry[69:75].strip())
                               *exptime_multiply]
        except ValueError:
            self['exptime'] = []
            
        self['emulsion'] = maindata_entry[75:86].strip()
        self['filter'] = maindata_entry[86:93].strip()
        self['spectral_band'] = maindata_entry[93:95].strip()

        try:
            self['plate_size1'] = int(maindata_entry[95:97])
        except ValueError:
            self['plate_size1'] = None

        try:
            self['plate_size2'] = int(maindata_entry[97:99])
        except ValueError:
            self['plate_size2'] = None

        if self['plate_size1'] and self['plate_size2']:
            self['plate_format'] = '{}x{}'.format(min([self['plate_size1'], self['plate_size2']]),
                                                 max([self['plate_size1'], self['plate_size2']]))

    def parse_notes(self, notes_entry):
        """
        Extract data from the WFPDB notes entry.

        """

        obsnotes = []
        notes = []

        for note in notes_entry[15:].split(';'):
            if not note.strip():
                continue

            # Apply default keyword 'notes'
            if not ':' in note:
                note = 'notes: {}'.format(note)

            if self.conf.has_section('WFPDB'):
                if self.conf.has_option('WFPDB', 'notes_timezone'):
                    notes_tz = self.conf.get('WFPDB', 'notes_timezone')

                    if notes_tz.upper() == 'ST' or notes_tz.upper() == 'UT':
                        self['tz_orig'] = notes_tz.upper()
                    else:
                        self['tz_orig'] = notes_tz

            key, val = (s.strip() for s in note.split(':', 1))

            if key.lower() == 'exposure time':
                self['exptime'] = [str_to_num(x.strip()) 
                                   for x in val.split(',')]
            elif key.lower() == 'exposure start':
                self['tms_orig'] = ['{} {}'.format(self['tz_orig'], 
                                                   x.strip()).strip() 
                                    for x in val.split(',')]

                if self['tz_orig'] == 'ST':
                    self['st_start_orig'] = [x.strip() for x in val.split(',')]
                elif self['tz_orig'] == 'UT':
                    self['ut_start_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'exposure end':
                self['tme_orig'] = ['{} {}'.format(self['tz_orig'], 
                                                   x.strip()).strip() 
                                    for x in val.split(',')]
                #self['tme_orig'] = [x.strip() for x in val.split(',')]

                if self['tz_orig'] == 'ST':
                    self['st_end_orig'] = [x.strip() for x in val.split(',')]
                elif self['tz_orig'] == 'UT':
                    self['ut_end_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'ut-start':
                self['ut_start_orig'] = [x.strip() for x in val.split(',')]
                #self.tms_orig = ['UT {}'.format(x.strip()) for x in val.split(',')]
            elif key.lower() == 'ut-end':
                self['ut_end_orig'] = [x.strip() for x in val.split(',')]
                #self.tme_orig = ['UT {}'.format(x.strip()) for x in val.split(',')]
            elif key.lower() == 'st-start':
                self['st_start_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'st-end':
                self['st_end_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'jd-avg':
                self['jd_avg_orig'] = str_to_num(val.strip())
            elif key.lower() == 'object' or key.lower() == 'object name':
                self['object_name'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'object type':
                self['object_type_code'] = [x.strip() for x in val.split(',')]
                self['object_type'] = []

                for obj_code in self['object_type_code']:
                    if obj_code in self._object_types:
                        self['object_type'].append(self._object_types[obj_code])
                    else:
                        self['object_type'].append(obj_code)
            elif key.lower() == 'ra':
                self['ra_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'dec':
                self['dec_orig'] = [x.strip() for x in val.split(',')]
            elif key.lower() == 'focus':
                self['focus'] = [str_to_num(x.strip()) 
                                 for x in val.split(',')]
            elif key.lower() == 'emulsion':
                self['emulsion'] = val.strip()
            elif key.lower() == 'instrument':
                self['instrument'] = val.strip()
            elif key.lower() == 'prism name':
                self['prism'] = val.strip()
            elif key.lower() == 'prism angle':
                self['prism_angle'] = val.strip()
            elif key.lower() == 'sky conditions':
                self['skycond'] = val.strip()
            elif key.lower() == 'original plate number':
                self['plate_num_orig'] = val.strip()
            elif key.lower() == 'bibcode':
                self['bibcode'] = val.strip()
            elif key.lower()[:13] == 'observer note':
                obsnotes.append(val.strip())
            elif key.lower() == 'notes':
                notes.append(val.strip())
            else:
                notes.append('{}: {}'.format(key.strip(), val.strip()))

        # Combine start times into tms_orig keyword
        self['tms_orig'] = _combine_orig_times(self['tms_orig'], 
                                               self['ut_start_orig'],
                                               self['st_start_orig'])

        # Combine end times into tme_orig keyword
        self['tme_orig'] = _combine_orig_times(self['tme_orig'], 
                                               self['ut_end_orig'],
                                               self['st_end_orig'])

        # Create a string from individual notes
        if obsnotes:
            self['obsnotes'] = '; '.join(obsnotes)

        if notes:
            self['notes'] = '; '.join(notes)

    def parse_quality(self, quality_entry):
        """
        Extract data from the WFPDB quality entry.

        """

        notes = []

        for keyval in quality_entry[15:].split(';'):
            if not ':' in keyval:
                if keyval.strip():
                    self['plate_quality'] = keyval.strip()
                continue

            key, val = [s.strip() for s in keyval.split(':', 1)]

            if key.lower() == 'temperature':
                try:
                    self['temperature'] = float(val.strip())
                except ValueError:
                    self['temperature'] = None
            elif key.lower() == 'calmness':
                self['calmness'] = val.strip()
            elif key.lower() == 'sharpness':
                self['sharpness'] = val.strip()
            elif key.lower() == 'transparency':
                self['transparency'] = val.strip()
            elif key.lower() == 'sky conditions':
                self['skycond'] = val.strip()
            else:
                notes.append('{}: {}'.format(key.strip(), val.strip()))

        # Create a string from individual key-value pairs
        if notes:
            self['notes'] = '; '.join(notes)

    def parse_observer(self, observer_entry):
        """
        Extract data from the WFPDB observer entry.

        """

        if observer_entry[15:].strip():
            self['observer'] = observer_entry[15:].strip()

    def parse_csv(self, val_list, csv_filename=None):
        """
        Extract data from a CSV row.

        """

        if self.conf.has_section(csv_filename):
            # Default list delimiter
            csv_list_delimiter = ','

            # Get list delimiter from configuration file
            if self.conf.has_option(csv_filename, 'csv_list_delimiter'):
                csv_list_delimiter = self.conf.get(csv_filename, 
                                                   'csv_list_delimiter')

            # Get timezone from configuration file
            if self.conf.has_option(csv_filename, 'csv_timezone'):
                csv_timezone = self.conf.get(csv_filename, 'csv_timezone')

                if (csv_timezone.upper() == 'ST' or 
                    csv_timezone.upper() == 'UT'):
                    self['tz_orig'] = csv_timezone.upper()
                else:
                    self['tz_orig'] = csv_timezone

            for (key, pos) in self.conf.items(csv_filename):
                if key in self:
                    try:
                        pos = int(pos)
                    except ValueError:
                        pos = 0

                    if (pos > 0 and pos <= len(val_list)):
                        if key == 'plate_id':
                            val = val_list[pos-1]
                        # Check if keyword can be a list
                        elif _keyword_meta[key][1]:
                            if (_keyword_meta[key][0] is int or
                                _keyword_meta[key][0] is float):
                                val = [str_to_num(x.strip()) for x in 
                                       val_list[pos-1].split(csv_list_delimiter)]
                            else:
                                val = [x.strip() for x in 
                                       val_list[pos-1].split(csv_list_delimiter)]
                                
                                if val == ['']:
                                    val = []
                        elif (_keyword_meta[key][0] is int or 
                              _keyword_meta[key][0] is float):
                            val = str_to_num(val_list[pos-1])
                        else:
                            val = val_list[pos-1]

                        if key == 'numexp' and not val:
                            val = 1

                        # Delete previous values for lists
                        if isinstance(self[key], list):
                            self[key] = []

                        try:
                            self[key].extend(val)
                        except TypeError:
                            self[key].append(val)
                        except AttributeError:
                            self[key] = val

            # Combine start times into tms_orig keyword
            self['tms_orig'] = _combine_orig_times(self['tms_orig'], 
                                                   self['ut_start_orig'],
                                                   self['st_start_orig'])

            # Combine end times into tme_orig keyword
            self['tme_orig'] = _combine_orig_times(self['tme_orig'], 
                                                   self['ut_end_orig'],
                                                   self['st_end_orig'])

    def parse_header(self, header):
        """
        Extract data from FITS header.

        """

        for k,v in _keyword_meta.items():
            if v[1] and v[4]:
                self[k] = []
                
                for i in np.arange(99):
                    nkey = v[4].replace('n', str(i+1))

                    if nkey in header:
                        val = header[nkey]

                        if isinstance(val, fits.card.Undefined):
                            val = None

                        self[k].append(val)
            elif k == 'fits_history':
                if 'HISTORY' in header:
                    self[k].extend(header['HISTORY'])
            elif v[3] and v[3] in header:
                val = header[v[3]]

                if isinstance(val, fits.card.Undefined):
                    val = None

                self[k] = val

        if ('fits_bzero' in self and 'fits_bitpix' in self and 
            self['fits_bzero'] and self['fits_bitpix']):
            if self['fits_bitpix'] == 8:
                self['fits_bscale'] = 1.0
                self['fits_bzero'] = -128
            elif self['fits_bitpix'] == 16:
                self['fits_bscale'] = 1.0
                self['fits_bzero'] = 32768
            elif self['fits_bitpix'] == 32:
                self['fits_bscale'] = 1.0
                self['fits_bzero'] = 2147483648
            elif self['fits_bitpix'] == 64:
                self['fits_bscale'] = 1.0
                self['fits_bzero'] = 9223372036854775808

    def parse_conf(self):
        """
        Extract data from configuration file.

        """

        if self.conf.has_section('Keyword values'):
            for key,val in self.conf.items('Keyword values'):
                if key in self:
                    if (_keyword_meta[key][0] is int or 
                        _keyword_meta[key][0] is float):
                        val = str_to_num(val)

                    try:
                        self[key].append(val)
                    except AttributeError:
                        self[key] = val

        if self.conf.has_section('Keyword lookup'):
            for key in self.conf.options('Keyword lookup'):
                try:
                    replace_key = self.conf.getboolean('Keyword lookup', key)
                except ValueError:
                    replace_key = False

                if (replace_key and key in self and self[key] and 
                    self.conf.has_section(key)):
                    if self.conf.has_option(key, self[key]):
                        self[key] = self.conf.get(key, self[key])

    def compute_values(self):
        """
        Compute UT time from sidereal time and precess coordinates from plate
        epoch to J2000.

        """

        # By default, assume that date refers to observation time, not evening
        evening_date = False

        if self.conf.has_section('Metadata'):
            if self.conf.has_option('Metadata', 'evening_date'):
                evening_date = self.conf.getboolean('Metadata', 'evening_date')

        # Check if observation date is given
        if self['date_orig']:
            tms_missing = False

            if isinstance(self['tms_orig'], list):
                ntimes = len(self['tms_orig'])

                if ntimes == 0:
                    tms_missing = True
                    ntimes = 1
            else:
                ntimes = 1

            # Update number of exposures if the number of timestamps is 
            # different
            if (ntimes > self['numexp'] or (ntimes < self['numexp'] and 
                                            not tms_missing)):
                self['numexp'] = ntimes
                
            self['numsub'] = []
            self['date_obs'] = []
            self['jd'] = []
            self['year'] = []
            self['date_avg'] = []
            self['jd_avg'] = []
            self['hjd_avg'] = []
            self['year_avg'] = []
            self['date_weighted'] = []
            self['jd_weighted'] = []
            self['hjd_weighted'] = []
            self['year_weighted'] = []
            self['date_end'] = []
            self['jd_end'] = []
            self['year_end'] = []
            self.exposures = []
            exptime_calc = []

            for iexp in np.arange(ntimes):
                if (isinstance(self['date_orig'], list) and 
                    len(self['date_orig']) > 1):
                    date_orig = unidecode.unidecode(self['date_orig'][iexp])
                elif isinstance(self['date_orig'], list):
                    date_orig = unidecode.unidecode(self['date_orig'][0])
                else:
                    date_orig = unidecode.unidecode(self['date_orig'])

                try:
                    date_test = Time(date_orig, scale='tai')
                except ValueError:
                    # Cannot work without valid date_orig
                    continue

                date_start_orig = date_orig
                date_end_orig = date_orig
                tz_orig = self['tz_orig']

                if self['tms_orig']:
                    tms_orig = self['tms_orig'][iexp]
                    tms_orig, uts_orig, sts_orig = _split_orig_times(tms_orig)
                else:
                    tms_orig = None

                if self['tme_orig']:
                    tme_orig = self['tme_orig'][iexp]
                    tme_orig, ute_orig, ste_orig = _split_orig_times(tme_orig)
                else:
                    tme_orig = None

                # If UT times are given separately
                if (not tms_orig and not tme_orig and 
                        (self['ut_start_orig'] or self['ut_end_orig'])):
                    tms_orig = self['ut_start_orig'][iexp]
                    tme_orig = self['ut_end_orig'][iexp]
                    tz_orig = 'UT'

                # If ST times are given separately
                if (not tms_orig and not tme_orig and 
                        (self['st_start_orig'] or self['st_end_orig'])):
                    tms_orig = self['st_start_orig'][iexp]
                    tme_orig = self['st_end_orig'][iexp]
                    tz_orig = 'ST'

                if tms_orig and '|' in tms_orig:
                    tms_orig = [x.strip() for x in tms_orig.split('|')]

                if tme_orig and '|' in tme_orig:
                    tme_orig = [x.strip() for x in tme_orig.split('|')]

                try:
                    exptime = self['exptime'][iexp]
                except Exception:
                    exptime = None
                            
                if (isinstance(tms_orig, list) or isinstance(tme_orig, list)):
                    expmeta = self.copy()
                    expmeta.conf = self.conf
                    expmeta['tms_orig'] = tms_orig
                    expmeta['tme_orig'] = tme_orig
                    expmeta['numexp'] = len(tms_orig)
                    expmeta['exptime'] = []
                    expmeta.compute_values()
                    self.exposures.append(expmeta)

                    # Copy sub-exposure values
                    self['numsub'].append(expmeta['numexp'])
                    self['date_obs'].append(expmeta['date_obs'][0])
                    self['jd'].append(expmeta['jd'][0])
                    self['year'].append(expmeta['year'][0])
                    self['date_end'].append(expmeta['date_end'][-1])
                    self['jd_end'].append(expmeta['jd_end'][-1])
                    self['year_end'].append(expmeta['year_end'][-1])

                    # Check if exptimes exist for all sub-exposures
                    if (len(filter(None, expmeta['exptime'])) == 
                        expmeta['numexp']):
                        exptime_calc.append(sum(expmeta['exptime']))
                        jd_weighted = np.average(expmeta['jd_avg'],
                                                 weights=expmeta['exptime'])
                        year_weighted = np.average(expmeta['year_avg'],
                                                   weights=expmeta['exptime'])
                        time_weighted = Time(jd_weighted, format='jd', 
                                             scale='ut1', precision=0)
                        self['date_weighted'].append(time_weighted.isot)
                        self['jd_weighted'].append(jd_weighted)
                        self['year_weighted'].append(year_weighted)
                    else:
                        exptime_calc.append(None)
                        self['date_weighted'].append(None)
                        self['jd_weighted'].append(None)
                        self['year_weighted'].append(None)

                    jd_avg = np.mean([expmeta['jd'][0], expmeta['jd_end'][-1]])
                    year_avg = np.mean([expmeta['year'][0],
                                        expmeta['year_end'][-1]])
                    time_avg = Time(jd_avg, format='jd', scale='ut1', 
                                    precision=0)
                    self['date_avg'].append(time_avg.isot)
                    self['jd_avg'].append(float('{:.5f}'.format(jd_avg)))
                    self['year_avg'].append(float('{:.8f}'.format(year_avg)))
                else:
                    self['numsub'].append(1)
                    self.exposures.append(None)

                    ut_start_isot = None
                    ut_end_isot = None

                    # Handle cases where time hours are larger than 24, 
                    # or date refers to evening
                    if tms_orig or tme_orig:
                        t_date_orig = Time(date_orig, scale='tai')

                        if tme_orig:
                            if tme_orig.count(':') == 1:
                                tme_orig += ':00'

                            tsec = pytimeparse.parse(tme_orig)

                            if evening_date and tsec < 43200:
                                tsec += 86400

                            if tsec >= 86400:
                                td_tme = TimeDelta(tsec, format='sec')
                                t_tme = t_date_orig + td_tme
                                date_orig = Time(t_tme, out_subfmt='date').iso
                                date_end_orig = date_orig

                                if '.' in tme_orig:
                                    tme_orig = Time(t_tme).iso.split()[-1]
                                else:
                                    tme_orig = Time(t_tme, 
                                                    precision=0).iso.split()[-1]

                        if tms_orig:
                            if tms_orig.count(':') == 1:
                                tms_orig += ':00'

                            tsec = pytimeparse.parse(tms_orig)
                            
                            if evening_date and tsec < 43200:
                                tsec += 86400

                            if tsec >= 86400:
                                td_tms = TimeDelta(tsec, format='sec')
                                t_tms = t_date_orig + td_tms
                                date_orig = Time(t_tms, out_subfmt='date').iso
                                date_start_orig = date_orig

                                if '.' in tms_orig:
                                    tms_orig = Time(t_tms).iso.split()[-1]
                                else:
                                    tms_orig = Time(t_tms, 
                                                    precision=0).iso.split()[-1]

                    if ((tz_orig == 'ST') and (tms_orig or tme_orig) and 
                        self['site_latitude'] and self['site_longitude']):
                        # Convert sidereal time to UT using pyEphem's next_transit
                        # Initialize location and date
                        loc = ephem.Observer()
                        loc.lat = str(self['site_latitude'])
                        loc.lon = str(self['site_longitude'])

                        if self['site_elevation']:
                            loc.elevation = self['site_elevation']
                            
                        loc.date = '%s %s' % (date_orig, '12:00:00')
                        # Define imaginary star with the RA value of the ST of observation,
                        # and the Dec value of 0. Then compute transit time.
                        st = ephem.FixedBody()
                        st._dec = 0.
                        st._epoch = date_orig

                        if tms_orig:
                            st._ra = ephem.hours(tms_orig)
                            st.compute()
                            ut_start_isot = '%04d-%02d-%02dT%02d:%02d:%02d' % loc.next_transit(st).tuple()

                        if tme_orig:
                            st._ra = ephem.hours(tme_orig)
                            st.compute()
                            ut_end_isot = '%04d-%02d-%02dT%02d:%02d:%02d' % loc.next_transit(st).tuple()

                    elif (tz_orig == 'UT' or 
                          (pytz_available and tz_orig in pytz.all_timezones)):
                        # Handle UT times and local times with specified 
                        # pytz-compatible timezones
                        ut_start_orig = None
                        ut_end_orig = None

                        if self['ut_start_orig'] or tms_orig:
                            if self['ut_start_orig'] and not tms_orig:
                                ut_start_orig = self['ut_start_orig'][iexp]
                            else:
                                if tz_orig == 'UT':
                                    ut_start_orig = tms_orig
                                else:
                                    str_start = '{} {}'.format(date_start_orig,
                                                               tms_orig)
                                    dt_start = (Time(str_start, scale='tai', 
                                                     format='iso')
                                                .datetime)
                                    dt_local = (pytz.timezone(tz_orig)
                                                .localize(dt_start))
                                    date_start_orig = (dt_local.astimezone(pytz.utc)
                                                       .strftime('%Y-%m-%d'))
                                    ut_start_orig = (dt_local.astimezone(pytz.utc)
                                                     .strftime('%H:%M:%S'))

                            if not ':' in ut_start_orig:
                                ut_start_orig += ':00'

                            ut_start_isot = '{}T{}'.format(date_start_orig, 
                                                           ut_start_orig)

                            # Make sure that ut_start_isot formatting is correct
                            if not '.' in ut_start_orig:
                                ut_start_isot = Time(ut_start_isot, format='isot', 
                                                     scale='ut1', precision=0).isot

                        if self['ut_end_orig'] or tme_orig:
                            if self['ut_end_orig'] and not tme_orig:
                                ut_end_orig = self['ut_end_orig'][iexp]
                            else:
                                if tz_orig == 'UT':
                                    ut_end_orig = tme_orig
                                else:
                                    str_end = '{} {}'.format(date_end_orig,
                                                             tme_orig)
                                    dt_end = (Time(str_end, scale='tai', 
                                                   format='iso')
                                              .datetime)
                                    dt_local = (pytz.timezone(tz_orig)
                                                .localize(dt_end))
                                    date_end_orig = (dt_local.astimezone(pytz.utc)
                                                     .strftime('%Y-%m-%d'))
                                    ut_end_orig = (dt_local.astimezone(pytz.utc)
                                                   .strftime('%H:%M:%S'))

                            if not ':' in ut_end_orig:
                                ut_end_orig += ':00'

                            ut_end_isot = '{}T{}'.format(date_end_orig, 
                                                         ut_end_orig)

                            # Check if end time is after midnight. If so, use 
                            # next date.
                            if ut_start_isot and ut_end_isot < ut_start_isot:
                                next_day = (Time(ut_end_isot, format='isot', 
                                                 scale='ut1', precision=0) 
                                            + TimeDelta(1, format='jd'))
                                date_end = next_day.isot.split('T')[0]
                                ut_end_isot = '{}T{}'.format(date_end, ut_end_orig)

                            # Make sure that ut_start_isot formatting is correct
                            if not '.' in ut_end_orig:
                                ut_end_isot = Time(ut_end_isot, format='isot',
                                                   scale='ut1', precision=0).isot

                    if not tms_orig and not self['ut_start_orig']:
                        ut_start_isot = Time(date_orig, format='iso', scale='ut1', 
                                             out_subfmt='date').iso

                    if ut_start_isot:
                        self['date_obs'].append(ut_start_isot)
                        time_start = Time(ut_start_isot, format='isot', scale='ut1')

                        if self['ut_start_orig'] or tms_orig:
                            self['jd'].append(float('%.5f' % time_start.jd))
                            self['year'].append(float('%.8f' % time_start.jyear))
                        else:
                            self['jd'].append(float('%.0f' % time_start.jd))
                            self['year'].append(float('%.3f' % time_start.jyear))
                    else:
                        self['date_obs'].append(None)
                        self['jd'].append(None)
                        self['year'].append(None)

                    if ut_end_isot:
                        self['date_end'].append(ut_end_isot)
                        time_end = Time(ut_end_isot, format='isot', scale='ut1')

                        if self['ut_end_orig'] or tme_orig:
                            self['jd_end'].append(float('{:.5f}'
                                                        .format(time_end.jd)))
                            self['year_end'].append(float('{:.8f}'
                                                          .format(time_end.jyear)))
                        else:
                            self['jd_end'].append(None)
                            self['year_end'].append(None)
                    else:
                        self['date_end'].append(None)
                        self['jd_end'].append(None)
                        self['year_end'].append(None)

                    if ut_start_isot and ut_end_isot:
                        jd_avg = np.mean([time_start.jd, time_end.jd])
                        year_avg = np.mean([time_start.jyear, time_end.jyear])
                        time_avg = Time(jd_avg, format='jd', scale='ut1', 
                                        precision=0)
                        self['date_avg'].append(time_avg.isot)
                        self['date_weighted'].append(time_avg.isot)
                        jd_avg = float('{:.5f}'.format(jd_avg))
                        self['jd_avg'].append(jd_avg)
                        self['jd_weighted'].append(jd_avg)
                        year_avg = float('{:.8f}'.format(year_avg))
                        self['year_avg'].append(year_avg)
                        self['year_weighted'].append(year_avg)
                    elif ut_start_isot and exptime:
                        time_avg = Time(time_start.jd + 0.5 * exptime / 86400.,
                                        format='jd', scale='ut1', precision=0)
                        jd_avg = float('{:.5f}'.format(time_avg.jd))
                        year_avg = float('{:.8f}'.format(time_avg.jyear))
                        self['date_avg'].append(time_avg.isot)
                        self['date_weighted'].append(time_avg.isot)
                        self['jd_avg'].append(jd_avg)
                        self['jd_weighted'].append(jd_avg)
                        self['year_avg'].append(year_avg)
                        self['year_weighted'].append(year_avg)
                    else:
                        self['date_avg'].append(None)
                        self['date_weighted'].append(None)
                        self['jd_avg'].append(None)
                        self['jd_weighted'].append(None)
                        self['year_avg'].append(None)
                        self['year_weighted'].append(None)

                    if ut_start_isot and ut_end_isot:
                        tdiff = (Time(ut_end_isot, format='isot', 
                                      scale='tai') -
                                 Time(ut_start_isot, format='isot', 
                                      scale='tai'))
                        exptime_calc.append(tdiff.sec)
                    else:
                        exptime_calc.append(None)

            if self['exptime'] == [] and filter(None, exptime_calc) != []:
                self['exptime'] = exptime_calc

        if self['ra_orig'] and self['dec_orig'] and self['date_orig']:
            for iexp in np.arange(len(self['ra_orig'])):
                if len(self['date_orig']) > 1:
                    date_orig = self['date_orig'][iexp]
                else:
                    date_orig = self['date_orig'][0]

                pointing = ephem.FixedBody()
                pointing._ra = ephem.hours(str(self['ra_orig'][iexp]))
                pointing._dec = ephem.degrees(str(self['dec_orig'][iexp]))
                pointing._epoch = str(date_orig)
                pointing.compute(ephem.J2000)

                try:
                    ra_str = Angle(pointing.ra, units.radian)\
                            .to_string(unit=units.hour, sep=':', precision=1, 
                                       pad=True)
                    dec_str = Angle(pointing.dec, units.radian)\
                            .to_string(unit=units.deg, sep=':', precision=1, 
                                       pad=True)
                    ra_deg = float(Angle(pointing.ra, units.radian)
                                   .to_string(unit=units.deg, decimal=True,
                                              precision=4))
                    dec_deg = float(Angle(pointing.dec, units.radian)
                                   .to_string(unit=units.deg, decimal=True, 
                                              precision=4))
                except AttributeError:
                    ra_str = Angle(pointing.ra, units.radian)\
                            .format(unit='hour', sep=':', precision=1, pad=True)
                    dec_str = Angle(pointing.dec, units.radian)\
                            .format(sep=':', precision=1, pad=True)
                    ra_deg = float('{:.4f}'.format(Angle(pointing.ra, 
                                                         units.radian).degrees))
                    dec_deg = float('{:.4f}'.format(Angle(pointing.dec, 
                                                          units.radian).degrees))

                self['ra'].append(ra_str)
                self['dec'].append(dec_str)
                self['ra_deg'].append(ra_deg)
                self['dec_deg'].append(dec_deg)

                # Calculate heliocentric correction for the plate center
                if (self['site_latitude'] and self['site_longitude']):
                    lon = self['site_longitude']
                    lat = self['site_latitude']
                    elev = 0.

                    if self['site_elevation']:
                        elev = self['site_elevation']

                    loc = EarthLocation.from_geodetic(lon, lat, elev)
                    coord = SkyCoord(ra_deg, dec_deg, unit='deg')

                    if self['jd_avg'][iexp]:
                        tavg = Time(self['jd_avg'][iexp], format='jd', 
                                    scale='ut1')
                        tavg_ltt = tavg.light_travel_time(coord, 
                                                          'heliocentric',
                                                          location=loc)
                        hjd_avg = float('{:.5f}'.format((tavg + tavg_ltt).jd))
                        self['hjd_avg'].append(hjd_avg)
                    else:
                        self['hjd_avg'].append(None)

                    if self['jd_weighted'][iexp]:
                        tw = Time(self['jd_weighted'][iexp], format='jd', 
                                  scale='ut1')
                        tw_ltt = tw.light_travel_time(coord, 'heliocentric',
                                                      location=loc)
                        hjd_weighted = float('{:.5f}'.format((tw + tw_ltt).jd))
                        self['hjd_weighted'].append(hjd_weighted)
                    else:
                        self['hjd_weighted'].append(None)

        self['date'] = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

    # Create alias for compute_values
    calculate = compute_values


class PlateHeader(fits.Header):
    """
    Plate header class. Based on FITS header class from astropy.io.

    """

    def __init__(self, *args, **kwargs):
        fits.Header.__init__(self, *args, **kwargs)
        self.platemeta = PlateMeta()
        self.conf = ConfigParser.ConfigParser()
        self.fits_dir = ''
        self.write_fits_dir = ''
        self.write_header_dir = ''
        self.create_checksum = False
        self.fits_datetime = None
        self.fits_size = None
        self.fits_checksum = None
        self.fits_datasum = None
        self.german_transliteration = False
        
    _default_comments = {'SIMPLE':   'file conforms to FITS standard',
        'BITPIX':   'number of bits per data pixel',
        'NAXIS':    'number of data axes',
        'NAXIS1':   'length of data axis 1',
        'NAXIS2':   'length of data axis 2',
        'BSCALE':   'physical_value = BZERO + BSCALE * array_value',
        'BZERO':    'physical_value = BZERO + BSCALE * array_value',
        'MINVAL':   'minimum image value',
        'MAXVAL':   'maximum image value',
        'EXTEND':   'file may contain extensions',
        'DATEORIG': 'recorded date of the observation',
        'DATEORn':  'recorded date of exposure {}',
        'TMS-ORIG': 'recorded time of the start of exposure 1',
        'TMS-ORn':  'recorded time of the start of exposure {}',
        'TME-ORIG': 'recorded time of the end of exposure 1',
        'TME-ORn':  'recorded time of the end of exposure {}',
        'TZ-ORIG':  'time zone of the recorded time (ST = sidereal)',
        'JDA-ORIG': 'recorded Julian date, mid-point of exposure 1',
        'JDA-ORn':  'recorded Julian date, mid-point of exposure {}',
        'TIMEFLAG': 'quality flag of recorded time',
        #'RA-ORIG':  'recorded right ascension of telescope pointing',
        #'DEC-ORIG': 'recorded declination of telescope pointing',
        'RA-ORIG':  'recorded right ascension of exposure 1',
        'DEC-ORIG': 'recorded declination of exposure 1',
        'RA-ORn':   'recorded right ascension of exposure {}',
        'DEC-ORn':  'recorded declination of exposure {}',
        'COORFLAG': 'quality flag of recorded coordinates',
        #'OBJECT':   'name of the observed object or field',
        'OBJECT':   'observed object or field (exposure 1)',
        'OBJECTn':  'observed object or field (exposure {})',
        'OBJTYPE':  'object type',
        'OBJTYPn':  'object type (exposure {})',
        'EXPTIME':  '[s] exposure time of exposure 1',
        'EXPTIMn':  '[s] exposure time of exposure {}',
        'NUMEXP':   'number of exposures of the plate',
        'OBSERVAT': 'observatory name',
        'SITENAME': 'observatory site',
        'SITELONG': '[deg] East longitude of the observatory',
        'SITELAT':  '[deg] latitude of the observatory',
        'SITEELEV': '[m] elevation of the observatory',
        'TELESCOP': 'telescope name',
        'OTA-NAME': 'optical tube assembly (OTA)',
        'OTA-DIAM': '[m] diameter of the OTA',
        'OTA-APER': '[m] clear aperture of the OTA',
        'FOCLEN':   '[m] focal length of the OTA',
        'PLTSCALE': '[arcsec/mm] plate scale of the OTA',
        'INSTRUME': 'instrument',
        'DETNAM':   'detector',
        'METHOD':   'method of observation',
        'FILTER':   'filter type',
        'PRISM':    'objective prism',
        'PRISMANG': 'prism angle "deg:min"',
        'DISPERS':  '[Angstrom/mm] dispersion',
        'GRATING':  'grating',
        'FOCUS':    'focus position',
        'FOCUSn':   'focus position (exposure {})',
        'TEMPERAT': '[deg C] air temperature (degrees Celsius)',
        'CALMNESS': 'sky calmness (scale 1-5)',
        'SHARPNES': 'sky sharpness (scale 1-5)',
        'TRANSPAR': 'sky transparency (scale 1-5)',
        'SKYCOND':  'other sky conditions',
        'OBSERVER': 'observer name',
        'OBSNOTES': 'observer notes',
        'NOTES':    'miscellaneous notes',
        'BIBCODE':  'bibcode of a related paper',
        'PLATENUM': 'plate number in archive',
        'PNUMORIG': 'original plate number in archive',
        'WFPDB-ID': 'plate identification in the WFPDB',
        'SERIES':   'plate series',
        'PLATEFMT': 'plate format in cm',
        'PLATESZ1': '[cm] plate size along axis 1',
        'PLATESZ2': '[cm] plate size along axis 2',
        'FOV1':     '[deg] field of view along axis 1',
        'FOV2':     '[deg] field of view along axis 2',
        'EMULSION': 'photographic emulsion type',
        'DEVELOP':  'plate development details',
        'PQUALITY': 'quality of plate',
        'PLATNOTE': 'plate notes',
        #'DATE-OBS': 'UT date of the start of the observation',
        'DATE-OBS': 'UT date of the start of exposure 1',
        'DT-OBSn':  'UT date of the start of exposure {}',
        'DATE-AVG': 'UT date of the mid-point of exposure 1',
        'DT-AVGn':  'UT date of the mid-point of exposure {}',
        'DATE-END': 'UT date of the end of exposure 1',
        'DT-ENDn':  'UT date of the end of exposure {}',
        'YEAR':     'decimal year of the start of exposure 1',
        'YEARn':    'decimal year of the start of exposure {}',
        'YEAR-AVG': 'decimal year of the mid-point of exposure 1',
        'YR-AVGn':  'decimal year of the mid-point of exposure {}',
        'YEAR-END': 'decimal year of the end of exposure 1',
        'YR-ENDn':  'decimal year of the end of exposure {}',
        'JD':       'Julian date at the start of exposure 1',
        'JDn':      'Julian date at the start of exposure {}',
        'JD-AVG':   'Julian date at the mid-point of exposure 1',
        'JD-AVGn':  'Julian date at the mid-point of exposure {}',
        'JD-END':   'Julian date at the end of exposure 1',
        'JD-ENDn':  'Julian date at the end of exposure {}',
        'HJD-AVG':  'heliocentric JD at the mid-point of exposure 1',
        'HJD-AVn':  'heliocentric JD at the mid-point of exposure {}',
        'RA':       'right ascension of pointing (J2000) "h:m:s"',
        'DEC':      'declination of pointing (J2000) "d:m:s"',
        'RAn':      'right ascension of exposure {} (J2000) "h:m:s"',
        'DECn':     'declination of exposure {} (J2000) "d:m:s"',
        'RA_DEG':   '[deg] right ascension of pointing (J2000)',
        'DEC_DEG':  '[deg] declination of pointing (J2000)',
        'RA_DEGn':  '[deg] right ascension of exposure {} (J2000)',
        'DEC_DEn':  '[deg] declination of exposure {} (J2000)',
        'SCANNER':  'scanner name',
        'SCANRES1': '[dpi] scan resolution along axis 1',
        'SCANRES2': '[dpi] scan resolution along axis 2',
        'PIXSIZE1': '[um] pixel size along axis 1',
        'PIXSIZE2': '[um] pixel size along axis 2',
        'SCANSOFT': 'scanning software',
        'SCANHCUT': 'scan high-cut value',
        'SCANLCUT': 'scan low-cut value',
        'SCANGAM':  'scan gamma value',
        'SCANFOC':  'scan focus',
        'WEDGE':    'type of photometric step-wedge',
        'DATESCAN': 'scan date and time',
        'SCANAUTH': 'author of scan',
        'SCANNOTE': 'scan notes',
        'FILENAME': 'filename of the plate scan',
        'FN-SCNn':  'filename of scan {}',
        'FN-WEDGE': 'filename of the wedge scan',
        'FN-PRE':   'filename of the preview image',
        'FN-COVER': 'filename of the plate cover image',
        #'FILENOTE': 'filename of the observer notes image',
        'FN-LOGn':  'filename of logbook image {}',
        #'LICENCE':  'licence of data files',
        'DATE':     'last change of this file',
        'WCSAXES':  'number of axes in the WCS description',
        'RADESYS':  'name of the reference frame',
        'EQUINOX':  'epoch of the mean equator and equinox in years',
        'CTYPE1':   'TAN (gnomonic) projection',
        'CTYPE2':   'TAN (gnomonic) projection',
        'CUNIT1':   'physical units of CRVAL and CDELT for axis 1',
        'CUNIT2':   'physical units of CRVAL and CDELT for axis 2',
        'CRPIX1':   'reference pixel for axis 1',
        'CRPIX2':   'reference pixel for axis 2',
        'CRVAL1':   'right ascension at the reference point',
        'CRVAL2':   'declination at the reference point',
        'CD1_1':    'transformation matrix',
        'CD1_2':    'transformation matrix',
        'CD2_1':    'transformation matrix',
        'CD2_2':    'transformation matrix',
        'LONPOLE':  'native longitude of the celestial pole'}

    _default_order = ['SIMPLE', 
        'BITPIX',
        'NAXIS',
        'NAXIS1',
        'NAXIS2',
        'BSCALE',
        'BZERO',
        'MINVAL',
        'MAXVAL',
        'EXTEND',
        'sep:Original data of the observation',
        'DATEORIG',
        'DATEORn',
        'TMS-ORIG',
        'TMS-ORn',
        'TME-ORIG',
        'TME-ORn',
        'TZ-ORIG',
        'JDA-ORIG',
        'JDA-ORn',
        'TIMEFLAG',
        'RA-ORIG',
        'DEC-ORIG',
        'RA-ORn',
        'DEC-ORn',
        'COORFLAG',
        'OBJECT',
        'OBJECTn',
        'OBJTYPE',
        'OBJTYPn',
        'EXPTIME',
        'NUMEXP',
        'EXPTIMn',
        'OBSERVAT',
        'SITENAME',
        'SITELONG',
        'SITELAT',
        'SITEELEV',
        'TELESCOP',
        'OTA-NAME',
        'OTA-DIAM',
        'OTA-APER',
        'FOCLEN',
        'PLTSCALE',
        'INSTRUME',
        'DETNAM',
        'METHOD',
        'FILTER',
        'PRISM',
        'PRISMANG',
        'DISPERS',
        'GRATING',
        'FOCUS',
        'FOCUSn',
        'TEMPERAT',
        'CALMNESS',
        'SHARPNES',
        'TRANSPAR',
        'SKYCOND',
        'OBSERVER',
        'OBSNOTES',
        'NOTES',
        'BIBCODE',
        'sep:Photographic plate',
        'PLATENUM',
        'PNUMORIG',
        'WFPDB-ID',
        'SERIES',
        'PLATEFMT',
        'PLATESZ1',
        'PLATESZ2',
        'FOV1',
        'FOV2',
        'EMULSION',
        'DEVELOP',
        'PQUALITY',
        'PLATNOTE',
        'sep:Computed data of the observation',
        'DATE-OBS',
        'DT-OBSn',
        'DATE-AVG',
        'DT-AVGn',
        'DATE-END',
        'DT-ENDn',
        'YEAR',
        'YEARn',
        'YEAR-AVG',
        'YR-AVGn',
        'YEAR-END',
        'YR-ENDn',
        'JD',
        'JDn',
        'JD-AVG',
        'JD-AVGn',
        'JD-END',
        'JD-ENDn',
        'HJD-AVG',
        'HJD-AVn',
        'RA',
        'DEC',
        'RAn',
        'DECn',
        'RA_DEG',
        'DEC_DEG',
        'RA_DEGn',
        'DEC_DEn',
        'sep:Scan',
        'SCANNER',
        'SCANRES1',
        'SCANRES2',
        'PIXSIZE1',
        'PIXSIZE2',
        'SCANSOFT',
        'SCANHCUT',
        'SCANLCUT',
        'SCANGAM',
        'SCANFOC',
        'WEDGE',
        'DATESCAN',
        'SCANAUTH',
        'SCANNOTE',
        'sep:Data files',
        'FILENAME',
        'FN-SCNn',
        'FN-WEDGE',
        'FN-PRE',
        'FN-COVER',
        'FN-LOGn',
        'ORIGIN',
        'DATE',
        'sep:WCS',
        'sep:Licence',
        'LICENCE',
        'sep:Acknowledgements',
        'sep:History',
        'HISTORY',
        'sep:Checksums',
        'CHECKSUM',
        'DATASUM',
        'sep:']

    def assign_conf(self, conf):
        """
        Assign configuration to the plate header.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        self.conf = conf

        for attr in ['fits_dir', 'write_fits_dir', 'write_header_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['create_checksum']:
            try:
                setattr(self, attr, conf.getboolean('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['german_transliteration']:
            try:
                setattr(self, attr, conf.getboolean('Metadata', attr))
            except ConfigParser.Error:
                pass

    def assign_platemeta(self, platemeta):
        """
        Assign plate metadata.

        """

        self.platemeta = platemeta

    @classmethod
    def from_fits(cls, filename):
        """
        Read header from FITS file.

        """

        pheader = cls.fromfile(filename)
        pheader.add_history('Header imported from FITS with PyPlate v{} at {}'
                            .format(__version__, dt.datetime.utcnow()
                                    .strftime('%Y-%m-%dT%H:%M:%S')))
        return pheader

    @classmethod
    def from_hdrf(cls, filename):
        """
        Read header from the output file of header2011.

        """

        pheader = cls.fromfile(filename, sep='\n', endcard=False, 
                               padding=False)
        pheader.add_history('Header imported from file with PyPlate v{} at {}'
                            .format(__version__, dt.datetime.utcnow()
                                    .strftime('%Y-%m-%dT%H:%M:%S')))
        return pheader

    def _update_keyword(self, key, valtype, value):
        """
        Update header keyword.

        """

        if value or (valtype is int and value == 0):
            if isinstance(value, unicode):
                if self.german_transliteration:
                    value = (value.replace(u'\xc4','Ae').replace(u'\xe4','ae')
                             .replace(u'\xd6','Oe').replace(u'\xf6','oe')
                             .replace(u'\xdc','Ue').replace(u'\xfc','ue'))

                value = unidecode.unidecode(value)

            if isinstance(value, str):
                if self.german_transliteration:
                    value = (value.replace('ä','ae').replace('Ä','Ae')
                             .replace('ö','oe').replace('Ö','Oe')
                             .replace('ü','ue').replace('Ü','Ue')
                             .replace('ß','ss'))

                value = (value.replace('ä','a').replace('Ä','A')
                         .replace('ö','o').replace('Ö','O')
                         .replace('ü','u').replace('Ü','U')
                         .replace('õ','o').replace('Õ','O')
                         .replace('š','s').replace('Š','S')
                         .replace('ž','z').replace('Ž','Z'))

                # Workaround for Astropy treating colon in string as assignment
                # of numerical value
                if ':' in value:
                    self.set(key, '')

            self.set(key, value)
        elif not key in self:
            if valtype is str:
                self.set(key, '')
            else:
                self.append(fits.Card.fromstring('{:8s}='.format(key)))

    def _update_keyword_list(self, skey, nkey, valtype, value):
        """
        Update a list of keywords in the header.

        """

        if value:
            if isinstance(value, list):
                if skey:
                    if skey == 'HISTORY':
                        for histitem in value:
                            self.add_history(histitem)
                    else:
                        self.set(skey, value[0])

                if (len(value) > 1 or (len(value) == 1 and not skey)) and nkey:
                    for iexp in np.arange(len(value)):
                        keyname = nkey.replace('n', '{}').format(iexp+1)

                        if value[iexp]:
                            self.set(keyname, value[iexp])
                        else:
                            self.append(fits.Card.fromstring('{:8s}='
                                                             .format(keyname)))
            elif skey:
                self.set(skey, value)
        elif skey and not skey in self:
            if valtype is str:
                self.set(skey, '')
            else:
                self.append(fits.Card.fromstring('{:8s}='.format(skey)))

    def populate(self):
        """
        Populate header with blank cards.

        """

        if self.__len__() == 0:
            self.add_history('Header created with PyPlate v{} at {}'
                             .format(__version__, dt.datetime.utcnow()
                                     .strftime('%Y-%m-%dT%H:%M:%S')))

        for k,v in _keyword_meta.items():
            if v[1]:
                self._update_keyword_list(v[3], v[4], v[0], None)
            elif v[3]:
                self._update_keyword(v[3], v[0], None)

        _default_header_values = OrderedDict([('SIMPLE', True),
                                              ('BITPIX', 16),
                                              ('NAXIS', 2),
                                              ('NAXIS1', 0),
                                              ('NAXIS2', 0),
                                              ('BSCALE', 1.0),
                                              ('BZERO', 32768),
                                              ('EXTEND', True)])

        for k in _default_header_values:
            if k not in self or (k in self and 
                                 isinstance(self[k], fits.card.Undefined)):
                v = _default_header_values[k]
                self._update_keyword(k, type(v), v)

        self.format()

    def update_from_platemeta(self, platemeta=None):
        """
        Update header with plate metadata.

        """

        if platemeta is None:
            platemeta = self.platemeta

        for k,v in _keyword_meta.items():
            if k in platemeta:
                if v[1]:
                    self._update_keyword_list(v[3], v[4], v[0], 
                                              platemeta[k])
                elif v[3]:
                    self._update_keyword(v[3], v[0], platemeta[k])
                elif k == 'fits_acknowledgements':
                    # Get acknowledgements from plate metadata
                    ack = platemeta[k]
                    ack = '\n\n'.join([textwrap.fill(ackpara, 72) 
                                       for ackpara in ack.split('\n\n')])
                    ack_sep = ' Acknowledgements'.rjust(72, '-')

                    # If acknowledgements section exists, remove it first
                    if ack_sep in self.values():
                        ack_ind = self.values().index(ack_sep) + 1

                        for i,c in enumerate(self.cards[ack_ind:]):
                            if c[0] == 'COMMENT':
                                del self[ack_ind]
                            else:
                                break

                        for ackline in ack.split('\n'):
                            self.insert(ack_ind, ('COMMENT', ackline))
                            ack_ind += 1
                    else:
                        # Start section with a separator/title line
                        self.append(('', ack_sep), end=True)

                        # Write acknowledgements with COMMENT keywords
                        for ackline in ack.split('\n'):
                            self.append(('COMMENT', ackline), end=True)

                        # End section with a separator line
                        self.append(('', '-' * 72), end=True)

        self.add_history('Header updated with PyPlate v{} at {}'
                         .format(__version__, dt.datetime.utcnow()
                                 .strftime('%Y-%m-%dT%H:%M:%S')))
        self.format()

    def update_from_fits(self, filename):
        """
        Update header with header values in a FITS file.

        """

        fn_fits = os.path.join(self.fits_dir, filename)

        try:
            h = fits.getheader(fn_fits)
        except IOError:
            print 'Error reading file {}'.format(fn_fits)
            return

        for k,v,c in h.cards:
            if k in self:
                self.set(k, v)
            else:
                self.append(c, bottom=True)

        self.add_history('Header updated from FITS with PyPlate v{} at {}'
                         .format(__version__, dt.datetime.utcnow()
                                 .strftime('%Y-%m-%dT%H:%M:%S')))
        self.format()
        
    def update_values(self):
        """
        Edit keyword values based on configuration.

        """

        self.append(fits.Card.fromstring('PLATESZ1='))
        self.append(fits.Card.fromstring('PLATESZ2='))
        
        if self.conf.has_section('Parse keywords'):
            if self.conf.has_option('Parse keywords', 'plate_size'):
                plate_size_key = self.conf.get('Parse keywords', 'plate_size')

                if plate_size_key in self:
                    if self[plate_size_key] != '':
                        plate_size = self[plate_size_key].split('x')
                        self.set('PLATESZ1', float(plate_size[0]))
                        self.set('PLATESZ2', float(plate_size[1]))

            if self.conf.has_option('Parse keywords', 'exptime_sec'):
                exptime_key = self.conf.get('Parse keywords', 'exptime_sec')

                if ((exptime_key in self) and 
                    not isinstance(self[exptime_key], fits.card.Undefined)):
                    self[exptime_key] = float(self[exptime_key]) * 60.
                else:
                    self.set(exptime_key, float(self.platemeta['exptime'][0]) * 60.)

            if self.conf.has_option('Parse keywords', 'observer'):
                observer_key = self.conf.get('Parse keywords', 'observer')

                # Add space after initial and capitalize first letter
                if observer_key in self:
                    self[observer_key] = self[observer_key].replace('.', '. ').title()

            if self.conf.has_option('Parse keywords', 'filter'):
                filter_key = self.conf.get('Parse keywords', 'filter')

                if filter_key in self:
                    self[filter_key] = self[filter_key].replace('NONE', 'none')
                    self[filter_key] = self[filter_key].replace('NO', 'none')

            if self.conf.has_option('Parse keywords', 'prism_angle'):
                prism_angle_key = self.conf.get('Parse keywords', 
                                                'prism_angle')

                if prism_angle_key in self:
                    self[prism_angle_key] = self[prism_angle_key].replace('NO', '')

            if self.conf.has_option('Parse keywords', 'ut_date_start'):
                ut_date_start_key = self.conf.get('Parse keywords', 
                                                  'ut_date_start')

                if ut_date_start_key in self:
                    ut_date_start = self[ut_date_start_key]

                if self.conf.has_option('Parse keywords', 'ut_time_start'):
                    ut_time_start_key = self.conf.get('Parse keywords', 
                                                      'ut_time_start')

                    if ut_time_start_key in self:
                        ut_time_start = self[ut_time_start_key]
                        ut_datetime_start = '%sT%s' % (ut_date_start, ut_time_start)
                    else:
                        ut_time_start = ''
                        ut_datetime_start = ut_date_start

                self[ut_date_start_key] = ut_datetime_start

            if self.conf.has_option('Parse keywords', 'orig_date_from_ut'):
                if self.conf.getboolean('Parse keywords', 'orig_date_from_ut'):
                    t_start = Time(ut_datetime_start, format='isot', scale='ut1')
                    orig_date = Time(t_start.jd - 0.5, format='jd',
                        scale='ut1', out_subfmt='date').iso
                    self.set('DATEORIG', orig_date)

    def rename_keywords(self):
        """
        Rename keywords in the header.

        """
        
        if self.conf.has_section('Rename FITS keywords'):
            for (oldkey, newkey) in self.conf.items('Rename FITS keywords'):
                if oldkey in self:
                    if (newkey == ''):
                        self.remove(oldkey)
                    else:
                        self.rename_keyword(oldkey, newkey)

    def assign_values(self):
        """
        Assign fixed keyword values (integers, floats and strings 
        separately).

        """

        if self.conf.has_section('Integer values'):
            for (key, value) in self.conf.items('Integer values'):
                self.set(key, int(value))

        if self.conf.has_section('Float values'):
            for (key, value) in self.conf.items('Float values'):
                self.set(key, float(value))

        if self.conf.has_section('String values'):
            for (key, value) in self.conf.items('String values'):
                self.set(key, value)

    def compute_values(self):
        """
        Compute keyword values.

        """

        self.append(fits.Card.fromstring('RA_DEG  ='))
        self.append(fits.Card.fromstring('DEC_DEG ='))

        if (('DATEORIG' in self) and (self['DATEORIG'] != '') and
            ('TMS-ORIG' in self) and (self['TMS-ORIG'] != '') and
            ('SITELAT' in self) and ('SITELONG' in self)):
            # Convert sidereal time to UT using pyEphem's next_transit
            # Initialize location and date
            loc = ephem.Observer()
            loc.lat = str(self['SITELAT'])
            loc.lon = str(self['SITELONG'])

            if ('SITEELEV' in self):
                loc.elevation = self['SITEELEV']
                
            loc.date = '%s %s' % (self['DATEORIG'], '12:00:00')
            # Define imaginary star with the RA value of the ST of observation,
            # and the Dec value of 0. Then compute transit time.
            st = ephem.FixedBody()
            st._ra = ephem.hours(self['TMS-ORIG'])
            st._dec = 0.
            st._epoch = self['DATEORIG']
            st.compute()
            ut_start_isot = '%04d-%02d-%02dT%02d:%02d:%02d' % loc.next_transit(st).tuple()
            self.set('DATE-OBS', ut_start_isot)
            time_start = Time(ut_start_isot, format='isot', scale='ut1')
            time_avg = Time(time_start.jd + 0.5 * self['EXPTIME'] / 86400., format='jd', 
                scale='ut1', precision=0)
            self.set('DATE-AVG', time_avg.isot)
            self.set('JD', float('%.5f' % time_start.jd))
            self.set('JD-AVG', float('%.5f' % time_avg.jd))
            self.set('YEAR-AVG', float('%.8f' % time_avg.jyear))

        if ('RA' in self) and (self['RA'] != ''):
            a = Angle(self['RA'], units.hour)
            self.set('RA_DEG', float('%.4f' % a.degrees))

        if ('DEC' in self) and (self['DEC'] != ''):
            a = Angle(self['DEC'], units.degree)
            self.set('DEC_DEG', float('%.4f' % a.degrees))

        self.set('DATE', dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    def update_comments(self):
        """
        Add/modify keyword comments based on configuration.

        """

        # Assign default comments
        for key in self:
            if key in self._default_comments:
                self.comments[key] = self._default_comments[key]
            elif key and key[-1].isdigit():
                # If keyword ends with a digit, replace the ending number 
                # with 'n', and look for such a keyword. For example, if 
                # keyword is 'EXPTIM1', look for 'EXPTIMn' instead.
                lastnumstr = re.findall(r'\d+', key)[-1]
                nkey = key.replace(lastnumstr, 'n')

                if nkey in self._default_comments:
                    self.comments[key] = self._default_comments[nkey].format(lastnumstr)

        # Assign comments from the configuration
        if self.conf.has_section('FITS keyword comments'):
            for (key, comment) in self.conf.items('FITS keyword comments'):
                if key in self:
                    self.comments[key] = comment

    def rewrite(self):
        """
        Rewrite header cards with proper formatting.

        """

        h = self.copy()
        self.clear()

        for (k,v,c) in h.cards:
            # Remove empty COMMENT or HISTORY card
            if (k == 'COMMENT' or k == 'HISTORY') and not v:
                continue

            # Rename blank keyword to COMMENT if it is not a separator
            if not k:
                if not re.match('---', v):
                    k = 'COMMENT'

            # Store cards in the new header
            # Treat null value separately to get proper comment alignment
            if isinstance(v, fits.card.Undefined):
                cardstr = k.ljust(8) + '='.ljust(22) + ' / ' + c
                self.append(fits.Card.fromstring(cardstr))
            else:
                # Workaround for Astropy treating colon in string as assignment
                # of numerical value
                if (isinstance(v, str) and (k != 'COMMENT') and (k != 'HISTORY') 
                    and ':' in v):
                    self.append((k, '', c), bottom=True)
                    self[k] = v
                else:
                    self.append((k, v, c), bottom=True)

            # Pad empty strings in card values
            if (isinstance(v, str) and not v and k and (k != 'COMMENT') 
                and (k != 'HISTORY')):
                self[k] = 'a'  # pyfits hack
                self[k] = ' '  # pyfits hack

    def reorder(self):
        """
        Reorder header cards based on configuration.

        """

        if self.conf.has_section('FITS keyword order'):
            orderkeys = self.conf.get('FITS keyword order',
                                      'keywords').split('\n')
        else:
            orderkeys = self._default_order

        h = self.copy()
        self.clear()

        for key in orderkeys:
            if key[:3] == 'sep':
                if key[4:] == '':
                    sepstr = '-' * 72
                else:
                    sepstr = (' ' + key[4:]).rjust(72, '-')
                    #sepstr = ('-' * 3 + ' ' + key[4:] + ' ').ljust(72, '-')

                self.append(('', sepstr), end=True)

                # Copy WCS block
                wcs_sep = ' WCS'.rjust(72, '-')

                if key.strip().endswith('WCS') and wcs_sep in h.values():
                    wcs_ind = h.values().index(wcs_sep) + 1

                    for i,c in enumerate(h.cards[wcs_ind:]):
                        if c[0]:
                            self.append(c, end=True)
                            del h[wcs_ind]
                        else:
                            break

                # Copy acknowledgements
                ack_sep = ' Acknowledgements'.rjust(72, '-')

                if (key.strip().endswith('Acknowledgements') and
                        ack_sep in h.values()):
                        #self.platemeta['fits_acknowledgements']):
                    ack_ind = h.values().index(ack_sep) + 1

                    for i,c in enumerate(h.cards[ack_ind:]):
                        if c[0] == 'COMMENT':
                            self.append(c, end=True)
                            del h[ack_ind]
                        else:
                            break
            elif 'n' in key:
                for n in np.arange(99)+1:
                    keystr = key.replace('n', str(n))

                    if keystr in h:
                        c = h.cards[h.index(keystr)]
                        self.append(c, end=True)
                        h.remove(keystr)
            elif key == 'HISTORY':
                if key in h:
                    for histitem in h[key]:
                        self.append(('HISTORY', histitem), end=True)
                        #self.add_history(histitem)
                        
                    del h[key]
            elif key in h:
                c = h.cards[h.index(key)]
                self.append(c, end=True)
                h.remove(key)

        # Copy the remaining cards
        for c in h.cards:
            k,v,comment = c

            # Copy cards that are not existing separators
            if k or v not in self['']:
                self.append(c, bottom=True)

    def format(self):
        """
        Format and reorder header cards.

        """

        self.update_comments()
        self.rewrite()
        self.reorder()

    def update_all(self):
        """
        Do all header updates.

        """

        self.update_from_platemeta()
        self.update_values()
        self.rename_keywords()
        self.assign_values()
        self.compute_values()
        self.update_comments()
        self.rewrite()
        self.reorder()

    def insert_wcs(self, wcshead):
        """
        Insert WCS header cards.

        """

        wcs_sep = ' WCS'.rjust(72, '-')

        if wcs_sep in self.values():
            wcs_ind = self.values().index(wcs_sep) + 1
            
            for c in wcshead.cards:
                if c[0] == 'HISTORY':
                    #self.insert(wcs_ind, ('COMMENT', c[1]))
                    self.insert(wcs_ind, c)
                    wcs_ind += 1
                elif c[0] == 'COMMENT':
                    pass
                elif c[0] not in self:
                    self.insert(wcs_ind, c)
                    wcs_ind += 1
                    
            self.add_history('WCS added with PyPlate v{} at {}'
                             .format(__version__, dt.datetime.utcnow()
                                     .strftime('%Y-%m-%dT%H:%M:%S')))
            self.set('DATE', 
                     dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'))

    def output_to_fits(self, filename, checksum=None):
        """
        Output header to FITS file.

        """

        if checksum is None:
            checksum = self.create_checksum

        fn_fits = os.path.join(self.fits_dir, filename)
        fn_out = os.path.join(self.write_fits_dir, filename)

        if os.path.exists(fn_out):
            fitsfile = fits.open(fn_out, mode='update', 
                                 do_not_scale_image_data=True, 
                                 ignore_missing_end=True)
            fitsfile[0].header = self.copy()
            fitsfile.flush()
        else:
            if not os.path.exists(fn_fits):
                print 'File does not exist: {}'.format(fn_fits)

            fitsfile = fits.open(fn_fits, do_not_scale_image_data=True, 
                                 ignore_missing_end=True)
            fitsfile[0].header = self.copy()

            try:
                os.makedirs(self.write_fits_dir)
            except OSError:
                if not os.path.isdir(self.write_fits_dir):
                    print ('Could not create directory {}'
                           .format(self.write_fits_dir))
        
            try:
                fitsfile.writeto(fn_out, checksum=checksum)
            except IOError:
                print 'Could not write to {}'.format(fn_out)
            
        fitsfile.close()
        del fitsfile

        # Get size and timestamp of the updated FITS file
        try:
            mtime = dt.datetime.utcfromtimestamp(os.path.getmtime(fn_out))
            mtime_str = mtime.strftime('%Y-%m-%dT%H:%M:%S')
            self.fits_datetime = mtime_str
            self.fits_size = os.path.getsize(fn_out)

            if checksum:
                h = fits.getheader(fn_out)
                self.fits_checksum = h['CHECKSUM']
                self.fits_datasum = h['DATASUM']
        except IOError:
            pass

    def output_to_file(self, filename):
        """
        Output header to a text file.

        """
        
        try:
            os.makedirs(self.write_header_dir)
        except OSError:
            if not os.path.isdir(self.write_header_dir):
                print ('Could not create directory {}'
                       .format(self.write_header_dir))

        fn_out = os.path.join(self.write_header_dir, filename)

        if os.path.exists(fn_out):
            os.remove(fn_out)

        try:
            self.tofile(fn_out, sep='', endcard=True, padding=True)
        except IOError:
            print 'Error writing {}'.format(fn_out)


