import os
import glob
import numpy as np
import ConfigParser
from datetime import datetime
from astropy.io import fits
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
    

class PlateConverter:
    """
    TIFF-to-FITS converter class

    """

    def __init__(self):
        self.tiff_dir = ''
        self.write_fits_dir = ''
        self.write_wedge_dir = ''
        self.scan_exif_timezone = None
        self.wedge_height = None
        self.cut_wedge = False

    def assign_conf(self, conf):
        """
        Parse configuration and set class attributes.

        """

        if isinstance(conf, str):
            conf = read_conf(conf)

        for attr in ['tiff_dir', 'write_fits_dir', 'write_wedge_dir']:
            try:
                setattr(self, attr, conf.get('Files', attr))
            except ConfigParser.Error:
                pass

        for attr in ['scan_exif_timezone']:
            try:
                setattr(self, attr, conf.get('Image', attr))
            except ConfigParser.Error:
                pass

        for attr in ['wedge_height']:
            try:
                setattr(self, attr, conf.getint('Image', attr))
            except ConfigParser.Error:
                pass

        for attr in ['cut_wedge']:
            try:
                setattr(self, attr, conf.getboolean('Image', attr))
            except ConfigParser.Error:
                pass

    def batch_tiff2fits(self):
        """
        Convert all TIFF images in the TIFF directory to FITS.

        """
        
        for fn_tiff in sorted(glob.glob(os.path.join(self.tiff_dir, '*.tif'))):
            self.tiff2fits(os.path.basename(fn_tiff))

    def tiff2fits(self, filename, cut_wedge=None, wedge_height=None):
        """
        Convert TIFF image to FITS.

        Parameters
        ----------
        filename : str
            Filename of the TIFF image
        cut_wedge : bool
            If True, a wedge image is separated from below plate image
        wedge_height : int
            Height of the wedge in pixels

        """

        if cut_wedge is None and self.cut_wedge is not None:
            cut_wedge = self.cut_wedge

        if wedge_height is None and self.wedge_height is not None:
            wedge_height = self.wedge_height

        fn_tiff = os.path.join(self.tiff_dir, filename)
        im_pil = Image.open(fn_tiff)
        exif_datetime = None

        if gexiv_available:
            exif = GExiv2.Metadata(fn_tiff)

            if 'Exif.Image.DateTime' in exif:
                exif_datetime = exif['Exif.Image.DateTime']
            elif 'Exif.Photo.DateTimeDigitized' in exif:
                exif_datetime = exif['Exif.Photo.DateTimeDigitized']
        else:
            try:
                exif_datetime = im_pil.tag[306]
            except Exception:
                pass

        if exif_datetime:
            if exif_datetime[4] == ':':
                exif_datetime = '{} {}'.format(exif_datetime[:10]
                                               .replace(':', '-'),
                                               exif_datetime[11:])

            if pytz_available and self.scan_exif_timezone:
                dt = datetime.strptime(exif_datetime, '%Y-%m-%d %H:%M:%S')

                try:
                    dt_local = (pytz.timezone(self.scan_exif_timezone)
                                .localize(dt))
                    exif_datetime = (dt_local.astimezone(pytz.utc)
                                     .strftime('%Y-%m-%dT%H:%M:%S'))
                except pytz.exceptions.UnknownTimeZoneError:
                    pass

        #print '{} Reading {}'.format(str(datetime.now()), fn_tiff)

        im = np.array(im_pil.getdata(),
                      dtype=np.uint16).reshape(im_pil.size[1],-1)
        imwidth = im.shape[1]
        imheight = im.shape[0]
        imblack = im.min()
        imwhite = im.max()

        # Cut wedge image if necessary
        if not cut_wedge or wedge_height == 0:
            im_plates = im
            im_wedge = None
        elif cut_wedge and wedge_height != 0:
            ycut_wedge = imheight - wedge_height
            im_wedge = im[ycut_wedge:,:]
            im_plates = im[:ycut_wedge,:]
        else:
            yedge = []
            yedge_plate = []

            for x in np.arange(100, imwidth-100, 10):
                # Take column, reverse it and use pixels from the 101st to
                # 80% of the image height.
                colrev = im[::-1,x][100:int(0.8*imheight)]
                # Find nearly white pixels
                ind_white = np.where(colrev-imblack > 0.95*(imwhite-imblack))[0]

                # If the first near-white pixel is significantly lighter than
                # the first 500 pixels in colrev, then use it as an edge of the
                # wedge.
                if (ind_white.size > 0 and
                    colrev[ind_white[0]]-imblack 
                    > 1.1*(np.median(colrev[:500])-imblack)):
                    yedge.append(imheight - 100 - ind_white[0])
                else:
                    col = im[int(0.2*imheight):,x]
                    ind_white = np.where(col-imblack 
                                         > 0.95*(imwhite-imblack))[0]

                    if (ind_white.size > 0 and
                        col[ind_white[0]]-imblack 
                        > 1.1*(np.median(col[:500])-imblack)):
                        yedge_plate.append(ind_white[0] + int(0.2*imheight))

            if len(yedge) > 0.01*imwidth:
                ycut_wedge = int(np.median(yedge))
                im_wedge = im[ycut_wedge:,:]
                im_plates = im[:ycut_wedge,:]
            else:
                try:
                    ycut_wedge = int(np.percentile(yedge_plate, 80))
                    im_wedge = im[ycut_wedge:,:]
                    im_plates = im[:ycut_wedge,:]
                except ValueError:
                    print 'Cannot separate wedge in {}'.format(fn_tiff)
                    im_wedge = None
                    im_plates = im

        del im
        history_line = ('TIFF image converted to FITS with '
                        'PyPlate v{} at {}'
                        .format(__version__, datetime.utcnow()
                                .strftime('%Y-%m-%dT%H:%M:%S')))

        if im_wedge is not None:
            hdu_wedge = fits.PrimaryHDU(np.flipud(im_wedge))

            if exif_datetime:
                hdu_wedge.header.set('DATESCAN', exif_datetime)

            hdu_wedge.header.add_history(history_line)

            # Create wedge output directory
            if self.write_wedge_dir:
                try:
                    os.makedirs(self.write_wedge_dir)
                except OSError:
                    if not os.path.isdir(self.write_wedge_dir):
                        print ('Could not create directory {}'
                               .format(write_wedge_dir))
                        raise

        # Create FITS image output directory
        if self.write_fits_dir:
            try:
                os.makedirs(self.write_fits_dir)
            except OSError:
                if not os.path.isdir(self.write_fits_dir):
                    print ('Could not create directory {}'
                           .format(write_fits_dir))
                    raise

        # If filename contains dash, assume that two plates have been scanned 
        # side by side.
        if '-' in os.path.basename(fn_tiff):
            xedge = []

            for y in np.arange(100, im_plates.shape[0]-100, 10):
                row = im_plates[y,:]
                row_mid = row[int(0.25*row.size):int(0.75*row.size)]

                if row_mid.max() > 1.1*np.median(row_mid):
                    xedge.append(np.argmax(row_mid)+int(0.25*row.size))

            xcut = int(np.median(xedge))
            im_left = im_plates[:,:xcut]
            im_right = im_plates[:,xcut:]
            del im_plates

            fn_two = os.path.splitext(os.path.basename(fn_tiff))[0]
            fn_parts = fn_two.split('-')
            fn_left = '{}{}.fits'.format(fn_parts[0][:7], fn_parts[1])
            fn_right = '{}.fits'.format(fn_parts[0])

            # Store left-side plate FITS
            hdu_left = fits.PrimaryHDU(np.flipud(im_left))
            hdu_left.header.set('MINVAL', im_left.min())
            hdu_left.header.set('MAXVAL', im_left.max())

            if exif_datetime:
                hdu_left.header.set('DATESCAN', exif_datetime)

            hdu_left.header.add_history(history_line)
            hdu_left.writeto(os.path.join(self.write_fits_dir, fn_left), 
                             overwrite=True)

            if im_wedge is not None:
                fn_wedge = os.path.splitext(fn_left)[0] + '_w.fits'
                hdu_wedge.writeto(os.path.join(self.write_wedge_dir, fn_wedge), 
                                  overwrite=True)

            # Store right-side plate FITS
            hdu_right = fits.PrimaryHDU(np.flipud(im_right))
            hdu_right.header.set('MINVAL', im_right.min())
            hdu_right.header.set('MAXVAL', im_right.max())

            if exif_datetime:
                hdu_right.header.set('DATESCAN', exif_datetime)

            hdu_right.header.add_history(history_line)
            hdu_right.writeto(os.path.join(self.write_fits_dir, fn_right), 
                              overwrite=True)

            if im_wedge is not None:
                fn_wedge = os.path.splitext(fn_right)[0] + '_w.fits'
                hdu_wedge.writeto(os.path.join(self.write_wedge_dir, fn_wedge), 
                                  overwrite=True)
        else:
            fn_plate = os.path.splitext(os.path.basename(fn_tiff))[0] + '.fits'
            hdu_plate = fits.PrimaryHDU(np.flipud(im_plates))
            hdu_plate.header.set('MINVAL', im_plates.min())
            hdu_plate.header.set('MAXVAL', im_plates.max())

            if exif_datetime:
                hdu_plate.header.set('DATESCAN', exif_datetime)

            hdu_plate.header.add_history(history_line)
            hdu_plate.writeto(os.path.join(self.write_fits_dir, fn_plate), 
                              overwrite=True)

            if im_wedge is not None:
                fn_wedge = os.path.splitext(fn_plate)[0] + '_w.fits'
                hdu_wedge.writeto(os.path.join(self.write_wedge_dir, fn_wedge),
                                  overwrite=True)

