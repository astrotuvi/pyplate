"""Create the binary inputs of the "full" mock archive in tests/ver4/metadata:
small FITS files (fits/), preview images (previews/), logpage images
(logpages/) and a plate cover image (covers/).

The files are committed; this script documents how they were made and can
recreate them. It needs only numpy, astropy and Pillow. Run from the
repository root:

    python tests/ver4/make_metadata_inputs.py

File modification times are not kept by git. gen_metadata_regression.py and
the test of the redesigned code set them to a fixed value before reading.
"""
import os

import numpy as np
from astropy.io import fits
from PIL import Image

META_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'metadata')


def make_fits(path, header_values, shape, offset):
    """16-bit unsigned image (BZERO 32768) with the given header keywords."""
    data = (np.arange(shape[0] * shape[1], dtype=np.uint32)
            .reshape(shape) * 7 + offset).astype(np.uint16)
    hdu = fits.PrimaryHDU(data)

    for key, value in header_values:
        hdu.header[key] = value

    hdu.writeto(path, overwrite=True)


def make_jpeg(path, size, exif_datetime=None):
    im = Image.new('L', size, color=128)
    kwargs = {}

    if exif_datetime:
        exif = Image.Exif()
        exif[306] = exif_datetime
        kwargs['exif'] = exif.tobytes()

    im.save(path, 'JPEG', quality=90, **kwargs)


def make_png(path, size):
    Image.new('L', size, color=64).save(path, 'PNG')


def make_tiff(path, size):
    # No DateTime tag: the old code cannot read it from a TIFF file
    # (Pillow returns a tuple, which the old code indexes as a string)
    Image.new('L', size, color=200).save(path, 'TIFF')


def main():
    for d in ['fits', 'previews', 'logpages', 'covers']:
        os.makedirs(os.path.join(META_DIR, d), exist_ok=True)

    make_fits(os.path.join(META_DIR, 'fits', 'mck001_000001.fits'),
              [('SCANNER', 'Mock Scanner 1'), ('SCANRES1', 2400),
               ('SCANRES2', 2400), ('PIXSIZE1', 10.58), ('PIXSIZE2', 10.58),
               ('SCANSOFT', 'MockScan 1.0'), ('SCANGAM', 1.0),
               ('ORIGIN', 'Mock Institute'), ('OBSERVER', 'FITS Observer'),
               ('EMULSION', 'FITS emulsion')],
              (30, 20), 100)
    make_fits(os.path.join(META_DIR, 'fits', 'mck001_000002a.fits'),
              [('SCANNER', 'Mock Scanner 2'), ('SCANRES1', 1200),
               ('SCANRES2', 1600), ('WEDGE', 'Mock wedge'),
               ('DATESCAN', '2019-12-31')],
              (25, 40), 5000)
    make_jpeg(os.path.join(META_DIR, 'previews', 'mck001_000001.jpg'),
              (60, 40), exif_datetime='2015:06:07 08:09:10')
    make_png(os.path.join(META_DIR, 'previews', 'mck001_000002a.png'),
             (50, 30))
    make_jpeg(os.path.join(META_DIR, 'covers', 'cover_000001.jpg'), (30, 45))
    make_jpeg(os.path.join(META_DIR, 'logpages', 'log1_p012.jpg'), (70, 90),
              exif_datetime='2016:01:02 10:20:30')
    make_tiff(os.path.join(META_DIR, 'logpages', 'log1_p013.tif'), (72, 96))
    make_png(os.path.join(META_DIR, 'logpages', 'log1_p014.png'), (64, 80))


if __name__ == '__main__':
    main()
