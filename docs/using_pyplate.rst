Using PyPlate
=============

Creating and managing a FITS header
-----------------------------------

Create an empty header::

    import pyplate

    pheader = pyplate.metadata.PlateHeader()
    pheader.populate()

Create header from a FITS file::

    pheader = pyplate.metadata.PlateHeader.from_fits('/path/to/file.fits')

Create header from a text file that contains a FITS header::

    pheader = pyplate.metadata.PlateHeader.from_hdrf('/path/to/file.hdrf')

Update header with metadata::

    pmeta = pyplate.metadata.PlateMeta()
    pheader.update_from_platemeta(pmeta)

Configure input and output directories::

    pheader.fits_dir = '/path/to/fits_dir'
    pheader.write_fits_dir = '/path/to/write_fits_dir'
    pheader.write_header_dir = '/path/to/write_header_dir'

Write header to a FITS file in ``write_fits_dir``::

    pheader.output_to_fits('file.fits')

If the FITS file exists, then its header is updated with ``pheader``. If the 
file does not exist, ``pyplate`` assumes that the file with the same name
exists in ``fits_dir``. The file is then opened for reading and an output
file is written to ``write_fits_dir`` with the current header.

Write header to a text file in ``write_header_dir``::

    pheader.output_to_fits('file.hdr')


