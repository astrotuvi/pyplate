"""Classifying sources as celestial or artifacts with the bundled Keras
model (algorithm and model by Gal Matijevic)."""
import os

import numpy as np
from astropy.io import fits

from .context import ProcessContext
from .sources import SourceTable

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.models import load_model
    have_keras = True
except ImportError:
    have_keras = False


def predict(ctx: ProcessContext, sources: SourceTable) -> np.ndarray:
    """
    Classify the sources on the inverted image (``<basefn>_inverted.fits``
    in the scratch directory).

    Returns
    -------
    np.ndarray
        For each source, the probability that it is a true source.

    """

    # Read model
    fn_model = os.path.join(os.path.dirname(__file__), 'artifact_model.keras')
    model = load_model(fn_model)

    # Read inverted FITS file
    fn_image = os.path.join(ctx.work_dir, ctx.basefn + '_inverted.fits')
    plate_fits = fits.open(fn_image)
    plate_data = plate_fits[0].data
    plate_fits.close()

    # 1/2 of the size of the thumbnail
    thumb_size = 10

    # number of thumbnails
    noft = len(sources)

    # initialize thumbnails array
    thumbs = np.zeros((noft, 1, thumb_size * 2, thumb_size * 2))

    for i, (x, y) in enumerate(zip(sources['x_image'], sources['y_image'])):
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

    return model.predict(thumbs).flatten()
