"""Finding multi-exposure patterns and scanner patterns."""
from __future__ import annotations
from collections.abc import Callable

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import lstsq

from ..context import ProcessContext

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
    import statsmodels.api as sm
    have_statsmodels = True
except ImportError:
    have_statsmodels = False


def find_scanner_pattern(coords_image: np.ndarray, coords_ref: np.ndarray
                         ) -> tuple[np.ndarray, Callable, Callable, float]:
    """
    Find if there is a pattern along one axis in the scan file.
    If found, fit a smooth curve to the difference between scan
    coordinates and reference coordinates along that axis.

    Parameters
    ----------
    coords_image
        The coordinates of sources (at least 10)
    coords_ref
        The coordinates of reference stars, one per source

    Returns
    -------
    coords_image
        The image coordinates corrected for the pattern (the input array,
        changed in place).
    s_x, s_y
        The shift as a function of the coordinate along each axis.
    std_ratio
        Ratio of the standard deviations of the shifts along y and x.

    """

    # All input arrays must have the same length
    assert len(coords_image) == len(coords_ref), 'coords_image and coords_ref must have the same length'
    assert len(coords_image) >= 10, 'Number of sources must be at least 10'

    # Prepare 1-dimensional arrays
    x_image = coords_image[:,0]
    y_image = coords_image[:,1]
    x_ref = coords_ref[:,0]
    y_ref = coords_ref[:,1]

    # Calculate differences
    dx = x_image - x_ref
    dy = y_image - y_ref

    # Make sure that lowess fraction includes at least 10 stars
    nstars = len(y_image)

    if nstars > 100:
        frac = 1. / np.sqrt(nstars)
    else:
        frac = 10. / len(y_image)

    # Find smooth curve along y-axis
    z = sm.nonparametric.lowess(dy, y_image, frac=frac, it=3,
                                return_sorted=True)
    _,uind = np.unique(z[:,0], return_index=True)
    s_y = InterpolatedUnivariateSpline(z[uind,0], z[uind,1], k=1)

    # Find smooth curve along x-axis
    z = sm.nonparametric.lowess(dx, x_image, frac=frac, it=3,
                                return_sorted=True)
    _,uind = np.unique(z[:,0], return_index=True)
    s_x = InterpolatedUnivariateSpline(z[uind,0], z[uind,1], k=1)

    y_range = np.max(y_image) - np.min(y_image)
    x_range = np.max(x_image) - np.min(x_image)
    yy = np.linspace(np.min(y_image) + 0.2 * y_range,
                     np.max(y_image) - 0.2 * y_range, 60)
    xx = np.linspace(np.min(x_image) + 0.2 * x_range,
                     np.max(x_image) - 0.2 * x_range, 60)

    # Scanner pattern exists if the standard deviation of
    # 60 points from the smooth curve along that axis is at least
    # 1.5 times as high as the standard deviation along the other axis
    std_ratio = s_y(yy).std() / s_x(xx).std()

    if std_ratio > 1.5:
        coords_image[:,1] = y_image - s_y(y_image)
    elif std_ratio < 2./3.:
        coords_image[:,0] = x_image - s_x(x_image)

    return coords_image, s_x, s_y, std_ratio


def find_peak_separation(coords: np.ndarray) -> float:
    """
    Find characteristic distance between nearest neighbors in coords.

    """

    n, dim = coords.shape
    assert dim == 2

    kdt = KDT(coords)
    ds,_ = kdt.query(coords, k=2)

    # Kernel density estimation (KDE) to distances
    kde = sm.nonparametric.KDEUnivariate(ds[:,1].astype(np.double))
    kde.fit()

    # Find peak in density
    a = kde.density.argmax()
    dist_peak = kde.support[a]

    # Return peak distance
    return dist_peak


def get_scale_rotation(xy1: np.ndarray, xy2: np.ndarray
                       ) -> tuple[float, float, np.ndarray]:
    """
    Find scale and rotation between two point clouds.

    """

    assert xy1.shape == xy2.shape
    n, dim = xy1.shape

    H = (xy1.T @ xy2) / n
    V, S, W = np.linalg.svd(H)

    if (np.linalg.det(V) * np.linalg.det(W)) < 0:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    R = V @ W
    scale = (np.sqrt((xy2**2).sum(axis=1)).sum()
             / np.sqrt((xy1**2).sum(axis=1)).sum())
    rot_angle = np.degrees(np.arctan2(R[1,0], R[0,0]))

    if rot_angle > 90:
        rot_angle -= 180.
    elif rot_angle < -90:
        rot_angle += 180.

    return scale, rot_angle, R


def find_pattern(ctx: ProcessContext, coords: np.ndarray, dist_peak: float,
                 numexp: int) -> np.ndarray | None:
    """
    Find the multi-exposure pattern in coords and number the sources by
    exposure.

    Sources are clustered into images of the same star, the clusters give
    the pattern of the exposures (with the scale and rotation across the
    plate when there are enough clusters), and every source where the whole
    pattern is found gets its exposure number.

    Parameters
    ----------
    ctx
        Context of the process (image size and log).
    coords
        Coordinates of the sources: x, y and the scaled magnitude.
    dist_peak
        Characteristic distance between nearest neighbours.
    numexp
        Number of exposures.

    Returns
    -------
    np.ndarray or None
        Exposure number of each source (0 if not in a found pattern), or
        None if the pattern was not found.

    """

    ctx.log.write('Finding multi-exposure pattern', level=3, event=31)

    n, dim = coords.shape
    assert dim == 3

    clusters = _find_star_clusters(ctx, coords, dist_peak, numexp)

    if clusters is None:
        return None

    coords_select, labels, labels_sel = clusters
    xy, xy_mean, m = _cluster_centres(coords_select, labels, labels_sel)

    # Analyse scale and rotation only if number of clusters is
    # above threshold
    if len(labels_sel) > 10 and numexp < 10:
        fit = _fit_scale_rotation(ctx, xy, xy_mean, m, labels, labels_sel)
    else:
        fit = None

    xy_local = _local_coordinates(xy, xy_mean, m, labels, labels_sel, fit)
    xy_mean_exp = _exposure_offsets(ctx, xy_local, len(labels_sel))

    if xy_mean_exp is None:
        return None

    return _assign_exposures(ctx, coords, xy_mean_exp, fit)


def _find_star_clusters(ctx, coords, dist_peak, numexp):
    """Cluster the brightest sources into the images of single stars and
    keep the clusters with about numexp members. Returns the clustered
    coordinates, the cluster label of each and the accepted labels, or None
    if fewer than 5 clusters are accepted."""

    # Choose optimal clustering parameters
    if numexp > 20:
        eps = dist_peak * 2
        min_samples = 1
        num_select = 50 * numexp
    elif numexp > 10:
        eps = dist_peak * 3
        min_samples = 2
        num_select = 100 * numexp
    else:
        eps = dist_peak * 4
        min_samples = 4
        num_select = 200 * numexp

    coords_select = coords[:num_select]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_select)
    labels = db.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ctx.log.write('Found {:d} source clusters'.format(num_clusters),
                  level=4, event=31, double_newline=False)

    # Find cluster sizes (label counts)
    lab, cnt = np.unique(labels[labels>-1], return_counts=True)

    # Select labels that have the accepted cluster size
    labels_sel = lab[np.abs(cnt - numexp) < 0.1 * numexp]
    num_accepted_clusters = len(labels_sel)

    if num_accepted_clusters < 5:
        ctx.log.write('Not enough source clusters found (<5)!',
                      level=2, event=31)
        return None
    else:
        ctx.log.write('Use {:d} source clusters'
                      .format(num_accepted_clusters), level=4, event=31,
                      double_newline=False)

    return coords_select, labels, labels_sel


def _cluster_centres(coords_select, labels, labels_sel):
    """Centres of the accepted clusters, and a mask of the clusters whose
    nearest-neighbour distances are not outliers. Returns also the x, y
    coordinates of the clustered sources."""

    # Create array for source coordinates relative to cluster center
    xy = coords_select.copy()[:,:2]
    xy_mean = np.zeros((len(labels_sel), 2))

    # Create arrays for nearest-neighbour distances
    nnd_min = np.zeros(len(labels_sel))
    nnd_max = np.zeros(len(labels_sel))

    for i,k in enumerate(labels_sel):
        class_member_mask = (labels == k)

        # Calculate mean x and y of a cluster by averaging the coordinates
        # of its extreme members
        xy_mean[i,0] = (np.min(xy[class_member_mask,0]) +
                        np.max(xy[class_member_mask,0])) / 2.
        xy_mean[i,1] = (np.min(xy[class_member_mask,1]) +
                        np.max(xy[class_member_mask,1])) / 2.

        xyi = xy[class_member_mask]
        kdt_xyi = KDT(xyi)
        ds_xyi,_ = kdt_xyi.query(xyi, k=2)
        nnd_min[i] = ds_xyi[:,1].min()
        nnd_max[i] = ds_xyi[:,1].max()

    # Exclude clusters with outlying (min_nnd, max_nnd) values
    iqr_nnd_min = np.subtract(*np.percentile(nnd_min, [75, 25]))
    iqr_nnd_max = np.subtract(*np.percentile(nnd_max, [75, 25]))
    m = ((nnd_min > np.percentile(nnd_min, 25) - 1.5 * iqr_nnd_min) &
         (nnd_min < np.percentile(nnd_min, 75) + 1.5 * iqr_nnd_min) &
         (nnd_max > np.percentile(nnd_max, 25) - 1.5 * iqr_nnd_max) &
         (nnd_max < np.percentile(nnd_max, 75) + 1.5 * iqr_nnd_max))

    return xy, xy_mean, m


def _fit_scale_rotation(ctx, xy, xy_mean, m, labels, labels_sel):
    """Fit planes to the scale and rotation of the clusters relative to the
    cluster nearest to the image centre. Returns the plane coefficients of
    the rotation angle and of the scale."""
    pattern_angle = np.zeros(m.sum())
    pattern_scale = np.zeros(m.sum()) + 1.

    # Find cluster nearest to image center
    im_center = np.array(((ctx.imwidth + 1.) / 2.,
                          (ctx.imheight + 1.) / 2.))
    dist_center = np.linalg.norm(xy_mean[m] - im_center, axis=1)
    ind_nearest = dist_center.argmin()

    # Find scale and rotation relative to the cluster closest to
    # image center
    class_member_mask = (labels == labels_sel[m][ind_nearest])
    xy00 = xy[class_member_mask] - xy_mean[m][ind_nearest]

    for i,k in enumerate(labels_sel[m]):
        class_member_mask = (labels == k)

        if i != ind_nearest:
            xyi = xy[class_member_mask] - xy_mean[m][i]
            res = get_scale_rotation(xyi, xy00)
            pattern_scale[i],pattern_angle[i],_ = res

    # Fit 2D plane
    angle_mask = np.abs(pattern_angle) < 10
    A_angle = np.c_[xy_mean[m][angle_mask,0], xy_mean[m][angle_mask,1],
                    np.ones(angle_mask.sum())]
    C_angle,_,_,_ = lstsq(A_angle, pattern_angle[angle_mask])
    A_scale = np.c_[xy_mean[m][:,0], xy_mean[m][:,1],
                    np.ones(xy_mean[m].shape[0])]
    C_scale,_,_,_ = lstsq(A_scale, pattern_scale)
    return C_angle, C_scale


def _local_coordinates(xy, xy_mean, m, labels, labels_sel, fit):
    """Coordinates of the members of the accepted clusters relative to their
    cluster centre, corrected for scale and rotation if fit is given."""

    # Subtract cluster center coordinates and apply scale/rotation
    for i,k in enumerate(labels_sel):
        class_member_mask = (labels == k)
        xy[class_member_mask,0] -= xy_mean[i,0]
        xy[class_member_mask,1] -= xy_mean[i,1]

        if fit is not None:
            C_angle, C_scale = fit
            theta = np.radians(C_angle[0] * xy_mean[i,0] +
                               C_angle[1] * xy_mean[i,1] + C_angle[2])
            rot = np.array(((np.cos(theta), -np.sin(theta)),
                            (np.sin(theta),  np.cos(theta))))
            scale = (C_scale[0] * xy_mean[i,0] +
                     C_scale[1] * xy_mean[i,1] + C_scale[2])
            xy[class_member_mask] = xy[class_member_mask].dot(scale*rot)

    label_mask = np.isin(labels, labels_sel[m])
    return xy[label_mask]


def _exposure_offsets(ctx, xy_local, num_accepted_clusters):
    """Offsets of the exposures from the first one, sorted along the
    pattern, found by clustering the local coordinates; None if no
    exposure cluster is found."""

    # Analyse clustering on xy_local
    eps = 5.

    if num_accepted_clusters > 4:
        min_samples = 4
    else:
        min_samples = num_accepted_clusters

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_local)
    noise_mask = (db.labels_ == -1)
    exp_labels = db.labels_
    num_exp_clusters = len(set(exp_labels)) - (1 if -1 in exp_labels else 0)
    ctx.log.write('Number of clusters within source pattern: {:d}'
                  .format(num_exp_clusters), level=4, event=31,
                  double_newline=False)

    if num_exp_clusters == 0:
        ctx.log.write('Finding multi-exposure pattern failed!',
                      level=2, event=31)
        return None

    # Find cluster centers and sort them
    xy_mean_exp = np.zeros((num_exp_clusters, 2))

    for i,e in enumerate(set(exp_labels[~noise_mask])):
        xy_exp = xy_local[exp_labels == e].copy()
        xy_mean_exp[i,0] = np.mean(xy_exp[:,0])
        xy_mean_exp[i,1] = np.mean(xy_exp[:,1])

    # Check whether exposure pattern is horizontal or vertical
    exp_pattern_ratio = ((xy_mean_exp[:,1].max() - xy_mean_exp[:,1].min()) /
                         (xy_mean_exp[:,0].max() - xy_mean_exp[:,0].min()))

    # Sort cluster centers
    sort_axis = 1 if exp_pattern_ratio > 1 else 0
    xy_mean_exp = xy_mean_exp[np.argsort(xy_mean_exp[:,sort_axis])]
    xy_mean_exp[:,0] -= xy_mean_exp[0,0]
    xy_mean_exp[:,1] -= xy_mean_exp[0,1]
    return xy_mean_exp


def _assign_exposures(ctx, coords, xy_mean_exp, fit):
    """Exposure numbers of the sources where the whole pattern is found;
    None if it is found fewer than 10 times."""
    num_exp_clusters = len(xy_mean_exp)

    # Find locations of pattern and assign sources to exposures
    exp_num = np.zeros(len(coords), dtype=int)
    kdt_coords = KDT(coords[:,:2])
    xy_found = np.empty((0, 2))

    for i in np.arange(len(coords)):
        # Take a source from list
        xy_eval = coords[i,:2]

        # Search for matches
        if fit is not None:
            C_angle, C_scale = fit
            theta = np.radians(C_angle[0] * xy_eval[0] + C_angle[2])
            rot = np.array(((np.cos(-theta), -np.sin(-theta)),
                            (np.sin(-theta),  np.cos(-theta))))
            scale = 1./(C_scale[1] * xy_eval[1] + C_scale[2])
            xy_transform = xy_eval + xy_mean_exp.dot(scale*rot)
            ds,ind = kdt_coords.query(xy_transform)
        else:
            ds,ind = kdt_coords.query(xy_eval+xy_mean_exp)

        if (ds < 10).sum() == num_exp_clusters:
            # Append found pattern to list of finds
            xy_found = np.vstack((xy_found, xy_eval))
            exp_num[ind] = np.arange(1, num_exp_clusters+1)

    num_found = len(xy_found)
    ctx.log.write('Found {:d} patterns'.format(num_found),
                  level=4, event=31)

    if num_found < 10:
        ctx.log.write('Not enough patterns found (<10)!',
                      level=2, event=31)
        exp_num = None

    return exp_num
