###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""FDK reconstruction: filtering on the CPU, backprojection on the chosen
backend."""

import numpy as np

from .filtering import preweight_and_filter


def fdk(projections, geom, vol_geom=None, filter_name='ram-lak', cutoff=1.0,
        short_scan='auto', backend='cpu', threads=None, progress=None,
        should_stop=None, log=None, overwrite=False):
    """Reconstruct a volume from cone-beam line-integral projections.

    projections: (n_angles, n_v, n_u) array of -log(I/I0) values.
    geom: ConeGeometry. vol_geom: VolumeGeometry (default: geom's default).
    backend: 'cpu', 'gpu' or 'auto', or a Backend from backend.get_backend.
    progress(fraction) is called with values in [0, 1].
    should_stop() is polled; FDK returns None when it says True.
    log(message) receives human-readable notes (which weights, backend).
    overwrite: filter in place, reusing the projections' memory, when they
    are a C-contiguous float32 array. Halves peak memory on large scans.

    Returns a float32 (n_z, n_y, n_x) volume in units of 1 / length.
    """
    log = log or (lambda msg: None)
    if vol_geom is None:
        vol_geom = geom.default_volume()

    from tomviz.projectors.backend import get_backend
    be = get_backend(backend) if isinstance(backend, str) else backend
    log(f'Backprojection backend: {be.description}')

    out = None
    if (overwrite and isinstance(projections, np.ndarray) and
            projections.dtype == np.float32 and
            projections.flags.c_contiguous and projections.flags.writeable):
        out = projections
    filtered, note = preweight_and_filter(projections, geom, filter_name,
                                          cutoff, short_scan, out=out)
    log(f'Redundancy weights: {note}')
    if should_stop is not None and should_stop():
        return None
    if progress is not None:
        progress(0.05)

    def bp_progress(done, total):
        if progress is not None:
            progress(0.05 + 0.95 * done / total)

    return be.backproject(filtered, geom, vol_geom, threads=threads,
                          progress=bp_progress, should_stop=should_stop)
