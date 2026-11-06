###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""SIRT reconstruction for any geometry (cone or parallel beam).

    x <- x + lambda * C * BP(R * (b - FP(x)))

with R = 1 / FP(1) (inverse ray lengths through the volume) and
C = 1 / BP(1). FP is the ray-driven forward projector and BP the
voxel-driven backprojector. They are not exact transposes of each other
(an unmatched pair, as in ASTRA's GPU SIRT), but BP is proportional to the
transpose up to a smooth weight, which C normalises away.
"""

import numpy as np


def _safe_inverse(a, rel=1e-6):
    """1 / a where a is meaningfully positive, 0 elsewhere."""
    out = np.zeros_like(a)
    ok = a > rel * float(a.max()) if a.size else a > 0
    out[ok] = 1.0 / a[ok]
    return out


def sirt(projections, geom, vol_geom=None, iterations=50, relaxation=1.0,
         nonnegative=True, backend='cpu', x0=None, progress=None,
         should_stop=None, callback=None, volume_needed=None, log=None):
    """Reconstruct a volume from line-integral projections.

    projections: (n_angles, n_v, n_u) array.
    geom: ConeGeometry or ParallelGeometry. vol_geom: VolumeGeometry
        (default: geom.default_volume()).
    relaxation: lambda, between 0 and 2; 1 is the usual choice.
    nonnegative: clip negative values after every iteration.
    backend: 'cpu', 'gpu' or 'auto', or a Backend.
    x0: starting volume (default zeros).
    progress(fraction) is called with values in [0, 1]. should_stop() is
    polled once per iteration; if it says True, SIRT returns None.
    callback(iteration, x) runs after every iteration (for live views);
    if it returns True, SIRT stops early and returns the current x.
    volume_needed(iteration), if given, says whether the callback will
    use x at that iteration; when it says False, x may be None. On a GPU
    that saves copying the volume back every iteration.
    log(message) receives human-readable notes.

    Returns a float32 (n_z, n_y, n_x) volume.
    """
    log = log or (lambda msg: None)
    should_stop = should_stop or (lambda: False)
    if vol_geom is None:
        vol_geom = geom.default_volume()
    if not 0 < relaxation < 2:
        raise ValueError('The relaxation must be between 0 and 2')

    from .backend import get_backend
    be = get_backend(backend) if isinstance(backend, str) else backend
    log(f'Projectors: {be.description}')

    b = np.ascontiguousarray(projections, dtype=np.float32)
    if b.shape != (geom.n_angles, geom.n_v, geom.n_u):
        raise ValueError(f'Projections of shape {b.shape} do not match the '
                         'geometry')

    def fp(x):
        return be.forward_project(x, geom, vol_geom,
                                  should_stop=should_stop)

    def bp(p):
        return be.backproject(p, geom, vol_geom, should_stop=should_stop)

    # Two extra projections' worth of work, so the progress bar moves.
    steps = iterations + 1
    row = fp(np.ones(vol_geom.shape, dtype=np.float32))
    if row is None:
        return None
    R = _safe_inverse(row)
    del row
    col = bp(np.ones_like(b))
    if col is None:
        return None
    C = _safe_inverse(col) * np.float32(relaxation)
    del col
    if progress is not None:
        progress(1 / steps)

    if x0 is None:
        x = np.zeros(vol_geom.shape, dtype=np.float32)
    else:
        x = np.array(x0, dtype=np.float32, copy=True)
        if x.shape != vol_geom.shape:
            raise ValueError('x0 does not match the volume')

    if be.sirt_loop is not None:
        def progress_it(it):
            if progress is not None:
                progress((it + 2) / steps)

        # b, R and C stay on the device; only the volume comes back.
        result = be.sirt_loop(b, R, C, x, geom, vol_geom, iterations,
                              nonnegative=nonnegative,
                              should_stop=should_stop,
                              progress=progress_it, callback=callback,
                              volume_needed=volume_needed, log=log)
        if result is not NotImplemented:
            return result
    for it in range(iterations):
        if should_stop():
            return None
        r = fp(x)
        if r is None:
            return None
        np.subtract(b, r, out=r)
        r *= R
        u = bp(r)
        if u is None:
            return None
        del r
        u *= C
        x += u
        del u
        if nonnegative:
            np.maximum(x, 0, out=x)
        if progress is not None:
            progress((it + 2) / steps)
        if callback is not None and callback(it, x):
            break
    return x
