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


def _available_memory():
    """Bytes of memory available now (Linux), else 8 GiB."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30


def _slice_projector(geom, vol_geom, be, cpu_matrix, memory_bytes, log):
    """A sparse.SliceProjector when there is no hardware GPU and the
    geometry and memory allow one, else None."""
    from . import sparse
    if cpu_matrix == 'none':
        return None
    if cpu_matrix not in sparse.MATRICES:
        raise ValueError(f'Unknown cpu_matrix {cpu_matrix!r}; choose one of '
                         f"{sparse.MATRICES + ('none',)}")
    if be.name != 'cpu' and not be.software:
        return None
    if memory_bytes is None:
        memory_bytes = _available_memory() // 4
    why = sparse.unsuitable(geom, vol_geom, cpu_matrix, memory_bytes)
    if why is not None:
        log(f'No sparse matrix per slice: {why}')
        return None
    return sparse.SliceProjector(geom, vol_geom, cpu_matrix)


def _safe_inverse(a, rel=1e-6):
    """1 / a where a is meaningfully positive, 0 elsewhere."""
    out = np.zeros_like(a)
    ok = a > rel * float(a.max()) if a.size else a > 0
    out[ok] = 1.0 / a[ok]
    return out


def sirt(projections, geom, vol_geom=None, iterations=50, relaxation=1.0,
         nonnegative=True, backend='cpu', x0=None, progress=None,
         should_stop=None, callback=None, volume_needed=None, log=None,
         cpu_matrix='joseph', cpu_memory_bytes=None):
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
    cpu_matrix: without a hardware GPU (the numpy backend or a software
        adapter), parallel beams perpendicular to the axis are projected
        with a sparse matrix per slice (tomviz.projectors.sparse), far
        faster: 'joseph' (the same interpolation as the other
        projectors) or 'siddon' (the matrix of Tomviz's ART and SIRT
        operators); 'none' keeps the backend's projectors. Used when the
        matrix fits in cpu_memory_bytes (default: a quarter of the memory
        available).

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
    slices = _slice_projector(geom, vol_geom, be, cpu_matrix,
                              cpu_memory_bytes, log)
    log(f'Projectors: {slices.description if slices else be.description}')

    b = np.ascontiguousarray(projections, dtype=np.float32)
    if b.shape != (geom.n_angles, geom.n_v, geom.n_u):
        raise ValueError(f'Projections of shape {b.shape} do not match the '
                         'geometry')

    if slices is not None:
        fp, bp = slices.forward, slices.back
    else:
        def fp(x):
            return be.forward_project(x, geom, vol_geom,
                                      should_stop=should_stop)

        def bp(p):
            return be.backproject(p, geom, vol_geom,
                                  should_stop=should_stop)

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

    # The GPU's resident loop, unless projecting slice by slice.
    sirt_loop = be.sirt_loop if slices is None else None
    if sirt_loop is not None:
        def progress_it(it):
            if progress is not None:
                progress((it + 2) / steps)

        # b, R and C stay on the device; only the volume comes back.
        result = sirt_loop(b, R, C, x, geom, vol_geom, iterations,
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
