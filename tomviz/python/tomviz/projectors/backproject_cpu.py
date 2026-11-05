###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Voxel-driven cone-beam backprojection in numpy.

This is the reference implementation: every other backend must match it.
Work is split into slabs of z slices that run on a thread pool (numpy
releases the GIL in the array arithmetic, so threads do run in parallel).
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


def default_threads():
    env = os.environ.get('TOMVIZ_CONEBEAM_THREADS')
    if env:
        return max(1, int(env))
    return os.cpu_count() or 1


def _bilinear(img, u, v):
    """Sample img (n_v, n_u) at fractional index coordinates; zero outside
    the detector."""
    n_v, n_u = img.shape
    u0 = np.floor(u)
    v0 = np.floor(v)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    u0 = u0.astype(np.int64)
    v0 = v0.astype(np.int64)
    flat = img.ravel()
    out = np.zeros(u.shape, dtype=np.float32)
    for dv, wv in ((0, 1 - fv), (1, fv)):
        vi = v0 + dv
        okv = (vi >= 0) & (vi < n_v)
        for du, wu in ((0, 1 - fu), (1, fu)):
            ui = u0 + du
            ok = okv & (ui >= 0) & (ui < n_u)
            idx = np.where(ok, vi * n_u + ui, 0)
            out += np.where(ok, flat[idx], 0.0) * (wu * wv)
    return out


def _backproject_slab(filtered, P, x, y, z, should_stop):
    """Backproject every projection into the voxels z (nzs,) x y x x."""
    acc = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    X = x[None, None, :]
    Y = y[None, :, None]
    Z = z[:, None, None]
    for k in range(len(P)):
        if should_stop():
            return None
        m = P[k]
        den = m[2, 0] * X + m[2, 1] * Y + m[2, 2] * Z + m[2, 3]
        inv = 1.0 / den
        u = (m[0, 0] * X + m[0, 1] * Y + m[0, 2] * Z + m[0, 3]) * inv
        v = (m[1, 0] * X + m[1, 1] * Y + m[1, 2] * Z + m[1, 3]) * inv
        acc += _bilinear(filtered[k], u, v) * (inv * inv).astype(np.float32)
    return acc


def backproject(filtered, geom, vol_geom, threads=None, slab=None,
                progress=None, should_stop=None):
    """FDK backprojection of pre-weighted, filtered projections
    (n_angles, n_v, n_u) into a new (n_z, n_y, n_x) float32 volume.

    progress(done_slices, total_slices) is called as slabs finish.
    should_stop() is polled; if it returns True the function returns None.
    """
    threads = threads or default_threads()
    should_stop = should_stop or (lambda: False)
    filtered = np.ascontiguousarray(filtered, dtype=np.float32)
    P = geom.projection_matrices()
    nz, ny, nx = vol_geom.shape
    x, y, z = vol_geom.x_coords(), vol_geom.y_coords(), vol_geom.z_coords()
    if slab is None:
        # About 2 M voxels per slab: big enough for numpy to be efficient,
        # small enough to spread over the threads and keep memory low.
        slab = max(1, min(nz, (1 << 21) // max(1, ny * nx)))
        slab = max(1, min(slab, -(-nz // threads)))
    starts = list(range(0, nz, slab))
    vol = np.zeros(vol_geom.shape, dtype=np.float32)
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(_backproject_slab, filtered, P, x, y,
                        z[s:s + slab], should_stop): s
            for s in starts
        }
        for fut in futures:
            s = futures[fut]
            res = fut.result()
            if res is None:
                return None
            vol[s:s + slab] = res
            done += len(res)
            if progress is not None:
                progress(done, nz)
    if should_stop():
        return None
    return vol
