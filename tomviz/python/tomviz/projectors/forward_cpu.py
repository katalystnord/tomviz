###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Ray-driven forward projection in numpy (Joseph's method).

Every detector pixel's ray is sampled once per voxel plane along its
dominant axis (the axis it is most nearly parallel to), with bilinear
interpolation inside the plane, and the samples are summed times the ray
length per plane. Voxels outside the volume count as zero. Works for any
geometry with ray_parameters() (cone and parallel beams).

This is the reference implementation; the WebGPU backend matches it.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

import numpy as np

from .backproject_cpu import default_threads


def index_space_rays(geom, vol_geom):
    """ray_parameters() converted to continuous voxel index coordinates
    (ix, iy, iz), where voxel centres are integers, plus the voxel size
    (dx, dy, dz) to convert index steps back to physical lengths."""
    dz, dy, dx = vol_geom.voxel_size
    scale = np.array([dx, dy, dz])
    first = np.array([vol_geom.x_coords()[0], vol_geom.y_coords()[0],
                      vol_geom.z_coords()[0]])
    o0, o_u, o_v, d0, d_u, d_v = geom.ray_parameters()
    return ((o0 - first) / scale, o_u / scale, o_v / scale, d0 / scale,
            d_u / scale, d_v / scale), scale


def _project_one(vol, rays, scale, k, n_u, n_v):
    """Forward projection k, shape (n_v, n_u)."""
    nz, ny, nx = vol.shape
    sizes = np.array([nx, ny, nz])
    strides = np.array([1, nx, nx * ny])
    flat = vol.ravel()
    o0, o_u, o_v, d0, d_u, d_v = (r[k] for r in rays)
    iv, iu = np.mgrid[0:n_v, 0:n_u]
    iu = iu.ravel().astype(np.float64)
    iv = iv.ravel().astype(np.float64)
    o = o0 + iu[:, None] * o_u + iv[:, None] * o_v
    d = d0 + iu[:, None] * d_u + iv[:, None] * d_v
    dom = np.argmax(np.abs(d), axis=1)
    out = np.zeros(n_u * n_v, dtype=np.float64)
    for a in range(3):
        sel = np.nonzero(dom == a)[0]
        if len(sel) == 0:
            continue
        b, c = [i for i in range(3) if i != a]
        oa, ob, oc = o[sel, a], o[sel, b], o[sel, c]
        da, db, dc = d[sel, a], d[sel, b], d[sel, c]
        # Physical length of ray per unit step along axis a.
        step = np.linalg.norm(d[sel] * scale, axis=1) / np.abs(da)
        acc = np.zeros(len(sel))
        for p in range(sizes[a]):
            t = (p - oa) / da
            qb = ob + t * db
            qc = oc + t * dc
            b0 = np.floor(qb)
            c0 = np.floor(qc)
            fb = qb - b0
            fc = qc - c0
            b0 = b0.astype(np.int64)
            c0 = c0.astype(np.int64)
            for db_, wb in ((0, 1 - fb), (1, fb)):
                bi = b0 + db_
                okb = (bi >= 0) & (bi < sizes[b])
                for dc_, wc in ((0, 1 - fc), (1, fc)):
                    ci = c0 + dc_
                    ok = okb & (ci >= 0) & (ci < sizes[c])
                    idx = np.where(ok, p * strides[a] + bi * strides[b] +
                                   ci * strides[c], 0)
                    acc += np.where(ok, flat[idx], 0.0) * (wb * wc)
        out[sel] = acc * step
    return out.reshape(n_v, n_u).astype(np.float32)


def forward_project(volume, geom, vol_geom, threads=None, progress=None,
                    should_stop=None):
    """Line integrals of a (n_z, n_y, n_x) volume through every detector
    pixel: a new float32 (n_angles, n_v, n_u) array.

    progress(done, total) is called as projections finish. should_stop()
    is polled; if it returns True the function returns None.
    """
    threads = threads or default_threads()
    should_stop = should_stop or (lambda: False)
    vol = np.ascontiguousarray(volume, dtype=np.float32)
    if vol.shape != vol_geom.shape:
        raise ValueError(f'Volume shape {vol.shape} does not match '
                         f'{vol_geom.shape}')
    rays, scale = index_space_rays(geom, vol_geom)
    n = geom.n_angles
    out = np.empty((n, geom.n_v, geom.n_u), dtype=np.float32)
    stop = threading.Event()

    def work(k):
        if stop.is_set():
            return None
        out[k] = _project_one(vol, rays, scale, k, geom.n_u, geom.n_v)
        return k

    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        pending = {pool.submit(work, k) for k in range(n)}
        while pending:
            finished, pending = wait(pending, timeout=0.2,
                                     return_when=FIRST_COMPLETED)
            if should_stop():
                stop.set()
                return None
            done += len(finished)
            for fut in finished:
                fut.result()
            if progress is not None and finished:
                progress(done, n)
    return out
