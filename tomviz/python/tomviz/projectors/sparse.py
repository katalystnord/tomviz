###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Parallel-beam projection on the CPU with one sparse matrix per slice.

When the rays are perpendicular to the rotation axis (det_u level, det_v
along the axis), every detector row sees one plane of the volume, so a
projection is a 2D system matrix applied to every slice at once (one
sparse-times-dense product, threaded over slices) followed by linear
interpolation from slices to detector rows. Backprojection is the exact
transpose. For SIRT without a GPU this is far faster than the numpy
projectors, and faster than Tomviz's existing SIRT, which applies the
same kind of matrix one slice at a time (256^3 from 180 projections,
24 threads: 0.46 s for a projection and backprojection against 13.3 s
per iteration).

Two matrices:

* 'joseph': Joseph's method, the interpolation of forward_cpu and of the
  GPU projector (equal to forward_cpu to float rounding);
* 'siddon': path lengths through each pixel, the matrix of Tomviz's ART,
  SIRT and TV minimisation operators (tomviz.projectors.siddon), for
  square slices of square pixels with the rotation axis at the detector
  centre. It treats pixels as flat, so a ray within a few degrees of an
  axis reads the nearest pixel row instead of interpolating: on a smooth
  test object 3 % from the true line integrals at 2 degrees, against
  0.2 % for Joseph (equal elsewhere).

The matrix grows with the slice size, the detector width and the number
of projections (about 2 * n_angles * n_u * max(nx, ny) entries): 512^2
slices from 720 projections take 4.5 GB, 1024^2 from 900 about 23 GB.
unsuitable() refuses what does not fit the memory given.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .geometry import ParallelGeometry

MATRICES = ('joseph', 'siddon')
BYTES_PER_ENTRY = 8   # float32 value and int32 column index


def _level(geom, vol_geom):
    """Why the geometry does not separate into slices, or None."""
    if not isinstance(geom, ParallelGeometry):
        return 'not a parallel beam'
    tol = 1e-9
    if np.any(np.abs(geom.ray[:, 2]) >
              tol * np.linalg.norm(geom.ray, axis=1)):
        return 'the rays are not perpendicular to the rotation axis'
    if np.any(np.abs(geom.det_u[:, 2]) >
              tol * np.linalg.norm(geom.det_u, axis=1)):
        return 'the detector rows are tilted'
    if np.any(np.linalg.norm(geom.det_v[:, :2], axis=1) >
              tol * np.linalg.norm(geom.det_v, axis=1)):
        return 'the detector columns are not along the rotation axis'
    rows_z = geom.det_center[:, 2]
    if np.ptp(rows_z) > tol * max(1.0, np.abs(rows_z).max()) or \
            np.ptp(geom.det_v[:, 2]) > tol * np.abs(geom.det_v[:, 2]).max():
        return 'the detector moves along the axis between projections'
    return None


def matrix_entries(geom, vol_geom):
    """Roughly how many non-zeros the slice matrix has."""
    _, ny, nx = vol_geom.shape
    return 2 * geom.n_angles * geom.n_u * max(nx, ny)


def unsuitable(geom, vol_geom, matrix='joseph', memory_bytes=None):
    """Why SliceProjector cannot be used here, or None."""
    if matrix not in MATRICES:
        return f'unknown matrix {matrix!r}; choose one of {MATRICES}'
    why = _level(geom, vol_geom)
    if why:
        return why
    if matrix == 'siddon':
        why = _siddon_unsuitable(geom, vol_geom)
        if why:
            return why
    # The matrix and its transpose.
    need = 2 * BYTES_PER_ENTRY * matrix_entries(geom, vol_geom)
    if memory_bytes is not None and need > memory_bytes:
        return (f'the slice matrix would take {need / 1e9:.1f} GB, more '
                f'than the {memory_bytes / 1e9:.1f} GB allowed')
    return None


def joseph_matrix(geom, vol_geom):
    """(n_angles * n_u, ny * nx) Joseph weights for one slice: row
    k * n_u + u is ray u of projection k, column y * nx + x a pixel."""
    import scipy.sparse as ss
    from .forward_cpu import index_space_rays
    (o0, o_u, _, d0, _, _), scale = index_space_rays(geom, vol_geom)
    _, ny, nx = vol_geom.shape
    n, n_u = geom.n_angles, geom.n_u
    iu = np.arange(n_u)
    rows, cols, vals = [], [], []
    for k in range(n):
        o = o0[k][None, :2] + iu[:, None] * o_u[k][None, :2]
        d = d0[k][:2]
        a = 0 if abs(d[0]) >= abs(d[1]) else 1   # one sample per plane
        b = 1 - a
        na, nb = (nx, ny) if a == 0 else (ny, nx)
        step = np.linalg.norm(d * scale[:2]) / abs(d[a])
        p = np.arange(na)
        qb = o[:, b:b + 1] + (p[None, :] - o[:, a:a + 1]) / d[a] * d[b]
        fb = np.floor(qb)
        w1 = qb - fb
        b0 = fb.astype(np.int64)
        r = np.broadcast_to((k * n_u + iu)[:, None], b0.shape)
        planes = np.broadcast_to(p[None, :], b0.shape)
        for bi, w in ((b0, 1 - w1), (b0 + 1, w1)):
            ok = (bi >= 0) & (bi < nb) & (w > 0)
            if a == 0:
                c = bi[ok] * nx + planes[ok]
            else:
                c = planes[ok] * nx + bi[ok]
            rows.append(r[ok])
            cols.append(c)
            vals.append((w[ok] * step).astype(np.float32))
    return ss.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n_u, ny * nx), dtype=np.float32)


def _angles(geom):
    """Rotation angle of each projection in radians, 0 with the rays
    along -x (as ParallelGeometry.circular)."""
    return np.unwrap(np.arctan2(-geom.ray[:, 1], -geom.ray[:, 0]))


def _siddon_unsuitable(geom, vol_geom):
    _, ny, nx = vol_geom.shape
    _, dy, dx = vol_geom.voxel_size
    if nx != ny or abs(dx - dy) > 1e-9 * dx:
        return 'the Siddon matrix needs square slices of square pixels'
    if any(abs(c) > 1e-9 for c in vol_geom.center[1:]):
        return 'the Siddon matrix needs the volume centred on the axis'
    pitch = np.linalg.norm(geom.det_u, axis=1)
    if np.ptp(pitch) > 1e-9 * pitch.max():
        return 'the Siddon matrix needs one detector pitch'
    # The rotation axis must project onto the detector centre, and det_u
    # point the way ParallelGeometry.circular has it at each angle.
    along = np.einsum('ij,ij->i', geom.det_center[:, :2],
                      geom.det_u[:, :2]) / pitch
    if np.abs(along).max() > 1e-6 * pitch.max():
        return 'the Siddon matrix needs the axis at the detector centre'
    beta = _angles(geom)
    e_u = np.stack([-np.sin(beta), np.cos(beta)], axis=1)
    if np.any(np.einsum('ij,ij->i', geom.det_u[:, :2], e_u) <= 0):
        return 'the Siddon matrix needs the usual detector orientation'
    return None


def siddon_matrix(geom, vol_geom):
    """Tomviz's ART/SIRT matrix (tomviz.projectors.siddon.parallelRay),
    with its rows and columns in joseph_matrix's order: same angles and
    ray order; parallelRay numbers pixels with x and y the other way
    round (found by comparing the two on test images: 0.2 % apart, the
    difference between path lengths and interpolation)."""
    from .siddon import parallelRay
    _, ny, nx = vol_geom.shape
    pitch = float(np.linalg.norm(geom.det_u[0]))
    A = parallelRay(nx, vol_geom.voxel_size[2], np.rad2deg(_angles(geom)),
                    geom.n_u, pitch).tocsc()
    # Column r * nx + c of parallelRay is pixel (y = c, x = r) here.
    perm = np.arange(ny * nx).reshape(ny, nx).T.ravel()
    return A[:, perm].tocsr()


def row_weights(geom, vol_geom):
    """(n_v, nz) linear interpolation from slices to detector rows, or
    None when every row lies on a slice."""
    import scipy.sparse as ss
    from .forward_cpu import index_space_rays
    (o0, _, o_v, _, _, _), _ = index_space_rays(geom, vol_geom)
    nz = vol_geom.shape[0]
    z = o0[0, 2] + np.arange(geom.n_v) * o_v[0, 2]   # slice coordinates
    if geom.n_v == nz and np.allclose(z, np.arange(nz), atol=1e-6):
        return None
    f = np.floor(z)
    w1 = z - f
    rows, cols, vals = [], [], []
    for c, w in ((f, 1 - w1), (f + 1, w1)):
        ok = (c >= 0) & (c < nz) & (w > 0)
        rows.append(np.flatnonzero(ok))
        cols.append(c[ok].astype(np.int64))
        vals.append(w[ok])
    return ss.csr_matrix((np.concatenate(vals).astype(np.float32),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(geom.n_v, nz), dtype=np.float32)


_cache = {}


def _key(geom, vol_geom, matrix):
    _, ny, nx = vol_geom.shape
    return (matrix, geom.n_u, ny, nx, vol_geom.voxel_size[1:],
            vol_geom.center[1:], geom.ray[:, :2].tobytes(),
            geom.det_center[:, :2].tobytes(), geom.det_u[:, :2].tobytes())


class SliceProjector:
    """Forward projection and its transpose for a level parallel beam.
    The slice matrix is cached (the last one only), so the slabs of a
    streamed reconstruction share it."""

    def __init__(self, geom, vol_geom, matrix='joseph', threads=None):
        why = unsuitable(geom, vol_geom, matrix)
        if why:
            raise ValueError(f'Cannot project slice by slice: {why}')
        from .backproject_cpu import default_threads
        self.geom, self.vol_geom = geom, vol_geom
        self.threads = threads or default_threads()
        key = _key(geom, vol_geom, matrix)
        if key not in _cache:
            _cache.clear()
            A = joseph_matrix(geom, vol_geom) if matrix == 'joseph' else \
                siddon_matrix(geom, vol_geom)
            _cache[key] = (A, A.T.tocsr())
        self.A, self.AT = _cache[key]
        self.Z = row_weights(geom, vol_geom)
        self.description = (f'CPU, {self.threads} threads, a {matrix} '
                            f'matrix per slice ({self.A.nnz / 1e6:.0f} M '
                            'non-zeros)')

    def _threaded(self, M, X):
        """M @ X, with the columns of X (one per slice) split over
        threads; scipy's sparse products release the GIL."""
        parts = [c for c in np.array_split(np.arange(X.shape[1]),
                                           self.threads) if len(c)]
        out = np.empty((M.shape[0], X.shape[1]), dtype=np.float32)

        def run(c):
            out[:, c] = M @ np.ascontiguousarray(X[:, c])

        with ThreadPoolExecutor(len(parts)) as pool:
            list(pool.map(run, parts))
        return out

    def forward(self, x):
        nz, ny, nx = self.vol_geom.shape
        n, n_v, n_u = self.geom.n_angles, self.geom.n_v, self.geom.n_u
        X = np.asarray(x, dtype=np.float32).reshape(nz, ny * nx).T
        S = self._threaded(self.A, X)                     # (n * n_u, nz)
        if self.Z is not None:
            S = np.asarray((self.Z @ S.T).T, dtype=np.float32)
        return np.ascontiguousarray(
            S.reshape(n, n_u, n_v).transpose(0, 2, 1))

    def back(self, p):
        nz, ny, nx = self.vol_geom.shape
        n, n_v, n_u = self.geom.n_angles, self.geom.n_v, self.geom.n_u
        P = np.asarray(p, dtype=np.float32).transpose(0, 2, 1).reshape(
            n * n_u, n_v)
        if self.Z is not None:
            P = np.asarray((self.Z.T @ P.T).T, dtype=np.float32)
        U = self._threaded(self.AT, P)                    # (ny * nx, nz)
        return np.ascontiguousarray(U.T.reshape(nz, ny, nx))
