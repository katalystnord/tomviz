###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""FDK for scans larger than memory: projections read from a file, the
volume written to a file, a slab of slices at a time.

For each slab of the volume only the detector rows it projects onto are
read, pre-weighted and filtered (FDK's filter runs along rows, so rows are
independent), and backprojected through a row-cropped copy of the
geometry. Memory use is bounded by the slab, not by the data. A reader
thread fetches the next slab's rows while the current slab is
reconstructed, and a writer thread stores finished slabs, so disk and GPU
work at the same time. On a fast NVMe drive (about 7 GB/s here) reading is
a small part of the work.

The result is identical to the in-memory fdk(), up to float rounding.
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np

from tomviz.projectors.geometry import ConeGeometry, VolumeGeometry

from .filtering import preweight_and_filter


class RawStack:
    """A C-ordered float32 (or other dtype) array in a file, read and
    written with positioned I/O (which releases the GIL, so reads and
    writes overlap the reconstruction). offset: header bytes to skip, as
    in .npy files."""

    def __init__(self, path, shape, dtype=np.float32, offset=0,
                 mode='r'):
        self.path = path
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.offset = int(offset)
        flags = os.O_RDONLY if mode == 'r' else os.O_RDWR | os.O_CREAT
        self.fd = os.open(path, flags, 0o644)
        if mode != 'r':
            size = self.offset + int(np.prod(self.shape)) * \
                self.dtype.itemsize
            os.ftruncate(self.fd, size)

    @classmethod
    def npy(cls, path, mode='r'):
        """Open an existing .npy file (C order) without loading it."""
        arr = np.load(path, mmap_mode='r')
        if not arr.flags.c_contiguous:
            raise ValueError(f'{path}: needs a C-ordered array')
        return cls(path, arr.shape, arr.dtype, arr.offset, mode)

    def close(self):
        os.close(self.fd)

    def read_rows(self, r0, r1, out=None):
        """All frames, rows r0..r1: an (n, r1 - r0, n_u) array."""
        n, n_v, n_u = self.shape
        rows = r1 - r0
        if out is None:
            out = np.empty((n, rows, n_u), dtype=self.dtype)
        size = rows * n_u * self.dtype.itemsize
        for k in range(n):
            start = self.offset + ((k * n_v + r0) * n_u) * \
                self.dtype.itemsize
            view = memoryview(out[k]).cast('B')
            got = os.preadv(self.fd, [view], start)
            if got != size:
                raise IOError(f'{self.path}: short read')
        return out

    def write_slab(self, z0, data):
        """data (nz, ny, nx) at slices z0.. of a (n_z, n_y, n_x) file."""
        data = np.ascontiguousarray(data, dtype=self.dtype)
        start = self.offset + z0 * int(np.prod(self.shape[1:])) * \
            self.dtype.itemsize
        view = memoryview(data).cast('B')
        done = 0
        while done < len(view):
            done += os.pwrite(self.fd, view[done:], start + done)


def crop_rows(geom, r0, r1):
    """The geometry of detector rows r0..r1 only: the same source and
    pixel steps, the detector centre moved to the middle of those rows."""
    shift = (r0 + (r1 - r0 - 1) / 2) - (geom.n_v - 1) / 2
    return ConeGeometry(geom.source, geom.det_center + shift * geom.det_v,
                        geom.det_u, geom.det_v, geom.n_u, r1 - r0,
                        nominal_angles=geom.nominal_angles)


def slab_geometry(vol_geom, z0, z1):
    """The volume slices z0..z1 as a volume of their own."""
    z = vol_geom.z_coords()
    _, cy, cx = vol_geom.center
    return VolumeGeometry((z1 - z0, vol_geom.shape[1], vol_geom.shape[2]),
                          vol_geom.voxel_size,
                          center=(float(z[z0:z1].mean()), cy, cx))


def rows_for(geom, vol_geom, z0, z1, margin=2):
    """The detector rows that slices z0..z1 project onto, over all
    projections (with a margin for interpolation)."""
    corners = vol_geom.slab_corners(z0, z1)
    h = np.einsum('kij,cj->kci', geom.projection_matrices(),
                  np.concatenate([corners, np.ones((8, 1))], axis=1))
    if np.any(h[:, :, 2] <= 0):
        raise RuntimeError('Part of the volume is behind the source; make '
                           'the volume smaller')
    v = h[:, :, 1] / h[:, :, 2]
    r0 = int(np.clip(np.floor(v.min()) - margin, 0, geom.n_v))
    r1 = int(np.clip(np.ceil(v.max()) + margin + 1, 0, geom.n_v))
    return r0, max(r1, r0 + 1)


def default_memory_bytes(fraction=0.5, floor=1 << 30):
    """Half the memory available now (Linux /proc/meminfo; 4 GiB where
    that cannot be read). The rows a slab needs are the costly part: with
    a cone beam even one slice near the top or bottom of the volume
    projects onto a tall band of detector rows, so a small budget forces
    slabs of a slice or two and the per-slab overheads dominate (2048^3
    from 720 projections: 1408 s within 4 GiB, 193 s within 20 GiB)."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return max(floor, int(int(line.split()[1]) * 1024 *
                                          fraction))
    except OSError:
        pass
    return 4 << 30


def plan_slabs(geom, vol_geom, memory_bytes):
    """Slabs (z0, z1, r0, r1) whose working set fits in memory_bytes
    (approximately: library overheads come on top)."""
    nz, ny, nx = vol_geom.shape
    n, n_u = geom.n_angles, geom.n_u
    slabs = []
    z0 = 0
    while z0 < nz:
        lo, hi = 1, nz - z0
        while lo < hi:  # the largest slab that fits
            mid = (lo + hi + 1) // 2
            r0, r1 = rows_for(geom, vol_geom, z0, z0 + mid)
            # Rows: being read ahead, filtered, kept from the last slab.
            # Volume slab: the result, the GPU's read-back copy, and the
            # previous slab still being written.
            need = 4 * (3 * n * (r1 - r0) * n_u + 3 * mid * ny * nx)
            if need <= memory_bytes:
                lo = mid
            else:
                hi = mid - 1
        r0, r1 = rows_for(geom, vol_geom, z0, z0 + lo)
        slabs.append((z0, z0 + lo, r0, r1))
        z0 += lo
    return slabs


def new_pieces(window, previous):
    """Row ranges of window (r0, r1) not in previous (p0, p1): at most two
    pieces, below and above the overlap."""
    r0, r1 = window
    if previous is None:
        return [(r0, r1)]
    p0, p1 = previous
    lo, hi = max(r0, p0), min(r1, p1)
    if lo >= hi:
        return [(r0, r1)]
    return [(a, b) for a, b in ((r0, lo), (hi, r1)) if b > a]


def fdk_stream(source, geom, vol_geom, target, filter_name='ram-lak',
               cutoff=1.0, short_scan='auto', backend='auto',
               memory_bytes=None, progress=None, should_stop=None,
               log=None):
    """FDK from source (a RawStack of (n_angles, n_v, n_u) line integrals)
    into target (a RawStack of vol_geom.shape float32), a slab at a time
    within memory_bytes (default: half the memory available now).

    Neighbouring slabs need overlapping detector rows. Filtering works row
    by row, so the previous slab's filtered rows are kept and each slab
    reads and filters only the rows new to it (like the resident pages of
    a paged pool). A reader thread fetches the next slab's new rows while
    the current slab is filtered and backprojected; a writer thread stores
    finished slabs.

    Returns timings in seconds: read (waiting for rows), filter,
    backproject, write (waiting for the writer), total, and the number of
    detector rows read and filtered (rows)."""
    import time
    from tomviz.projectors.backend import get_backend
    log = log or (lambda msg: None)
    be = get_backend(backend) if isinstance(backend, str) else backend
    if tuple(source.shape) != (geom.n_angles, geom.n_v, geom.n_u):
        raise ValueError(f'Projections of {source.shape} do not match the '
                         'geometry')
    if tuple(target.shape) != tuple(vol_geom.shape):
        raise ValueError('The output file does not match the volume')
    if memory_bytes is None:
        memory_bytes = default_memory_bytes()
    slabs = plan_slabs(geom, vol_geom, memory_bytes)
    windows = [(r0, r1) for _, _, r0, r1 in slabs]
    pieces = [new_pieces(w, windows[i - 1] if i else None)
              for i, w in enumerate(windows)]
    log(f'{len(slabs)} slabs; backprojection: {be.description}')
    t = dict(read=0.0, filter=0.0, backproject=0.0, write=0.0, total=0.0,
             rows=0)

    def read(i):
        return [source.read_rows(a, b) for a, b in pieces[i]]

    start = time.perf_counter()
    kept = None   # (r0, r1, filtered rows) of the previous slab
    with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as writer:
        pending_write = None
        next_raw = reader.submit(read, 0)
        for i, (z0, z1, r0, r1) in enumerate(slabs):
            if should_stop is not None and should_stop():
                return None
            a = time.perf_counter()
            raw = next_raw.result()
            t['read'] += time.perf_counter() - a
            if i + 1 < len(slabs):
                next_raw = reader.submit(read, i + 1)

            # Filter the new rows; take the rest from the previous slab.
            a = time.perf_counter()
            filtered = np.empty((geom.n_angles, r1 - r0, geom.n_u),
                                dtype=np.float32)
            for (p0, p1), rows in zip(pieces[i], raw):
                rows = rows.astype(np.float32, copy=False)
                preweight_and_filter(rows, crop_rows(geom, p0, p1),
                                     filter_name, cutoff, short_scan,
                                     out=filtered[:, p0 - r0:p1 - r0])
                t['rows'] += p1 - p0
            if kept is not None:
                k0, k1, kept_rows = kept
                lo, hi = max(r0, k0), min(r1, k1)
                if lo < hi:
                    filtered[:, lo - r0:hi - r0] = kept_rows[:, lo - k0:
                                                             hi - k0]
            del raw
            t['filter'] += time.perf_counter() - a

            a = time.perf_counter()
            vol = be.backproject(filtered, crop_rows(geom, r0, r1),
                                 slab_geometry(vol_geom, z0, z1))
            t['backproject'] += time.perf_counter() - a
            kept = (r0, r1, filtered)

            a = time.perf_counter()
            if pending_write is not None:
                pending_write.result()
            t['write'] += time.perf_counter() - a
            pending_write = writer.submit(target.write_slab, z0, vol)
            if progress is not None:
                progress((i + 1) / len(slabs))
        a = time.perf_counter()
        if pending_write is not None:
            pending_write.result()
        t['write'] += time.perf_counter() - a
    t['total'] = time.perf_counter() - start
    return t
