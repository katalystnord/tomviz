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
import threading

import numpy as np

from tomviz.projectors.geometry import (ConeGeometry, ParallelGeometry,
                                        VolumeGeometry)

from .filtering import preweight_and_filter

# Windows opens files in text mode (newline translation) without O_BINARY.
O_BINARY = getattr(os, 'O_BINARY', 0)
_seek_lock = threading.Lock()


def read_at(fd, view, offset):
    """Fill view (a writable byte memoryview) from the file at offset.
    Positioned reads release the GIL and leave the file position alone,
    so threads can share a file: os.preadv (Linux, BSD), else os.pread
    (other Unix, macOS), else a seek and read under a lock (Windows).
    Partial reads are continued; only the end of the file stops it."""
    done, size = 0, len(view)
    while done < size:
        if hasattr(os, 'preadv'):
            got = os.preadv(fd, [view[done:]], offset + done)
        else:
            if hasattr(os, 'pread'):
                data = os.pread(fd, size - done, offset + done)
            else:
                with _seek_lock:
                    os.lseek(fd, offset + done, os.SEEK_SET)
                    data = os.read(fd, size - done)
            got = len(data)
            view[done:done + got] = data
        if got == 0:
            raise IOError(f'unexpected end of file at byte {offset + done}')
        done += got


def write_at(fd, view, offset):
    """Write view (a byte memoryview) to the file at offset, as read_at:
    os.pwrite where it exists (Unix), else a seek and write under a lock."""
    done = 0
    while done < len(view):
        if hasattr(os, 'pwrite'):
            done += os.pwrite(fd, view[done:], offset + done)
        else:
            with _seek_lock:
                os.lseek(fd, offset + done, os.SEEK_SET)
                done += os.write(fd, view[done:])


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
        flags = {'r': os.O_RDONLY, 'r+': os.O_RDWR,
                 'w': os.O_RDWR | os.O_CREAT}[mode]
        self.fd = os.open(path, flags | O_BINARY, 0o644)
        self.transpose = None
        if mode == 'w':
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
        for k in range(n):
            start = self.offset + ((k * n_v + r0) * n_u) * \
                self.dtype.itemsize
            try:
                read_at(self.fd, memoryview(out[k]).cast('B'), start)
            except IOError as e:
                raise IOError(f'{self.path}: {e}') from e
        return out

    def write_slab(self, z0, data):
        """data (nz, ny, nx) at slices z0.. of a (n_z, n_y, n_x) file."""
        if self.transpose is not None:
            data = np.transpose(data, self.transpose)
        data = np.ascontiguousarray(data, dtype=self.dtype)
        start = self.offset + z0 * int(np.prod(self.shape[1:])) * \
            self.dtype.itemsize
        write_at(self.fd, memoryview(data).cast('B'), start)


def create_emd(path, vol_geom, units='mm', name='FDK'):
    """An EMD file (the HDF5 layout Tomviz opens) for a vol_geom volume,
    with its data set allocated contiguously, returned as a RawStack
    that writes slabs straight into it with positioned I/O.

    Tomviz stores a volume with x along the rotation axis (our z), then
    our x and y; on disk that is (n_z, n_x, n_y) in C order, so a slab of
    slices is one block of the file."""
    import h5py
    nz, ny, nx = vol_geom.shape
    dz, dy, dx = vol_geom.voxel_size
    with h5py.File(path, 'w') as f:
        f.attrs.create('version_major', 0, dtype='uint32')
        f.attrs.create('version_minor', 2, dtype='uint32')
        group = f.create_group('data').create_group('tomography')
        group.attrs.create('emd_group_type', 1, dtype='uint32')
        space = h5py.h5s.create_simple((nz, nx, ny))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.CONTIGUOUS)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        h5py.h5d.create(group.id, b'data', h5py.h5t.IEEE_F32LE, space,
                        dcpl=dcpl)
        data = group['data']
        data.attrs['name'] = name
        offset = data.id.get_offset()
        axes = (('x', nz, dz), ('y', nx, dx), ('z', ny, dy))
        for i, (axis, n, step) in enumerate(axes):
            d = group.create_dataset(f'dim{i + 1}', data=np.arange(
                n, dtype=np.float32) * np.float32(step))
            d.attrs['name'] = axis
            d.attrs['units'] = f'[{units}]'
        scalars = group.create_group('tomviz_scalars')
        scalars[name] = h5py.SoftLink(data.name)
    out = RawStack(path, (nz, nx, ny), np.float32, offset, mode='r+')
    out.transpose = (0, 2, 1)   # slabs arrive as (nz, ny, nx)
    out.shape = (nz, ny, nx)    # what fdk_stream checks against
    return out


def crop_rows(geom, r0, r1):
    """The geometry of detector rows r0..r1 only: the same source and
    pixel steps, the detector centre moved to the middle of those rows."""
    shift = (r0 + (r1 - r0 - 1) / 2) - (geom.n_v - 1) / 2
    center = geom.det_center + shift * geom.det_v
    if isinstance(geom, ParallelGeometry):
        return ParallelGeometry(geom.ray, center, geom.det_u, geom.det_v,
                                geom.n_u, r1 - r0)
    return ConeGeometry(geom.source, center, geom.det_u, geom.det_v,
                        geom.n_u, r1 - r0,
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
    # Always a window inside the detector, even for slices that project
    # off it (they then backproject nothing, as in memory).
    r0 = int(np.clip(np.floor(v.min()) - margin, 0, geom.n_v - 1))
    r1 = int(np.clip(np.ceil(v.max()) + margin + 1, r0 + 1, geom.n_v))
    return r0, r1


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
            # Rows: read ahead, the next slab being filtered, this slab
            # on the GPU, and a margin. Volume slab: the result, the GPU's
            # read-back copy, and the previous slab still being written.
            need = 4 * (4 * n * (r1 - r0) * n_u + 3 * mid * ny * nx)
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
    a paged pool). Four stages run at once: a reader thread fetches rows
    two slabs ahead, a filter thread prepares the next slab on the CPU,
    the GPU backprojects the current one, and a writer thread stores the
    previous one.

    Returns timings in seconds: read (the filter thread waiting for
    rows), filter (CPU time filtering), waiting for filter (the GPU
    waiting for filtered rows), backproject, write (waiting for the
    writer), total; and rows, the detector rows read and filtered."""
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

    reader = ThreadPoolExecutor(1)
    filterer = ThreadPoolExecutor(1)
    writer = ThreadPoolExecutor(1)
    reads = {0: reader.submit(read, 0)}
    state = {'kept': None}   # the last filtered slab, for its rows

    def filter_slab(i):
        """Runs in the filter thread, one slab ahead of the GPU: the new
        rows read for slab i, filtered, plus the overlap with slab i - 1."""
        if i + 1 < len(slabs):
            reads[i + 1] = reader.submit(read, i + 1)
        a = time.perf_counter()
        raw = reads.pop(i).result()
        wait = time.perf_counter() - a
        a = time.perf_counter()
        _, _, r0, r1 = slabs[i]
        filtered = np.empty((geom.n_angles, r1 - r0, geom.n_u),
                            dtype=np.float32)
        for (p0, p1), rows in zip(pieces[i], raw):
            rows = rows.astype(np.float32, copy=False)
            preweight_and_filter(rows, crop_rows(geom, p0, p1),
                                 filter_name, cutoff, short_scan,
                                 out=filtered[:, p0 - r0:p1 - r0])
        if state['kept'] is not None:
            k0, k1, kept_rows = state['kept']
            lo, hi = max(r0, k0), min(r1, k1)
            if lo < hi:
                filtered[:, lo - r0:hi - r0] = kept_rows[:, lo - k0:hi - k0]
        state['kept'] = (r0, r1, filtered)
        return filtered, wait, time.perf_counter() - a, \
            sum(q1 - q0 for q0, q1 in pieces[i])

    start = time.perf_counter()
    try:
        pending_write = None
        next_filtered = filterer.submit(filter_slab, 0)
        for i, (z0, z1, r0, r1) in enumerate(slabs):
            if should_stop is not None and should_stop():
                return None
            a = time.perf_counter()
            filtered, wait, busy, rows = next_filtered.result()
            t['waiting for filter'] = t.get('waiting for filter', 0.0) + \
                time.perf_counter() - a
            t['read'] += wait
            t['filter'] += busy
            t['rows'] += rows
            if i + 1 < len(slabs):
                next_filtered = filterer.submit(filter_slab, i + 1)

            a = time.perf_counter()
            vol = be.backproject(filtered, crop_rows(geom, r0, r1),
                                 slab_geometry(vol_geom, z0, z1))
            t['backproject'] += time.perf_counter() - a
            del filtered

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
    finally:
        for pool in (reader, filterer, writer):
            pool.shutdown(wait=True, cancel_futures=True)
    t['total'] = time.perf_counter() - start
    return t


def plan_sirt_slabs(geom, vol_geom, memory_bytes, halo,
                    max_slab_voxels=1 << 27):
    """Slabs (z0, z1, e0, e1, r0, r1) for sirt_stream: core slices z0..z1,
    reconstructed with halo slices e0..e1, from detector rows r0..r1.

    SIRT holds the rows (data, row weights, a residual) and the slab
    (volume, column weights, an update) at once; max_slab_voxels keeps a
    slab within one GPU buffer, so every iteration stays on the GPU
    (512 MB here, as tomviz.projectors.webgpu allows)."""
    nz, ny, nx = vol_geom.shape
    n, n_u = geom.n_angles, geom.n_u

    def extent(z0, z1):
        e0, e1 = max(0, z0 - halo), min(nz, z1 + halo)
        r0, r1 = rows_for(geom, vol_geom, e0, e1)
        return e0, e1, r0, r1

    slabs = []
    z0 = 0
    while z0 < nz:
        lo, hi = 1, nz - z0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            e0, e1, r0, r1 = extent(z0, z0 + mid)
            voxels = (e1 - e0) * ny * nx
            need = 4 * (4 * n * (r1 - r0) * n_u + 4 * voxels)
            if need <= memory_bytes and voxels <= max_slab_voxels:
                lo = mid
            else:
                hi = mid - 1
        slabs.append((z0, z0 + lo) + extent(z0, z0 + lo))
        z0 += lo
    return slabs


def sirt_stream(source, geom, vol_geom, target, iterations=50,
                relaxation=1.0, nonnegative=True, halo=8, backend='auto',
                memory_bytes=None, max_slab_voxels=1 << 27, progress=None,
                should_stop=None, log=None):
    """SIRT (tomviz.projectors.sirt) from source into target a slab at a
    time, for parallel beams whose rays are perpendicular to the rotation
    axis (synchrotron scans).

    Such rays never leave their slice, so slabs are independent problems
    except where interpolation between detector rows and slices couples
    neighbours. Each slab is reconstructed with halo extra slices on each
    side and only its own slices are kept; with the slices on the detector
    row grid (voxel size equal to the row pitch, as by default) slabs are
    exactly independent. Off the grid, the largest difference from SIRT
    of the whole volume was 5e-3 of the peak with 4 halo slices, 4e-5
    with 8 (the default) and 7e-7 with 12, after 10 to 200 iterations.
    Cone beams couple every slice to every other through each ray, so
    they are not streamed this way.

    Every iteration of a slab runs on the GPU when the slab fits in one
    buffer (max_slab_voxels). A reader thread fetches the next slab's rows
    and a writer thread stores finished slices meanwhile.

    Returns timings (read, sirt, write, total) in seconds."""
    import time
    from tomviz.projectors.backend import get_backend
    from tomviz.projectors.sirt import sirt
    log = log or (lambda msg: None)
    if not isinstance(geom, ParallelGeometry) or np.any(
            np.abs(geom.ray[:, 2]) >
            1e-9 * np.linalg.norm(geom.ray, axis=1)):
        raise ValueError('Streamed SIRT needs a parallel beam with rays '
                         'perpendicular to the rotation axis')
    be = get_backend(backend) if isinstance(backend, str) else backend
    if tuple(source.shape) != (geom.n_angles, geom.n_v, geom.n_u):
        raise ValueError(f'Projections of {source.shape} do not match the '
                         'geometry')
    if tuple(target.shape) != tuple(vol_geom.shape):
        raise ValueError('The output file does not match the volume')
    if memory_bytes is None:
        memory_bytes = default_memory_bytes()
    slabs = plan_sirt_slabs(geom, vol_geom, memory_bytes, halo,
                            max_slab_voxels)
    log(f'{len(slabs)} slabs of up to '
        f'{max(z1 - z0 for z0, z1, *_ in slabs)} slices; {be.description}')
    t = dict(read=0.0, sirt=0.0, write=0.0, total=0.0)
    reader = ThreadPoolExecutor(1)
    writer = ThreadPoolExecutor(1)
    start = time.perf_counter()
    try:
        reads = {0: reader.submit(source.read_rows, *slabs[0][4:])}
        pending = None
        for i, (z0, z1, e0, e1, r0, r1) in enumerate(slabs):
            if should_stop is not None and should_stop():
                return None
            a = time.perf_counter()
            rows = reads.pop(i).result()
            t['read'] += time.perf_counter() - a
            if i + 1 < len(slabs):
                reads[i + 1] = reader.submit(source.read_rows,
                                             *slabs[i + 1][4:])

            def slab_progress(f, i=i):
                if progress is not None:
                    progress((i + f) / len(slabs))

            a = time.perf_counter()
            x = sirt(rows, crop_rows(geom, r0, r1),
                     slab_geometry(vol_geom, e0, e1), iterations,
                     relaxation=relaxation, nonnegative=nonnegative,
                     backend=be, progress=slab_progress,
                     should_stop=should_stop)
            t['sirt'] += time.perf_counter() - a
            if x is None:
                return None
            del rows
            a = time.perf_counter()
            if pending is not None:
                pending.result()
            t['write'] += time.perf_counter() - a
            pending = writer.submit(target.write_slab, z0,
                                    np.ascontiguousarray(x[z0 - e0:z1 - e0]))
            del x
        a = time.perf_counter()
        if pending is not None:
            pending.result()
        t['write'] += time.perf_counter() - a
    finally:
        reader.shutdown(wait=True, cancel_futures=True)
        writer.shutdown(wait=True, cancel_futures=True)
    t['total'] = time.perf_counter() - start
    return t
