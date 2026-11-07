"""Benchmark FDK streamed from and to files on a fast drive.

    python dev/stream_bench.py DIR N_PROJ DET VOX [MEMORY_GB] [--compare]

Writes N_PROJ projections of DET x DET pixels (float32) into DIR, drops
them from the page cache so reads come from the drive, reconstructs a VOX^3
volume into DIR with fdk_stream within MEMORY_GB, and reports where the
time went and the peak memory. --compare also runs the in-memory fdk() (if
it fits) to show the cost of streaming. The projection content is a smooth
synthetic pattern: only the timing matters here; correctness is tested in
tests/python/stream_test.py.
"""

import os
import resource
import sys
import time

import numpy as np

from tomviz.conebeam import fdk
from tomviz.conebeam.stream import RawStack, fdk_stream, plan_slabs
from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.projectors.backend import get_backend


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def drop_cache(path):
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def make_projections(path, n, det):
    """A smooth pattern, written a projection at a time."""
    u = np.linspace(-1, 1, det, dtype=np.float32)
    uu, vv = np.meshgrid(u, u)
    out = RawStack(path, (n, det, det), mode='w')
    t = time.perf_counter()
    frame = np.empty((det, det), np.float32)
    for k in range(n):
        a = 2 * np.pi * k / n
        np.exp(-((uu - 0.3 * np.cos(a)) ** 2 + vv ** 2) * 6, out=frame)
        out.write_slab(k, frame[None])
    out.close()
    drop_cache(path)
    return time.perf_counter() - t


def main():
    folder, n, det, vox = sys.argv[1], *map(int, sys.argv[2:5])
    memory = float(sys.argv[5]) if len(sys.argv) > 5 and \
        not sys.argv[5].startswith('--') else 4.0
    compare = '--compare' in sys.argv
    os.makedirs(folder, exist_ok=True)
    proj_path = os.path.join(folder, 'proj.raw')
    vol_path = os.path.join(folder, 'vol.raw')
    g = ConeGeometry.circular(np.arange(n) * 360.0 / n, 300.0, 600.0,
                              0.1, 0.1, det, det)
    vg = VolumeGeometry((vox, vox, vox), 0.05 * det / vox)
    be = get_backend('auto')
    print(f'backend: {be.description}')
    gb_in = n * det * det * 4 / 1e9
    gb_out = vox ** 3 * 4 / 1e9
    print(f'projections {n} x {det}^2 = {gb_in:.1f} GB, volume {vox}^3 = '
          f'{gb_out:.1f} GB')
    print(f'writing projections: {make_projections(proj_path, n, det):.1f}'
          ' s')
    slabs = plan_slabs(g, vg, int(memory * 2 ** 30))
    print(f'{len(slabs)} slabs within {memory:g} GiB')
    src = RawStack(proj_path, (n, det, det))
    dst = RawStack(vol_path, vg.shape, mode='w')
    t = fdk_stream(src, g, vg, dst, backend=be,
                   memory_bytes=int(memory * 2 ** 30))
    src.close()
    dst.close()
    rows = t.pop('rows')
    print('streamed: ' + ', '.join(f'{k} {v:.1f} s' for k, v in t.items()) +
          f'; detector rows read and filtered: {rows} of {det}')
    print(f'peak memory (RSS): {peak_rss_gb():.1f} GB')
    if compare:
        drop_cache(proj_path)
        a = time.perf_counter()
        proj = np.fromfile(proj_path, np.float32).reshape(n, det, det)
        read = time.perf_counter() - a
        a = time.perf_counter()
        fdk(proj, g, vg, backend=be, overwrite=True)
        print(f'in memory: read {read:.1f} s, fdk '
              f'{time.perf_counter() - a:.1f} s, peak memory '
              f'{peak_rss_gb():.1f} GB')
    os.remove(proj_path)
    os.remove(vol_path)


if __name__ == '__main__':
    main()
