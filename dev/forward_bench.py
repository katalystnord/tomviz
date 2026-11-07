"""Forward projection and SIRT speed on the GPU backend.

    python dev/forward_bench.py N N_PROJ [ITERATIONS]

Times the texture forward projector against the storage-buffer one on an
N^3 volume and N_PROJ projections of N^2 (parallel and cone beam), and
SIRT per iteration, with everything resident on the GPU. Set
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json for lavapipe.
"""

import sys
import time

import numpy as np

from tomviz.projectors import ConeGeometry, ParallelGeometry, VolumeGeometry
from tomviz.projectors.backend import get_backend
from tomviz.projectors.sirt import sirt


def timed(f, *args, **kw):
    f(*args, **kw)  # warm up: shader compilation, allocation
    a = time.perf_counter()
    out = f(*args, **kw)
    return time.perf_counter() - a, out


def main():
    N, n = int(sys.argv[1]), int(sys.argv[2])
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    be = get_backend('gpu')
    bp = be.forward_project.__self__
    print(be.description)
    vg = VolumeGeometry((N, N, N), 0.05)
    x = np.random.default_rng(0).random(vg.shape, dtype=np.float32)
    geoms = {
        'parallel': ParallelGeometry.circular(np.arange(n) * 180 / n, 0.05,
                                              0.05, N, N, offset_u=1.3),
        'cone': ConeGeometry.circular(np.arange(n) * 360 / n, 300, 600,
                                      0.1, 0.1, N, N, offset_u=1.3),
    }
    for name, g in geoms.items():
        t_buf, ref = timed(bp._forward_buffer, x, g, vg)
        t_tex, out = timed(bp._forward_texture, x, g, vg)
        err = np.abs(out - ref).max() / np.abs(ref).max()
        print(f'{name} {N}^3, {n} projections: forward projection '
              f'{t_buf:.2f} s -> {t_tex:.2f} s ({t_buf / t_tex:.1f}x), '
              f'largest difference {err:.1e} of the peak')
        p = ref
        a = time.perf_counter()
        sirt(p, g, vg, iterations=1, backend=be)
        one = time.perf_counter() - a
        a = time.perf_counter()
        sirt(p, g, vg, iterations=1 + iterations, backend=be)
        per = (time.perf_counter() - a - one) / iterations
        print(f'  SIRT: {per:.2f} s per iteration (setup and first '
              f'iteration {one:.1f} s)')


if __name__ == '__main__':
    main()
