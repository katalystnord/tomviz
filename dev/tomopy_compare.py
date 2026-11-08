"""Compare TomoPy with our reconstruction on the problem of
dev/upstream_compare.py (Shepp-Logan on N^3 unit voxels, parallel beam,
N_PROJ projections over 180 degrees).

    python dev/tomopy_compare.py N N_PROJ TOMOPY_PYTHON

TOMOPY_PYTHON is the python of an environment with TomoPy installed (kept
apart from Tomviz's), for example
~/micromamba/envs/tomopy-bench/bin/python. This script makes the
projections and scores the results in the Tomviz environment, and runs
itself in TomoPy's environment for the reconstructions.

TomoPy's analytic methods are run as its documentation shows: projections
padded (gridrec and fbp otherwise leave a large offset and blown-up
corners), the result masked to the reconstruction circle. Every method,
ours included, is scored inside that circle (95 % of the radius), the
part of a slice that the detector covers at every angle.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ITERATIONS = (1, 21, 50)


def worker(folder):
    """In TomoPy's environment: reconstruct and time."""
    import tomopy
    folder = Path(folder)
    proj = np.load(folder / 'proj.npy')         # (angle, slice, column)
    theta = np.load(folder / 'theta.npy')       # radians
    times = {}
    n = proj.shape[2]
    pad = n // 2
    for name, kw in [('gridrec', dict(algorithm='gridrec',
                                      filter_name='shepp')),
                     ('fbp', dict(algorithm='fbp', filter_name='shepp'))]:
        def run(p):
            padded = tomopy.pad(p, axis=2, npad=2 * pad, mode='edge')
            rec = tomopy.recon(padded, theta, center=(n - 1) / 2 + pad,
                               **kw)
            rec = rec[:, pad:pad + n, pad:pad + n]
            return tomopy.circ_mask(rec, axis=0, ratio=0.95)
        run(proj[:, :2])  # warm up
        a = time.perf_counter()
        rec = run(proj)
        times[name] = time.perf_counter() - a
        np.save(folder / f'{name}.npy', rec.astype(np.float32))
    for n in ITERATIONS:
        a = time.perf_counter()
        rec = tomopy.recon(proj, theta, algorithm='sirt', num_iter=n)
        times[f'sirt{n}'] = time.perf_counter() - a
        np.save(folder / f'sirt{n}.npy', rec.astype(np.float32))
    times['version'] = tomopy.__version__
    (folder / 'times.json').write_text(json.dumps(times))


def disc(n, ratio=0.95):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2
    return x ** 2 + y ** 2 <= (ratio * n / 2) ** 2


def rmse(rec, truth):
    m = disc(truth.shape[-1])
    return float(np.sqrt(np.mean((rec[:, m] - truth[:, m]) ** 2)))


def oriented(rec, truth):
    """rec in the orientation (of the 8 a slice can take) closest to
    truth, and its RMSE inside the reconstruction circle."""
    best = None
    for t in (False, True):
        a = np.swapaxes(rec, 1, 2) if t else rec
        for f1 in (False, True):
            for f2 in (False, True):
                b = a[:, ::-1] if f1 else a
                b = b[:, :, ::-1] if f2 else b
                e = rmse(b, truth)
                if best is None or e < best[0]:
                    best = (e, b)
    return best


def main():
    import importlib.util
    N, n_proj, tomopy_python = int(sys.argv[1]), int(sys.argv[2]), \
        sys.argv[3]
    spec = importlib.util.spec_from_file_location(
        'upstream_compare', Path(__file__).with_name('upstream_compare.py'))
    uc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uc)
    from tomviz.conebeam import fdk
    from tomviz.projectors.backend import get_backend
    from tomviz.projectors.sirt import sirt
    _, vg, angles, g, truth, proj = uc.problem(N, n_proj)
    folder = Path(tempfile.mkdtemp())
    np.save(folder / 'proj.npy', proj)
    np.save(folder / 'theta.npy', np.deg2rad(angles))
    subprocess.run([tomopy_python, '-I', __file__, '--worker', str(folder)],
                   check=True)
    times = json.loads((folder / 'times.json').read_text())
    # TomoPy reconstructs a disc of N x N per slice, in pixel units: the
    # same grid as the phantom's unit voxels.
    print(f'TomoPy {times["version"]} (CPU, all cores), {N}^3 from '
          f'{n_proj} projections')
    for name in ('gridrec', 'fbp'):
        rec = np.load(folder / f'{name}.npy')
        e, _ = oriented(rec, truth)
        print(f'  {name}: {times[name]:.2f} s, RMSE {e:.4f}')
    per = (times[f'sirt{ITERATIONS[1]}'] - times[f'sirt{ITERATIONS[0]}']) / \
        (ITERATIONS[1] - ITERATIONS[0])
    for n in ITERATIONS:
        e, _ = oriented(np.load(folder / f'sirt{n}.npy'), truth)
        print(f'  sirt, {n} iterations: {times[f"sirt{n}"]:.1f} s, '
              f'RMSE {e:.4f}')
    print(f'  sirt: {per:.2f} s per iteration')
    be = get_backend('auto')
    try:
        fdk(proj[:, :1].copy(), uc.ParallelGeometry.circular(
            angles, 1.0, 1.0, N, 1), uc.VolumeGeometry((1, N, N), 1.0),
            backend=be)
        a = time.perf_counter()
        ours = fdk(proj, g, vg, backend=be)
        t = time.perf_counter() - a
        print(f'our FBP ({be.description}): {t:.2f} s, RMSE '
              f'{rmse(ours, truth):.4f}')
    except AttributeError:
        print('our FBP: parallel-beam FBP is on the large-data branch')
    x = sirt(proj, g, vg, iterations=50, backend=be)
    print(f'our SIRT, 50 iterations ({be.description}): RMSE '
          f'{rmse(x, truth):.4f}')


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
