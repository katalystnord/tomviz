"""Electron tomography check on real data: the star nanoparticle tilt
series that ships with Tomviz as sample data (doi 10.1021/nl103400a).

Reconstructs it with the original slice-by-slice WBP, the same WBP on the
shared backprojector, and parallel-beam SIRT, reports timings and how
closely the WBPs agree, and optionally writes a PNG of central slices.

    python dev/nanoparticle_check.py TiltSeries_NanoParticle_...emd \\
        [--backend gpu] [--sirt-iterations 50] [--png out.png]
"""

import argparse
import importlib.util
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.join(HERE, '..', 'tomviz', 'python')
sys.path.insert(0, PYDIR)

from tomviz.projectors import ParallelGeometry, VolumeGeometry  # noqa: E402
from tomviz.projectors.backend import get_backend  # noqa: E402
from tomviz.projectors.sirt import sirt  # noqa: E402


def load_wbp():
    spec = importlib.util.spec_from_file_location(
        'Recon_WBP', os.path.join(PYDIR, 'Recon_WBP.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('emd')
    ap.add_argument('--backend', default='auto')
    ap.add_argument('--sirt-iterations', type=int, default=50)
    ap.add_argument('--png')
    args = ap.parse_args()

    with h5py.File(args.emd, 'r') as f:
        data = f['data/tomography/data'][:].astype(np.float32)
        angles = f['data/tomography/dim1'][:].astype(np.float64)
    # (angle, y, x) -> tomviz tilt series (x: tilt axis, y, angle).
    ts = np.ascontiguousarray(data.transpose(2, 1, 0))
    n_slice, n_ray, n_proj = ts.shape
    print(f'Tilt series: {n_slice} slices x {n_ray} rays, {n_proj} tilts '
          f'from {angles[0]:g} to {angles[-1]:g} degrees')

    wbp = load_wbp()
    t = time.time()
    old = np.stack([wbp.wbp2(ts[i], angles, n_ray, 'ramp', 'linear')
                    for i in range(n_slice)])
    t_old = time.time() - t
    print(f'Original WBP (one thread):  {t_old:6.2f} s')

    be = get_backend(args.backend)
    print(f'Backend: {be.description}')
    t = time.time()
    new = wbp.wbp3(ts, angles, n_ray, 'ramp', be)
    t_new = time.time() - t
    X, Y = np.mgrid[0:n_ray, 0:n_ray]
    inside = np.hypot(X - n_ray // 2, Y - n_ray // 2) < n_ray // 2 - 2
    diff = abs(new - old)[:, inside].max() / abs(old).max()
    print(f'WBP on shared backprojector: {t_new:6.2f} s '
          f'({t_old / t_new:.1f}x); max difference inside the field of '
          f'view {diff:.1e} of the peak')

    # SIRT with the same geometry as wbp3.
    proj = np.ascontiguousarray(ts.transpose(2, 0, 1))
    geom = ParallelGeometry.circular(
        angles + 90.0, 1.0, 1.0, n_ray, n_slice,
        offset_u=n_ray // 2 - (n_ray - 1) / 2, clockwise=True)
    c = (n_ray - 1) / 2 - n_ray // 2
    vg = VolumeGeometry((n_slice, n_ray, n_ray), 1.0, center=(0.0, c, c))
    t = time.time()
    rec = sirt(proj, geom, vg, iterations=args.sirt_iterations, backend=be)
    t_sirt = time.time() - t
    res = np.linalg.norm(be.forward_project(rec, geom, vg) - proj) / \
        np.linalg.norm(proj)
    print(f'SIRT, {args.sirt_iterations} iterations: {t_sirt:6.2f} s; '
          f'relative projection residual {res:.3f}')

    if args.png:
        from PIL import Image

        def norm(img):
            lo, hi = np.percentile(img, [0.5, 99.5])
            return (255 * np.clip((img - lo) / (hi - lo), 0, 1)).astype(
                np.uint8)

        mid = n_slice // 2
        tiles = [norm(old[mid]), norm(new[mid]), norm(rec[mid]),
                 norm(old[:, n_ray // 2, :]), norm(new[:, n_ray // 2, :]),
                 norm(rec[:, n_ray // 2, :])]
        h = max(t.shape[0] for t in tiles[:3])
        rows = [np.hstack(tiles[:3]), np.hstack(tiles[3:])]
        w = max(r.shape[1] for r in rows)
        canvas = np.zeros((h + rows[1].shape[0] + 4, w), np.uint8)
        canvas[:rows[0].shape[0], :rows[0].shape[1]] = rows[0]
        canvas[h + 4:, :rows[1].shape[1]] = rows[1]
        Image.fromarray(canvas).save(args.png)
        print(f'Wrote {args.png}: original WBP, new WBP, SIRT; top: slice '
              f'across the tilt axis, bottom: slice along it')


if __name__ == '__main__':
    main()
