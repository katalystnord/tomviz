"""Compare our SIRT and forward projection with what Tomviz users run
today: upstream's Recon_SIRT.py and GenerateTiltSeries.py, taken
unchanged from upstream/master (git show), on the same parallel-beam
Shepp-Logan problem.

    python dev/upstream_compare.py sirt-time N N_PROJ BACKEND...
    python dev/upstream_compare.py sirt-quality N N_PROJ ITERATIONS BACKEND
    python dev/upstream_compare.py tilt-series N N_PROJ BACKEND...

BACKEND is upstream, gpu, cpu (numpy) or lavapipe (gpu with
VK_DRIVER_FILES pointing at Mesa's lavapipe, set by the caller). Run from
the repository root with the upstream remote fetched.
"""

import importlib.util
import inspect
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tomviz._internal import OperatorWrapper
from tomviz.operators import Operator
from tomviz.projectors import ParallelGeometry, VolumeGeometry
from tomviz.projectors.phantom import EllipsoidPhantom
from tomviz_pipeline.dataset import LegacyDataset as Dataset


def upstream_module(name):
    """tomviz/python/<name>.py exactly as on upstream/master."""
    code = subprocess.run(
        ['git', 'show', f'upstream/master:tomviz/python/{name}.py'],
        check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / f'upstream_{name}.py'
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operator(module):
    for v in module.__dict__.values():
        if inspect.isclass(v) and issubclass(v, Operator) and \
                v.__module__ == module.__name__:
            op = v()
            op._operator_wrapper = OperatorWrapper()
            return op


def problem(N, n_proj):
    """A Shepp-Logan phantom on an N^3 grid of unit voxels, and its exact
    parallel-beam projections over 180 degrees (tilt angles -90..90)."""
    ph = EllipsoidPhantom.shepp_logan(radius=0.45 * N)
    vg = VolumeGeometry((N, N, N), 1.0)
    angles = -90 + np.arange(n_proj) * 180.0 / n_proj
    g = ParallelGeometry.circular(angles, 1.0, 1.0, N, N)
    return ph, vg, angles, g, ph.voxelize(vg, supersample=2).astype(
        np.float32), ph.project(g).astype(np.float32)


def tilt_series(proj):
    """Our (angle, v, u) projections as a Tomviz tilt series."""
    return np.asfortranarray(np.moveaxis(proj, 0, 2))


def best_rmse(slices, truth):
    """RMSE against the phantom, over the eight ways a slice can be
    flipped or transposed (the two codes orient slices differently),
    and which one it was."""
    best = None
    for t in (False, True):
        a = np.swapaxes(slices, 1, 2) if t else slices
        for f1 in (False, True):
            for f2 in (False, True):
                b = a[:, ::-1] if f1 else a
                b = b[:, :, ::-1] if f2 else b
                e = float(np.sqrt(np.mean((b - truth) ** 2)))
                if best is None or e < best[0]:
                    best = (e, (t, f1, f2))
    return best


def orient(slices, how):
    t, f1, f2 = how
    a = np.swapaxes(slices, 1, 2) if t else slices
    a = a[:, ::-1] if f1 else a
    return a[:, :, ::-1] if f2 else a


def sirt_time(N, n_proj, backends):
    _, vg, angles, g, truth, proj = problem(N, n_proj)
    for be in backends:
        if be == 'upstream':
            m = upstream_module('Recon_SIRT')
            a = time.perf_counter()
            A = m.parallelRay(N, 1.0, angles, N, 1.0)
            t_matrix = time.perf_counter() - a
            op = operator(m)
            ds = Dataset({'scalars': tilt_series(proj)})
            ds.tilt_angles = angles
            times = []
            for niter in (1, 3):
                a = time.perf_counter()
                op.transform(ds, Niter=niter)
                times.append(time.perf_counter() - a)
            per = (times[1] - times[0]) / 2
            print(f'upstream SIRT, {N}^3 from {n_proj}: matrix '
                  f'{t_matrix:.1f} s ({A.nnz / 1e6:.0f} M non-zeros), '
                  f'{per:.2f} s per iteration')
        else:
            from tomviz.projectors.backend import get_backend
            from tomviz.projectors.sirt import sirt
            b = get_backend('gpu' if be == 'lavapipe' else be)
            times = []
            for niter in (1, 3):
                a = time.perf_counter()
                sirt(proj, g, vg, iterations=niter, backend=b)
                times.append(time.perf_counter() - a)
            per = (times[1] - times[0]) / 2
            print(f'our SIRT on {be} ({b.description}), {N}^3 from '
                  f'{n_proj}: setup and first iteration {times[0]:.1f} s, '
                  f'{per:.2f} s per iteration')


def sirt_quality(N, n_proj, iterations, backend):
    """RMSE against the phantom after each iteration."""
    _, vg, angles, g, truth, proj = problem(N, n_proj)
    marks = sorted({1, 2, 5, 10, 20, 50, 100, 200, iterations})
    marks = [k for k in marks if k <= iterations]
    if backend == 'upstream':
        m = upstream_module('Recon_SIRT')
        A = m.parallelRay(N, 1.0, angles, N, 1.0).tocsr()
        # Landweber diverges for steps above 2 / ||A||^2; ||A||^2 by
        # power iteration.
        v = np.random.default_rng(0).random(A.shape[1])
        for _ in range(50):
            v = A.T @ (A @ v)
            v /= np.linalg.norm(v)
        L = float(np.linalg.norm(A.T @ (A @ v)))
        print(f'upstream, {N}^3 from {n_proj}: ||A||^2 = {L:.0f}, so '
              f'Landweber converges for steps below {2 / L:.2e}')
        tilt = tilt_series(proj)
        for step in (1e-4, 1.0 / L):
            r = m.SIRT(A, 'landweber', N)
            r.initialize()
            recon = np.zeros((N, N, N), np.float32)
            how = None
            line = []
            for it in range(1, iterations + 1):
                for s in range(N):
                    b = tilt[s, :, :].transpose().flatten()
                    recon[s] = r.recon2(b, recon[s].flatten(), step,
                                        s).reshape(N, N)
                recon[recon < 0] = 0
                if it in marks:
                    if not np.all(np.isfinite(recon)):
                        line.append(f'{it}: diverged')
                        break
                    if how is None:
                        e, how = best_rmse(recon, truth)
                    else:
                        e = float(np.sqrt(np.mean(
                            (orient(recon, how) - truth) ** 2)))
                    line.append(f'{it}: {e:.4f}')
            label = 'default step 1e-4' if step == 1e-4 else \
                f'step 1/||A||^2 = {step:.2e}'
            print(f'  upstream SIRT, {label}: RMSE ' + ', '.join(line))
    else:
        from tomviz.projectors.backend import get_backend
        from tomviz.projectors.sirt import sirt
        b = get_backend('gpu' if backend == 'lavapipe' else backend)
        out = {}

        def record(it, x):
            if it + 1 in marks:
                out[it + 1] = float(np.sqrt(np.mean((x - truth) ** 2)))

        sirt(proj, g, vg, iterations=iterations, backend=b,
             callback=record, volume_needed=lambda it: it + 1 in marks)
        print(f'  our SIRT ({b.description}): RMSE ' +
              ', '.join(f'{k}: {v:.4f}' for k, v in sorted(out.items())))
    print(f'  (phantom RMSE against zero: '
          f'{float(np.sqrt(np.mean(truth ** 2))):.4f})')


def tilt_series_bench(N, n_proj, backends):
    ph, vg, angles, g, truth, proj = problem(N, n_proj)
    exact = tilt_series(proj)
    # Tomviz volumes are (x along the axis, y, z); ours are (z, y, x).
    volume = np.asfortranarray(truth)
    for be in backends:
        ds = Dataset({'scalars': volume.copy(order='F')})
        ds.spacing = [1.0, 1.0, 1.0]
        args = dict(start_angle=-90.0, angle_increment=180.0 / n_proj,
                    num_tilts=n_proj)
        if be == 'upstream':
            op = operator(upstream_module('GenerateTiltSeries'))
        else:
            spec = importlib.util.spec_from_file_location(
                'GenerateTiltSeries', 'tomviz/python/GenerateTiltSeries.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            op = operator(module)
            args['backend'] = 'gpu' if be == 'lavapipe' else be
        a = time.perf_counter()
        op.transform(ds, **args)
        t = time.perf_counter() - a
        out = np.asarray(ds.active_scalars)
        # Upstream pads the detector to the volume's diagonal: compare the
        # middle N columns.
        c0 = (out.shape[1] - N) // 2
        got = out[:, c0:c0 + N, :]
        # Same orientation and angle convention in both codes (checked:
        # no flip or angle offset matches better). The sharp-edged
        # phantom's voxelisation limits both to a few percent.
        m = exact > 0.5 * exact.max()
        err = float(np.mean(np.abs(got - exact)[m] / exact[m]))
        print(f'Generate Tilt Series, {be}: {N}^3, {n_proj} tilts in '
              f'{t:.1f} s; mean error on long chords {100 * err:.1f} % '
              'of the exact projection')


def main():
    what = sys.argv[1]
    N, n_proj = int(sys.argv[2]), int(sys.argv[3])
    if what == 'sirt-time':
        sirt_time(N, n_proj, sys.argv[4:])
    elif what == 'sirt-quality':
        sirt_quality(N, n_proj, int(sys.argv[4]), sys.argv[5])
    elif what == 'tilt-series':
        tilt_series_bench(N, n_proj, sys.argv[4:])


if __name__ == '__main__':
    main()
