"""Benchmark ring artefact removal on real cone-beam data.

Takes CWI Walnut 1 (see walnut_check.py), injects realistic detector
defects into the raw intensities, and measures for each removal method:

- residual: RMSE between the reconstruction of the corrected defective
  data and the reconstruction of the clean data (how much ring error is
  left);
- harm: RMSE between the reconstruction of the corrected clean data and
  the reconstruction of the clean data (what the method does to data
  without injected rings).

Both in units of the clean reconstruction's standard deviation inside the
walnut. Three stages, so methods from other packages (TomoPy) can run in
their own Python:

    python dev/ring_bench.py prepare ~/data/Walnut1 OUTDIR
    python dev/ring_bench.py correct OUTDIR METHOD [METHOD...]
    python dev/ring_bench.py score OUTDIR [--backend gpu] [--png DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tomviz' / 'python'))
sys.path.insert(0, str(HERE))


def inject_defects(intensity, flat, dark, seed=0):
    """Return a copy of raw intensities with detector defects that cause
    the common kinds of rings (Vo et al., Opt. Express 26:28396, 2018):

    - full stripes: a few per cent of pixels with a constant gain error;
    - partial stripes: pixels whose response is slightly nonlinear, so
      the error grows with attenuation;
    - large stripes: clusters of adjacent pixels with a common gain error;
    - dead pixels: no response to the beam.

    The defects are not in the flat field (as when the detector drifts
    after the flats were taken), so flat-field correction leaves them.
    """
    rng = np.random.default_rng(seed)
    n_v, n_u = intensity.shape[1:]
    gain = np.ones((n_v, n_u), np.float32)
    full = rng.random((n_v, n_u)) < 0.03
    gain[full] += rng.normal(0, 0.01, full.sum()).astype(np.float32)
    for _ in range(max(1, n_u // 60)):
        u0 = rng.integers(0, n_u - 12)
        w = rng.integers(4, 12)
        v0 = rng.integers(0, n_v // 2)
        h = rng.integers(n_v // 8, n_v // 2)
        gain[v0:v0 + h, u0:u0 + w] *= 1 + rng.choice([-1, 1]) * 0.02
    gamma = np.ones((n_v, n_u), np.float32)
    partial = rng.random((n_v, n_u)) < 0.01
    gamma[partial] += rng.normal(0, 0.03, partial.sum()).astype(np.float32)
    dead = rng.random((n_v, n_u)) < 0.0005

    out = np.empty_like(intensity)
    span = np.maximum(flat - dark, 1e-3)
    for k in range(intensity.shape[0]):
        t = np.clip((intensity[k] - dark) / span, 1e-6, None)
        t = gain * t ** gamma
        t[dead] = 1.0
        out[k] = dark + t * span
    return out


METHODS = {}


def method(name):
    def reg(f):
        METHODS[name] = f
        return f
    return reg


@method('none')
def _none(p):
    return p


def _tomopy():
    import tomopy
    return tomopy


@method('tomopy_all')
def _all(p):
    return _tomopy().remove_all_stripe(p, ncore=8)


@method('tomopy_sorting')
def _sorting(p):
    return _tomopy().remove_stripe_based_sorting(p, ncore=8)


@method('tomopy_filtering')
def _filtering(p):
    return _tomopy().remove_stripe_based_filtering(p, ncore=8)


@method('tomopy_fw')
def _fw(p):
    return _tomopy().remove_stripe_fw(p, ncore=8)


@method('tomopy_ti')
def _ti(p):
    return _tomopy().remove_stripe_ti(p, ncore=8)


def _tomviz_with(**kw):
    def run(p):
        from tomviz.conebeam.rings import remove_rings
        return remove_rings(p, threads=8, **kw)
    return run


METHODS['tomviz'] = _tomviz_with()
METHODS['tomviz_nodead'] = _tomviz_with(dead=False)
METHODS['tomviz_size9'] = _tomviz_with(size=9)


def prepare(args):
    from walnut_check import load
    from tomviz.conebeam import flat_dark_correct
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    projs, flat, dark, vecs = load(args.root, 2, args.bin, args.step, 8)
    bad = inject_defects(projs, flat, dark)
    np.save(out / 'clean.npy', flat_dark_correct(projs, flat, dark))
    np.save(out / 'defects.npy', flat_dark_correct(bad, flat, dark))
    np.save(out / 'vecs.npy', vecs)
    print(f'{projs.shape} written to {out}')


def correct(args):
    import time
    out = Path(args.outdir)
    for name in args.methods:
        for which in ('clean', 'defects'):
            p = np.load(out / f'{which}.npy')
            t = time.perf_counter()
            r = np.asarray(METHODS[name](p), dtype=np.float32)
            dt = time.perf_counter() - t
            np.save(out / f'{name}.{which}.npy', r)
            print(f'{name} {which}: {dt:.1f} s')


def score(args):
    from tomviz.projectors import ConeGeometry, VolumeGeometry
    from tomviz.conebeam import fdk
    from tomviz.projectors.backend import get_backend
    out = Path(args.outdir)
    vecs = np.load(out / 'vecs.npy')
    clean = np.load(out / 'clean.npy', mmap_mode='r')
    g = ConeGeometry.from_astra_vectors(vecs, clean.shape[2], clean.shape[1])
    n = 501 // args.bin
    vg = VolumeGeometry((n, n, n), 0.1 * args.bin)
    be = get_backend(args.backend)

    def rec(path):
        return fdk(np.load(path), g, vg, backend=be)

    ref = rec(out / 'clean.npy')
    inside = ref > 0.3 * np.percentile(ref, 99.5)
    # Score inside the walnut's bounding cylinder, away from the volume
    # edges where cone-beam data is incomplete.
    z = slice(n // 6, n - n // 6)
    scale = ref[z][inside[z]].std()

    def err(v):
        return np.sqrt(np.mean((v[z] - ref[z]) ** 2)) / scale

    names = sorted({p.name.split('.')[0] for p in out.glob('*.defects.npy')})
    print(f'{"method":18s} {"residual":>9s} {"harm":>7s}')
    base = rec(out / 'defects.npy')
    print(f'{"(no correction)":18s} {err(base):9.4f} {0:7.4f}')
    images = {'reference': ref, 'no correction': base}
    for name in names:
        r_def = rec(out / f'{name}.defects.npy')
        r_cln = rec(out / f'{name}.clean.npy')
        print(f'{name:18s} {err(r_def):9.4f} {err(r_cln):7.4f}')
        images[name] = r_def
    if args.png:
        from PIL import Image
        lo, hi = np.percentile(ref[n // 2], [1, 99.8])
        Path(args.png).mkdir(parents=True, exist_ok=True)
        for name, v in images.items():
            img = np.clip((v[n // 2] - lo) / (hi - lo), 0, 1)
            Image.fromarray((255 * img).astype(np.uint8)).save(
                Path(args.png) / f'{name.replace(" ", "_")}.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('root')
    p.add_argument('outdir')
    p.add_argument('--bin', type=int, default=2)
    p.add_argument('--step', type=int, default=2)
    p = sub.add_parser('correct')
    p.add_argument('outdir')
    p.add_argument('methods', nargs='+', choices=sorted(METHODS))
    p = sub.add_parser('score')
    p.add_argument('outdir')
    p.add_argument('--bin', type=int, default=2)
    p.add_argument('--backend', default='auto')
    p.add_argument('--png')
    args = ap.parse_args()
    {'prepare': prepare, 'correct': correct, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
