"""End-to-end check of tomviz.conebeam on real cone-beam data.

Uses the CWI walnut collection (Der Sarkissian et al., "A cone-beam X-ray
computed tomography data collection designed for machine learning",
Sci. Data 6:215, 2019; Zenodo 2686726, CC-BY-4.0). Each walnut has raw
projections for three source heights, flats, darks, the measured
per-projection geometry, and the authors' own FDK reconstruction.

This loads one source height, does flat/dark correction and FDK, and
compares the result with the authors' reconstruction.

Usage:
    python dev/walnut_check.py ~/data/Walnut1 --bin 2 --step 2 \
        --backend cpu --png out.png

Data conventions (found by checking, see the notes printed with --verbose):
- The detector is mounted rotated: the TIFF's 768-pixel axis (axis 0) is
  detector u and its 972-pixel axis is detector v, so frames are
  transposed. With that, the geometry file applies with no flips.
- scan_geom_corrected.geom has one ASTRA cone_vec row per file; file 1200
  repeats file 0 and is skipped.
- The reference reconstructions are 501^3 at 0.1 mm. They match ours after
  flipping our x axis, which is the reference's own axis convention: ours is
  pinned by the phantom tests (world +x is increasing x index).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import tifffile

from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.conebeam import fdk, flat_dark_correct


def bin2d(a, b):
    if b == 1:
        return a.astype(np.float32)
    h, w = a.shape[0] // b * b, a.shape[1] // b * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(
        axis=(1, 3), dtype=np.float32)


def load(root, tube, bin_, step, threads):
    d = Path(root) / 'Projections' / f'tubeV{tube}'
    vecs = np.loadtxt(d / 'scan_geom_corrected.geom')
    idx = np.arange(0, 1200, step)

    def read(name):
        return bin2d(tifffile.imread(d / name).astype(np.float32).T, bin_)

    with ThreadPoolExecutor(threads) as ex:
        projs = np.stack(list(ex.map(
            read, [f'scan_{i:06d}.tif' for i in idx])))
    flat = np.mean([read('io000000.tif'), read('io000001.tif')], axis=0)
    dark = read('di000000.tif')
    v = vecs[idx].copy()
    v[:, 6:12] *= bin_
    return projs, flat, dark, v


def load_reference(root, tube, bin_):
    d = Path(root) / 'Reconstructions'
    files = sorted(d.glob(f'fdk_pos{tube}_*.tiff'))
    vol = np.stack([tifffile.imread(f) for f in files])
    if bin_ > 1:
        n = vol.shape[0] // bin_ * bin_
        vol = vol[:n, :n, :n].reshape(
            n // bin_, bin_, n // bin_, bin_, n // bin_, bin_).mean(
                axis=(1, 3, 5))
    return vol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='Walnut directory, e.g. .../Walnut1')
    ap.add_argument('--tube', type=int, default=2, choices=(1, 2, 3))
    ap.add_argument('--bin', type=int, default=2,
                    help='detector and volume binning (1, 2, 4...)')
    ap.add_argument('--step', type=int, default=2,
                    help='use every n-th projection')
    ap.add_argument('--backend', default='cpu')
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--png', help='write a comparison image here')
    args = ap.parse_args()
    threads = args.threads or 8

    t0 = time.perf_counter()
    projs, flat, dark, vecs = load(args.root, args.tube, args.bin,
                                   args.step, threads)
    line = flat_dark_correct(projs, flat, dark)
    del projs
    g = ConeGeometry.from_astra_vectors(vecs, line.shape[2], line.shape[1])
    n = 501 // args.bin
    vg = VolumeGeometry((n, n, n), 0.1 * args.bin)
    t1 = time.perf_counter()
    rec = fdk(line, g, vg, backend=args.backend, threads=args.threads,
              log=print)
    t2 = time.perf_counter()
    rec = np.flip(rec, 2)  # reference x axis convention, see docstring
    ref = load_reference(args.root, args.tube, args.bin)
    slope, offset = np.polyfit(ref.ravel(), rec.ravel(), 1)
    corr = np.corrcoef(ref.ravel(), rec.ravel())[0, 1]
    rmse = np.sqrt(np.mean((rec - ref) ** 2))
    print(f'{g.n_angles} projections of {line.shape[1]}x{line.shape[2]}, '
          f'volume {vg.shape} at {vg.voxel_size[0]} mm')
    print(f'load + correct {t1 - t0:.1f} s, FDK {t2 - t1:.1f} s')
    print(f'vs reference: correlation {corr:.4f}, slope {slope:.4f}, '
          f'offset {offset:.5f}, RMSE {rmse:.5f} '
          f'(reference std {ref.std():.5f})')

    if args.png:
        from PIL import Image
        c = n // 2
        hi = float(np.percentile(ref, 99.9))

        def u8(a):
            return (np.clip(a / hi, 0, 1) * 255).astype(np.uint8)

        rows = [np.concatenate([u8(v[c]), u8(v[:, c]), u8(v[:, :, c])], 1)
                for v in (ref, rec)]
        Image.fromarray(np.concatenate(rows, 0)).save(args.png)
        print(f'wrote {args.png} (top: reference, bottom: tomviz)')


if __name__ == '__main__':
    main()
