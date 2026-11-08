"""Our streamed FBP of DM4CT F3_1 against the authors' ASTRA FBP.
    python -I compare.py OUR.emd REFERENCE_DIR PNG"""
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile

ours_path, ref_dir, png = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
f = h5py.File(ours_path, 'r')
d = f['data/tomography/data']          # Tomviz (x, y, z) = our (z, x, y)
print('our volume', d.shape)
n = d.shape[1]
yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
disc = xx ** 2 + yy ** 2 <= (0.9 * n / 2) ** 2   # away from the edges


def orientations(a):
    for t in (False, True):
        b = a.T if t else a
        for f1 in (False, True):
            for f2 in (False, True):
                c = b[::-1] if f1 else b
                c = c[:, ::-1] if f2 else c
                # a transpose or one flip is a mirror; two flips, or a
                # transpose and one flip, is a rotation
                mirror = (t + f1 + f2) % 2 == 1
                yield (t, f1, f2), mirror, c


rows = []
pairs = []
for path in sorted(ref_dir.glob('recon_*.tif')):
    z = int(path.stem.split('_')[1])
    ref = tifffile.imread(path).astype(np.float64)
    ours = d[z].T.astype(np.float64)     # back to our (y, x)
    best = None
    for how, mirror, o in orientations(ours):
        c = np.corrcoef(o[disc], ref[disc])[0, 1]
        if best is None or c > best[0]:
            best = (c, how, mirror, o)
    c, how, mirror, o = best
    k = np.vdot(o[disc], ref[disc]) / np.vdot(o[disc], o[disc])
    rows.append((z, c, how, mirror, k))
    pairs.append((z, ref, o))
    print(f'slice {z}: correlation {c:.4f} with orientation {how} '
          f'({"mirror" if mirror else "rotation"}); scale ours x {k:.4f}')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(len(pairs), 2, figsize=(10, 5 * len(pairs)))
for r, (z, ref, o) in enumerate(pairs):
    lo, hi = np.percentile(ref[disc], [1, 99])
    for c, (im, title) in enumerate(((ref, f'authors (ASTRA FBP), slice {z}'),
                                     (o, f'ours (streamed FBP), slice {z}'))):
        a = ax[r, c] if len(pairs) > 1 else ax[c]
        a.imshow(im * (1 if c == 0 else rows[r][4]), cmap='gray',
                 vmin=lo, vmax=hi)
        a.set_title(title)
        a.axis('off')
plt.tight_layout()
plt.savefig(png, dpi=60)
