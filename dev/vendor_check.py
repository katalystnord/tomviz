"""Check the lab CT scanner readers against the vendors' own reconstructions.

Reconstructs public scans with Tomviz's readers and FDK, and compares the
central slice with the reconstruction the scanner software made of the same
scan (normalised cross-correlation, best over the eight flips/rotations of
a square and over small shifts):

    nikon    Zenodo 4912635, "Crystals in clay" (Nikon XT H), CC-BY-4.0:
             clay.xtekct + TIFFs, and clay_recon/clay.vol (float32, VG).
    waygate  Zenodo 14993754, "Waygate plug 750 acq.zip" and
             "Waygate plug 750 recon.zip" (phoenix v|tome|x m), CC-BY-4.0.
    zeiss    Zenodo 4822516, usb.zip (Xradia Versa): the scan .txrm and
             ZEISS's reconstruction *_recon.txm, CC-BY-4.0.

    python dev/vendor_check.py nikon   /path/to/clay-folder
    python dev/vendor_check.py waygate /path/to/plug-folder /path/to/recon
    python dev/vendor_check.py zeiss   /path/to/scan.txrm /path/to/recon.txm

Results when the readers were written (RX 9070):
    nikon:   our slices at several heights against Nikon's central
             slice: peak 0.86 at the matching height (half resolution,
             every third view), as a transpose. Matching every height
             against every Nikon slice (0.84 to 0.87) shows the slice
             order reversed: with the transpose a rotation, not a mirror.
    waygate: correlation 0.9935 (every second view); a 90 degree rotation.
    zeiss:   correlation 0.968 over the circuit board (ZEISS's slice has a
             bright field inside a black circle where ours is dark, so the
             object region is compared); half resolution, every 2nd view.
"""

import glob
import os
import sys

import numpy as np
import tifffile

from tomviz.conebeam import fdk
from tomviz.conebeam.operator_support import geometry_from_metadata
from tomviz.projectors import VolumeGeometry


def best_match(mine, ref, size=2048):
    """Best normalised cross-correlation of ref with any of the eight
    flips/rotations of mine, over all shifts."""
    def norm(a):
        return (a - a.mean()) / (a.std() + 1e-12)
    ref = norm(ref)
    out = []
    for name, v in (('same', mine), ('flip x', mine[:, ::-1]),
                    ('flip y', mine[::-1]), ('rot180', mine[::-1, ::-1]),
                    ('T', mine.T), ('T flip x', mine.T[:, ::-1]),
                    ('T flip y', mine.T[::-1]),
                    ('T rot180', mine.T[::-1, ::-1])):
        # No cropping: ref may cover only part of our slice (vendors crop
        # their reconstructions), so search for it anywhere in ours.
        v = norm(v)
        shape = (size, size)
        f = np.fft.rfft2(v, shape) * np.conj(np.fft.rfft2(ref, shape))
        out.append((float(np.fft.irfft2(f, (size, size)).max() / ref.size),
                    name))
    return sorted(out, reverse=True)


def line_integrals(raw, level):
    return -np.log(np.clip(raw / level, 1e-6, None))


def nikon(folder):
    from tomviz.io.formats import nikon as m
    path = glob.glob(os.path.join(glob.escape(folder), '*.xtekct'))
    path = [p for p in path if not p.endswith('_recon.xtekct')][0]
    params = m.read_xtekct(path)
    recon = os.path.splitext(path)[0] + '_recon.xtekct'
    angles, meta = m.scan_metadata(path, params, m.read_xtekct(recon))
    step, b = 3, 2
    raw = tifffile.imread(m.projection_files(path, params)[::step])
    n, r, c = raw.shape
    raw = raw.reshape(n, r // b, b, c // b, b).mean((2, 4))
    proj = line_integrals(raw, meta['white_level'])
    geom = geometry_from_metadata(meta, n, c // b, r // b, angles[::step])

    recon_dir = os.path.join(glob.escape(folder), '*_recon')
    vgi = open(glob.glob(os.path.join(recon_dir, '*.vgi'))[0]).read()
    size = [int(v) for v in vgi.split('Size = ')[1].split()[:3]]
    vox = float(vgi.split('resolution = ')[1].split()[0])
    vol = np.fromfile(glob.glob(os.path.join(recon_dir, '*.vol'))[0],
                      np.float32).reshape(size[::-1])
    k = 2 * (min(size[:2]) // 2)
    ref = vol[size[2] // 2, :k, :k].reshape(k // b, b, k // b, b).mean((1, 3))
    zs = np.arange(-30.0, 15.01, 5.0)
    vg = VolumeGeometry((274, k // b, k // b), vox * b,
                        center=(-7.5, 0.0, -0.5))
    mine = fdk(proj, geom, vg, backend='auto')
    for z in zs:
        i = int(np.argmin(abs(vg.z_coords() - z)))
        print(f'z {z:+6.1f} mm:', best_match(mine[i], ref)[0])


def waygate(folder, recon):
    from tomviz.io.formats import waygate as m
    path = glob.glob(os.path.join(glob.escape(folder), '*.pca'))[0]
    angles, meta = m.scan_metadata(path, m.read_pca(path))
    step = 2
    raw = tifffile.imread(m.projection_files(path, len(angles))[::step])
    n, r, c = raw.shape
    proj = line_integrals(raw.astype(np.float32), meta['white_level'])
    geom = geometry_from_metadata(meta, n, c, r, angles[::step])
    pcr = open(glob.glob(os.path.join(glob.escape(recon), '*.pcr'))[0],
               encoding='latin-1').read()
    vox = float(pcr.split('VoxelSizeRec=')[1].split()[0])
    slices = sorted(glob.glob(os.path.join(glob.escape(recon), '*.tif')))
    ref = tifffile.imread(slices[len(slices) // 2]).astype(np.float32)
    mine = fdk(proj, geom, VolumeGeometry((3, *ref.shape), vox),
               backend='auto')[1]
    print('central slice:', best_match(mine, ref)[:2])


def zeiss(scan, recon):
    from tomviz.io.formats import zeiss as m
    ole = m._open(scan)
    meta = m.read_metadata(ole)
    images = m.read_images(ole, meta).astype(np.float32)
    flat = m.read_reference(ole, meta).astype(np.float32)
    ole.close()
    b = 2
    idx = np.arange(0, meta['n'] - 1, 2)  # the last view repeats the first
    t = np.clip(images[idx] / flat, 1e-6, None)
    n, h, w = t.shape
    proj = -np.log(t.reshape(n, h // b, b, w // b, b).mean((2, 4)))
    g = m.scan_geometry(meta)
    g['vectors'] = [g['vectors'][i] for i in idx]
    geom = geometry_from_metadata({'geometry': g}, n, w // b, h // b, None)
    ole = m._open(recon)
    rmeta = m.read_metadata(ole)
    rec = m.read_images(ole, rmeta).astype(np.float32)
    ole.close()
    ref = rec[len(rec) // 2]
    ref = ref.reshape(ref.shape[0] // b, b, ref.shape[1] // b, b).mean((1, 3))
    H, W = ref.shape
    box = ref[int(0.33 * H):int(0.75 * H), int(0.3 * W):int(0.75 * W)]
    vox = meta['voxel_um'] / 1000 * b
    mine = fdk(proj, geom, VolumeGeometry((3, w // b, w // b), vox),
               backend='auto')[1]

    def norm(a):
        return (a - a.mean()) / a.std()
    a = norm(box)
    best = []
    for name, cand in (('T', mine.T), ('T flip y', mine.T[::-1]),
                       ('T flip x', mine.T[:, ::-1]), ('same', mine)):
        f = np.fft.rfft2(norm(cand), (1024, 1024)) * np.conj(
            np.fft.rfft2(a, (1024, 1024)))
        y, x = np.unravel_index(np.argmax(np.fft.irfft2(f, (1024, 1024))),
                                (1024, 1024))
        win = cand[y:y + a.shape[0], x:x + a.shape[1]]
        if win.shape == a.shape:
            best.append((float((norm(win) * a).mean()), name))
    print('object region:', sorted(best, reverse=True)[:2])


if __name__ == '__main__':
    {'nikon': nikon, 'waygate': waygate,
     'zeiss': zeiss}[sys.argv[1]](*sys.argv[2:])
