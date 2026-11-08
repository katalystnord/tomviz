###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Reconstruct a scan on disk into a file, never holding all of either in
memory (FDK streamed a slab at a time; see docs/large-data.md).

    python -m tomviz.conebeam.reconstruct_file SCAN OUTPUT.emd [options]

SCAN is a scanner project (Nikon .xtekct, Waygate .pca, SkyScan .log,
which carry their geometry), a folder of TIFF files, a multi-page TIFF,
.npy, or an HDF5 file (NXtomo, Data Exchange, or --dataset). OUTPUT.emd
opens in Tomviz like any other volume. Run with --help for the options.
"""

import argparse
import os
import sys
import time

import numpy as np

from tomviz.projectors.geometry import ParallelGeometry

from .operator_support import build_geometry, build_volume
from .scan import axis_offset_from_opposite, line_integrals, open_scan
from .stream import (RawStack, create_emd, default_memory_bytes,
                     fdk_stream, sirt_stream)

# Formats that come from synchrotrons, whose beams are parallel.
PARALLEL_KINDS = ('NXtomo', 'Data Exchange')


def reconstruct_to_file(path, output, beam='auto', method='fbp',
                        iterations=50, geometry_source='auto', sod=100.0,
                        sdd=200.0, pixel_size=0.0, offset_u=0.0,
                        offset_v=0.0, find_axis=False, clockwise=False,
                        geometry_file=None, arc=360.0, intensity='auto',
                        i0=0.0,
                        dark_level=0.0, filter_name='ram-lak',
                        short_scan='auto', volume_size_xy=0,
                        volume_size_axial=0, voxel_size=0.0,
                        backend='auto', memory_gb=0.0, dataset=None,
                        raw_shape=None, raw_dtype='float32', progress=None,
                        should_stop=None, log=None):
    """Reconstruct the scan at path into output (.emd, or a raw float32
    file in (z, y, x) order otherwise). Parameters as the FDK operator's,
    plus:

    beam: 'cone', 'parallel', or 'auto': parallel for synchrotron formats
    (NXtomo, Data Exchange), else cone.
    method: 'fbp' (FDK for cone beams) or 'sirt' (parallel beams only),
    with iterations.
    geometry_source: 'auto' takes the scanner's geometry from its file
    when there is one, else 'circular' from sod, sdd, pixel_size...
    (parallel beams use pixel_size, offset_u, offset_v and clockwise).
    pixel_size: 0 takes it from the file; parallel beams without one are
    reconstructed in pixel units.
    find_axis: for parallel beams, find offset_u from projections 180
    degrees apart; when that is not possible, the given offset_u is used.
    arc: degrees covered, when the file stores no angles (evenly spaced,
    the last projection one step short of the arc).
    intensity: 'auto', 'none', 'flat' or 'open-beam' (scan.line_integrals).
    memory_gb: working memory; 0 means half of what is available now.

    Returns (vol_geom, timings)."""
    log = log or (lambda msg: None)
    scan = open_scan(path, dataset=dataset, raw_shape=raw_shape,
                     raw_dtype=raw_dtype)
    try:
        n, n_v, n_u = scan.shape
        log(f'{scan.kind}: {n} projections of {n_v} x {n_u}')
        angles = scan.angles
        if angles is None or len(angles) != n:
            angles = np.arange(n) * (arc / n)
        if beam == 'auto':
            beam = 'parallel' if scan.kind in PARALLEL_KINDS else 'cone'
            log(f'{beam} beam')
        if pixel_size <= 0:
            pixel_size = float(scan.metadata.get('pixel_size_mm') or 0)
        source = line_integrals(scan, intensity, i0=i0,
                                dark_level=dark_level, log=log)
        if beam == 'parallel':
            if pixel_size <= 0:
                pixel_size = 1.0
                log('no pixel size in the file: lengths in pixels')
            if find_axis:
                try:
                    found = axis_offset_from_opposite(source, angles)
                except RuntimeError as e:
                    log(f'rotation axis not found ({e}); using the given '
                        f'offset, {offset_u:+.2f} pixels')
                else:
                    note = f' (replaces the given {offset_u:+.2f})' if \
                        offset_u else ''
                    offset_u = found
                    log(f'rotation axis {offset_u:+.2f} pixels from the '
                        f'detector centre{note}')
            geom = ParallelGeometry.circular(
                angles, pixel_size, pixel_size, n_u, n_v,
                offset_u=offset_u, offset_v=offset_v, clockwise=clockwise)
        else:
            if method == 'sirt':
                raise RuntimeError(
                    'SIRT of scans larger than memory works for parallel '
                    'beams only; use FDK, or Cone-Beam SIRT in memory')
            if geometry_source == 'auto':
                geometry_source = 'metadata' if \
                    scan.metadata.get('geometry') else 'circular'
            if geometry_source == 'circular' and pixel_size <= 0:
                raise RuntimeError('Give the detector pixel size')
            geom = build_geometry(n, n_u, n_v, angles, geometry_source,
                                  sod, sdd, pixel_size, offset_u, offset_v,
                                  clockwise, geometry_file,
                                  metadata=scan.metadata)
        vol_geom = build_volume(geom, volume_size_xy, volume_size_axial,
                                voxel_size)
        nz, ny, nx = vol_geom.shape
        log(f'volume {nz} x {ny} x {nx} voxels of '
            f'{vol_geom.voxel_size[0]:.4g} mm, {nz * ny * nx * 4 / 1e9:.1f}'
            ' GB')
        output = os.fspath(output)
        if output.lower().endswith(('.emd', '.h5', '.hdf5')):
            target = create_emd(output, vol_geom)
        else:
            target = RawStack(output, vol_geom.shape, mode='w')
        try:
            memory = int(memory_gb * 2 ** 30) if memory_gb > 0 else \
                default_memory_bytes()
            short = {'auto': 'auto', 'on': True, 'off': False}.get(
                short_scan, short_scan)
            if method == 'sirt':
                timings = sirt_stream(source, geom, vol_geom, target,
                                      iterations=iterations,
                                      backend=backend, memory_bytes=memory,
                                      progress=progress,
                                      should_stop=should_stop, log=log)
            else:
                timings = fdk_stream(source, geom, vol_geom, target,
                                     filter_name=filter_name,
                                     short_scan=short, backend=backend,
                                     memory_bytes=memory,
                                     progress=progress,
                                     should_stop=should_stop, log=log)
        finally:
            target.close()
    finally:
        scan.source.close()
    return vol_geom, timings


def read_preview(path, max_voxels=256 ** 3):
    """A strided view of a reconstructed .emd small enough to show:
    (array in Tomviz's (x, y, z) layout, spacing, stride)."""
    import h5py
    with h5py.File(path, 'r') as f:
        d = f['data/tomography/data']
        total = int(np.prod(d.shape))
        stride = max(1, int(np.ceil((total / max_voxels) ** (1 / 3))))
        # Stored with Tomviz's (x, y, z) axes in C order, dim1 to dim3.
        data = d[::stride, ::stride, ::stride]
        spacing = []
        for name in ('dim1', 'dim2', 'dim3'):
            ax = f[f'data/tomography/{name}'][()]
            spacing.append(float(ax[1] - ax[0]) * stride if len(ax) > 1
                           else 1.0)
    return np.asfortranarray(data), spacing, stride


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='python -m tomviz.conebeam.reconstruct_file',
        description='Reconstruct a scan on disk into a file, a slab at a '
        'time, for scans larger than memory: FDK for cone beams, FBP or '
        'SIRT for parallel beams.')
    p.add_argument('scan', help='scanner project, TIFF folder or file, '
                   '.npy, or HDF5 file')
    p.add_argument('output', help='.emd (opens in Tomviz) or raw float32')
    p.add_argument('--dataset', help='HDF5 dataset holding the frames')
    p.add_argument('--raw-shape', type=int, nargs=3,
                   metavar=('N', 'ROWS', 'COLUMNS'))
    p.add_argument('--raw-dtype', default='float32')
    p.add_argument('--beam', default='auto',
                   choices=['auto', 'cone', 'parallel'],
                   help='auto: parallel for NXtomo and Data Exchange')
    p.add_argument('--method', default='fbp', choices=['fbp', 'sirt'],
                   help='sirt: parallel beams only')
    p.add_argument('--iterations', type=int, default=50)
    g = p.add_argument_group('geometry (when the scan does not carry it)')
    g.add_argument('--geometry', default='auto',
                   choices=['auto', 'metadata', 'circular', 'file'])
    g.add_argument('--sod', type=float, default=100.0,
                   help='source to rotation axis, mm')
    g.add_argument('--sdd', type=float, default=200.0,
                   help='source to detector, mm')
    g.add_argument('--pixel-size', type=float, default=0.0,
                   help='detector pixel pitch, mm (0: from the file)')
    g.add_argument('--offset-u', type=float, default=0.0,
                   help='detector offset across the axis, pixels')
    g.add_argument('--offset-v', type=float, default=0.0,
                   help='detector offset along the axis, pixels')
    g.add_argument('--find-axis', action='store_true',
                   help='parallel beams: find the axis offset from '
                   'projections 180 degrees apart')
    g.add_argument('--clockwise', action='store_true')
    g.add_argument('--vectors', help='ASTRA cone_vec file (12 per row)')
    g.add_argument('--arc', type=float, default=360.0,
                   help='degrees covered, when no angles are stored')
    i = p.add_argument_group('intensities')
    i.add_argument('--intensity', default='auto',
                   choices=['auto', 'none', 'flat', 'open-beam'])
    i.add_argument('--i0', type=float, default=0.0,
                   help='open-beam level (0: from the file or estimated)')
    i.add_argument('--dark-level', type=float, default=0.0)
    r = p.add_argument_group('reconstruction')
    r.add_argument('--filter', default='ram-lak')
    r.add_argument('--short-scan', default='auto',
                   choices=['auto', 'on', 'off'])
    r.add_argument('--volume-xy', type=int, default=0)
    r.add_argument('--volume-z', type=int, default=0)
    r.add_argument('--voxel-size', type=float, default=0.0)
    r.add_argument('--backend', default='auto')
    r.add_argument('--memory-gb', type=float, default=0.0,
                   help='working memory (0: half of what is free)')
    a = p.parse_args(argv)

    start = time.perf_counter()
    last = [0.0]

    def progress(fraction):
        now = time.perf_counter()
        if now - last[0] > 2 or fraction >= 1:
            last[0] = now
            print(f'  {100 * fraction:5.1f} %  {now - start:7.1f} s',
                  flush=True)

    try:
        vol_geom, t = reconstruct_to_file(
            a.scan, a.output, beam=a.beam, method=a.method,
            iterations=a.iterations, geometry_source=a.geometry, sod=a.sod,
            sdd=a.sdd, pixel_size=a.pixel_size, offset_u=a.offset_u,
            offset_v=a.offset_v, find_axis=a.find_axis,
            clockwise=a.clockwise,
            geometry_file=a.vectors, arc=a.arc, intensity=a.intensity,
            i0=a.i0, dark_level=a.dark_level, filter_name=a.filter,
            short_scan=a.short_scan, volume_size_xy=a.volume_xy,
            volume_size_axial=a.volume_z, voxel_size=a.voxel_size,
            backend=a.backend, memory_gb=a.memory_gb, dataset=a.dataset,
            raw_shape=a.raw_shape, raw_dtype=a.raw_dtype,
            progress=progress, log=print)
    except (RuntimeError, ValueError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    rows = t.pop('rows', None)
    print('done: ' + ', '.join(f'{k} {v:.1f} s' for k, v in t.items())
          + (f'; {rows} detector rows read' if rows else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
