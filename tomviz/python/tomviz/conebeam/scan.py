###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Opening a scan on disk for streamed reconstruction, without loading it:
a row source (tomviz.conebeam.sources), the projection angles, and what
the file says about the scanner (dataset.metadata layout, including
'geometry' and 'white_level' where the format stores them).

Scanner projects (Nikon .xtekct, Waygate .pca, SkyScan .log) are opened
through Tomviz's file readers for them when those are installed; plain
stacks (a folder or list of TIFF files, a multi-page TIFF, .npy, an HDF5
dataset) carry no geometry, which then comes from parameters.
"""

import glob
import os

import numpy as np

from .sources import FrameFiles, HDF5Frames, LineIntegrals, \
    open_beam_levels
from .stream import RawStack


class Scan:
    def __init__(self, source, angles=None, metadata=None, flat=None,
                 dark=None, kind='projections'):
        self.source = source        # raw intensities or line integrals
        self.angles = angles        # degrees, or None
        self.metadata = metadata or {}
        self.flat = flat            # averaged reference frames, if any
        self.dark = dark
        self.kind = kind            # a description for messages

    @property
    def shape(self):
        return self.source.shape


def _vendor(path):
    """A scanner project through the Tomviz readers for it, if present."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.xtekct':
            from tomviz.io.formats import nikon
            params = nikon.read_xtekct(path)
            recon = os.path.splitext(path)[0] + '_recon.xtekct'
            recon_params = nikon.read_xtekct(recon) if \
                os.path.exists(recon) and \
                not path.endswith('_recon.xtekct') else None
            angles, meta = nikon.scan_metadata(path, params, recon_params)
            files = nikon.projection_files(path, params)
            name = 'Nikon XT scan'
        elif ext == '.pca':
            from tomviz.io.formats import waygate
            pca = waygate.read_pca(path)
            angles, meta = waygate.scan_metadata(path, pca)
            files = waygate.projection_files(path, len(angles))
            name = 'Waygate phoenix scan'
        elif ext == '.log':
            from tomviz.io.formats import skyscan
            log = skyscan.read_log(path)
            files = skyscan.projection_files(path, log)
            angles, meta = skyscan.scan_metadata(path, log, len(files))
            name = 'SkyScan scan'
        else:
            return None
    except ImportError as e:
        raise RuntimeError(f'{path}: reading this scanner format needs '
                           f"Tomviz's reader for it ({e})") from e
    return Scan(FrameFiles(files), np.asarray(angles, float), meta,
                kind=name)


def _tiff_files(path):
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, '*'))
                       if f.lower().endswith(('.tif', '.tiff')))
        if not files:
            raise RuntimeError(f'{path}: no TIFF files in the folder')
        return files
    if any(c in path for c in '*?['):
        files = sorted(glob.glob(path))
        if not files:
            raise RuntimeError(f'{path}: no files match')
        return files
    return [path]


def open_scan(path, dataset=None, raw_shape=None, raw_dtype='float32',
              raw_offset=0):
    """A Scan of the projections at path:

    * .xtekct, .pca, .log: a scanner project (geometry included);
    * a folder of TIFF files, a glob pattern, or a (multi-page) TIFF;
    * .npy (C order);
    * .h5, .hdf5, .nxs, .emd: dataset names the frames, else an NXtomo
      entry's projections (with its flats and darks) or Data Exchange's
      /exchange/data (with data_white and data_dark);
    * anything else: raw C-ordered frames, raw_shape (n, rows, columns).
    """
    path = os.fspath(path)
    scan = _vendor(path)
    if scan is not None:
        return scan
    low = path.lower()
    if os.path.isdir(path) or low.endswith(('.tif', '.tiff')) or \
            any(c in path for c in '*?['):
        return Scan(FrameFiles(_tiff_files(path)), kind='TIFF frames')
    if low.endswith('.npy'):
        return Scan(RawStack.npy(path), kind='.npy stack')
    if low.endswith(('.h5', '.hdf5', '.nxs', '.emd', '.hdf')):
        return _open_hdf5(path, dataset)
    if raw_shape is None:
        raise RuntimeError(f'{path}: unknown format; give the shape of the '
                           'raw frames')
    return Scan(RawStack(path, raw_shape, raw_dtype, raw_offset),
                kind='raw frames')


def _open_hdf5(path, dataset):
    import h5py
    if dataset:
        return Scan(HDF5Frames(path, dataset), kind=f'{dataset}')
    with h5py.File(path, 'r') as f:
        if 'exchange/data' in f:
            names = {k: f'exchange/{n}' for k, n in
                     (('data', 'data'), ('flat', 'data_white'),
                      ('dark', 'data_dark')) if f'exchange/{n}' in f}
            angles = f['exchange/theta'][()] if 'exchange/theta' in f \
                else None
            index = None
        else:
            try:
                from tomviz.io.formats import nxtomo
            except ImportError:
                nxtomo = None
            entry = nxtomo.find_nxtomo(f) if nxtomo else None
            if entry is None:
                raise RuntimeError(f'{path}: name the dataset that holds '
                                   'the frames')
            g = f[entry]
            data = nxtomo._first(g, ['instrument/detector/data',
                                     'data/data'])
            keys = nxtomo._first(g, ['instrument/detector/image_key',
                                     'data/image_key'])
            n = data.shape[0]
            keys = np.zeros(n, int) if keys is None else \
                np.asarray(keys[()]).astype(int).ravel()
            theta = nxtomo._first(g, ['sample/rotation_angle',
                                      'data/rotation_angle'])
            units = nxtomo._text(theta.attrs.get('units', 'degree')) \
                .strip().lower() if theta is not None else 'degree'
            scale = nxtomo._ANGLE.get(units, 1.0)
            index = np.flatnonzero(keys == nxtomo.PROJECTION)
            angles = None if theta is None else \
                np.asarray(theta[()], float).ravel()[index] * scale
            names = {'data': data.name}
            refs = {'flat': np.flatnonzero(keys == nxtomo.FLAT),
                    'dark': np.flatnonzero(keys == nxtomo.DARK)}
    if index is None:
        source = HDF5Frames(path, names['data'])
        flat = dark = None
        with h5py.File(path, 'r') as f:
            if 'flat' in names:
                flat = np.asarray(f[names['flat']][()], np.float32) \
                    .reshape(-1, *source.shape[1:]).mean(axis=0)
            if 'dark' in names:
                dark = np.asarray(f[names['dark']][()], np.float32) \
                    .reshape(-1, *source.shape[1:]).mean(axis=0)
        return Scan(source, angles, flat=flat, dark=dark,
                    kind='Data Exchange')
    source = HDF5Frames(path, names['data'], index)
    flat = source.frames(refs['flat']).mean(axis=0) \
        if len(refs['flat']) else None
    dark = source.frames(refs['dark']).mean(axis=0) \
        if len(refs['dark']) else None
    return Scan(source, angles, flat=flat, dark=dark, kind='NXtomo')


def line_integrals(scan, mode='auto', i0=0.0, dark_level=0.0,
                   percentile=99.5, log=None):
    """The scan's projections as line integrals.

    mode: 'none' when they already are; 'flat' to divide by the scan's
    own flat (and dark) frames; 'open-beam' for one open-beam level:
    i0 if given, else the scanner's stored white level, else estimated
    per projection from its brightest pixels; 'auto' picks 'flat' when
    there are flats, 'none' for float data without them, else
    'open-beam'."""
    log = log or (lambda msg: None)
    if mode == 'auto':
        if scan.flat is not None:
            mode = 'flat'
        elif np.issubdtype(getattr(scan.source, 'dtype', np.float32),
                           np.floating) and \
                not scan.metadata.get('white_level') and i0 <= 0:
            mode = 'none'
        else:
            mode = 'open-beam'
    if mode == 'none':
        return scan.source
    if mode == 'flat':
        if scan.flat is None:
            raise RuntimeError(f'This {scan.kind} has no flat fields')
        log('line integrals from the flat'
            + (' and dark fields' if scan.dark is not None else ' field'))
        return LineIntegrals(scan.source, scan.flat,
                             scan.dark if scan.dark is not None
                             else dark_level)
    if i0 <= 0:
        i0 = float(scan.metadata.get('white_level') or 0)
        if i0 > 0:
            log(f"open-beam level {i0:g} from the scanner's file")
    if i0 > 0:
        return LineIntegrals(scan.source, np.float32(i0), dark_level)
    levels = open_beam_levels(scan.source, percentile)
    log(f'open-beam levels estimated per projection ({levels.min():g} to '
        f'{levels.max():g})')
    return LineIntegrals(scan.source, levels, dark_level)
