###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Projections read a band of detector rows at a time, for fdk_stream.

A source has .shape (n_projections, n_rows, n_columns) and
read_rows(r0, r1), which returns rows r0..r1 of every projection as an
(n, r1 - r0, n_columns) float32 array. Frames are (rows, columns) as
detectors store them; rows run along the rotation axis, as in
tomviz.io.image_from_frames.

  FrameFiles      TIFF files, one or many frames each (and any image
                  format Pillow reads, decoded whole)
  HDF5Frames      a 3D dataset in an HDF5 file (NXtomo, Data Exchange)
  LineIntegrals   -ln((I - dark) / (flat - dark)) of another source
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


class FrameFiles:
    """Projections in image files, in order: one file per projection
    (Nikon, Waygate, SkyScan, a TIFF folder) or multi-page TIFF stacks.

    Uncompressed TIFF pages, which scanners write, are read row band by
    row band with positioned reads at the page's offset in the file, from
    several threads (on an NVMe drive, parallel small reads are much
    faster than one after another). Compressed pages and other formats are
    decoded whole for every band: correct, but slower."""

    def __init__(self, files, threads=8):
        import tifffile
        self.files = [os.fspath(f) for f in files]
        if not self.files:
            raise ValueError('No projection files')
        # (path, byte offset or None, page index, byte order)
        self.frames = []
        shape = dtype = None
        for path in self.files:
            if not path.lower().endswith(('.tif', '.tiff')):
                self.frames.append((path, None, 0, None))
                continue
            with tifffile.TiffFile(path) as tif:
                for i, page in enumerate(tif.pages):
                    page_shape = tuple(page.shape)
                    if shape is None:
                        shape, dtype = page_shape, page.dtype
                    elif page_shape != shape:
                        raise ValueError(
                            f'{path} page {i}: {page_shape} pixels, the '
                            f'first frame has {shape}')
                    plain = page.is_memmappable and len(page_shape) == 2
                    self.frames.append(
                        (path, page.dataoffsets[0] if plain else None, i,
                         tif.byteorder))
        if shape is None:   # no TIFF at all: decode the first frame
            first = self._decode(self.frames[0])
            shape, dtype = first.shape, first.dtype
        if len(shape) != 2:
            raise ValueError(f'Expected greyscale frames, got {shape}')
        self.shape = (len(self.frames),) + tuple(shape)
        self.dtype = np.dtype(dtype)
        self.threads = threads

    @staticmethod
    def _decode(frame):
        path, _, page, _ = frame
        if path.lower().endswith(('.tif', '.tiff')):
            import tifffile
            return tifffile.imread(path, key=page)
        from PIL import Image
        return np.asarray(Image.open(path))

    def _read_frame(self, k, r0, r1, out):
        path, offset, _, byteorder = self.frames[k]
        if offset is None:
            out[...] = self._decode(self.frames[k])[r0:r1]
            return
        n_u = self.shape[2]
        file_dtype = self.dtype.newbyteorder(byteorder)
        buf = np.empty((r1 - r0, n_u), dtype=file_dtype)
        from .stream import O_BINARY, read_at
        fd = os.open(path, os.O_RDONLY | O_BINARY)
        try:
            read_at(fd, memoryview(buf).cast('B'),
                    offset + r0 * n_u * file_dtype.itemsize)
        except IOError as e:
            raise IOError(f'{path}: {e}') from e
        finally:
            os.close(fd)
        out[...] = buf

    def read_rows(self, r0, r1):
        out = np.empty((self.shape[0], r1 - r0, self.shape[2]), np.float32)
        with ThreadPoolExecutor(self.threads) as pool:
            for f in [pool.submit(self._read_frame, k, r0, r1, out[k])
                      for k in range(self.shape[0])]:
                f.result()
        return out

    def close(self):
        pass


class HDF5Frames:
    """Frames (n, rows, columns) of an HDF5 dataset; index picks some of
    them (the projections among NXtomo's flats and darks). Row bands come
    from h5py hyperslab reads; a dataset chunked or compressed in whole
    frames is decoded whole for every band."""

    def __init__(self, path, dataset, index=None):
        import h5py
        self.file = h5py.File(path, 'r')
        self.data = self.file[dataset]
        if self.data.ndim != 3:
            raise ValueError(f'{path}:{dataset} is not a stack of frames')
        self.dtype = self.data.dtype
        n = self.data.shape[0]
        if index is None:
            self.select = slice(None)
            count = n
        else:
            index = np.asarray(index, dtype=np.int64)
            if len(index) and np.all(np.diff(index) == 1):
                self.select = slice(int(index[0]), int(index[-1]) + 1)
            else:
                self.select = index
            count = len(index)
        self.shape = (count,) + tuple(self.data.shape[1:])

    def read_rows(self, r0, r1):
        return np.asarray(self.data[self.select, r0:r1, :], np.float32)

    def frames(self, index):
        """Whole frames (flats, darks) as float32."""
        return np.asarray(self.data[np.asarray(index)], np.float32)

    def close(self):
        self.file.close()


def _as_band(value, r0, r1):
    """A scalar, a frame (rows, columns) or per-projection levels (n,)
    cut to rows r0..r1, shaped to broadcast over (n, rows, columns)."""
    if value is None:
        return np.float32(0)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        return value
    if value.ndim == 1:
        return value[:, None, None]
    return value[r0:r1]


class LineIntegrals:
    """-ln((I - dark) / (flat - dark)) of the intensities another source
    reads, computed band by band in the reader thread.

    flat: a frame (rows, columns), one open-beam level for all, or one per
    projection (n,) (see open_beam_levels). dark: a frame or a level.
    As flat_dark_correct: transmission clipped to min_transmission, and
    dead pixels (flat <= dark) give no attenuation."""

    def __init__(self, source, flat, dark=None, min_transmission=1e-6):
        self.source = source
        self.shape = source.shape
        self.flat = flat
        self.dark = dark
        self.min_transmission = np.float32(min_transmission)
        gain = np.asarray(flat, np.float32) - np.asarray(
            0 if dark is None else dark, np.float32)
        if gain.ndim == 0 and gain <= 0:
            raise ValueError('The open beam must be brighter than the dark '
                             'level')

    def read_rows(self, r0, r1):
        t = self.source.read_rows(r0, r1)
        dark = _as_band(self.dark, r0, r1)
        gain = _as_band(self.flat, r0, r1) - dark
        dead = gain <= 0
        gain = np.where(dead, np.float32(1), gain)
        t -= dark
        t /= gain
        np.maximum(t, self.min_transmission, out=t)
        np.log(t, out=t)
        np.negative(t, out=t)
        if np.any(dead):
            t[np.broadcast_to(dead, t.shape)] = 0
        return t

    def close(self):
        self.source.close()


def open_beam_levels(source, percentile=99.5, n_rows=32):
    """One open-beam level per projection, from its brightest pixels, as
    the IntensityToLineIntegrals operator estimates them; from n_rows
    detector rows spread over the frame instead of every pixel, so that
    estimating it costs a small fraction of one pass over the data."""
    n_v = source.shape[1]
    rows = np.unique(np.linspace(0, n_v - 1, min(n_rows, n_v)).astype(int))
    sample = np.concatenate([source.read_rows(r, r + 1) for r in rows],
                            axis=1)
    return np.percentile(sample.reshape(len(sample), -1), percentile,
                         axis=1).astype(np.float32)
