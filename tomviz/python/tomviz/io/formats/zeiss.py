###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""ZEISS Xradia (Versa, Ultra) files: OLE2 compound documents.

* ``.txrm``: projections, with the flat field (``ReferenceData/Image``).
* ``.txm``: a reconstructed volume, one image per slice.
* ``.xrm``: a single image.

There is no public specification. Stream names follow dxchange's reader
(Argonne BSD-style licence) and CIL's ZEISSDataReader (Apache-2.0),
checked against a real Versa scan and ZEISS's own reconstruction of it
(Zenodo 4822516, CC-BY-4.0):

* ImageInfo/NoOfImages, ImageWidth, ImageHeight, DataType (5 uint16,
  10 float32); images in ``ImageData{ceil(i/100)}/Image{i}``.
* ImageInfo/PixelSize: the voxel size at the object, in micrometres.
* ImageInfo/Angles (degrees), StoRADistance and DtoRADistance (mm,
  signed; their magnitudes are the source-axis and axis-detector
  distances), XrayGeometry (1 = cone beam), AcquisitionMode (0 =
  projections).
* alignment/x-shifts, y-shifts: per-view image shifts in pixels that the
  ZEISS software applies (for example to undo dithering). CIL rolls the
  images by their whole-pixel part; here they become per-view detector
  offsets in the geometry, which keeps the sub-pixel part and leaves the
  projections and flat field as recorded.

Needs the olefile package (BSD).
"""

import struct

import numpy as np

from tomviz.io import (FileType, Reader, image_from_array,
                       image_from_frames)

DTYPES = {5: np.dtype('<u2'), 10: np.dtype('<f4')}


def _open(path):
    try:
        import olefile
    except ImportError:
        raise RuntimeError('Reading ZEISS files needs the olefile package '
                           '(conda install -c conda-forge olefile)')
    return olefile.OleFileIO(path)


def _values(ole, name, fmt, count=None):
    if not ole.exists(name):
        return None
    data = ole.openstream(name).read()
    size = struct.calcsize('<' + fmt)
    n = len(data) // size if count is None else count
    values = struct.unpack('<' + fmt * n, data[:n * size])
    return np.array(values)


def _value(ole, name, fmt, default=None):
    v = _values(ole, name, fmt, 1)
    return default if v is None else v[0].item()


def read_metadata(ole):
    n = _value(ole, 'ImageInfo/NoOfImages', 'I')
    meta = {
        'n': n,
        'width': _value(ole, 'ImageInfo/ImageWidth', 'I'),
        'height': _value(ole, 'ImageInfo/ImageHeight', 'I'),
        'dtype': DTYPES.get(_value(ole, 'ImageInfo/DataType', 'I')),
        'voxel_um': _value(ole, 'ImageInfo/PixelSize', 'f'),
        'cone': _value(ole, 'ImageInfo/XrayGeometry', 'I', 1) == 1,
        'projections': _value(ole, 'ImageInfo/AcquisitionMode', 'I', 0) == 0,
        'angles': _values(ole, 'ImageInfo/Angles', 'f', n),
        'sod': _values(ole, 'ImageInfo/StoRADistance', 'f', n),
        'odd': _values(ole, 'ImageInfo/DtoRADistance', 'f', n),
        'x_shifts': _values(ole, 'alignment/x-shifts', 'f', n),
        'y_shifts': _values(ole, 'alignment/y-shifts', 'f', n),
    }
    if meta['dtype'] is None:
        raise ValueError('Unsupported ZEISS image data type')
    return meta


def read_images(ole, meta):
    out = np.empty((meta['n'], meta['height'], meta['width']), meta['dtype'])
    for i in range(meta['n']):
        stream = 'ImageData{}/Image{}'.format(-(-(i + 1) // 100), i + 1)
        out[i] = np.frombuffer(ole.openstream(stream).read(), meta['dtype'],
                               meta['height'] * meta['width']).reshape(
                                   meta['height'], meta['width'])
    return out


def read_reference(ole, meta):
    if not ole.exists('ReferenceData/Image'):
        return None
    dtype = DTYPES.get(_value(ole, 'ReferenceData/DataType', 'I'),
                       meta['dtype'])
    data = np.frombuffer(ole.openstream('ReferenceData/Image').read(), dtype)
    if data.size != meta['height'] * meta['width']:
        return None
    return data.reshape(meta['height'], meta['width'])


def circular_orbit(angles_deg, sod, sdd, pixel, clockwise):
    """Per-view source, detector centre, column step and row step
    ((n, 3) each, mm) for a circular orbit about the z axis: the source
    at angle beta is at sod (cos beta, sin beta, 0), the detector centre
    opposite at distance sdd - sod from the axis, columns along
    (-sin beta, cos beta, 0) and rows along z."""
    beta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64).ravel())
    if clockwise:
        beta = -beta
    c, s = np.cos(beta), np.sin(beta)
    zero = np.zeros_like(beta)
    e_s = np.stack([c, s, zero], axis=1)
    e_u = np.stack([-s, c, zero], axis=1)
    e_v = np.tile([0.0, 0.0, 1.0], (len(beta), 1))
    return sod * e_s, -(sdd - sod) * e_s, pixel * e_u, pixel * e_v


def scan_geometry(meta):
    """The geometry dict for dataset.metadata: per-view vectors (ASTRA
    cone_vec layout: source, detector centre, column step, row step)
    that include the alignment shifts."""
    w, h = meta['width'], meta['height']
    sod = abs(float(meta['sod'][0]))
    odd = abs(float(meta['odd'][0]))
    pixel = meta['voxel_um'] / 1000.0 * (sod + odd) / sod
    source, det_center, det_u, det_v = circular_orbit(
        meta['angles'], sod, sod + odd, pixel, CLOCKWISE)
    if meta['x_shifts'] is not None and meta['y_shifts'] is not None:
        det_center = det_center + SHIFT_SIGN * (
            meta['x_shifts'][:, None] * det_u +
            meta['y_shifts'][:, None] * det_v)
    vectors = np.concatenate([source, det_center, det_u, det_v], axis=1)
    return {
        'sod': sod,
        'sdd': sod + odd,
        'pixel_size_u': pixel,
        'pixel_size_v': pixel,
        'detector_shape': [h, w],
        'vectors': vectors.tolist(),
        'units': 'mm',
        'source': 'ZEISS .txrm (alignment shifts as per-view offsets)',
    }


# Set by reconstructing the USB stick scan of Zenodo 4822516 and comparing
# a slice with ZEISS's own reconstruction in the same dataset (a box
# around the circuit board): clockwise, with the
# alignment shifts added as written, correlates at 0.968. Without the
# shifts it is 0.70, with them reversed 0.56, and counter-clockwise at
# most 0.64. (The slices match as a transpose; the direction along the
# rotation axis was not checked separately for ZEISS.)
CLOCKWISE = True
SHIFT_SIGN = 1.0


class ZeissReader(Reader):

    @staticmethod
    def file_type():
        return FileType('ZEISS Xradia', ['txrm', 'txm', 'xrm'])

    def read(self, path):
        ole = _open(path)
        try:
            meta = read_metadata(ole)
            images = read_images(ole, meta)
            reference = read_reference(ole, meta)
        finally:
            ole.close()
        voxel = (meta['voxel_um'] or 1000.0) / 1000.0
        if not (meta['projections'] and meta['cone'] and
                meta['angles'] is not None and meta['n'] > 1):
            # A reconstructed volume (or a single image): slices along z.
            return image_from_array(np.transpose(images, (2, 1, 0)),
                                    spacing=(voxel, voxel, voxel))
        geometry = scan_geometry(meta)
        pixel = geometry['pixel_size_u']
        data = image_from_frames(images, (pixel, pixel),
                                 angles=meta['angles'],
                                 metadata={'geometry': geometry})
        if reference is None:
            return data
        return {'data': data, 'flat': image_from_frames(reference,
                                                        (pixel, pixel))}
