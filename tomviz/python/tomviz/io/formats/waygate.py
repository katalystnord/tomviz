###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Waygate Technologies (GE) phoenix datos|x CT scans: a ``.pca`` file
(INI) next to one TIFF per projection, ``<name>NNNNN.tif`` from 1, and a
``.pcp`` table with the rotation position of every image.

There is no public specification and no licensed open reader; the keys
were read from real scans (Zenodo 14993754 and 14993749, v|tome|x m):

* [Geometry] FDD, FOD (mm), cx, cy (image centre, pixels), and
  VoxelSizeX/Y: the voxel size datos|x reconstructs at, which is the
  stored pixel pitch over the magnification FDD/FOD. That gives the
  pixel pitch; the [Image] region (Left, Right, ...) and [Detector]
  PixelsizeX did not (on the plug scan, the pitch from the region was
  twice too large and the reconstruction showed it at once).
* [Image] DimX, DimY: the stored image size.
* [Image] FreeRay: the open-beam grey value (the flat field and offset
  images named in [CalibImages] were applied during acquisition).
* [CT] NumberImages; the scanner writes one image more (360 degrees).
* .pcp: ImgNr and RotPos (degrees) per image.
"""

import configparser
import glob
import os
import re

import numpy as np

from tomviz.io import FileType, Reader, image_from_frames


def read_pca(path):
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    with open(path, encoding='latin-1') as f:
        parser.read_string(f.read())
    return parser


def _get(pca, section, key, default=None):
    try:
        return float(pca[section][key])
    except (KeyError, ValueError):
        if default is None:
            raise ValueError(f'The .pca file has no [{section}] {key}')
        return default


def rotation_positions(path, n):
    """RotPos (degrees) of the first n images from the .pcp table, or
    None when there is no table."""
    pcp = os.path.splitext(path)[0] + '.pcp'
    if not os.path.exists(pcp):
        return None
    with open(pcp, encoding='latin-1') as f:
        lines = [line.split('\t') for line in f.read().splitlines()]
    for i, cols in enumerate(lines):
        if 'ImgNr' in cols and 'RotPos' in cols:
            nr, rot = cols.index('ImgNr'), cols.index('RotPos')
            table = {int(c[nr]): float(c[rot]) for c in lines[i + 1:]
                     if len(c) > max(nr, rot) and c[nr].strip().isdigit()}
            if all(k in table for k in range(1, n + 1)):
                return np.array([table[k] for k in range(1, n + 1)])
    return None


def projection_files(path, n):
    base = os.path.splitext(path)[0]
    pattern = re.compile(re.escape(os.path.basename(base)) + r'(\d+)\.tiff?$',
                         re.IGNORECASE)
    files = {}
    for f in glob.glob(glob.escape(base) + '*.tif*'):
        m = pattern.match(os.path.basename(f))
        if m:
            files[int(m.group(1))] = f
    if not all(k in files for k in range(1, n + 1)):
        raise ValueError(f'{n} projections expected next to {path}, found '
                         f'{len(files)} image files')
    return [files[k] for k in range(1, n + 1)]


def scan_metadata(path, pca):
    n = int(_get(pca, 'CT', 'NumberImages'))
    angles = rotation_positions(path, n)
    if angles is None:
        sector = _get(pca, 'CT', 'RotationSector', 360.0)
        angles = np.arange(n) * sector / n
    dim_x = int(_get(pca, 'Image', 'DimX'))
    dim_y = int(_get(pca, 'Image', 'DimY'))
    mag = _get(pca, 'Geometry', 'FDD') / _get(pca, 'Geometry', 'FOD')
    pu = _get(pca, 'Geometry', 'VoxelSizeX') * mag
    pv = _get(pca, 'Geometry', 'VoxelSizeY', pu / mag) * mag
    cx = _get(pca, 'Geometry', 'cx', (dim_x - 1) / 2)
    cy = _get(pca, 'Geometry', 'cy', (dim_y - 1) / 2)
    # [Geometry] CalibValue (-0.428 on the plug scan) is undocumented; the
    # reconstruction file repeats it as CorrectionValue. Compared with
    # Waygate's own reconstruction of the plug, the best centre lies at
    # -0.2 to -0.3 pixel (correlation 0.99359, against 0.99351 at 0 and
    # 0.99354 at -0.4), so it is applied as a pixel offset, sign as
    # written. Read as millimetres it visibly doubles every pin.
    calib = _get(pca, 'Geometry', 'CalibValue', 0.0)
    notes = []
    for key in ('DetectorRot', 'Tilt'):
        if abs(_get(pca, 'Geometry', key, 0.0)) > 0.01:
            notes.append(f'{key} = {pca["Geometry"][key]} is not applied')
    geometry = {
        'sod': _get(pca, 'Geometry', 'FOD'),
        'sdd': _get(pca, 'Geometry', 'FDD'),
        'pixel_size_u': pu,
        'pixel_size_v': pv,
        'offset_u': cx - (dim_x - 1) / 2 + calib,
        'offset_v': cy - (dim_y - 1) / 2,
        'clockwise': CLOCKWISE,
        'detector_shape': [dim_y, dim_x],
        'units': 'mm',
        'source': 'Waygate phoenix .pca' + (': ' + '; '.join(notes)
                                            if notes else ''),
    }
    metadata = {'geometry': geometry,
                'white_level': _get(pca, 'Image', 'FreeRay', 0.0) or None}
    return angles, metadata


# Direction of rotation relative to Tomviz's circular orbit. Found on the
# plug scan (Zenodo 14993754): counter-clockwise reconstructs the pins as
# sharp dots; clockwise turns every one into a ring of arcs. With it, the
# central slice matches Waygate's own reconstruction at correlation
# 0.9935 (a 90 degree in-plane rotation of the array; not mirrored).
CLOCKWISE = False


class WaygateReader(Reader):

    @staticmethod
    def file_type():
        return FileType('Waygate phoenix CT scan', ['pca'])

    def read(self, path):
        pca = read_pca(path)
        angles, metadata = scan_metadata(path, pca)

        import tifffile
        frames = tifffile.imread(projection_files(path, len(angles)))
        g = metadata['geometry']
        return image_from_frames(frames, (g['pixel_size_v'],
                                          g['pixel_size_u']),
                                 angles=angles, metadata=metadata)
