###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Bruker micro-CT (SkyScan) scans: an INI-style ``.log`` file next to one
image per projection, ``<prefix>NNNN.tif`` (or ``.bmp``) from 0000.

There is no public specification; the keys are the ones TIGRE's
BrukerDataLoader (BSD-3-Clause) uses, checked against a real SkyScan 1172
scan (Zenodo 12744181):

* [System] Camera Pixel Size (um), CameraXYRatio.
* [Acquisition] Filename Prefix, Number of Files (one more than there are
  projections), Number of Rows/Columns, Camera binning ("2x2"),
  Object to Source (mm), Camera to Source (mm), Optical Axis (line),
  Rotation Step (deg), Rotation Direction (CC or CW),
  Flat Field Correction (ON: the images are already normalised).

The horizontal centre of rotation is not in the acquisition log (NRecon
finds it when reconstructing), so it starts at the detector centre.
"""

import configparser
import glob
import os
import re

import numpy as np

from tomviz.io import FileType, Reader, image_from_frames


def read_log(path):
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str
    try:
        with open(path, encoding='latin-1') as f:
            parser.read_string(f.read())
    except configparser.Error:
        parser = None
    if parser is None or not parser.has_section('Acquisition'):
        raise ValueError(f'{path} is not a Bruker SkyScan scan log (no '
                         '[Acquisition] section)')
    return parser


def _get(log, section, key, default=None):
    # Key case differs between software versions ("Number Of Files").
    for k, v in log[section].items() if log.has_section(section) else ():
        if k.lower() == key.lower():
            return v.strip()
    if default is None:
        raise ValueError(f'The scan log has no [{section}] {key}')
    return default


def _num(log, section, key, default=None):
    return float(_get(log, section, key, default))


def projection_files(path, log):
    folder = os.path.dirname(path)
    prefix = _get(log, 'Acquisition', 'Filename Prefix',
                  os.path.splitext(os.path.basename(path))[0])
    pattern = re.compile(re.escape(prefix) + r'(\d{4,})\.(tiff?|bmp)$',
                         re.IGNORECASE)
    files = {}
    for f in glob.glob(os.path.join(glob.escape(folder),
                                    glob.escape(prefix) + '*')):
        m = pattern.match(os.path.basename(f))
        if m:
            files[int(m.group(1))] = f
    n = int(_num(log, 'Acquisition', 'Number of Files')) - 1
    numbers = sorted(files)[:n]
    if not numbers or numbers != list(range(numbers[0],
                                            numbers[0] + len(numbers))):
        raise ValueError(f'No consecutive {prefix}NNNN images next to {path}')
    return [files[i] for i in numbers]


def scan_metadata(path, log, n):
    step = _num(log, 'Acquisition', 'Rotation Step (deg)')
    angles = step * np.arange(n)
    binning = _get(log, 'Acquisition', 'Camera binning', '1x1').lower()
    bu, _, bv = binning.partition('x')
    pixel = _num(log, 'System', 'Camera Pixel Size (um)') / 1000.0
    ratio = _num(log, 'System', 'CameraXYRatio', 1.0)
    pu = pixel * float(bu or 1)
    pv = pixel * float(bv or bu or 1) * ratio
    rows = int(_num(log, 'Acquisition', 'Number of Rows'))
    cols = int(_num(log, 'Acquisition', 'Number of Columns'))
    axis_row = _num(log, 'Acquisition', 'Optical Axis (line)',
                    (rows - 1) / 2)
    direction = _get(log, 'Acquisition', 'Rotation Direction', 'CC').upper()
    geometry = {
        'sod': _num(log, 'Acquisition', 'Object to Source (mm)'),
        'sdd': _num(log, 'Acquisition', 'Camera to Source (mm)'),
        'pixel_size_u': pu,
        'pixel_size_v': pv,
        'offset_u': 0.0,
        'offset_v': OPTICAL_AXIS_SIGN * (axis_row - (rows - 1) / 2),
        'clockwise': CLOCKWISE_FOR_CC if direction == 'CC'
        else not CLOCKWISE_FOR_CC,
        'detector_shape': [rows, cols],
        'units': 'mm',
        'source': 'Bruker SkyScan .log: the horizontal centre of rotation '
                  'is not in the log; refine offset_u if edges look doubled',
    }
    flat_corrected = _get(log, 'Acquisition', 'Flat Field Correction',
                          'OFF').upper() == 'ON'
    return angles, {'geometry': geometry,
                    'flat_field_corrected': flat_corrected}


# Direction: on the grains scan (Zenodo 12744181, "Rotation Direction=CC")
# counter-clockwise reconstructs sharp grain outlines and clockwise
# doubles them. Its log has no centre of rotation; the sharpest centre
# was within a few pixels of the detector centre.
CLOCKWISE_FOR_CC = False
# Optical Axis (line) -> vertical offset: TIGRE's convention. On the
# grains scan either sign gives equally sharp slices (it mainly moves
# them up or down by up to twice the offset), so the sign is NOT
# verified on data; there is no Bruker reconstruction to compare with.
OPTICAL_AXIS_SIGN = 1.0


class SkyScanReader(Reader):

    @staticmethod
    def file_type():
        return FileType('Bruker SkyScan scan log', ['log'])

    def read(self, path):
        log = read_log(path)
        files = projection_files(path, log)
        angles, metadata = scan_metadata(path, log, len(files))

        import tifffile
        if files[0].lower().endswith('.bmp'):
            from PIL import Image
            frames = np.stack([np.asarray(Image.open(f)) for f in files])
        else:
            frames = tifffile.imread(files)
        g = metadata['geometry']
        return image_from_frames(frames, (g['pixel_size_v'],
                                          g['pixel_size_u']),
                                 angles=angles, metadata=metadata)
