###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Nikon Metrology XT H / X-TEK CT scans: a ``.xtekct`` file (INI, section
[XTekCT]) next to one TIFF per projection, ``<Name>_0001.tif`` onwards.

There is no public specification; the keys used here are the ones the
open readers rely on (CIL's NikonDataReader, Apache-2.0, and TIGRE,
BSD-3-Clause), checked against a real scan (Zenodo 6874123):

* SrcToObject, SrcToDetector (mm); DetectorPixelsX/Y, DetectorPixelSizeX/Y
  (mm); DetectorOffsetX/Y (mm).
* Projections, InitialAngle, AngularStep (degrees). Like CIL, the .ang
  and _ctdata.txt angle files are not used.
* The centre of rotation: ObjectOffsetX (mm, at the object) when present,
  else the mean of CentreOfRotationTop/Bottom (mm, at the detector). A
  scan's own file often has 0 there and AutomaticCentreOfRotation=1;
  Nikon's software then stores the centre it found in
  ``<Name>_recon.xtekct``, which is used when it exists.
* WhiteLevel: the open-beam intensity the projections are divided by.
* The scanner writes one projection more than Projections (the 360
  degree repeat of the first); only the first Projections are read.
"""

import configparser
import glob
import os
import re

import numpy as np

from tomviz.io import FileType, Reader, image_from_frames


def read_xtekct(path):
    """The [XTekCT] section of a .xtekct file as a dict of strings."""
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # keep the key case
    with open(path, encoding='latin-1') as f:
        parser.read_string(f.read())
    if not parser.has_section('XTekCT'):
        raise ValueError(f'{path}: no [XTekCT] section')
    return dict(parser['XTekCT'])


def _num(params, key, default=None):
    value = params.get(key)
    if value is None or value.strip() == '':
        if default is None:
            raise ValueError(f'The .xtekct file has no {key}')
        return default
    return float(value)


def centre_of_rotation(params, recon_params=None):
    """The rotation axis offset at the detector, in mm, and where it came
    from. ObjectOffsetX is at the object, so it is magnified."""
    for source, p in (('reconstruction file', recon_params),
                      ('scan file', params)):
        if not p:
            continue
        if p.get('ObjectOffsetX', '').strip():
            mag = _num(p, 'SrcToDetector') / _num(p, 'SrcToObject')
            return _num(p, 'ObjectOffsetX') * mag, source
        top = _num(p, 'CentreOfRotationTop', 0.0)
        bottom = _num(p, 'CentreOfRotationBottom', 0.0)
        if top or bottom or source == 'scan file':
            return 0.5 * (top + bottom), source
    return 0.0, 'scan file'


def projection_files(path, params):
    folder = os.path.dirname(path)
    sub = params.get('InputFolderName', '').strip()
    if sub:
        folder = os.path.join(folder, sub)
    name = params.get('Name', '').strip() or \
        os.path.splitext(os.path.basename(path))[0]
    files = {}
    pattern = re.compile(re.escape(name) + r'_(\d+)\.tiff?$', re.IGNORECASE)
    for f in glob.glob(os.path.join(glob.escape(folder), '*.tif*')):
        m = pattern.match(os.path.basename(f))
        if m:
            files[int(m.group(1))] = f
    n = int(_num(params, 'Projections'))
    numbers = sorted(files)[:n]
    if len(numbers) < n:
        raise ValueError(f'{n} projections expected, {len(numbers)} '
                         f'{name}_NNNN.tif files found in {folder}')
    return [files[i] for i in numbers]


def scan_metadata(path, params, recon_params=None):
    """Angles (degrees) and the geometry dict for dataset.metadata."""
    n = int(_num(params, 'Projections'))
    angles = _num(params, 'InitialAngle', 0.0) + \
        _num(params, 'AngularStep') * np.arange(n)
    pu = _num(params, 'DetectorPixelSizeX')
    pv = _num(params, 'DetectorPixelSizeY')
    cor, cor_source = centre_of_rotation(params, recon_params)
    offset_u = (cor + _num(params, 'DetectorOffsetX', 0.0)) / pu
    offset_v = _num(params, 'DetectorOffsetY', 0.0) / pv
    notes = [f'centre of rotation from the {cor_source}']
    for key in ('ObjectRoll', 'ObjectTilt'):
        if _num(params, key, 0.0):
            notes.append(f'{key} = {params[key]} is not applied')
    geometry = {
        'sod': _num(params, 'SrcToObject'),
        'sdd': _num(params, 'SrcToDetector'),
        'pixel_size_u': pu,
        'pixel_size_v': pv,
        'offset_u': offset_u,
        'offset_v': offset_v,
        'clockwise': CLOCKWISE,
        'detector_shape': [int(_num(params, 'DetectorPixelsY')),
                           int(_num(params, 'DetectorPixelsX'))],
        'units': 'mm',
        'source': 'Nikon .xtekct: ' + '; '.join(notes),
    }
    metadata = {'geometry': geometry,
                'white_level': _num(params, 'WhiteLevel', 0.0) or None}
    return angles, metadata


# Direction of rotation relative to Tomviz's circular orbit. Found on the
# Korn scan (Zenodo 6874123): clockwise, with the centre offset as Nikon
# gives it, reconstructs sharp seeds; the other direction smears every
# edge, and the other offset sign blurs them.
# On the crystals-in-clay scan (Zenodo 4912635) our slices then match
# Nikon's own reconstruction at correlation 0.84 to 0.87 (half
# resolution, every third view), with the slice order reversed and the
# slice transposed: a proper rotation of the axes, not a mirror image,
# if Nikon's .vol axes are right-handed (Volume Graphics does not say).
CLOCKWISE = True


class NikonReader(Reader):

    @staticmethod
    def file_type():
        return FileType('Nikon XT CT scan', ['xtekct'])

    def read(self, path):
        params = read_xtekct(path)
        recon = os.path.splitext(path)[0] + '_recon.xtekct'
        recon_params = read_xtekct(recon) if os.path.exists(recon) and \
            not path.endswith('_recon.xtekct') else None
        angles, metadata = scan_metadata(path, params, recon_params)

        import tifffile
        frames = tifffile.imread(projection_files(path, params))
        return image_from_frames(
            frames, (metadata['geometry']['pixel_size_v'],
                     metadata['geometry']['pixel_size_u']),
            angles=angles, metadata=metadata)
