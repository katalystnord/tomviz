###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""NeXus NXtomo files (ESRF, Diamond and other synchrotrons): HDF5 with a
group whose ``definition`` is ``NXtomo``.

Following the NXtomo application definition
(manual.nexusformat.org/classes/applications/NXtomo.html):

* ``instrument/detector/data``: all frames, (n_frames, rows, columns).
* ``instrument/detector/image_key``: 0 projection, 1 flat field,
  2 dark field, 3 invalid.
* ``sample/rotation_angle``: one angle per frame, with ``units``.
* ``instrument/detector/x_pixel_size``, ``y_pixel_size`` and ``distance``
  (optional, with units); ESRF also writes ``real_x_pixel_size``.

Like httomo's loader, the first group with ``definition == NXtomo`` is
used, wherever it is in the file. Projections open as a tilt series, and
flat and dark frames as datasets of their own for Flat/Dark Field
Correction.
"""

import numpy as np

from tomviz.io import FileType, Reader, image_from_frames

PROJECTION, FLAT, DARK = 0, 1, 2

# Length units to millimetres, angle units to degrees.
_LENGTH = {'m': 1e3, 'metre': 1e3, 'meter': 1e3, 'mm': 1.0,
           'millimetre': 1.0, 'millimeter': 1.0, 'um': 1e-3, 'µm': 1e-3,
           'micron': 1e-3, 'micrometre': 1e-3, 'micrometer': 1e-3,
           'nm': 1e-6, 'nanometre': 1e-6, 'nanometer': 1e-6}
_ANGLE = {'degree': 1.0, 'degrees': 1.0, 'deg': 1.0,
          'radian': np.degrees(1.0), 'radians': np.degrees(1.0),
          'rad': np.degrees(1.0)}


def _text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value[()])
    return str(value)


def find_nxtomo(f):
    """The path of the first group whose definition is NXtomo, or None."""
    found = []

    def visit(name, obj):
        if found or not hasattr(obj, 'keys') or 'definition' not in obj:
            return None
        try:
            if _text(obj['definition'][()]).strip() == 'NXtomo':
                found.append(name)
        except (TypeError, ValueError, KeyError):
            pass
        return None
    f.visititems(visit)
    return found[0] if found else None


def is_nxtomo(path):
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            return find_nxtomo(f) is not None
    except (OSError, ImportError):
        return False


def _first(group, names):
    for name in names:
        if name in group:
            return group[name]
    return None


def _length_mm(dataset):
    if dataset is None:
        return None
    units = _text(dataset.attrs.get('units', 'm')).strip()
    scale = _LENGTH.get(units)
    if scale is None:
        return None
    return float(np.ravel(dataset[()])[0]) * scale


class NXtomoReader(Reader):

    @staticmethod
    def file_type():
        return FileType('NeXus NXtomo', ['nxs'])

    def read(self, path):
        import h5py
        with h5py.File(path, 'r') as f:
            name = find_nxtomo(f)
            if name is None:
                raise ValueError(f'{path} has no NXtomo group')
            g = f[name]
            data = _first(g, ['instrument/detector/data', 'data/data'])
            keys = _first(g, ['instrument/detector/image_key',
                              'data/image_key',
                              'instrument/detector/image_key_control'])
            angles = _first(g, ['sample/rotation_angle',
                                'data/rotation_angle'])
            if data is None or angles is None:
                raise ValueError(f'{path}: the NXtomo group has no detector '
                                 'data or no rotation angles')
            n = data.shape[0]
            keys = np.zeros(n, int) if keys is None else \
                np.asarray(keys[()]).astype(int).ravel()
            units = _text(angles.attrs.get('units', 'degree')).strip()
            scale = _ANGLE.get(units.lower(), 1.0)
            theta = np.asarray(angles[()], dtype=np.float64).ravel() * scale

            det = g.get('instrument/detector', {})
            px_u = _length_mm(_first(det, ['real_x_pixel_size',
                                           'x_pixel_size']))
            px_v = _length_mm(_first(det, ['real_y_pixel_size',
                                           'y_pixel_size'])) or px_u
            px_u = px_u or px_v or 1.0
            px_v = px_v or px_u
            metadata = {'nxtomo': {
                'entry': name,
                'pixel_size_mm': [px_v, px_u],
                'sample_detector_distance_mm': _length_mm(
                    _first(det, ['distance'])),
            }}

            def frames(key):
                index = np.flatnonzero(keys == key)
                if len(index) == 0:
                    return None, index
                return data[index], index

            proj, index = frames(PROJECTION)
            if proj is None:
                raise ValueError(f'{path}: no projections (image_key 0)')
            flat, _ = frames(FLAT)
            dark, _ = frames(DARK)
            result = {'data': image_from_frames(proj, (px_v, px_u),
                                                angles=theta[index],
                                                metadata=metadata)}
            if flat is not None:
                result['flat'] = image_from_frames(flat, (px_v, px_u))
            if dark is not None:
                result['dark'] = image_from_frames(dark, (px_v, px_u))
            return result
