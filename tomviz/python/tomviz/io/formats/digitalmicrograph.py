###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Gatan DigitalMicrograph files (.dm3, .dm4), read with Tomviz's own
parser (tomviz.io.dm).

A stack whose third axis is calibrated in degrees, as Gatan's tomography
acquisition writes it, opens as a tilt series: angle = (index - origin)
x scale. Checked on ETSpy's HAADF.dm3 test file (NIST): 91 images from
-90 to +90 degrees in steps of 2.

Images are laid out as Tomviz opens MRC stacks: x along the image
columns, y along the rows, z the image index. Pixel sizes are converted
to nanometres when their unit is known.
"""

import numpy as np

from tomviz.io import FileType, Reader, image_from_array

_TO_NM = {'nm': 1.0, 'µm': 1e3, 'um': 1e3, 'micron': 1e3,
          'Å': 0.1, 'A': 0.1, 'angstrom': 0.1, 'pm': 1e-3, 'mm': 1e6,
          'm': 1e9}


def calibration(dataset):
    """(scale, origin, unit) per axis, in the order of the data array."""
    scales = np.atleast_1d(dataset.get('pixelSize', []))
    origins = np.atleast_1d(dataset.get('pixelOrigin', []))
    units = list(dataset.get('pixelUnit', []))
    out = []
    for i in range(np.ndim(dataset['data'])):
        out.append((float(scales[i]) if i < len(scales) else 1.0,
                    float(origins[i]) if i < len(origins) else 0.0,
                    str(units[i]).strip() if i < len(units) else ''))
    return out


class DigitalMicrographReader(Reader):

    @staticmethod
    def file_type():
        return FileType('Gatan DigitalMicrograph', ['dm3', 'dm4'])

    def read(self, path):
        from tomviz.io import dm
        f = dm.FileDM(path, on_memory=False)
        try:
            dataset = f.getDataset(0)
            data = np.array(dataset['data'])  # a copy, not the file's map
            cal = calibration(dataset)
            del dataset
        finally:
            del f
        if data.ndim == 2:
            data = data[None]
            cal = [(1.0, 0.0, '')] + cal
        if data.ndim != 3:
            raise ValueError(f'{path}: expected an image or a stack, got '
                             f'{data.ndim} dimensions')
        (sz, oz, uz), (sy, _, uy), (sx, _, ux) = cal
        spacing = (sx * _TO_NM.get(ux, 1.0), sy * _TO_NM.get(uy, 1.0), 1.0)
        angles = None
        if uz.lower() in ('deg', 'degree', 'degrees', '°'):
            angles = (np.arange(data.shape[0]) - oz) * sz
        else:
            spacing = spacing[:2] + (sz * _TO_NM.get(uz, 1.0),)
        # (z, y, x) array -> Tomviz's (x, y, z).
        return image_from_array(np.transpose(data, (2, 1, 0)),
                                spacing=spacing, tilt_angles=angles)
