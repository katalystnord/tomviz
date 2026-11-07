###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Thermo Fisher Velox EMD files: HDF5 whose ``/Version`` JSON says
``"format": "Velox"`` (unlike Berkeley EMD, which Tomviz reads in C++).

* Images: ``/Data/Image/<uuid>/Data``, shape (y, x, frames).
* Per-frame metadata: ``/Data/Image/<uuid>/Metadata``, a uint8 array of
  shape (n, frames); column i is the NUL-padded JSON of frame i, with
  ``Stage.AlphaTilt`` (radians, as a string) and
  ``BinaryResult.PixelSize.width/height`` (metres).

Checked on a Velox stack (rosettasciio test data). A stack opens as a tilt
series when its frames' alpha tilts differ. With several images in one
file (simultaneous detectors), the first is read and the detector named
in the metadata.
"""

import json

import numpy as np

from tomviz.io import FileType, Reader, image_from_array


def is_velox(path):
    try:
        import h5py
        with h5py.File(path, 'r') as f:
            if 'Version' not in f:
                return False
            version = np.ravel(f['Version'][()])[0]
            if isinstance(version, bytes):
                version = version.decode('utf-8', 'replace')
            return json.loads(version).get('format') == 'Velox'
    except (OSError, ImportError, ValueError, AttributeError):
        return False


def frame_metadata(metadata, i):
    raw = bytes(np.asarray(metadata[:, i], dtype=np.uint8))
    return json.loads(raw.rstrip(b'\0').decode('utf-8', 'replace'))


class VeloxReader(Reader):

    @staticmethod
    def file_type():
        return FileType('Thermo Fisher Velox EMD', ['emd'])

    def read(self, path):
        import h5py
        with h5py.File(path, 'r') as f:
            images = f.get('Data/Image')
            if not images or not len(images):
                raise ValueError(f'{path}: no images in this Velox file')
            count = len(images)
            group = images[sorted(images)[0]]
            data = np.asarray(group['Data'][()])
            if data.ndim == 2:
                data = data[:, :, None]
            frames = [frame_metadata(group['Metadata'], i)
                      for i in range(group['Metadata'].shape[1])]
        first = frames[0] if frames else {}
        size = first.get('BinaryResult', {}).get('PixelSize', {})
        unit = first.get('BinaryResult', {}).get('PixelUnitX', 'm')
        scale = {'m': 1e9, 'nm': 1.0, 'pm': 1e-3}.get(unit, 1.0)
        px = float(size.get('width', 1.0)) * scale
        py = float(size.get('height', size.get('width', 1.0))) * scale
        tilts = np.array([float(fr.get('Stage', {}).get('AlphaTilt', 'nan'))
                          for fr in frames])
        angles = None
        if len(tilts) == data.shape[2] and len(tilts) > 1 and \
                np.all(np.isfinite(tilts)) and np.ptp(tilts) > 0:
            angles = np.degrees(tilts)
        metadata = {'velox': {
            'detector': first.get('BinaryResult', {}).get('Detector'),
            'images_in_file': count,
        }}
        # (y, x, n) -> Tomviz's (x, y, n).
        return image_from_array(np.transpose(data, (1, 0, 2)),
                                spacing=(px, py, 1.0), tilt_angles=angles,
                                metadata=metadata)
