###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Tilt angles for MRC tilt series (.mrc, .st, .ali, ...).

The MRC format has no tilt angle field. Acquisition and processing
programs put them in one of these places, all tried here:

* ``<name>.tlt`` next to an aligned stack (``.ali``): IMOD's refined
  angles, which belong to the aligned images.
* The extended header:
    - ``FEI1`` / ``FEI2`` (Thermo Fisher Tomography, MRC2014 spec
      PN 306687): one metadata block per image, "Alpha tilt" as float64
      at byte 100, in the file's byte order.
    - ``SERI`` (SerialEM): per image ``nint`` bytes; if bit 1 of
      ``nreal`` is set the first field is the tilt angle times 100 as an
      int16 (IMOD mrc_format.txt).
    - The legacy FEI/Agard header (no EXTTYP, 1024 blocks of 128 bytes,
      ``nreal`` = 32): ``a_tilt`` as float32 at byte 0 of each block.
* A SerialEM / Thermo Fisher autodoc next to the file,
  ``<file>.mdoc`` or ``<name>.mdoc``: ``TiltAngle = ...`` in each
  ``[ZValue = n]`` section.
* IMOD's ``<name>.rawtlt`` (raw angles) or ``<name>.tlt``: one or more
  numbers per line.

Only a list with exactly one angle per image is used.
"""

import math
import os
import re
import struct

import numpy as np


def tilt_angles(path, count):
    """Tilt angles in degrees for the ``count`` images of an MRC stack,
    or None when none are found."""
    stem = os.path.splitext(path)[0]
    sources = []
    if path.lower().endswith('.ali'):
        sources.append(lambda: _text_angles(stem + '.tlt'))
    sources += [
        lambda: header_angles(path),
        lambda: mdoc_angles(path + '.mdoc'),
        lambda: mdoc_angles(stem + '.mdoc'),
        lambda: _text_angles(stem + '.rawtlt'),
        lambda: _text_angles(stem + '.tlt'),
    ]
    for source in sources:
        try:
            angles = source()
        except (OSError, ValueError, struct.error):
            angles = None
        if angles is not None and len(angles) == count:
            return [float(a) for a in angles]
    return None


def _text_angles(path):
    if not os.path.exists(path):
        return None
    values = []
    with open(path) as f:
        for line in f:
            values += [float(v) for v in line.split()]
    return values or None


def mdoc_angles(path):
    """TiltAngle per [ZValue = n] section of a SerialEM autodoc."""
    if not os.path.exists(path):
        return None
    angles = {}
    z = None
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\[\s*ZValue\s*=\s*(\d+)\s*\]', line)
            if m:
                z = int(m.group(1))
                continue
            if line.startswith('['):
                z = None
                continue
            key, _, value = line.partition('=')
            if z is not None and key.strip() == 'TiltAngle':
                angles[z] = float(value.split()[0])
    if not angles or sorted(angles) != list(range(len(angles))):
        return None
    return [angles[z] for z in range(len(angles))]


def header_angles(path):
    """Tilt angles from the extended header, or None."""
    with open(path, 'rb') as f:
        header = f.read(1024)
        if len(header) < 1024:
            return None
        # Machine stamp: 0x44 0x44 (or 0x44 0x41) little endian, 0x11 0x11
        # big endian. Old files may leave it zero: assume little endian.
        bo = '>' if header[212] == 0x11 else '<'
        nx, ny, nz = struct.unpack(bo + '3i', header[0:12])
        nsymbt = struct.unpack(bo + 'i', header[92:96])[0]
        exttyp = header[104:108]
        nint, nreal = struct.unpack(bo + '2h', header[128:132])
        if nsymbt <= 0 or nz <= 0:
            return None
        ext = f.read(nsymbt)
    if len(ext) < nsymbt:
        return None
    if exttyp in (b'FEI1', b'FEI2'):
        return _fei_angles(ext, nz, bo)
    if exttyp == b'SERI':
        return _serialem_angles(ext, nz, nint, nreal, bo)
    if exttyp in (b'\0\0\0\0', b'    ', b'MRCO', b'AGAR') and nreal == 32 \
            and nint == 0 and nsymbt >= 128 * nz:
        return _legacy_fei_angles(ext, nz, bo)
    return None


def _fei_angles(ext, nz, bo):
    # Every block is as long as the first one says ("Metadata size").
    size = struct.unpack(bo + 'i', ext[0:4])[0]
    if size < 108 or size * nz > len(ext):
        return None
    angles = np.array([
        struct.unpack(bo + 'd', ext[i * size + 100:i * size + 108])[0]
        for i in range(nz)])
    # IMOD notes that FEI software once wrote these in radians. A real
    # tilt series in degrees spans more than a few degrees.
    if np.all(np.abs(angles) <= math.pi / 2 + 1e-3) and \
            np.ptp(angles) > 0.05:
        angles = np.degrees(angles)
    return list(angles)


# SerialEM extended header fields, in order: (bit, bytes).
_SERI_FIELDS = ((1, 2), (2, 6), (4, 4), (8, 2), (16, 2), (32, 4))


def _serialem_angles(ext, nz, nint, nreal, bo):
    if nint <= 0 or not nreal & 1:
        return None
    flagged = sum(n for bit, n in _SERI_FIELDS if nreal & bit)
    if flagged != nint or nint * nz > len(ext):
        # Then nint/nreal are counts of ints and reals, not SerialEM.
        return None
    return [struct.unpack(bo + 'h', ext[i * nint:i * nint + 2])[0] / 100.0
            for i in range(nz)]


def _legacy_fei_angles(ext, nz, bo):
    angles = [struct.unpack(bo + 'f', ext[i * 128:i * 128 + 4])[0]
              for i in range(nz)]
    if not any(angles):
        return None
    return angles
