"""Tilt angles for MRC tilt series: extended headers (FEI1/FEI2, SerialEM,
legacy FEI) and the IMOD / SerialEM files next to the stack.

Checked by hand on real files too (ETSpy test data): the legacy FEI
header of HAADF.mrc gives the same 77 angles as its HAADF.rawtlt, and
SerialEM SERI headers agree with their .mdoc to 0.01 degree."""

import struct

import numpy as np
import pytest

from tomviz.io.mrc_tilts import tilt_angles

ANGLES = [-60.0, -30.5, 0.0, 29.75, 60.0]


def write_mrc(path, nz=len(ANGLES), exttyp=b'\0\0\0\0', ext=b'', nint=0,
              nreal=0, big_endian=False):
    """A small MRC2014 stack of nz 4 x 3 int16 images."""
    bo = '>' if big_endian else '<'
    h = bytearray(1024)
    struct.pack_into(bo + '4i', h, 0, 4, 3, nz, 1)       # nx ny nz mode
    struct.pack_into(bo + '3i', h, 28, 4, 3, nz)         # mx my mz
    struct.pack_into(bo + '3f', h, 40, 40.0, 30.0, nz)   # cella (A)
    struct.pack_into(bo + 'i', h, 92, len(ext))          # nsymbt
    h[104:108] = exttyp
    struct.pack_into(bo + 'i', h, 108, 20140)
    struct.pack_into(bo + '2h', h, 128, nint, nreal)
    h[208:212] = b'MAP '
    h[212:214] = b'\x11\x11' if big_endian else b'\x44\x44'
    data = np.zeros((nz, 3, 4), dtype=bo + 'i2')
    path.write_bytes(bytes(h) + ext + data.tobytes())
    return str(path)


def fei_ext(angles, version=b'FEI2', size=1000, big_endian=False):
    # MRC2014 spec PN 306687: per-image block, size at byte 0, "Alpha
    # tilt" float64 at byte 100, in the file's byte order.
    bo = '>' if big_endian else '<'
    ext = bytearray(size * len(angles))
    for i, a in enumerate(angles):
        struct.pack_into(bo + 'i', ext, i * size, size)
        struct.pack_into(bo + 'd', ext, i * size + 100, a)
    return bytes(ext)


@pytest.mark.parametrize('big_endian', [False, True])
def test_fei_extended_header(tmp_path, big_endian):
    path = write_mrc(tmp_path / 'a.mrc', exttyp=b'FEI2',
                     ext=fei_ext(ANGLES, big_endian=big_endian),
                     big_endian=big_endian)
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_fei_angles_written_in_radians(tmp_path):
    # Some old FEI software wrote radians (IMOD extraheader docs).
    path = write_mrc(tmp_path / 'a.mrc', exttyp=b'FEI1',
                     ext=fei_ext(np.radians(ANGLES), size=768))
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_serialem_extended_header(tmp_path):
    # nreal 61 = tilt (2 bytes) + stage (4) + mag (2) + intensity (2) +
    # dose (4) = 14 bytes, as SerialEM writes it.
    ext = b''.join(struct.pack('<h', int(round(a * 100))) + bytes(12)
                   for a in ANGLES)
    path = write_mrc(tmp_path / 'a.st', exttyp=b'SERI', ext=ext, nint=14,
                     nreal=61)
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_serialem_header_without_tilts_is_skipped(tmp_path):
    ext = bytes(4 * len(ANGLES))
    path = write_mrc(tmp_path / 'a.st', exttyp=b'SERI', ext=ext, nint=4,
                     nreal=4)  # stage position only
    assert tilt_angles(path, len(ANGLES)) is None


def test_legacy_fei_header(tmp_path):
    ext = bytearray(1024 * 128)
    for i, a in enumerate(ANGLES):
        struct.pack_into('<f', ext, i * 128, a)
    path = write_mrc(tmp_path / 'a.mrc', ext=bytes(ext), nreal=32)
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_mdoc(tmp_path):
    path = write_mrc(tmp_path / 'a.mrc')
    lines = ['PixelSpacing = 7.88', 'ImageFile = a.mrc', '',
             '[T = SerialEM: test]']
    for z, a in enumerate(ANGLES):
        lines += ['', f'[ZValue = {z}]', f'TiltAngle = {a}',
                  'StagePosition = 1 2']
    (tmp_path / 'a.mrc.mdoc').write_text('\n'.join(lines) + '\n')
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_rawtlt(tmp_path):
    path = write_mrc(tmp_path / 'a.st')
    (tmp_path / 'a.rawtlt').write_text(''.join(f' {a:.2f}\n'
                                               for a in ANGLES))
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(ANGLES)


def test_aligned_stack_prefers_refined_angles(tmp_path):
    path = write_mrc(tmp_path / 'a.ali', exttyp=b'FEI2', ext=fei_ext(ANGLES))
    refined = [a + 0.3 for a in ANGLES]
    (tmp_path / 'a.tlt').write_text('\n'.join(map(str, refined)))
    assert tilt_angles(path, len(ANGLES)) == pytest.approx(refined)


def test_wrong_count_or_nothing_gives_none(tmp_path):
    path = write_mrc(tmp_path / 'a.st')
    assert tilt_angles(path, len(ANGLES)) is None
    (tmp_path / 'a.rawtlt').write_text('1\n2\n')
    assert tilt_angles(path, len(ANGLES)) is None
