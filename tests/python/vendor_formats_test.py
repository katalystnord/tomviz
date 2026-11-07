"""Readers for lab CT scanner formats (Nikon, Waygate, Bruker SkyScan),
on small scans laid out like real ones. The geometry conventions were set
on real public scans, compared with the vendors' own reconstructions."""

import numpy as np
import pytest

tifffile = pytest.importorskip('tifffile')

from tomviz import _boundary  # noqa: E402
from tomviz.io.formats.nikon import NikonReader  # noqa: E402
from tomviz.io.formats.skyscan import SkyScanReader  # noqa: E402
from tomviz.io.formats.waygate import WaygateReader  # noqa: E402

ROWS, COLS, N = 6, 8, 4


def frames(n=N + 1):
    """Distinct uint16 frames: 100 + 1000 * frame + 10 * row + column."""
    f = 100 + np.arange(n)[:, None, None] * 1000 + \
        np.arange(ROWS)[None, :, None] * 10 + np.arange(COLS)[None, None, :]
    return f.astype(np.uint16)


def wrap(result):
    image = result['data'] if isinstance(result, dict) else result
    return _boundary.wrap_vtk_image(image, legacy=True)


def check_frames(ds, n=N):
    # Tomviz tilt series: x = detector rows, y = columns, z = projection.
    assert ds.active_scalars.shape == (ROWS, COLS, n)
    np.testing.assert_array_equal(ds.active_scalars[:, :, 2], frames()[2])


# -- Nikon --------------------------------------------------------------------

XTEKCT = """[XTekCT]
Name=scan
InputFolderName=
SrcToObject=250.0
SrcToDetector=1000.0
DetectorPixelsX={cols}
DetectorPixelsY={rows}
DetectorPixelSizeX=0.2
DetectorPixelSizeY=0.25
DetectorOffsetX=0.1
DetectorOffsetY=0.0
CentreOfRotationTop={top}
CentreOfRotationBottom={bottom}
WhiteLevel=60000.0
Projections={n}
InitialAngle=10
AngularStep=90
VoxelSizeX=0.05
"""


def nikon_scan(folder, top=0.0, bottom=0.0, extra=''):
    (folder / 'scan.xtekct').write_text(
        XTEKCT.format(cols=COLS, rows=ROWS, n=N, top=top, bottom=bottom) +
        extra, encoding='latin-1')
    for i, f in enumerate(frames(), start=1):  # one extra, like the scanner
        tifffile.imwrite(folder / f'scan_{i:04d}.tif', f)
    return str(folder / 'scan.xtekct')


def test_nikon(tmp_path):
    ds = wrap(NikonReader().read(nikon_scan(tmp_path, top=-0.5, bottom=-0.3)))
    check_frames(ds)
    np.testing.assert_allclose(ds.tilt_angles, [10, 100, 190, 280])
    g = ds.metadata['geometry']
    assert (g['sod'], g['sdd']) == (250.0, 1000.0)
    assert (g['pixel_size_u'], g['pixel_size_v']) == (0.2, 0.25)
    # Centre (-0.4 mm) plus detector offset (0.1 mm), in 0.2 mm pixels.
    assert g['offset_u'] == pytest.approx(-1.5)
    assert g['detector_shape'] == [ROWS, COLS]
    assert ds.metadata['white_level'] == 60000.0


def test_nikon_centre_from_the_reconstruction_file(tmp_path):
    path = nikon_scan(tmp_path)
    (tmp_path / 'scan_recon.xtekct').write_text(XTEKCT.format(
        cols=COLS, rows=ROWS, n=N, top=-2.0, bottom=-1.0))
    g = wrap(NikonReader().read(path)).metadata['geometry']
    assert g['offset_u'] == pytest.approx((-1.5 + 0.1) / 0.2)
    assert 'reconstruction file' in g['source']


def test_nikon_object_offset_is_magnified(tmp_path):
    path = nikon_scan(tmp_path, extra='ObjectOffsetX=0.25\n')
    g = wrap(NikonReader().read(path)).metadata['geometry']
    assert g['offset_u'] == pytest.approx((0.25 * 4 + 0.1) / 0.2)


def test_nikon_missing_projections(tmp_path):
    path = nikon_scan(tmp_path)
    (tmp_path / 'scan_0003.tif').unlink()
    (tmp_path / 'scan_0005.tif').unlink()
    with pytest.raises(ValueError, match='4 projections expected'):
        NikonReader().read(path)


# -- Waygate ------------------------------------------------------------------

PCA = """[Geometry]
FDD=800.0
FOD=200.0
VoxelSizeX=0.05
VoxelSizeY=0.05
CalibValue=-0.4
cx={cx}
cy={cy}

[CT]
NumberImages={n}
RotationSector=360.0

[Image]
DimX={cols}
DimY={rows}
FreeRay=8877
"""


def waygate_scan(folder, table=True):
    (folder / 'Plug.pca').write_text(PCA.format(
        cx=(COLS - 1) / 2 + 1.0, cy=(ROWS - 1) / 2, n=N, cols=COLS,
        rows=ROWS))
    if table:
        rows = ['Datos|x Acquisition', 'ImgNr\tRotPos\tU\tI']
        rows += [f'{i}\t{(i - 1) * 91.0:.3f}\t250\t290' for i in
                 range(1, N + 2)]
        (folder / 'Plug.pcp').write_text('\r\n'.join(rows) + '\r\n')
    for i, f in enumerate(frames(), start=1):
        tifffile.imwrite(folder / f'Plug{i:05d}.tif', f)
    return str(folder / 'Plug.pca')


def test_waygate(tmp_path):
    ds = wrap(WaygateReader().read(waygate_scan(tmp_path)))
    check_frames(ds)
    np.testing.assert_allclose(ds.tilt_angles, [0, 91, 182, 273])
    g = ds.metadata['geometry']
    # Pixel pitch = reconstruction voxel x magnification FDD / FOD.
    assert g['pixel_size_u'] == pytest.approx(0.2)
    assert g['offset_u'] == pytest.approx(1.0 - 0.4)
    assert ds.metadata['white_level'] == 8877.0


def test_waygate_without_table_spreads_the_sector(tmp_path):
    ds = wrap(WaygateReader().read(waygate_scan(tmp_path, table=False)))
    np.testing.assert_allclose(ds.tilt_angles, [0, 90, 180, 270])


# -- Bruker SkyScan -----------------------------------------------------------

LOG = """[System]
Camera Pixel Size (um)=   10.0
CameraXYRatio=1.0
[Acquisition]
Filename Prefix=grain
Number Of Files=  {files}
Number of Rows= {rows}
Number of Columns= {cols}
Camera binning=2x2
Object to Source (mm)=100.0
Camera to Source (mm)=250.0
Optical Axis (line)= {axis}
Rotation Step (deg)=0.500
Rotation Direction=CC
Flat Field Correction=ON
"""


def skyscan_scan(folder):
    (folder / 'grain.log').write_text(LOG.format(
        files=N + 1, rows=ROWS, cols=COLS, axis=(ROWS - 1) / 2 - 1))
    for i, f in enumerate(frames(N)):
        tifffile.imwrite(folder / f'grain{i:04d}.tif', f)
    tifffile.imwrite(folder / 'grain_arc.tif', frames(1)[0])
    return str(folder / 'grain.log')


def test_skyscan(tmp_path):
    ds = wrap(SkyScanReader().read(skyscan_scan(tmp_path)))
    check_frames(ds)
    np.testing.assert_allclose(ds.tilt_angles, [0, 0.5, 1.0, 1.5])
    g = ds.metadata['geometry']
    assert g['pixel_size_u'] == pytest.approx(0.02)  # 10 um x binning 2
    assert g['offset_v'] == pytest.approx(-1.0)
    assert g['clockwise'] is False
    assert ds.metadata['flat_field_corrected'] is True


def test_skyscan_refuses_other_log_files(tmp_path):
    (tmp_path / 'build.log').write_text('compiling...\n')
    with pytest.raises(ValueError, match='not a Bruker SkyScan scan log'):
        SkyScanReader().read(str(tmp_path / 'build.log'))


# -- ZEISS --------------------------------------------------------------------

def zeiss_meta(n=4):
    return {'n': n, 'width': COLS, 'height': ROWS, 'voxel_um': 40.0,
            'angles': np.arange(n) * 90.0, 'sod': np.full(n, -100.0),
            'odd': np.full(n, 50.0),
            'x_shifts': np.array([0.0, 1.5, -2.0, 0.25]),
            'y_shifts': np.array([0.0, 0.0, 1.0, -0.5])}


def test_zeiss_geometry_keeps_the_sub_pixel_shifts():
    from tomviz.io.formats import zeiss
    meta = zeiss_meta()
    g = zeiss.scan_geometry(meta)
    assert (g['sod'], g['sdd']) == (100.0, 150.0)
    assert g['pixel_size_u'] == pytest.approx(0.04 * 150 / 100)
    vecs = np.array(g['vectors'])
    source, centre, det_u, det_v = zeiss.circular_orbit(
        meta['angles'], 100.0, 150.0, 0.06, zeiss.CLOCKWISE)
    np.testing.assert_allclose(vecs[:, 0:3], source)
    expected = zeiss.SHIFT_SIGN * (meta['x_shifts'][:, None] * det_u +
                                   meta['y_shifts'][:, None] * det_v)
    np.testing.assert_allclose(vecs[:, 3:6] - centre, expected, atol=1e-9)
    # The circular orbit: source and detector on opposite sides at 0 deg.
    np.testing.assert_allclose(source[0], [100, 0, 0])
    np.testing.assert_allclose(centre[0], [-50, 0, 0])


def test_zeiss_needs_olefile(monkeypatch):
    import builtins
    from tomviz.io.formats import zeiss
    real_import = builtins.__import__

    def no_olefile(name, *args, **kwargs):
        if name == 'olefile':
            raise ImportError(name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_olefile)
    with pytest.raises(RuntimeError, match='needs the olefile package'):
        zeiss._open('scan.txrm')


@pytest.mark.skipif(not __import__('os').environ.get('TOMVIZ_ZEISS_TXRM'),
                    reason='set TOMVIZ_ZEISS_TXRM to a real .txrm to run')
def test_zeiss_real_scan():
    import os
    from tomviz.io.formats.zeiss import ZeissReader
    result = ZeissReader().read(os.environ['TOMVIZ_ZEISS_TXRM'])
    ds = wrap(result)
    n = ds.active_scalars.shape[2]
    assert len(ds.tilt_angles) == n
    assert len(ds.metadata['geometry']['vectors']) == n
    flat = _boundary.wrap_vtk_image(result['flat'], legacy=True)
    assert flat.active_scalars.shape[:2] == ds.active_scalars.shape[:2]
