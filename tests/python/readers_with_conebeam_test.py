"""Where the file readers meet the cone-beam operators (separate changes):
the geometry a reader writes must be what the reconstruction expects."""

import numpy as np
import pytest

from tomviz.projectors import ConeGeometry


@pytest.mark.parametrize('clockwise', [False, True])
def test_zeiss_orbit_matches_cone_geometry(clockwise):
    # zeiss.py writes the circular orbit out so it does not depend on the
    # cone-beam code; it must stay identical to ConeGeometry.circular.
    from tomviz.io.formats.zeiss import circular_orbit
    angles = np.linspace(-180, 180, 13)
    g = ConeGeometry.circular(angles, 95.0, 150.0, 0.066, 0.066, 64, 48,
                              clockwise=clockwise)
    source, centre, du, dv = circular_orbit(angles, 95.0, 150.0, 0.066,
                                            clockwise)
    np.testing.assert_allclose(source, g.source, atol=1e-12)
    np.testing.assert_allclose(centre, g.det_center, atol=1e-12)
    np.testing.assert_allclose(du, g.det_u, atol=1e-12)
    np.testing.assert_allclose(dv, g.det_v, atol=1e-12)


def test_nikon_open_beam_level_reaches_line_integrals(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    from tomviz import _boundary
    from tomviz.io.formats.nikon import NikonReader
    from utils import load_operator_module
    (tmp_path / 'scan.xtekct').write_text(
        '[XTekCT]\nName=scan\nSrcToObject=250\nSrcToDetector=1000\n'
        'DetectorPixelsX=4\nDetectorPixelsY=3\nDetectorPixelSizeX=0.2\n'
        'DetectorPixelSizeY=0.2\nWhiteLevel=60000\nProjections=2\n'
        'InitialAngle=0\nAngularStep=180\n')
    frames = np.array([[[30000] * 4] * 3, [[15000] * 4] * 3], np.uint16)
    for i, f in enumerate(frames, start=1):
        tifffile.imwrite(tmp_path / f'scan_{i:04d}.tif', f)
    ds = _boundary.wrap_vtk_image(
        NikonReader().read(str(tmp_path / 'scan.xtekct')), legacy=True)
    load_operator_module('IntensityToLineIntegrals').transform(ds)
    np.testing.assert_allclose(ds.active_scalars[0, 0], [np.log(2),
                                                         np.log(4)],
                               rtol=1e-5)
