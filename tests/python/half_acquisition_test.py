"""Stitching a 360 degree half-acquisition scan: a sample wider than the
detector, projected exactly on a wide virtual detector, then cut to what a
detector with the axis near one edge sees. Stitching must give back the
full-width 180 degree projections."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

ROWS = 3
R = 120                    # full sample half-width on the virtual detector
DISCS = [(30.0, -10.0, 70.0, 1.0), (-60.0, 40.0, 25.0, 2.0),
         (85.0, 20.0, 20.0, 1.5)]


def projections(angles_deg, width, axis):
    """Exact, pixel-integrated parallel projections on a detector of
    `width` pixels with the axis at pixel position `axis`."""
    sub = (np.arange(8) + 0.5) / 8 - 0.5
    s = np.arange(width)[:, None] + sub[None, :] - axis
    out = np.zeros((ROWS, width, len(angles_deg)))
    for k, a in enumerate(np.radians(angles_deg)):
        p = np.zeros(width)
        for x0, y0, r, rho in DISCS:
            s0 = x0 * np.cos(a) + y0 * np.sin(a)
            p += rho * (2 * np.sqrt(np.clip(r ** 2 - (s - s0) ** 2, 0,
                                            None))).mean(axis=1)
        out[:, :, k] = p
    return out


@pytest.fixture(scope='module')
def module():
    return load_operator_module('StitchHalfAcquisition')


@pytest.mark.parametrize('side', ['right', 'left'])
def test_stitch_recovers_the_full_projections(module, side):
    angles = np.arange(0, 360, 2.0)
    full = projections(angles, 2 * R + 1, R)       # axis in the middle
    overlap = 30
    n = R + 1 + overlap                             # detector width
    if side == 'right':      # axis 30 px from the right edge
        half = full[:, :n]
        axis_offset = R - (n - 1) / 2
    else:                    # axis 30 px from the left edge
        half = full[:, -n:]
        axis_offset = (n - 1) / 2 - overlap
        axis_offset = -axis_offset
    out, new_angles = module.stitch(half, angles, axis_offset)
    assert out.shape == (ROWS, 2 * R + 1, len(angles) // 2)
    np.testing.assert_allclose(new_angles, angles[angles < 180])
    expected = full[:, :, angles < 180]
    np.testing.assert_allclose(out, expected, atol=1e-4 * full.max())


def test_finds_the_axis(module):
    angles = np.arange(0, 360, 2.0)
    full = projections(angles, 2 * R + 1, R + 0.4)
    n = R + 1 + 40
    half = np.asfortranarray(full[:, :n])
    true = (R + 0.4) - (n - 1) / 2
    found = module.find_axis(half, module.opposite_pairs(angles))
    assert found == pytest.approx(true, abs=0.05)


def test_operator_end_to_end(module):
    angles = np.arange(0, 360, 2.0)
    full = projections(angles, 2 * R + 1, R)
    n = R + 1 + 30
    ds = Dataset({'scalars': np.asfortranarray(full[:, :n])})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = angles
    load_operator_class(module).transform(ds)        # finds the axis
    assert ds.active_scalars.shape[2] == len(angles) // 2
    assert len(ds.tilt_angles) == len(angles) // 2
    width = ds.active_scalars.shape[1]
    centre = full[:, R - width // 2:R + width // 2 + 1, angles < 180]
    np.testing.assert_allclose(ds.active_scalars, centre,
                               atol=2e-3 * full.max())


def test_refuses_a_180_degree_scan(module):
    angles = np.arange(0, 180, 2.0)
    with pytest.raises(RuntimeError, match='360 degree scan'):
        module.stitch(projections(angles, 100, 70), angles, 20.5)
