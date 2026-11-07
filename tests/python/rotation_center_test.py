"""Finding the rotation axis from projections 180 degrees apart (parallel
beam), on exact projections of discs with the axis off centre."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

N_DET, N_ROWS = 160, 6
# (x, y, radius, density) in pixels, in the plane across the axis.
DISCS = [(10.0, -5.0, 22.0, 1.0), (-30.0, 18.0, 9.0, 2.5),
         (25.0, 30.0, 6.0, 1.5)]


def tilt_series(angles_deg, axis_offset):
    """Exact parallel projections: x along the axis (rows), y across it
    (detector), z the angle; the axis projects axis_offset pixels from
    the detector centre."""
    # Each detector pixel integrates over its width (16 sub-samples), as
    # a real detector does.
    sub = (np.arange(16) + 0.5) / 16 - 0.5
    s = (np.arange(N_DET)[:, None] + sub[None, :] - (N_DET - 1) / 2 -
         axis_offset)
    out = np.zeros((N_ROWS, N_DET, len(angles_deg)))
    for k, a in enumerate(np.radians(angles_deg)):
        p = np.zeros(N_DET)
        for x0, y0, r, rho in DISCS:
            s0 = x0 * np.cos(a) + y0 * np.sin(a)
            chord = 2 * np.sqrt(np.clip(r ** 2 - (s - s0) ** 2, 0, None))
            p += rho * chord.mean(axis=1)
        out[:, :, k] = p * (1 + 0.05 * np.arange(N_ROWS))[:, None]
    return np.asfortranarray(out)


@pytest.fixture(scope='module')
def module():
    return load_operator_module('AutoTiltAxisShiftAlignment')


@pytest.mark.parametrize('angles', [np.arange(0, 180.5, 1.0),
                                    np.arange(0, 360, 1.5)],
                         ids=['0-180 with both ends', '0-360'])
@pytest.mark.parametrize('offset,tolerance', [
    (7.3, 0.01), (-15.6, 0.01), (30.4, 0.01),
    # The discs reach 45 px from the axis: with the axis 40 px off centre
    # part of the object leaves the 160 px detector in some views, and
    # truncated projections are no longer exact mirror images.
    (40.2, 0.15)])
def test_finds_the_axis(module, angles, offset, tolerance):
    found = module.opposite_projection_offset(tilt_series(angles, offset),
                                              angles)
    assert found == pytest.approx(offset, abs=tolerance)


def test_finds_the_axis_in_transmission_and_with_blur(module):
    from scipy import ndimage
    angles = np.arange(0, 180.5, 1.0)
    ts = tilt_series(angles, 7.3)
    blurred = ndimage.gaussian_filter1d(ts, 1.0, axis=1)  # 1 px PSF
    for data in (np.exp(-ts / 40.0), blurred):
        found = module.opposite_projection_offset(data, angles)
        assert found == pytest.approx(7.3, abs=0.01)


def test_needs_opposite_projections(module):
    angles = np.arange(0, 180, 1.0)  # 180 itself is missing
    with pytest.raises(RuntimeError, match='180 degrees apart'):
        module.opposite_projection_offset(tilt_series(angles, 3.0), angles)


def test_operator_centres_the_axis(module):
    angles = np.arange(0, 180.5, 1.0)
    ds = Dataset({'scalars': tilt_series(angles, 12.4)})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = angles
    load_operator_class(module).transform(ds, method='opposite')
    after = module.opposite_projection_offset(ds.active_scalars, angles)
    # The shift is applied with linear interpolation, which blurs a
    # little; the axis ends up centred well within a pixel.
    assert after == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize('true_offset', [-9.0, 6.0])
def test_trial_reconstructions_with_gradient_energy(module, true_offset):
    # The trial method with gradient energy lands on the nearest whole
    # pixel. With the maximum (the default) it picked +2 instead of +9,
    # and -13 instead of -6, on this phantom.
    angles = np.arange(0, 180.5, 2.0)
    ds = Dataset({'scalars': tilt_series(angles, true_offset)})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = angles
    load_operator_class(module).transform(ds, method='reconstruction',
                                          sharpness_measure='gradient',
                                          num_slices=2)
    after = module.opposite_projection_offset(ds.active_scalars, angles)
    assert after == pytest.approx(0.0, abs=0.6)
