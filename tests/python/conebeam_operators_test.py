"""Tests for the cone-beam operators: flat/dark correction, intensity to
line integrals, and cone-beam FDK, driven the way tomviz runs them."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom

SOD, SDD, PITCH = 200.0, 400.0, 0.8
N_U, N_V = 64, 56
ANGLES = np.arange(0, 360, 2.0)


def to_tomviz(stack):
    """(n, v, u) frames -> tomviz tilt series (v, u, n), Fortran order."""
    return np.asfortranarray(np.moveaxis(stack, 0, 2).astype(np.float32))


def make_dataset(array, angles=None):
    ds = Dataset({'scalars': array})
    ds.spacing = [1.0, 1.0, 1.0]
    if angles is not None:
        ds.tilt_angles = np.asarray(angles, dtype=np.float64)
    return ds


@pytest.fixture(scope='module')
def phantom():
    # Shepp-Logan is almost mirror-symmetric, so add a blob off every axis
    # to make a mirrored or transposed reconstruction clearly wrong.
    marker = [0.6, 0.15, 0.15, 0.15, 0.45, 0.35, 0.3, 0.0]
    return EllipsoidPhantom(np.vstack([SHEPP_LOGAN_3D, marker]), radius=9.0)


@pytest.fixture(scope='module')
def geometry():
    return ConeGeometry.circular(ANGLES, SOD, SDD, PITCH, PITCH, N_U, N_V)


@pytest.fixture(scope='module')
def line_integrals(phantom, geometry):
    return phantom.project(geometry)


def run_fdk(dataset, **kw):
    operator = load_operator_class(load_operator_module('Recon_ConeBeam_FDK'))
    args = dict(sod=SOD, sdd=SDD, pixel_size=PITCH, backend='cpu')
    args.update(kw)
    return operator.transform(dataset, **args)


def reference_in_tomviz_order(phantom, shape_zyx, voxel):
    ref = phantom.voxelize(VolumeGeometry(shape_zyx, voxel), supersample=2)
    return ref.transpose(0, 2, 1)


def test_flat_dark_operator(line_integrals):
    rng = np.random.default_rng(3)
    gain = rng.uniform(900, 1100, (N_V, N_U)).astype(np.float32)
    dark = np.full((N_V, N_U), 100.0, dtype=np.float32)
    raw = gain * np.exp(-line_integrals) + dark
    flats = np.stack([gain + dark] * 4)

    ds = make_dataset(to_tomviz(raw), ANGLES)
    flat_ds = make_dataset(to_tomviz(flats))
    dark_ds = make_dataset(dark[:, :, None].copy(order='F'))
    module = load_operator_module('FlatDarkCorrection')
    module.transform(ds, flat=flat_ds, dark=dark_ds)
    assert np.allclose(ds.active_scalars, to_tomviz(line_integrals),
                       atol=1e-3)
    assert np.allclose(ds.tilt_angles, ANGLES)


def test_flat_dark_operator_without_dark(line_integrals):
    # The dark input is optional: unconnected, it counts as 0.
    rng = np.random.default_rng(5)
    gain = rng.uniform(900, 1100, (N_V, N_U)).astype(np.float32)
    ds = make_dataset(to_tomviz(gain * np.exp(-line_integrals)), ANGLES)
    module = load_operator_module('FlatDarkCorrection')
    module.transform(ds, flat=make_dataset(gain[:, :, None].copy(order='F')))
    assert np.allclose(ds.active_scalars, to_tomviz(line_integrals),
                       atol=1e-3)


def test_flat_dark_operator_needs_a_flat(line_integrals):
    ds = make_dataset(to_tomviz(np.exp(-line_integrals)), ANGLES)
    module = load_operator_module('FlatDarkCorrection')
    with pytest.raises(RuntimeError, match='Connect a flat field'):
        module.transform(ds)


@pytest.mark.parametrize('layout', ['single, count in the middle',
                                    'stack, count in the middle',
                                    'single, transposed'])
def test_flat_dark_operator_any_image_axis(line_integrals, layout):
    # A single image loaded from a file can arrive as (x, 1, y): seen in
    # the application with an EMD flat field. Any axis order must work.
    rng = np.random.default_rng(4)
    gain = rng.uniform(900, 1100, (N_V, N_U)).astype(np.float32)
    dark = np.full((N_V, N_U), 100.0, dtype=np.float32)
    raw = gain * np.exp(-line_integrals) + dark
    if layout == 'single, count in the middle':
        flat = (gain + dark)[:, None, :]
    elif layout == 'stack, count in the middle':
        flat = np.stack([gain + dark] * 3, axis=1)
    else:
        flat = (gain + dark).T[:, :, None]
    ds = make_dataset(to_tomviz(raw), ANGLES)
    module = load_operator_module('FlatDarkCorrection')
    module.transform(ds, flat=make_dataset(np.asfortranarray(flat)),
                     dark=make_dataset(np.asfortranarray(dark[:, None, :])))
    assert np.allclose(ds.active_scalars, to_tomviz(line_integrals),
                       atol=1e-3)


def test_flat_dark_operator_rejects_wrong_size(line_integrals):
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    bad = make_dataset(np.ones((N_V + 1, N_U, 1), dtype=np.float32,
                               order='F'))
    module = load_operator_module('FlatDarkCorrection')
    with pytest.raises(RuntimeError):
        module.transform(ds, flat=bad, dark=bad)


@pytest.mark.parametrize('i0', [1000.0, 0.0])
def test_intensity_to_line_integrals(line_integrals, i0):
    intensity = 1000.0 * np.exp(-line_integrals) + 20.0
    # Fixed or estimated I0 (each projection has air at its edges).
    ds = make_dataset(to_tomviz(intensity), ANGLES)
    module = load_operator_module('IntensityToLineIntegrals')
    module.transform(ds, i0=i0 + 20.0 if i0 else 0.0, dark_level=20.0,
                     air_percentile=99.9)
    assert np.allclose(ds.active_scalars, to_tomviz(line_integrals),
                       atol=1e-3)


def test_fdk_operator_circular(phantom, line_integrals):
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    result = run_fdk(ds, volume_size_xy=40, volume_size_axial=36,
                     voxel_size=0.5)
    recon = result['reconstruction']
    vol = recon.active_scalars
    assert vol.shape == (36, 40, 40)
    assert vol.flags.f_contiguous
    assert list(recon.spacing) == pytest.approx([0.5, 0.5, 0.5])
    ref = reference_in_tomviz_order(phantom, (36, 40, 40), 0.5)
    err = np.sqrt(np.mean((vol - ref) ** 2))
    assert err < 0.045
    # Orientation: the marker must be where it belongs, not at any of
    # its mirror images. World (x, y, z) is tomviz (y, z, x).
    mx, my, mz = 0.45 * 9.0, 0.35 * 9.0, 0.3 * 9.0

    def at(wx, wy, wz):
        i = [int(round(c / 0.5 + (n - 1) / 2))
             for c, n in zip((wz, wx, wy), vol.shape)]
        return vol[i[0] - 1:i[0] + 2, i[1] - 1:i[1] + 2,
                   i[2] - 1:i[2] + 2].mean()

    here = at(mx, my, mz)
    assert here > 0.5
    for sx, sy, sz in [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]:
        assert at(sx * mx, sy * my, sz * mz) < here - 0.4
    assert at(my, mx, mz) < here - 0.4


def test_fdk_operator_default_volume(line_integrals):
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    vol = run_fdk(ds)['reconstruction'].active_scalars
    assert vol.shape == (N_V, N_U, N_U)


def test_fdk_operator_geometry_file(tmp_path, phantom, geometry,
                                    line_integrals):
    vecs = np.concatenate([geometry.source, geometry.det_center,
                           geometry.det_u, geometry.det_v], axis=1)
    path = tmp_path / 'scan.geom'
    np.savetxt(path, vecs)
    ds = make_dataset(to_tomviz(line_integrals))  # no angles needed
    vol = run_fdk(ds, geometry_source='file', geometry_file=str(path),
                  volume_size_xy=40, volume_size_axial=36,
                  voxel_size=0.5)['reconstruction'].active_scalars
    ref = reference_in_tomviz_order(phantom, (36, 40, 40), 0.5)
    assert np.sqrt(np.mean((vol - ref) ** 2)) < 0.045

    np.savetxt(path, vecs[:-1])
    with pytest.raises(RuntimeError, match='rows'):
        run_fdk(ds, geometry_source='file', geometry_file=str(path))


def test_fdk_operator_needs_angles(line_integrals):
    ds = make_dataset(to_tomviz(line_integrals))
    with pytest.raises(RuntimeError, match='angle'):
        run_fdk(ds)


def test_fdk_operator_bad_geometry(line_integrals):
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    with pytest.raises(RuntimeError):
        run_fdk(ds, sod=500.0, sdd=400.0)


def test_fdk_operator_leaves_single_row_input_alone(phantom):
    # A single detector row moves into a C-contiguous view without a copy;
    # FDK filters its working copy in place, so it must be a real copy.
    g = ConeGeometry.circular(ANGLES, SOD, SDD, PITCH, PITCH, N_U, 1)
    tilt = to_tomviz(phantom.project(g))
    ds = make_dataset(tilt.copy(order='F'), ANGLES)
    run_fdk(ds, volume_size_xy=40, volume_size_axial=1, voxel_size=0.5)
    np.testing.assert_array_equal(ds.active_scalars, tilt)


GEOMETRY = {'sod': SOD, 'sdd': SDD, 'pixel_size_u': PITCH,
            'pixel_size_v': PITCH, 'offset_u': 0.0, 'offset_v': 0.0,
            'detector_shape': [N_V, N_U], 'source': 'test'}


def test_fdk_operator_geometry_from_the_data(line_integrals):
    typed = run_fdk(make_dataset(to_tomviz(line_integrals), ANGLES),
                    volume_size_xy=32, volume_size_axial=28,
                    voxel_size=0.5)['reconstruction'].active_scalars
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    ds.metadata = {'geometry': dict(GEOMETRY)}
    stored = run_fdk(ds, geometry_source='metadata', sod=1.0, sdd=2.0,
                     pixel_size=9.0, volume_size_xy=32, volume_size_axial=28,
                     voxel_size=0.5)['reconstruction'].active_scalars
    np.testing.assert_allclose(stored, typed, atol=1e-6)


def test_geometry_from_the_data_follows_binning():
    from tomviz.conebeam.operator_support import geometry_from_metadata
    g = dict(GEOMETRY, offset_u=4.0, offset_v=-2.0)
    full = geometry_from_metadata({'geometry': g}, len(ANGLES), N_U, N_V,
                                  ANGLES)
    binned = geometry_from_metadata({'geometry': g}, len(ANGLES), N_U // 2,
                                    N_V // 2, ANGLES)
    np.testing.assert_allclose(binned.det_u, 2 * full.det_u)
    np.testing.assert_allclose(binned.det_v, 2 * full.det_v)
    # The physical detector position does not change with binning.
    np.testing.assert_allclose(binned.det_center, full.det_center)


def test_geometry_from_the_data_refuses_missing_or_cropped(line_integrals):
    ds = make_dataset(to_tomviz(line_integrals), ANGLES)
    with pytest.raises(RuntimeError, match='carries no scanner geometry'):
        run_fdk(ds, geometry_source='metadata')
    ds.metadata = {'geometry': dict(GEOMETRY, detector_shape=[N_V, N_U + 3])}
    with pytest.raises(RuntimeError, match='were cropped'):
        run_fdk(ds, geometry_source='metadata')


def test_intensity_uses_the_stored_open_beam_level(line_integrals):
    # A scanner file reader may store the open-beam level (Nikon
    # WhiteLevel, Waygate FreeRay); with I0 = 0 it beats estimating.
    raw = 60000.0 * np.exp(-line_integrals)
    ds = make_dataset(to_tomviz(raw), ANGLES)
    ds.metadata = {'white_level': 60000.0}
    load_operator_module('IntensityToLineIntegrals').transform(ds)
    np.testing.assert_allclose(ds.active_scalars, to_tomviz(line_integrals),
                               atol=1e-4)


def drifting_scan(line_integrals, mid=False):
    """A beam whose gain drifts linearly over the scan, with flats taken
    before and after it (and, with mid, half way through)."""
    rng = np.random.default_rng(7)
    g0 = rng.uniform(900, 1100, (N_V, N_U)).astype(np.float32)
    g1 = (g0 * rng.uniform(0.7, 0.9, (N_V, N_U))).astype(np.float32)
    # The gain drifts linearly in acquisition position, from the flats
    # taken just before the first projection (position -1) to those just
    # after the last (position n).
    n = len(line_integrals)
    t = (np.arange(n) + 1) / (n + 1)
    gain = g0[None] * (1 - t[:, None, None]) + g1[None] * t[:, None, None]
    raw = gain * np.exp(-line_integrals)
    flats = [g0, g0, g1, g1, g1]
    positions = [-1, -1, n, n, n]
    if mid:
        half = n // 2
        flats.insert(2, gain[half])
        positions.insert(2, half)
    return raw, np.stack(flats), positions


def run_flat_dark(raw, flats, **kwargs):
    ds = make_dataset(to_tomviz(raw), ANGLES)
    flat_ds = make_dataset(to_tomviz(flats))
    meta = kwargs.pop('meta', None)
    if meta:
        ds.metadata = {'frame_index': meta[0]}
        flat_ds.metadata = {'frame_index': meta[1]}
    load_operator_module('FlatDarkCorrection').transform(
        ds, flat=flat_ds, **kwargs)
    return ds.active_scalars


def test_flat_dark_interpolates_a_drifting_beam(line_integrals):
    raw, flats, positions = drifting_scan(line_integrals)
    truth = to_tomviz(line_integrals)
    averaged = run_flat_dark(raw, flats)
    assert np.abs(averaged - truth).max() > 0.05     # drift left behind
    by_count = run_flat_dark(raw, flats, flat_mode='interpolate',
                             flats_before=2)
    np.testing.assert_allclose(by_count, truth, atol=2e-3)
    n = len(line_integrals)
    by_file = run_flat_dark(raw, flats, flat_mode='interpolate',
                            meta=(list(range(n)), positions))
    np.testing.assert_allclose(by_file, truth, atol=2e-3)


def test_flat_dark_uses_flats_taken_during_the_scan(line_integrals):
    # A third flat group half way: interpolation runs through it.
    raw, flats, positions = drifting_scan(line_integrals, mid=True)
    n = len(line_integrals)
    out = run_flat_dark(raw, flats, flat_mode='interpolate',
                        meta=(list(range(n)), positions))
    np.testing.assert_allclose(out, to_tomviz(line_integrals), atol=2e-3)


def test_flat_dark_interpolation_needs_positions(line_integrals):
    raw, flats, _ = drifting_scan(line_integrals)
    with pytest.raises(RuntimeError, match='were taken before it'):
        run_flat_dark(raw, flats, flat_mode='interpolate')
