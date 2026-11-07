"""Parallel-beam filtered backprojection (the FDK code with no fan), in
memory and streamed a slab at a time."""

import numpy as np
import pytest

from tomviz.conebeam import fdk
from tomviz.conebeam.filtering import redundancy_weights
from tomviz.conebeam.stream import RawStack, fdk_stream, plan_slabs
from tomviz.projectors import ParallelGeometry, VolumeGeometry
from tomviz.projectors.phantom import EllipsoidPhantom

N = 48


@pytest.fixture(scope='module')
def phantom():
    return EllipsoidPhantom.shepp_logan(radius=10.0)


@pytest.fixture(scope='module')
def vol_geom():
    return VolumeGeometry((N, N, N), 0.5)


@pytest.fixture(scope='module')
def reference(phantom, vol_geom):
    return phantom.voxelize(vol_geom, supersample=2)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


PITCH = 0.25   # half the voxel size: no magnification as in cone beams


def parallel(angles, **kw):
    return ParallelGeometry.circular(angles, PITCH, PITCH, 128, 2 * N, **kw)


def test_angles_follow_the_rotation():
    g = parallel(np.arange(0, 360, 10.0))
    np.testing.assert_allclose(np.rad2deg(g.angles()), np.arange(0, 360, 10))
    g = parallel(np.arange(0, 360, 10.0), clockwise=True)
    np.testing.assert_allclose(np.rad2deg(g.angles()), -np.arange(0, 360,
                                                                  10))


@pytest.mark.parametrize('arc', [180, 270, 360])
def test_every_ray_counts_once(arc):
    # A ray at angle b and its opposite at b + 180 degrees are the same
    # line: their weights must add up to 1.
    step = 1.0
    angles = np.arange(0, arc, step)
    w, _ = redundancy_weights(parallel(angles))
    w = w[:, 0]
    total = np.zeros(int(180 / step))
    for a, wk in zip(angles, w):
        total[int(round(a / step)) % len(total)] += wk
    np.testing.assert_allclose(total, 1.0, atol=0.02)


@pytest.mark.parametrize('angles,kw', [
    (np.arange(0, 360, 1.5), {}),
    (np.arange(0, 180, 1.0), {}),
    (np.arange(0, 180, 1.0), dict(clockwise=True)),
    (np.arange(0, 250, 1.0), {}),
    (np.arange(0, 360, 1.5), dict(offset_u=4.3)),
], ids=['full', 'half', 'half-clockwise', 'partial', 'off-centre'])
def test_fbp_matches_the_phantom(phantom, vol_geom, reference, angles, kw):
    g = parallel(angles, **kw)
    rec = fdk(phantom.project(g), g, vol_geom)
    assert rmse(rec, reference) < 0.03
    mid = N // 2
    body = reference[mid] > 0.1
    assert abs(rec[mid][body].mean() - reference[mid][body].mean()) < 0.003


@pytest.mark.parametrize('voxel', [0.5, 0.37])
def test_streamed_equals_in_memory(phantom, tmp_path, voxel):
    g = parallel(np.arange(0, 180, 1.0), offset_u=2.2, offset_v=0.6)
    vg = VolumeGeometry((int(N * 0.5 / voxel), N, N), voxel)
    proj = phantom.project(g).astype(np.float32)
    np.save(tmp_path / 'proj.npy', proj)
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = RawStack(str(tmp_path / 'vol.raw'), vg.shape, mode='w')
    budget = 2_000_000
    slabs = plan_slabs(g, vg, memory_bytes=budget)
    assert len(slabs) >= 4
    # Parallel rays: a slab needs only its own rows, and a margin.
    for z0, z1, r0, r1 in slabs:
        assert r1 - r0 <= (z1 - z0) * voxel / PITCH + 8
    fdk_stream(src, g, vg, out, backend='cpu', memory_bytes=budget)
    src.close()
    out.close()
    streamed = np.fromfile(tmp_path / 'vol.raw', np.float32).reshape(
        vg.shape)
    np.testing.assert_allclose(streamed, fdk(proj, g, vg),
                               atol=1e-5 * np.abs(streamed).max())
