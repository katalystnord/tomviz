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


def small_sirt_case(phantom, voxel):
    """SIRT on the CPU is slow (numpy forward projection): a small case."""
    g = ParallelGeometry.circular(np.arange(0, 180, 4.0), 0.5, 0.5, 48, 24,
                                  offset_u=1.5)
    vg = VolumeGeometry((int(24 * 0.5 / voxel), 24, 24), (voxel, 1.0, 1.0))
    return g, vg, phantom.project(g).astype(np.float32)


def stream_sirt(g, vg, proj, tmp_path, backend, budget):
    from tomviz.conebeam.stream import plan_sirt_slabs, sirt_stream
    assert len(plan_sirt_slabs(g, vg, budget, halo=8)) >= 3
    np.save(tmp_path / 'proj.npy', proj)
    src = RawStack.npy(str(tmp_path / 'proj.npy'))
    out = RawStack(str(tmp_path / 'vol.raw'), vg.shape, mode='w')
    sirt_stream(src, g, vg, out, iterations=10, backend=backend,
                memory_bytes=budget)
    src.close()
    out.close()
    return np.fromfile(tmp_path / 'vol.raw', np.float32).reshape(vg.shape)


@pytest.mark.parametrize('voxel', [0.5, 0.37])
def test_streamed_sirt_matches_in_memory_sirt(phantom, tmp_path, voxel):
    # 'auto': a GPU, else lavapipe, else numpy (much slower, same result).
    from tomviz.projectors.backend import get_backend
    from tomviz.projectors.sirt import sirt
    be = get_backend('auto')
    g, vg, proj = small_sirt_case(phantom, voxel)
    streamed = stream_sirt(g, vg, proj, tmp_path, be, 300_000)
    whole = sirt(proj, g, vg, iterations=10, backend=be)
    err = np.abs(streamed - whole).max() / np.abs(whole).max()
    print(f'voxel {voxel}: {err:.1e} of the peak ({be.description})')
    # On the detector row grid slabs are independent problems: the same
    # to rounding. Off it the 8 halo slices absorb the coupling.
    assert err < (1e-6 if voxel == 0.5 else 2e-4), err


def test_streamed_sirt_refuses_cone_beams(tmp_path):
    from tomviz.conebeam.stream import sirt_stream
    from tomviz.projectors import ConeGeometry
    g = ConeGeometry.circular(np.arange(0, 360, 10.0), 50, 100, 1, 1, 8, 8)
    with pytest.raises(ValueError, match='parallel beam'):
        sirt_stream(None, g, VolumeGeometry((4, 4, 4), 1.0), None)
