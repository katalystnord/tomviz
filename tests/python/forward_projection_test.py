"""Forward projection (numpy reference and WebGPU) for cone and parallel
beams, checked against exact phantom projections, and the Generate Tilt
Series operator that uses it."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ConeGeometry, ParallelGeometry, VolumeGeometry
from tomviz.projectors.forward_cpu import forward_project
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom


def gpu_backend():
    pytest.importorskip('wgpu')
    from tomviz.projectors.backend import get_backend
    try:
        return get_backend('gpu')
    except RuntimeError as e:
        pytest.skip(str(e))


def tilted_cone():
    g = ConeGeometry.circular(np.arange(0, 360, 30), 40, 80, 1.2, 1.2, 64,
                              56)
    g.det_u = g.det_u + 0.2 * g.det_v
    g.source = g.source + [0, 0, 6]
    return g


GEOMETRIES = {
    'cone': lambda: ConeGeometry.circular(np.arange(0, 360, 30), 60, 120,
                                          0.6, 0.6, 96, 96, offset_u=1.3),
    'parallel': lambda: ParallelGeometry.circular(np.arange(0, 180, 30),
                                                  0.3, 0.3, 96, 96),
}


@pytest.fixture(scope='module')
def smooth():
    """Two off-centre, rotated ellipsoids voxelised on an off-centre box,
    so offset and orientation errors cannot hide."""
    ph = EllipsoidPhantom(np.array([
        [1.0, 0.6, 0.45, 0.5, 0.2, -0.1, 0.15, 30.0],
        [0.5, 0.2, 0.3, 0.25, -0.4, 0.3, -0.2, -20.0]]), radius=9.0)
    vg = VolumeGeometry((44, 44, 44), 0.5, center=(0.3, -0.2, 0.1))
    return ph, vg, ph.voxelize(vg, supersample=4).astype(np.float32)


@pytest.mark.parametrize('name', sorted(GEOMETRIES))
def test_matches_exact_projections(smooth, name):
    ph, vg, vol = smooth
    g = GEOMETRIES[name]()
    fp = forward_project(vol, g, vg, threads=2)
    exact = ph.project(g)
    # Mass is conserved, and long chords are right to discretisation
    # error (which halves with the voxel size). Rays grazing an edge have
    # the largest errors, so look at the 99th percentile, not the maximum.
    assert abs(fp.sum() / exact.sum() - 1) < 0.005
    m = exact > 0.5 * exact.max()
    rel = (fp - exact)[m] / exact[m]
    assert abs(rel.mean()) < 0.01
    assert np.percentile(abs(rel), 99) < 0.05


def test_outside_volume_projects_to_zero():
    vg = VolumeGeometry((4, 6, 6), 1.0)
    g = ParallelGeometry.circular([0, 45, 90], 1.0, 1.0, 20, 4)
    fp = forward_project(np.ones(vg.shape, np.float32), g, vg, threads=1)
    assert np.all(fp[:, :, :3] == 0) and np.all(fp[:, :, -3:] == 0)
    # Straight through the middle at angle 0: six voxels of length 1.
    np.testing.assert_allclose(fp[0, :, 9:11], 6.0, rtol=1e-6)


def test_cancel():
    vg = VolumeGeometry((8, 8, 8), 1.0)
    g = ParallelGeometry.circular(np.arange(0, 180, 1.0), 1.0, 1.0, 12, 8)
    vol = np.ones(vg.shape, np.float32)
    assert forward_project(vol, g, vg, threads=2,
                           should_stop=lambda: True) is None


@pytest.mark.parametrize('name', sorted(GEOMETRIES) + ['tilted cone'])
def test_gpu_matches_cpu(smooth, name):
    be = gpu_backend()
    ph, vg, vol = smooth
    g = tilted_cone() if name == 'tilted cone' else GEOMETRIES[name]()
    ref = forward_project(vol, g, vg, threads=2)
    # Slabs of one slice exercise the overlap between slabs.
    for slab in (None, 7, 1):
        got = be.forward_project(vol, g, vg, slab=slab)
        np.testing.assert_allclose(got, ref, atol=2e-5 * abs(ref).max())


def test_gpu_cancel():
    be = gpu_backend()
    vg = VolumeGeometry((8, 8, 8), 1.0)
    g = ParallelGeometry.circular(np.arange(0, 180, 1.0), 1.0, 1.0, 12, 8)
    assert be.forward_project(np.ones(vg.shape, np.float32), g, vg,
                              should_stop=lambda: True) is None


# -- Generate Tilt Series --------------------------------------------------

def generate(volume, backend, **kw):
    ds = Dataset({'scalars': np.asfortranarray(volume)})
    ds.spacing = [1.0, 1.0, 1.0]
    op = load_operator_class(load_operator_module('GenerateTiltSeries'))
    args = dict(start_angle=-60.0, angle_increment=10.0, num_tilts=13,
                backend=backend)
    args.update(kw)
    op.transform(ds, **args)
    return ds.active_scalars, ds.tilt_angles


@pytest.mark.parametrize('shape', [(12, 32, 31), (12, 30, 36)])
def test_generate_tilt_series_gpu_matches_cpu(shape):
    gpu_backend()
    # Shepp-Logan plus a blob off every axis, so a mirrored or shifted
    # result differs by far more than the interpolation (about 2%).
    marker = [0.6, 0.15, 0.15, 0.15, 0.45, 0.35, 0.3, 0.0]
    ph = EllipsoidPhantom(np.vstack([SHEPP_LOGAN_3D, marker]), radius=14)
    vol = ph.voxelize(VolumeGeometry(shape, 1.0), supersample=2)
    cpu, angles_cpu = generate(vol, 'cpu')
    gpu, angles_gpu = generate(vol, 'gpu')
    np.testing.assert_array_equal(angles_cpu, angles_gpu)
    assert gpu.shape == cpu.shape and gpu.dtype == cpu.dtype
    rel = np.sqrt(np.mean((gpu - cpu)**2) / np.mean(cpu**2))
    assert rel < 0.03


def test_generate_tilt_series_without_gpu_uses_original_code(monkeypatch):
    from tomviz.projectors import backend
    monkeypatch.setattr(backend, 'get_backend',
                        lambda kind: backend.cpu_backend('no GPU'))
    vol = np.random.default_rng(0).random((4, 10, 9))
    auto, _ = generate(vol, 'auto')
    cpu, _ = generate(vol, 'cpu')
    np.testing.assert_array_equal(auto, cpu)


def test_gpu_ray_almost_parallel_to_slabs(smooth):
    # Rays tilted by 1e-8 out of the z planes: the slab clip range is
    # astronomically large and must not overflow when converted to int.
    # (Found on RADV, where fused multiply-adds left such a tiny z
    # component on rays that are exactly level in float32.)
    be = gpu_backend()
    _, vg, vol = smooth
    g = ParallelGeometry.circular(np.arange(0, 180, 15.0), 0.5, 0.5, 48, 40)
    g.ray = g.ray + np.array([0.0, 0.0, 1e-8])
    ref = forward_project(vol, g, vg, threads=2)
    got = be.forward_project(vol, g, vg)
    np.testing.assert_allclose(got, ref, atol=2e-5 * abs(ref).max())


@pytest.fixture(params=['texture', 'buffer'])
def kernel(request):
    """Each GPU forward-projection kernel in turn, whatever the adapter
    (software adapters normally use the buffer one, GPUs the texture)."""
    be = gpu_backend()
    bp = be.forward_project.__self__
    saved = bp.prefer_texture
    bp.prefer_texture = request.param == 'texture'
    yield be
    bp.prefer_texture = saved


@pytest.mark.parametrize(
    'name', sorted(GEOMETRIES) + ['tilted cone', 'almost level'])
def test_each_gpu_kernel_matches_cpu(smooth, kernel, name):
    _, vg, vol = smooth
    if name == 'tilted cone':
        g = tilted_cone()
    elif name == 'almost level':
        g = ParallelGeometry.circular(np.arange(0, 180, 15.0), 0.5, 0.5,
                                      48, 40)
        g.ray = g.ray + np.array([0.0, 0.0, 1e-8])
    else:
        g = GEOMETRIES[name]()
    ref = forward_project(vol, g, vg, threads=2)
    for slab in (None, 7, 1):
        got = kernel.forward_project(vol, g, vg, slab=slab)
        np.testing.assert_allclose(got, ref, atol=2e-5 * abs(ref).max())


def test_rays_at_45_degrees_and_off_the_volume(kernel):
    # Rays at exactly 45 degrees (a tie between the dominant axes) and
    # rays passing beside the volume, where the texture kernel's clipping
    # and zero border do the work.
    vg = VolumeGeometry((10, 12, 14), (0.5, 0.4, 0.3), center=(0.2, 0, 0))
    vol = np.random.default_rng(3).random(vg.shape, dtype=np.float32)
    g = ParallelGeometry.circular(np.arange(0, 360, 45.0), 0.35, 0.35, 40,
                                  24, offset_u=2.5)
    ref = forward_project(vol, g, vg, threads=2)
    got = kernel.forward_project(vol, g, vg)
    np.testing.assert_allclose(got, ref, atol=2e-5 * abs(ref).max())
    assert np.all(got[:, :, :2] == 0)


def test_resident_sirt_same_with_either_kernel(smooth):
    from tomviz.projectors.sirt import sirt
    be = gpu_backend()
    bp = be.forward_project.__self__
    ph, vg, _ = smooth
    g = GEOMETRIES['cone']()
    proj = ph.project(g).astype(np.float32)
    saved = bp.prefer_texture
    try:
        out = {}
        for k in (True, False):
            bp.prefer_texture = k
            out[k] = sirt(proj, g, vg, iterations=5, backend=be)
    finally:
        bp.prefer_texture = saved
    np.testing.assert_allclose(out[True], out[False],
                               atol=1e-5 * abs(out[False]).max())
