"""SIRT on the shared projectors, for cone and parallel beams, and the
Cone-Beam SIRT operator."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ConeGeometry, ParallelGeometry, VolumeGeometry
from tomviz.conebeam import fdk
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom
from tomviz.projectors.sirt import sirt

RADIUS = 9.0
# A blob off every axis, so a mirrored reconstruction is clearly wrong.
MARKER = [0.6, 0.15, 0.15, 0.15, 0.45, 0.35, 0.3, 0.0]


def gpu_backend():
    pytest.importorskip('wgpu')
    from tomviz.projectors.backend import get_backend
    try:
        return get_backend('gpu')
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture(scope='module')
def phantom():
    return EllipsoidPhantom(np.vstack([SHEPP_LOGAN_3D, MARKER]),
                            radius=RADIUS)


@pytest.fixture(scope='module')
def sparse_cone(phantom):
    """24 views over 360 degrees: few enough that SIRT beats FDK."""
    g = ConeGeometry.circular(np.arange(0, 360, 15), 100, 200, 1.2, 1.2,
                              40, 32)
    vg = VolumeGeometry((20, 28, 28), 0.75)
    return g, vg, phantom.project(g), phantom.voxelize(vg, supersample=2)


def rmse(vol, ref):
    mid = slice(5, 15)
    return np.sqrt(np.mean((vol[mid] - ref[mid]) ** 2))


def test_sirt_converges_and_beats_fdk_on_sparse_views(sparse_cone):
    g, vg, proj, ref = sparse_cone
    history = []
    x = sirt(proj, g, vg, iterations=20, backend='cpu',
             callback=lambda it, x: history.append(rmse(x, ref)))
    assert len(history) == 20
    assert all(b <= a + 1e-6 for a, b in zip(history, history[1:]))
    assert rmse(x, ref) < 0.9 * rmse(fdk(proj, g, vg), ref)
    assert x.min() >= 0


def test_sirt_parallel_limited_angle(phantom):
    # Electron tomography: parallel beam, tilts from -60 to +60 degrees.
    g = ParallelGeometry.circular(np.arange(-60, 61, 5.0), 0.75, 0.75,
                                  32, 20)
    vg = VolumeGeometry((20, 28, 28), 0.75)
    ref = phantom.voxelize(vg, supersample=2)
    x = sirt(phantom.project(g), g, vg, iterations=15, backend='cpu')
    assert rmse(x, ref) < 0.5 * rmse(np.zeros_like(ref), ref)


def test_sirt_stops(sparse_cone):
    g, vg, proj, _ = sparse_cone
    calls = []

    def finish_after_two(it, x):
        calls.append(it)
        return it == 1

    x = sirt(proj, g, vg, iterations=10, backend='cpu',
             callback=finish_after_two)
    assert calls == [0, 1] and x is not None
    assert sirt(proj, g, vg, iterations=10, backend='cpu',
                should_stop=lambda: True) is None


def test_sirt_rejects_bad_input(sparse_cone):
    g, vg, proj, _ = sparse_cone
    with pytest.raises(ValueError):
        sirt(proj, g, vg, relaxation=2.0)
    with pytest.raises(ValueError):
        sirt(proj[:, :-1], g, vg)


def test_sirt_gpu_matches_cpu(sparse_cone):
    be = gpu_backend()
    g, vg, proj, _ = sparse_cone
    cpu = sirt(proj, g, vg, iterations=5, backend='cpu')
    gpu = sirt(proj, g, vg, iterations=5, backend=be)
    np.testing.assert_allclose(gpu, cpu, atol=1e-4 * abs(cpu).max())


# -- Operator --------------------------------------------------------------

def test_sirt_operator(phantom, sparse_cone):
    g, vg, proj, _ = sparse_cone
    angles = np.arange(0, 360, 15.0)
    ds = Dataset({'scalars': np.asfortranarray(
        np.moveaxis(proj, 0, 2).astype(np.float32))})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = angles
    op = load_operator_class(load_operator_module('Recon_ConeBeam_SIRT'))
    out = op.transform(ds, sod=100.0, sdd=200.0, pixel_size=1.2,
                       iterations=15, volume_size_xy=28,
                       volume_size_axial=20, voxel_size=0.75,
                       backend='cpu')
    recon = out['reconstruction']
    vol = recon.active_scalars
    assert vol.shape == (20, 28, 28) and vol.flags.f_contiguous
    assert list(recon.spacing) == pytest.approx([0.75] * 3)
    # Tomviz order is (z, x, y); the marker sits at world x, y, z =
    # (0.45, 0.35, 0.3) * RADIUS.
    mx, my, mz = (np.array(MARKER[4:7]) * RADIUS)

    def at(wx, wy, wz):
        i = [int(round(c / 0.75 + (n - 1) / 2))
             for c, n in zip((wz, wx, wy), vol.shape)]
        return vol[i[0], i[1], i[2]]

    here = at(mx, my, mz)
    assert here > 0.45
    for sx, sy, sz in [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]:
        assert at(sx * mx, sy * my, sz * mz) < here - 0.3


def test_sirt_parallel_default_volume(phantom):
    g = ParallelGeometry.circular(np.arange(-60, 61, 10.0), 0.75, 0.75,
                                  32, 20)
    x = sirt(phantom.project(g), g, iterations=2, backend='cpu')
    assert x.shape == (20, 32, 32)
