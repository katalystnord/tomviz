"""Parallel-beam geometry and Weighted Back Projection on the shared
backprojectors (numpy and WebGPU), checked against exact phantom
projections and against the original slice-by-slice WBP."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ParallelGeometry, VolumeGeometry
from tomviz.projectors import backproject_cpu
from tomviz.projectors.phantom import SHEPP_LOGAN_3D, EllipsoidPhantom


@pytest.fixture(scope='module')
def wbp():
    return load_operator_module('Recon_WBP')


def gpu_backend():
    pytest.importorskip('wgpu')
    from tomviz.projectors.backend import get_backend
    try:
        return get_backend('gpu')
    except RuntimeError as e:
        pytest.skip(str(e))


# -- ParallelGeometry ------------------------------------------------------

def test_matrices_map_pixels_back_to_their_indices():
    rng = np.random.default_rng(3)
    n = 5
    ray = rng.normal(size=(n, 3))
    u = rng.normal(size=(n, 3))
    v = rng.normal(size=(n, 3))  # not orthogonal to u or to the ray
    g = ParallelGeometry(ray, rng.normal(size=(n, 3)), u, v, 7, 4)
    P = g.projection_matrices()
    for k in range(n):
        pix = g.pixel_positions(k)
        # Any point along the ray through a pixel projects onto it.
        pts = pix + rng.normal(size=pix.shape[:2] + (1,)) * g.ray[k]
        h = pts @ P[k, :, :3].T + P[k, :, 3]
        iv, iu = np.mgrid[0:4, 0:7]
        np.testing.assert_allclose(h[..., 0], iu, atol=1e-9)
        np.testing.assert_allclose(h[..., 1], iv, atol=1e-9)
        np.testing.assert_allclose(h[..., 2], 1.0)


def test_circular_convention_and_astra_round_trip():
    g = ParallelGeometry.circular([0, 90], 0.5, 2.0, 10, 6, offset_u=1.5)
    P = g.projection_matrices()
    # At angle 0 det_u is +y: a point 1 unit along +y lands 2 pixels to
    # the right of where the axis lands, which is centre + offset.
    axis_u = (10 - 1) / 2 + 1.5
    np.testing.assert_allclose(P[0] @ [0, 1, 0, 1], [axis_u + 2, 2.5, 1])
    # At 90 degrees det_u is -x.
    np.testing.assert_allclose(P[1] @ [-1, 0, 2, 1], [axis_u + 2, 3.5, 1])
    vecs = np.hstack([g.ray, g.det_center, g.det_u, g.det_v])
    g2 = ParallelGeometry.from_astra_vectors(vecs, 10, 6)
    np.testing.assert_allclose(g2.projection_matrices(), P)


def test_rejects_rays_parallel_to_detector():
    with pytest.raises(ValueError):
        ParallelGeometry([[1, 0, 0]], [[0, 0, 0]], [[1, 0, 0]],
                         [[0, 0, 1]], 4, 4)


def test_volume_center_shifts_coordinates():
    vg = VolumeGeometry((2, 3, 4), 0.5, center=(1.0, -2.0, 3.0))
    np.testing.assert_allclose(vg.x_coords(), [2.25, 2.75, 3.25, 3.75])
    np.testing.assert_allclose(vg.y_coords(), [-2.5, -2.0, -1.5])
    corners = vg.slab_corners(0, 2)
    np.testing.assert_allclose(corners.min(0), [2.0, -2.75, 0.5])
    np.testing.assert_allclose(corners.max(0), [4.0, -1.25, 1.5])


# -- Filtered backprojection of an exact phantom ---------------------------

N_DET, N_ROWS = 64, 24


@pytest.fixture(scope='module')
def phantom():
    # Shepp-Logan is almost mirror-symmetric, so add a blob off every axis.
    marker = [0.6, 0.15, 0.15, 0.15, 0.45, 0.35, 0.3, 0.0]
    table = np.vstack([SHEPP_LOGAN_3D, marker])
    return EllipsoidPhantom(table, radius=20.0)


@pytest.fixture(scope='module')
def parallel_case(phantom, wbp):
    angles = np.arange(0, 180, 1.0)
    g = ParallelGeometry.circular(angles, 1.0, 1.0, N_DET, N_ROWS)
    proj = phantom.project(g)
    # WBP's ramp is 2|f|, hence pi / (2 n_angles) for 180 degrees. Ask for
    # a filter twice the detector width so the convolution does not wrap.
    F = wbp.makeFilter(2 * N_DET, 'ramp')[:, 0]
    n2 = F.size
    spec = np.fft.rfft(proj, n=n2, axis=2) * F[:n2 // 2 + 1]
    filtered = np.fft.irfft(spec, n=n2, axis=2)[..., :N_DET]
    filtered = (filtered * np.pi / 2 / len(angles)).astype(np.float32)
    vg = VolumeGeometry((N_ROWS, 48, 48), 1.0)
    ref = phantom.voxelize(vg, supersample=2)
    return g, vg, filtered, ref


def check_against_phantom(vol, ref):
    nz = ref.shape[0]
    mid = slice(nz // 4, 3 * nz // 4)
    err = vol[mid] - ref[mid]
    assert np.sqrt(np.mean(err**2)) < 0.06
    # The marker (density 0.6 at x, y, z = 0.45, 0.35, 0.3 of the radius)
    # must come back where it was, not mirrored.
    z = int(round(0.3 * 20 + (nz - 1) / 2))
    x0 = int(round(0.45 * 20 + 23.5))
    y0 = int(round(0.35 * 20 + 23.5))
    assert vol[z, y0, x0] > 0.5
    assert vol[z, y0, 47 - x0] < 0.4
    assert vol[z, 47 - y0, x0] < 0.4


def test_parallel_fbp_reconstructs_phantom(parallel_case):
    g, vg, filtered, ref = parallel_case
    vol = backproject_cpu.backproject(filtered, g, vg, threads=2)
    check_against_phantom(vol, ref)


def test_parallel_gpu_matches_cpu(parallel_case):
    be = gpu_backend()
    g, vg, filtered, ref = parallel_case
    cpu = backproject_cpu.backproject(filtered, g, vg, threads=2)
    gpu = be.backproject(filtered, g, vg)
    np.testing.assert_allclose(gpu, cpu, atol=1e-5 * abs(cpu).max())
    check_against_phantom(gpu, ref)


# -- WBP -------------------------------------------------------------------

def tilt_series(n_slice=5, n_ray=64, n_proj=41, seed=1):
    rng = np.random.default_rng(seed)
    ts = rng.random((n_slice, n_ray, n_proj)).astype(np.float32)
    angles = np.linspace(-70, 70, n_proj) + rng.normal(0, 0.3, n_proj)
    return ts, angles


def per_slice(wbp, ts, angles, N, filt='ramp', interp='linear'):
    return np.stack([wbp.wbp2(ts[i], angles, N, filt, interp)
                     for i in range(len(ts))])


def interior(n, n_ray):
    """Voxels whose rays stay on the detector for every angle."""
    X, Y = np.mgrid[0:n, 0:n]
    return np.hypot(X - n // 2, Y - n // 2) < n_ray // 2 - 2


@pytest.mark.parametrize('n_ray,N', [(64, None), (65, None), (64, 64),
                                     (63, 50)])
def test_wbp3_matches_slice_by_slice(wbp, n_ray, N):
    from tomviz.projectors.backend import cpu_backend
    ts, angles = tilt_series(n_ray=n_ray)
    ref = per_slice(wbp, ts, angles, N, 'shepp-logan')
    got = wbp.wbp3(ts, angles, N, 'shepp-logan', cpu_backend())
    m = interior(ref.shape[1], n_ray)
    np.testing.assert_allclose(got[:, m], ref[:, m],
                               atol=1e-5 * abs(ref).max())


def test_wbp3_gpu_matches_slice_by_slice(wbp):
    be = gpu_backend()
    ts, angles = tilt_series()
    ref = per_slice(wbp, ts, angles, None)
    got = wbp.wbp3(ts, angles, None, 'ramp', be)
    m = interior(ref.shape[1], 64)
    np.testing.assert_allclose(got[:, m], ref[:, m],
                               atol=1e-5 * abs(ref).max())


def make_dataset(ts, angles):
    ds = Dataset({'scalars': np.asfortranarray(ts)})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = np.asarray(angles, dtype=np.float64)
    return ds


def run_operator(wbp, ts, angles, **kw):
    op = load_operator_class(wbp)
    args = dict(Nrecon=48, filter=1, interp=0, Nupdates=0)
    args.update(kw)
    out = op.transform(make_dataset(ts, angles), **args)
    return out['reconstruction'].active_scalars


def assert_original(got, ref):
    """Equal to the original code up to FFT rounding (the dataset is in
    Fortran order). The shared backprojectors differ by far more than this
    near the detector edges, so this tells the two paths apart."""
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * abs(ref).max())


def test_operator_cpu_is_the_original_code(wbp):
    ts, angles = tilt_series()
    got = run_operator(wbp, ts, angles, backend='cpu')
    assert_original(got, per_slice(wbp, ts, angles, 48))


def test_operator_without_gpu_uses_original_code(wbp, monkeypatch):
    from tomviz.projectors import backend
    monkeypatch.setattr(backend, 'get_backend',
                        lambda kind: backend.cpu_backend('no GPU'))
    ts, angles = tilt_series()
    got = run_operator(wbp, ts, angles, backend='auto')
    assert_original(got, per_slice(wbp, ts, angles, 48))


def test_operator_nearest_and_cubic_unchanged(wbp):
    ts, angles = tilt_series(n_slice=2)
    for interp, name in ((1, 'nearest'), (3, 'cubic')):
        got = run_operator(wbp, ts, angles, interp=interp, backend='auto')
        assert_original(got, per_slice(wbp, ts, angles, 48, interp=name))


def test_operator_gpu_with_live_updates(wbp):
    gpu_backend()
    ts, angles = tilt_series(n_slice=7)
    ref = per_slice(wbp, ts, angles, 48)
    m = interior(48, 64)
    for updates in (0, 3):
        got = run_operator(wbp, ts, angles, backend='gpu', Nupdates=updates)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got[:, m], ref[:, m],
                                   atol=1e-5 * abs(ref).max())
