"""Tests for the headless cone-beam package, tomviz.conebeam.

Everything is checked against an ellipsoid phantom whose projections are
exact analytic line integrals, so the tests do not depend on any projector.
"""

import numpy as np
import pytest

from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.conebeam import fdk, flat_dark_correct
from tomviz.conebeam import filtering
from tomviz.projectors.phantom import EllipsoidPhantom

N = 48
SOD, SDD = 200.0, 400.0
PITCH = 0.8
N_U, N_V = 72, 60
THREADS = 4


@pytest.fixture(scope='module')
def phantom():
    return EllipsoidPhantom.shepp_logan(radius=10.0)


@pytest.fixture(scope='module')
def vol_geom():
    return VolumeGeometry((N, N, N), 0.5)


@pytest.fixture(scope='module')
def reference(phantom, vol_geom):
    return phantom.voxelize(vol_geom, supersample=2)


def circular(angles, **kw):
    args = dict(sod=SOD, sdd=SDD, pixel_u=PITCH, pixel_v=PITCH,
                n_u=N_U, n_v=N_V)
    args.update(kw)
    return ConeGeometry.circular(angles, **args)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def body_mean_error(rec, ref):
    mid = N // 2
    body = ref[mid] > 0.1
    return abs(rec[mid][body].mean() - ref[mid][body].mean())


# -- geometry -------------------------------------------------------------

def test_projection_matrix_maps_points_onto_pixels():
    g = circular(np.arange(0, 360, 30), offset_u=3.5, offset_v=-2.0)
    P = g.projection_matrices()
    for k in range(g.n_angles):
        pix = g.pixel_positions(k)
        # A point between the source and a known pixel must project onto
        # that pixel, with depth weight 1 on the rotation axis.
        for (iv, iu) in [(0, 0), (10, 40), (N_V - 1, N_U - 1)]:
            pt = g.source[k] + 0.4 * (pix[iv, iu] - g.source[k])
            h = P[k] @ np.append(pt, 1.0)
            assert h[0] / h[2] == pytest.approx(iu, abs=1e-9)
            assert h[1] / h[2] == pytest.approx(iv, abs=1e-9)
        assert (P[k] @ [0, 0, 0, 1])[2] == pytest.approx(1.0)


def test_circular_geometry_distances_and_offsets():
    g = circular([0, 90, 200], offset_u=4.0, offset_v=-3.0)
    assert np.allclose(g.sod(), SOD)
    assert np.allclose(g.sdd(), SDD)
    pu, pv = g.principal_point()
    assert np.allclose(pu, (N_U - 1) / 2 + 4.0)
    assert np.allclose(pv, (N_V - 1) / 2 - 3.0)


def test_astra_vectors_round_trip():
    g = circular(np.arange(0, 360, 45), offset_u=2.0)
    vecs = np.concatenate([g.source, g.det_center, g.det_u, g.det_v], 1)
    g2 = ConeGeometry.from_astra_vectors(vecs, N_U, N_V)
    assert np.allclose(g2.projection_matrices(), g.projection_matrices())
    assert np.allclose(g2.angles(), np.deg2rad(np.arange(0, 360, 45)))


@pytest.mark.parametrize('kw', [
    dict(sod=0), dict(sod=500), dict(pixel_u=0), dict(n_u=0)])
def test_invalid_geometry_rejected(kw):
    with pytest.raises(ValueError):
        circular([0, 1], **kw)


def test_default_volume_uses_magnified_pitch():
    g = circular([0, 1])
    vg = g.default_volume()
    assert vg.voxel_size == pytest.approx((PITCH * SOD / SDD,) * 3)
    assert vg.shape == (N_V, N_U, N_U)


# -- filtering ------------------------------------------------------------

def test_ramp_kernel_has_no_dc_and_rises():
    for name in filtering.FILTERS:
        H, n_fft = filtering.ramp_kernel(N_U, name)
        assert n_fft >= 2 * N_U
        # Built from the exact spatial ramp, so DC is small but not zero.
        assert 0 <= H[0] < 1.0 / n_fft
        assert H[len(H) // 4] > H[1] > 0


def test_angular_steps():
    full = filtering.angular_steps(np.deg2rad(np.arange(0, 360, 2)))
    assert np.allclose(full, np.deg2rad(2))
    assert full.sum() == pytest.approx(2 * np.pi)
    part = filtering.angular_steps(np.deg2rad(np.arange(0, 200, 1.0)))
    assert np.allclose(part, np.deg2rad(1))


def test_parker_weights_make_rays_count_once():
    g = circular(np.arange(0, 230, 0.25), sod=120, pixel_u=1.6)
    w = filtering.parker_weights(g)
    assert w.min() >= 0 and w.max() <= 1
    # Every ray (beta, gamma) has a conjugate (beta + pi + 2 gamma, -gamma).
    # Parker weights of a ray and its conjugate sum to one. Rays and
    # conjugates are matched on the sampled grid by nearest angle.
    beta = g.angles()
    pu, _ = g.principal_point()
    gam = np.arctan((np.arange(N_U) - pu[0]) * 1.6 / SDD)
    checked = 0
    for k in range(0, len(beta), 37):
        for iu in range(0, N_U, 7):
            # Detector side for a counter-clockwise orbit: gamma at column
            # iu corresponds to -gamma at the mirrored column.
            bc = beta[k] + np.pi - 2 * gam[iu]
            kc = int(np.argmin(np.abs(beta - bc)))
            if abs(beta[kc] - bc) > np.deg2rad(0.13):
                continue
            iu_c = int(np.argmin(np.abs(gam + gam[iu])))
            assert w[k, iu] + w[kc, iu_c] == pytest.approx(1.0, abs=0.03)
            checked += 1
    assert checked > 20


# -- FDK on the phantom ---------------------------------------------------

def test_fdk_full_scan(phantom, vol_geom, reference):
    g = circular(np.arange(0, 360, 1.5))
    rec = fdk(phantom.project(g), g, vol_geom, threads=THREADS)
    assert rec.shape == vol_geom.shape and rec.dtype == np.float32
    assert rmse(rec, reference) < 0.035
    assert body_mean_error(rec, reference) < 0.01


@pytest.mark.parametrize('angles,kw', [
    (np.arange(0, 360, 1.5), dict(clockwise=True)),
    (np.arange(0, 360, 1.5), dict(offset_u=5.3, offset_v=-3.0)),
    (np.arange(0, 196, 1.0), {}),
    (-np.arange(0, 196, 1.0) + 40, {}),
    (np.arange(0, 196, 1.0), dict(offset_u=4.0)),
], ids=['clockwise', 'offsets', 'short', 'short-reversed',
        'short-offset'])
def test_fdk_variants(phantom, vol_geom, reference, angles, kw):
    g = circular(angles, **kw)
    rec = fdk(phantom.project(g), g, vol_geom, threads=THREADS)
    assert rmse(rec, reference) < 0.045
    assert body_mean_error(rec, reference) < 0.01


def test_parker_sign_matters(phantom, vol_geom):
    """A short scan with a wide fan: correct Parker weights must be much
    closer to the full-scan result than the same weights mirrored across
    the detector. Comparing with the full scan cancels the errors both
    share (voxelisation, cone-beam artefacts)."""
    kw = dict(sod=120, pixel_u=1.6, pixel_v=1.6, n_u=48, n_v=40)
    g_full = circular(np.arange(0, 360, 1.0), **kw)
    full = fdk(phantom.project(g_full), g_full, vol_geom, threads=THREADS)
    g = circular(np.arange(0, 205, 1.0), **kw)
    proj = phantom.project(g)
    good = rmse(fdk(proj, g, vol_geom, threads=THREADS), full)
    orig = filtering.parker_weights
    try:
        filtering.parker_weights = lambda geom: orig(geom)[:, ::-1]
        bad = rmse(fdk(proj, g, vol_geom, threads=THREADS), full)
    finally:
        filtering.parker_weights = orig
    assert good < 0.6 * bad


def test_fdk_measured_vector_geometry(phantom, vol_geom, reference):
    """A perturbed per-projection geometry (wobbling source and tilted
    detector) reconstructs as well as an ideal one when FDK is given the
    same vectors the data were simulated with."""
    rng = np.random.default_rng(1)
    g = circular(np.arange(0, 360, 1.5))
    n = g.n_angles
    vecs = np.concatenate([g.source, g.det_center, g.det_u, g.det_v], 1)
    vecs[:, 0:3] += rng.normal(0, 0.5, (n, 3))
    vecs[:, 3:6] += rng.normal(0, 0.5, (n, 3))
    tilt = np.deg2rad(1.0)
    vecs[:, 11] *= np.cos(tilt)
    vecs[:, 9:11] = np.sin(tilt) * vecs[:, 6:8]  # lean det_v along u
    gv = ConeGeometry.from_astra_vectors(vecs, N_U, N_V)
    rec = fdk(phantom.project(gv), gv, vol_geom, threads=THREADS)
    assert rmse(rec, reference) < 0.035
    assert body_mean_error(rec, reference) < 0.01


def test_fdk_progress_and_cancel(phantom, vol_geom):
    g = circular(np.arange(0, 360, 6.0))
    proj = phantom.project(g)
    seen = []
    fdk(proj, g, vol_geom, threads=THREADS, progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert fdk(proj, g, vol_geom, threads=THREADS,
               should_stop=lambda: True) is None


def test_fdk_overwrite_gives_same_result(phantom, vol_geom):
    g = circular(np.arange(0, 360, 6.0))
    proj = phantom.project(g)
    normal = fdk(proj.copy(), g, vol_geom, threads=THREADS)
    inplace = fdk(proj, g, vol_geom, threads=THREADS, overwrite=True)
    assert np.array_equal(normal, inplace)


def test_fdk_rejects_mismatched_projections(vol_geom):
    g = circular(np.arange(0, 360, 6.0))
    with pytest.raises(ValueError):
        fdk(np.zeros((g.n_angles, N_V, N_U + 1)), g, vol_geom)


# -- preprocessing --------------------------------------------------------

def test_flat_dark_correction_recovers_line_integrals():
    rng = np.random.default_rng(0)
    line = rng.uniform(0, 3, (5, 8, 9)).astype(np.float32)
    gain = rng.uniform(800, 1200, (8, 9)).astype(np.float32)
    dark = rng.uniform(90, 110, (8, 9)).astype(np.float32)
    raw = gain * np.exp(-line) + dark
    flats = np.stack([gain + dark] * 3)
    out = flat_dark_correct(raw, flat=flats, dark=dark)
    assert np.allclose(out, line, atol=1e-4)
    t = flat_dark_correct(raw, flat=flats, dark=dark, log=False)
    assert np.allclose(t, np.exp(-line), atol=1e-5)


def test_flat_dark_correction_is_finite_on_bad_pixels():
    raw = np.array([[[0.0, 50.0, 200.0]]], dtype=np.float32)
    flat = np.array([[100.0, 100.0, 5.0]], dtype=np.float32)
    dark = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    out = flat_dark_correct(raw, flat=flat, dark=dark)
    assert np.all(np.isfinite(out))
    assert out[0, 0, 2] == 0.0  # dead flat pixel: no attenuation
    assert out[0, 0, 0] > 5  # starved pixel: large but finite


def test_flat_dark_correction_shape_checks():
    with pytest.raises(ValueError):
        flat_dark_correct(np.ones((2, 3, 4)), flat=np.ones((3, 5)))
    with pytest.raises(ValueError):
        flat_dark_correct(np.ones((3, 4)))


# -- WebGPU backend -------------------------------------------------------

def _gpu_backend():
    """The WebGPU backend, or skip the test if this machine has none."""
    pytest.importorskip('wgpu')
    from tomviz.projectors.backend import get_backend
    try:
        return get_backend('gpu')
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.fixture
def no_cached_backend():
    from tomviz.projectors import webgpu
    saved = webgpu._cached, webgpu._failure
    webgpu._cached = webgpu._failure = None
    yield
    webgpu._cached, webgpu._failure = saved


def test_gpu_matches_cpu(phantom):
    from tomviz.projectors import backproject_cpu, webgpu
    be = _gpu_backend()
    g = circular(np.arange(0, 360, 3.0), offset_u=2.5, offset_v=-1.5)
    vg = VolumeGeometry((20, 26, 30), 0.6)  # deliberately not cubic
    filt, _ = filtering.preweight_and_filter(phantom.project(g), g)
    ref = backproject_cpu.backproject(filt, g, vg, threads=THREADS)
    got = be.backproject(filt, g, vg)
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() < 1e-4 * scale
    # Many slabs and tiny batches exercise the streaming and row windows.
    orig = webgpu.MAX_UPDATES_PER_DISPATCH
    try:
        webgpu.MAX_UPDATES_PER_DISPATCH = 1 << 12
        got = be.backproject(filt, g, vg, slab=3)
    finally:
        webgpu.MAX_UPDATES_PER_DISPATCH = orig
    assert np.abs(got - ref).max() < 1e-4 * scale


def test_gpu_fdk_and_cancel(phantom, vol_geom, reference):
    _gpu_backend()
    g = circular(np.arange(0, 360, 1.5))
    proj = phantom.project(g)
    rec = fdk(proj, g, vol_geom, backend='gpu')
    assert rmse(rec, reference) < 0.035
    assert fdk(proj, g, vol_geom, backend='gpu',
               should_stop=lambda: True) is None


def test_auto_falls_back_to_cpu(monkeypatch, no_cached_backend):
    from tomviz.projectors.backend import get_backend
    monkeypatch.setenv('TOMVIZ_WGPU_ADAPTER', 'no-such-adapter')
    be = get_backend('auto')
    assert be.name == 'cpu'
    assert 'GPU unavailable' in be.description
    with pytest.raises(RuntimeError, match='GPU backend unavailable'):
        get_backend('gpu')


def test_auto_remembers_why_the_gpu_failed(monkeypatch, no_cached_backend):
    # A failing adapter costs device setup and a self-test; pay it once.
    pytest.importorskip('wgpu')
    from tomviz.projectors import webgpu
    from tomviz.projectors.backend import get_backend
    calls = []

    def broken(wgpu):
        calls.append(1)
        raise RuntimeError('broken adapter')

    monkeypatch.setattr(webgpu, '_choose_adapter', broken)
    for _ in range(3):
        be = get_backend('auto')
        assert be.name == 'cpu'
        assert 'broken adapter' in be.description
    assert len(calls) == 1


def test_volume_behind_the_source_is_rejected():
    # The box reaches 30 past the axis, the source is only 20 away. The
    # CPU path used to add mirrored data there instead of stopping.
    from tomviz.projectors import backproject_cpu
    g = ConeGeometry.circular(np.arange(0, 360, 30), 20, 40, 1.0, 1.0,
                              16, 8)
    vg = VolumeGeometry((4, 60, 60), 1.0)
    filt = np.ones((g.n_angles, 8, 16), dtype=np.float32)
    with pytest.raises(RuntimeError, match='behind the source'):
        backproject_cpu.backproject(filt, g, vg, threads=THREADS)
    try:
        be = _gpu_backend()
    except pytest.skip.Exception:
        return
    with pytest.raises(RuntimeError, match='behind the source'):
        be.backproject(filt, g, vg)
