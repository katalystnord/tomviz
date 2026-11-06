"""Ring artefact removal (tomviz.conebeam.rings) and its operator.

The defects are the kinds that cause rings in practice: many pixels with
small gain errors, a few with a nonlinear response (partial stripes), and
dead pixels. dev/ring_bench.py measures the same on real data (CWI
Walnut 1)."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.projectors.phantom import EllipsoidPhantom
from tomviz.conebeam import fdk
from tomviz.conebeam.rings import _dead_pixels, remove_rings

THREADS = 4


def defects(shape, seed=0):
    """Per-pixel gain (as a line-integral offset), nonlinearity and dead
    pixels for a (n_v, n_u) detector."""
    rng = np.random.default_rng(seed)
    offset = np.where(rng.random(shape) < 0.05,
                      rng.normal(0, 0.02, shape), 0.0)
    gamma = np.where(rng.random(shape) < 0.02,
                     rng.normal(0, 0.05, shape), 0.0)
    dead = np.zeros(shape, bool)
    dead[:, [shape[1] // 3, shape[1] // 2 + 7]] = True
    return offset, gamma, dead


def apply(clean, offset, gamma, dead):
    bad = clean * (1 + gamma) + offset
    bad[:, dead] = 0.0
    return bad.astype(np.float32)


@pytest.fixture(scope='module')
def scan():
    n_u, n_v = 256, 4
    g = ConeGeometry.circular(np.arange(0, 360, 0.5), 200.0, 400.0,
                              0.2, 0.2, n_u, n_v)
    vg = VolumeGeometry((2, 180, 180), 0.12)
    ph = EllipsoidPhantom.shepp_logan(radius=10.0)
    clean = ph.project(g).astype(np.float32)
    rng = np.random.default_rng(1)
    clean += rng.normal(0, 0.005, clean.shape).astype(np.float32)
    return g, vg, clean


def rec_error(proj, ref, g, vg):
    v = fdk(proj, g, vg, threads=THREADS)
    return np.sqrt(np.mean((v - ref) ** 2))


def test_removes_rings_without_harming_clean_data(scan):
    g, vg, clean = scan
    bad = apply(clean, *defects(clean.shape[1:]))
    ref = fdk(clean, g, vg, threads=THREADS)
    before = rec_error(bad, ref, g, vg)
    after = rec_error(remove_rings(bad, threads=THREADS), ref, g, vg)
    harm = rec_error(remove_rings(clean, threads=THREADS), ref, g, vg)
    # Measured: before 0.1003, after 0.0085, harm 0.0083. The skull's
    # sharp, nearly round edge is the hard case for any stripe filter.
    assert after < 0.2 * before
    assert harm < 0.15 * before


def test_dead_pixels_found_exactly(scan):
    _, _, clean = scan
    offset, gamma, dead = defects(clean.shape[1:])
    bad = apply(clean, offset, gamma, dead)
    assert not _dead_pixels(clean).any()
    assert np.array_equal(_dead_pixels(bad), dead)


def test_smooth_signal_with_fixed_pattern():
    # Stripes on a smooth signal that is the same at every angle. The
    # median across the detector cannot do better than about one pixel's
    # worth of slope on a steep flank, so this asks for the bulk of the
    # stripes to go and for ring-free data to be left alone.
    rng = np.random.default_rng(5)
    n_a, n_v, n_u = 90, 3, 200
    smooth = np.tile(np.hanning(n_u)[None, None, :], (n_a, n_v, 1))
    offset = np.where(rng.random((n_v, n_u)) < 0.05, 0.05, 0.0)
    out = remove_rings((smooth + offset).astype(np.float32),
                       threads=THREADS)

    def rms(a):
        return np.sqrt(np.mean(a ** 2))

    # Measured: 0.44.
    assert rms(out - smooth) < 0.55 * rms(offset)
    assert np.abs(remove_rings(smooth.astype(np.float32)) - smooth).max() \
        < 1e-3


def test_threads_progress_out_and_stop(scan):
    _, _, clean = scan
    one = remove_rings(clean, threads=1)
    seen = []
    out = np.empty_like(clean)
    many = remove_rings(clean, threads=THREADS, progress=seen.append,
                        out=out)
    assert many is out
    np.testing.assert_array_equal(one, many)
    assert seen and seen[-1] == pytest.approx(1.0)
    assert remove_rings(clean, should_stop=lambda: True) is None


def test_callbacks_only_on_the_calling_thread(scan):
    # Tomviz progress objects are not meant to be poked from pool threads.
    import threading
    _, _, clean = scan
    me = threading.get_ident()
    seen = set()

    def note(*_):
        seen.add(threading.get_ident())
        return False

    remove_rings(clean, threads=THREADS, progress=note, should_stop=note)
    assert seen == {me}


def test_rejects_bad_input(scan):
    _, _, clean = scan
    with pytest.raises(ValueError):
        remove_rings(clean[0])
    with pytest.raises(ValueError):
        remove_rings(clean[:4])
    with pytest.raises(ValueError):
        remove_rings(clean, size=1)
    with pytest.raises(ValueError):
        remove_rings(clean, sigma=0)


def test_operator_axes(scan):
    # Tomviz tilt series: (x along the tilt axis, y across it, angle). The
    # operator must hand remove_rings (angle, v, u) and transpose back.
    _, _, clean = scan
    bad = apply(clean, *defects(clean.shape[1:]))
    data = np.asfortranarray(np.moveaxis(bad, 0, 2))
    ds = Dataset({'scalars': data})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = np.arange(bad.shape[0], dtype=np.float64) * 0.5
    op = load_operator_class(load_operator_module('RemoveRingArtifacts'))
    op.transform(ds)
    out = ds.active_scalars
    assert out.shape == data.shape and out.flags.f_contiguous
    expected = remove_rings(bad)
    np.testing.assert_array_equal(np.moveaxis(out, 2, 0), expected)
    assert np.abs(out - data).max() > 0.01  # it did change something
