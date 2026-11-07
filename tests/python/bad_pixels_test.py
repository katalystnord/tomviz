"""Remove Bad Pixels on a smooth noisy image with isolated spikes
(zingers) of 10 to 100 noise standard deviations."""

import numpy as np
import pytest

from utils import load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset


@pytest.fixture(scope='module')
def images():
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:128, :128]
    clean = 100 + 20 * np.sin(x / 9.0) * np.cos(y / 13.0)
    clean = np.repeat(clean[:, :, None], 3, axis=2).astype(np.float32)
    noisy = clean + rng.normal(0, 1.0, clean.shape).astype(np.float32)
    spiky = noisy.copy()
    spikes = []
    for k in range(60):
        i, j, a = rng.integers(2, 126), rng.integers(2, 126), rng.integers(3)
        amp = [10, 20, 50, 100][k % 4] * (1 if k % 2 else -1)
        spiky[i, j, a] += amp
        spikes.append((i, j, a))
    return noisy, spiky, spikes


def run(spiky, **kwargs):
    ds = Dataset({'scalars': np.asfortranarray(spiky.copy())})
    load_operator_module('RemoveBadPixelsTiltSeries').transform(ds, **kwargs)
    return ds.active_scalars


def test_removes_spikes_and_leaves_the_rest(images):
    noisy, spiky, spikes = images
    out = run(spiky, threshold=5.0)
    # Every spike (bright or dark, 10 sigma and up) is gone...
    for i, j, a in spikes:
        assert abs(out[i, j, a] - noisy[i, j, a]) < 4.0
    # ...and almost nothing else changed.
    mask = np.ones(noisy.shape, bool)
    for i, j, a in spikes:
        mask[i, j, a] = False
    assert np.mean(out[mask] != noisy[mask]) < 0.001


def test_default_threshold_is_used(images):
    # From the menu the operator used to get no threshold and fail.
    _, spiky, spikes = images
    out = run(spiky)
    assert np.array_equal(out, run(spiky, threshold=5.0))


def test_rejects_a_bad_threshold(images):
    with pytest.raises(RuntimeError, match='positive'):
        run(images[1], threshold=0)
