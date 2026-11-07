"""Paganin phase retrieval, checked against wave optics: a sphere of one
material is propagated with the Fresnel propagator, and the retrieval must
recover its projected attenuation and remove the edge fringes."""

import numpy as np
import pytest

from utils import load_operator_class, load_operator_module

from tomviz_pipeline.dataset import LegacyDataset as Dataset

HC = 1.23984198e-9  # keV m
N, PIXEL = 256, 1e-6  # 1 um pixels
ENERGY, DISTANCE = 20.0, 0.2
DELTA, BETA = 5e-7, 5e-10  # delta/beta = 1000, a light material
RADIUS = 40e-6


@pytest.fixture(scope='module')
def paganin():
    return load_operator_module('PhaseRetrievalPaganin')


def sphere_thickness():
    y, x = (np.mgrid[:N, :N] - (N - 1) / 2) * PIXEL
    return 2 * np.sqrt(np.clip(RADIUS ** 2 - x ** 2 - y ** 2, 0, None))


def propagate(thickness):
    """Intensity after free-space propagation (paraxial Fresnel), on a
    padded grid so the field does not wrap around."""
    lam = HC / ENERGY
    mu = 4 * np.pi * BETA / lam
    phase = 2 * np.pi * DELTA * thickness / lam
    field = np.exp(-mu * thickness / 2 - 1j * phase)
    pad = N // 2
    field = np.pad(field, pad, constant_values=1.0)
    f = np.fft.fftfreq(field.shape[0], d=PIXEL)
    h = np.exp(-1j * np.pi * lam * DISTANCE * (f[:, None] ** 2 +
                                               f[None, :] ** 2))
    out = np.fft.ifft2(np.fft.fft2(field) * h)
    return (np.abs(out) ** 2)[pad:pad + N, pad:pad + N], mu


def test_recovers_projected_attenuation_from_wave_optics(paganin):
    t = sphere_thickness()
    intensity, mu = propagate(t)
    truth = mu * t
    alpha = paganin.paganin_alpha(DISTANCE, ENERGY, DELTA / BETA)
    retrieved = -np.log(paganin.retrieve(intensity, PIXEL, alpha))
    raw = -np.log(intensity)
    inside = t > 0.6 * t.max()
    # Inside the sphere: within 5 % of the truth.
    rel = np.abs(retrieved[inside] - truth[inside]) / truth[inside]
    assert rel.mean() < 0.05
    # The raw image is dominated by edge fringes (negative attenuation
    # next to the edge, overshoots far beyond the truth); retrieval
    # removes them.
    assert raw.min() < -5 * truth.max() * 0.05
    assert retrieved.min() > -0.02 * truth.max()
    assert np.abs(raw - truth).max() > 5 * np.abs(retrieved - truth).max()


def test_inverts_its_own_forward_model_exactly(paganin):
    # With I = (1 - alpha laplacian) exp(-mu t), the TIE model behind the
    # method, retrieval gives mu t back.
    t = sphere_thickness()
    mu = 4 * np.pi * BETA / (HC / ENERGY)
    alpha = paganin.paganin_alpha(DISTANCE, ENERGY, DELTA / BETA)
    pad = N  # well clear of the borders
    att = np.pad(np.exp(-mu * t), pad, constant_values=1.0)
    k = 2 * np.pi * np.fft.fftfreq(att.shape[0], d=PIXEL)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    intensity = np.real(np.fft.ifft2(np.fft.fft2(att) * (1 + alpha * k2)))
    retrieved = -np.log(paganin.retrieve(intensity, PIXEL, alpha,
                                         pad=False))[pad:-pad, pad:-pad]
    np.testing.assert_allclose(retrieved, mu * t, atol=1e-9)


def test_operator_on_a_tilt_series(paganin):
    t = sphere_thickness()
    intensity, mu = propagate(t)
    stack = np.asfortranarray(np.stack([intensity] * 3, axis=2))
    ds = Dataset({'scalars': stack})
    ds.spacing = [1.0, 1.0, 1.0]
    ds.tilt_angles = np.array([0.0, 60.0, 120.0])
    operator = load_operator_class(paganin)
    operator.transform(ds, pixel_size=1.0, distance=DISTANCE, energy=ENERGY,
                       delta_beta=DELTA / BETA)
    out = ds.active_scalars
    assert out.shape == stack.shape and out.dtype == np.float32
    inside = t > 0.6 * t.max()
    assert np.allclose(out[:, :, 1][inside], (mu * t)[inside], rtol=0.06)

    # Line integrals in gives the same as transmission in.
    ds2 = Dataset({'scalars': np.asfortranarray(-np.log(stack))})
    ds2.spacing = [1.0, 1.0, 1.0]
    operator.transform(ds2, input_type='line_integrals', pixel_size=1.0,
                       distance=DISTANCE, energy=ENERGY,
                       delta_beta=DELTA / BETA)
    np.testing.assert_allclose(ds2.active_scalars, out, atol=1e-5)
