"""Single-distance phase retrieval (Paganin et al., J. Microsc. 206, 33,
2002) for propagation-based phase-contrast projections.

For one material with a known ratio delta/beta at propagation distance z,
the projected attenuation is

    mu t = -ln( F^-1[ F[I / I0] / (1 + z (delta / mu) |k|^2) ] )

with mu = 4 pi beta / lambda, so z delta / mu = z (delta / beta) lambda /
(4 pi), and k the angular spatial frequency. This is the low-pass filter
TomoPy's retrieve_phase applies as well.
"""

import numpy as np

import tomviz.operators

HC_KEV_M = 1.23984198e-9  # h c in keV m: lambda = HC_KEV_M / E


def paganin_alpha(distance_m, energy_kev, delta_beta):
    """z delta / mu in m^2."""
    wavelength = HC_KEV_M / energy_kev
    return distance_m * delta_beta * wavelength / (4 * np.pi)


def paganin_filter(shape, pixel_m, alpha):
    """1 / (1 + alpha |k|^2) on an rfft2 grid of the given (padded)
    shape."""
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=pixel_m)
    kx = 2 * np.pi * np.fft.rfftfreq(shape[1], d=pixel_m)
    return 1.0 / (1.0 + alpha * (ky[:, None] ** 2 + kx[None, :] ** 2))


def retrieve(transmission, pixel_m, alpha, pad=True, workers=-1):
    """Filtered transmission of one image (rows, columns). The image is
    padded (edge values, to at least twice its size) so the periodic FFT
    does not mix opposite borders."""
    import scipy.fft
    img = np.asarray(transmission, dtype=np.float64)
    r, c = img.shape
    if pad:
        pr = scipy.fft.next_fast_len(2 * r)
        pc = scipy.fft.next_fast_len(2 * c)
        top, left = (pr - r) // 2, (pc - c) // 2
        img = np.pad(img, ((top, pr - r - top), (left, pc - c - left)),
                     mode='edge')
    else:
        top = left = 0
    filt = paganin_filter(img.shape, pixel_m, alpha)
    out = scipy.fft.irfft2(scipy.fft.rfft2(img, workers=workers) * filt,
                           s=img.shape, workers=workers)
    return out[top:top + r, left:left + c]


class PhaseRetrievalPaganinOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, input_type='transmission', pixel_size=1.0,
                  distance=0.1, energy=20.0, delta_beta=1000.0,
                  line_integrals=True, min_transmission=1e-6):
        """Paganin phase retrieval of every projection of a tilt series.

        pixel_size in micrometres (at the sample), distance (sample to
        detector) in metres, energy in keV."""
        data = dataset.active_scalars
        if data is None:
            raise RuntimeError('No scalars found!')
        if data.ndim != 3:
            raise RuntimeError('Expected a stack of projections')
        if pixel_size <= 0 or distance < 0 or energy <= 0 or \
                delta_beta <= 0:
            raise RuntimeError('Pixel size, energy and delta/beta must be '
                               'positive, the distance not negative')
        alpha = paganin_alpha(distance, energy, delta_beta)
        pixel_m = pixel_size * 1e-6

        n = data.shape[2]
        out = np.empty(data.shape, dtype=np.float32, order='F')
        self.progress.maximum = n
        for i in range(n):
            if self.canceled:
                return
            frame = np.asarray(data[:, :, i], dtype=np.float64)
            if input_type == 'line_integrals':
                frame = np.exp(-frame)
            filtered = retrieve(frame, pixel_m, alpha)
            if line_integrals:
                filtered = -np.log(np.maximum(filtered, min_transmission))
            out[:, :, i] = filtered
            self.progress.value = i + 1
        dataset.active_scalars = out
