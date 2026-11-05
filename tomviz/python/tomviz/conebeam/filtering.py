###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""FDK pre-weighting, ramp filtering and redundancy weights.

Follows Feldkamp, Davis & Kress (1984) in the form given by Kak & Slaney,
"Principles of Computerized Tomographic Imaging", ch. 3: projections are
rescaled to a virtual detector through the rotation axis, cosine weighted,
ramp filtered along detector rows, and backprojected with the inverse square
distance weight.
"""

import numpy as np

FILTERS = ('ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann')

try:
    import pyfftw.interfaces.numpy_fft as _fft
    import pyfftw.interfaces.cache
    pyfftw.interfaces.cache.enable()
    _FFT_KW = {'threads': -1}
except ImportError:  # pyfftw is optional; numpy's FFT gives the same result
    _fft = np.fft
    _FFT_KW = {}


def _fft_len(n):
    """At least 2n (so the linear convolution does not wrap), rounded up to
    a size the FFT handles fast."""
    m = 2 * n
    best = 1 << int(np.ceil(np.log2(m)))
    # 2^a * 3^b * 5^c sizes are fast too and often much smaller.
    for p3 in (1, 3, 9, 27):
        for p5 in (1, 5, 25):
            base = p3 * p5
            k = 1
            while base * k < m:
                k *= 2
            best = min(best, base * k)
    return best


def ramp_kernel(n_u, filter_name='ram-lak', cutoff=1.0):
    """Frequency response (length n_fft // 2 + 1, for rfft) of the band-
    limited ramp filter for unit sample spacing, with an apodisation window.

    The ramp is built from its exact spatial-domain samples (Kak & Slaney
    eq. 3.61) rather than |f|, so it has no DC bias.
    """
    name = filter_name.lower()
    if name not in FILTERS:
        raise ValueError(f'Unknown filter {filter_name!r}; '
                         f'choose one of {FILTERS}')
    if not 0 < cutoff <= 1:
        raise ValueError('Cutoff must be in (0, 1]')
    n_fft = _fft_len(n_u)
    idx = np.concatenate([np.arange(0, n_fft // 2 + 1),
                          np.arange(-(n_fft // 2) + (n_fft % 2 == 0), 0)])
    h = np.zeros(n_fft)
    h[0] = 0.25
    odd = idx % 2 == 1
    h[odd] = -1.0 / (np.pi * idx[odd]) ** 2
    H = np.real(np.fft.rfft(h))
    # Normalised frequency: 0 at DC, 1 at Nyquist.
    f = np.linspace(0, 1, len(H))
    with np.errstate(divide='ignore', invalid='ignore'):
        if name == 'shepp-logan':
            x = np.pi * f / (2 * cutoff)
            w = np.where(f == 0, 1.0, np.sin(x) / x)
        elif name == 'cosine':
            w = np.cos(np.pi * f / (2 * cutoff))
        elif name == 'hamming':
            w = 0.54 + 0.46 * np.cos(np.pi * f / cutoff)
        elif name == 'hann':
            w = 0.5 + 0.5 * np.cos(np.pi * f / cutoff)
        else:
            w = np.ones_like(f)
    w = np.where(f > cutoff, 0.0, w)
    return H * w, n_fft


def angular_steps(beta):
    """Integration weight (radians) for each projection angle: half the
    distance to its neighbours. A full circle wraps around; a partial scan
    gives the end projections half a step beyond the ends."""
    beta = np.asarray(beta, dtype=np.float64)
    n = len(beta)
    if n == 1:
        return np.array([np.pi])
    order = np.argsort(beta)
    b = beta[order]
    gaps = np.diff(b)
    mean_gap = (b[-1] - b[0]) / (n - 1)
    span = b[-1] - b[0] + mean_gap
    steps = np.empty(n)
    steps[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    if span >= 2 * np.pi - 1e-6:
        wrap = 2 * np.pi - (b[-1] - b[0])
        steps[0] = (gaps[0] + wrap) / 2
        steps[-1] = (gaps[-1] + wrap) / 2
    else:
        steps[0] = gaps[0]
        steps[-1] = gaps[-1]
    out = np.empty(n)
    out[order] = steps
    return out


def scan_coverage(geom):
    """Angular coverage in radians, and the half fan angle of the
    detector, for deciding between full-scan and short-scan weights."""
    beta = geom.angles()
    n = len(beta)
    span = abs(beta[-1] - beta[0])
    coverage = span * n / (n - 1) if n > 1 else 0.0
    pu, _ = geom.principal_point()
    pitch = np.linalg.norm(geom.det_u, axis=1)
    sdd = geom.sdd()
    far = np.maximum(np.abs(pu + 0.5), np.abs(geom.n_u - 0.5 - pu))
    half_fan = float(np.max(np.arctan(far * pitch / sdd)))
    return coverage, half_fan


def parker_weights(geom):
    """Parker (1982) short-scan weights, shape (n_angles, n_u).

    They make every ray in a scan of pi + 2 * half_fan (or more) count
    once in total. The scan direction comes from the order of the angles.
    """
    beta = geom.angles()
    direction = 1.0 if beta[-1] >= beta[0] else -1.0
    b = (beta - beta[0]) * direction
    coverage, delta = scan_coverage(geom)
    # Stretch a scan longer than the minimum so the weights cover it.
    delta = max(delta, (coverage - np.pi) / 2)
    pu, _ = geom.principal_point()
    pitch = np.linalg.norm(geom.det_u, axis=1)
    iu = np.arange(geom.n_u)[None, :]
    # Parker's fan angle is positive on the side of the detector the source
    # is moving away from. Find which way det_u points relative to the
    # source's motion, so mirrored detectors and measured geometries work.
    motion = np.cross([0.0, 0.0, 1.0], geom.source) * direction
    side = np.sign(np.einsum('ij,ij->i', geom.det_u, motion))
    side[side == 0] = 1.0
    fan = (iu - pu[:, None]) * pitch[:, None] / geom.sdd()[:, None]
    gamma = -side[:, None] * np.arctan(fan)
    b = b[:, None]
    w = np.ones_like(gamma)
    with np.errstate(divide='ignore', invalid='ignore'):
        rise = b < 2 * (delta - gamma)
        w = np.where(rise, np.sin(np.pi / 4 * b / (delta - gamma)) ** 2, w)
        fall = b > np.pi - 2 * gamma
        x = (np.pi + 2 * delta - b) / (delta + gamma)
        w = np.where(fall, np.sin(np.pi / 4 * x) ** 2, w)
    w = np.where(b > np.pi + 2 * delta, 0.0, w)
    return np.clip(np.nan_to_num(w), 0.0, 1.0)


def redundancy_weights(geom, short_scan='auto'):
    """Weights that make every ray count once: 1/2 for a full circle,
    Parker weights for a short scan. Returns (n_angles, n_u) and a string
    saying which was used.

    short_scan: 'auto', True or False.
    """
    coverage, half_fan = scan_coverage(geom)
    full = coverage >= 2 * np.pi - 1e-3
    if short_scan == 'auto':
        use_parker = not full
    else:
        use_parker = bool(short_scan)
    if not use_parker:
        note = 'full scan'
        if not full:
            note += (f' (warning: only {np.rad2deg(coverage):.1f} degrees '
                     'covered; consider short-scan weights)')
        return np.full((geom.n_angles, geom.n_u), 0.5), note
    needed = np.pi + 2 * half_fan
    note = (f'short scan with Parker weights ({np.rad2deg(coverage):.1f} of '
            f'{np.rad2deg(needed):.1f} degrees needed)')
    if coverage < needed - 1e-3:
        note += '; warning: the scan is too short, expect artefacts'
    return parker_weights(geom), note


def preweight_and_filter(projections, geom, filter_name='ram-lak',
                         cutoff=1.0, short_scan='auto', out=None):
    """Turn raw line-integral projections (n_angles, n_v, n_u) into filtered
    projections ready for FDK backprojection. Every weight except the
    backprojection's own distance weight is folded in here, including the
    angular step, so backprojection is a plain weighted sum.

    Returns (filtered float32 array, note about the redundancy weights).
    """
    proj = np.asarray(projections)
    if proj.shape != (geom.n_angles, geom.n_v, geom.n_u):
        raise ValueError(
            f'Projections have shape {proj.shape}, geometry expects '
            f'{(geom.n_angles, geom.n_v, geom.n_u)}')
    if out is None:
        out = np.empty(proj.shape, dtype=np.float32)

    H, n_fft = ramp_kernel(geom.n_u, filter_name, cutoff)
    red, note = redundancy_weights(geom, short_scan)
    dbeta = angular_steps(geom.angles())
    pu, pv = geom.principal_point()
    sod, sdd = geom.sod(), geom.sdd()
    pitch_u = np.linalg.norm(geom.det_u, axis=1)
    iu = np.arange(geom.n_u, dtype=np.float64)
    iv = np.arange(geom.n_v, dtype=np.float64)

    for k in range(geom.n_angles):
        # Pixel offsets from the principal point, in world units.
        du = (iu - pu[k])[None, :, None] * geom.det_u[k]
        dv = (iv - pv[k])[:, None, None] * geom.det_v[k]
        r2 = np.einsum('vui,vui->vu', du + dv, du + dv)
        cos_w = sdd[k] / np.sqrt(sdd[k] ** 2 + r2)
        row = proj[k] * cos_w * red[k][None, :]
        # Sample spacing on the virtual detector through the axis.
        tau = pitch_u[k] * sod[k] / sdd[k]
        F = _fft.rfft(row, n=n_fft, axis=1, **_FFT_KW)
        F *= H
        filt = _fft.irfft(F, n=n_fft, axis=1, **_FFT_KW)[:, :geom.n_u]
        out[k] = filt * (dbeta[k] / tau)
    return out, note
