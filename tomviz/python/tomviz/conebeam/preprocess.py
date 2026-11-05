###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Turning raw detector counts into line integrals."""

import numpy as np


def _as_frame(ref, frame_shape, name):
    """A flat or dark reference: a scalar, one frame, or a stack of frames
    that is averaged. Returns float32 broadcastable to one frame."""
    if ref is None:
        return None
    ref = np.asarray(ref, dtype=np.float32)
    if ref.ndim == 0:
        return ref
    if ref.shape == frame_shape:
        return ref
    if ref.ndim == 3 and ref.shape[1:] == frame_shape:
        return ref.mean(axis=0, dtype=np.float64).astype(np.float32)
    raise ValueError(f'{name} has shape {ref.shape}; expected a single '
                     f'frame {frame_shape} or a stack (n, *{frame_shape})')


def flat_dark_correct(projections, flat=None, dark=None, log=True,
                      min_transmission=1e-6, out=None):
    """Flat-field and dark-field correction of a stack of frames
    (n_frames, n_v, n_u):

        T = (I - dark) / (flat - dark)

    then optionally the line integral -ln(T). Flat and dark may each be a
    scalar, one frame, or a stack of frames (averaged). Missing dark means
    0; missing flat means 1 (for data that are already normalised).
    Pixels where flat <= dark are dead and come out as no attenuation.

    Transmission is clipped to [min_transmission, inf) before the log so
    dead pixels and photon starvation give large finite values instead of
    inf or nan.
    """
    proj = np.asarray(projections)
    if proj.ndim != 3:
        raise ValueError('Projections must be a 3D stack of frames')
    frame_shape = proj.shape[1:]
    flat = _as_frame(flat, frame_shape, 'Flat field')
    dark = _as_frame(dark, frame_shape, 'Dark field')
    if dark is None:
        dark = np.float32(0)
    if flat is None:
        flat = np.float32(1)
    gain = np.asarray(flat - dark, dtype=np.float32)
    if gain.ndim == 0 and gain <= 0:
        raise ValueError('Flat must be brighter than dark')
    # Dead or saturated reference pixels (flat <= dark) carry no
    # information; report them as no attenuation instead of inf or nan.
    valid = gain > 0
    inv_gain = np.where(valid, 1.0 / np.where(valid, gain, 1.0),
                        0.0).astype(np.float32)

    if out is None:
        out = np.empty(proj.shape, dtype=np.float32)
    for k in range(proj.shape[0]):
        t = (proj[k].astype(np.float32) - dark) * inv_gain
        t = np.where(valid, t, np.float32(1))
        if log:
            np.maximum(t, np.float32(min_transmission), out=t)
            out[k] = -np.log(t)
        else:
            out[k] = t
    return out
