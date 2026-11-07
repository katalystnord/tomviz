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


def flat_groups(flat_frames, positions):
    """Flat frames (n, v, u) grouped into runs taken together: frames
    whose positions in the acquisition (frame numbers) follow on without
    a gap form one group. Returns (mean frames (g, v, u), group positions
    (g,)), sorted by position."""
    flat_frames = np.asarray(flat_frames, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float64).ravel()
    if flat_frames.ndim == 2:
        flat_frames = flat_frames[None]
    if len(positions) != len(flat_frames):
        raise ValueError(f'{len(positions)} positions for '
                         f'{len(flat_frames)} flat frames')
    order = np.argsort(positions, kind='stable')
    positions = positions[order]
    breaks = np.flatnonzero(np.diff(positions) > 1.0) + 1
    means, centres = [], []
    for run in np.split(np.arange(len(positions)), breaks):
        means.append(flat_frames[order[run]].mean(axis=0))
        centres.append(positions[run].mean())
    return np.stack(means), np.array(centres)


def flat_weights(group_positions, projection_positions):
    """For each projection, the two flat groups around it and the weight
    of the second: linear between the groups, the nearest group beyond
    the first or last."""
    g = np.asarray(group_positions, dtype=np.float64)
    p = np.asarray(projection_positions, dtype=np.float64).ravel()
    hi = np.clip(np.searchsorted(g, p), 1, max(len(g) - 1, 1))
    lo = hi - 1
    if len(g) == 1:
        return np.zeros(len(p), int), np.zeros(len(p), int), np.zeros(len(p))
    w = (p - g[lo]) / (g[hi] - g[lo])
    return lo, hi, np.clip(w, 0.0, 1.0)


def flat_dark_correct_interpolated(projections, flat_frames, flat_positions,
                                   projection_positions, dark=None,
                                   log=True, min_transmission=1e-6):
    """Flat-field correction with the flat for each projection
    interpolated between the flat groups taken before, during and after
    the scan (the beam drifts over a long synchrotron scan). Positions are
    frame numbers in the acquisition, or any monotonic stand-in for time.
    Returns a new float32 stack like flat_dark_correct."""
    proj = np.asarray(projections)
    if proj.ndim != 3:
        raise ValueError('Projections must be a 3D stack of frames')
    groups, centres = flat_groups(flat_frames, flat_positions)
    if groups.shape[1:] != proj.shape[1:]:
        raise ValueError(f'Flat frames of {groups.shape[1:]} do not match '
                         f'projections of {proj.shape[1:]}')
    lo, hi, w = flat_weights(centres, projection_positions)
    if len(w) != len(proj):
        raise ValueError('One position per projection is needed')
    out = np.empty(proj.shape, dtype=np.float32)
    for k in range(len(proj)):
        flat = (1 - w[k]) * groups[lo[k]] + w[k] * groups[hi[k]]
        flat_dark_correct(proj[k:k + 1], flat=flat, dark=dark, log=log,
                          min_transmission=min_transmission,
                          out=out[k:k + 1])
    return out
