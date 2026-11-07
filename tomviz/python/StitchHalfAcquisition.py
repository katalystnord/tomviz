"""Join a 360 degree half-acquisition (offset rotation axis) scan into an
ordinary 180 degree scan.

To image samples wider than the detector, synchrotrons put the rotation
axis near one edge of the field of view and rotate 360 degrees. For a
parallel beam the projection at theta + 180 is the one at theta mirrored
about the axis, so each pair together covers the whole sample:

    full(theta, t) = p(theta, c + t)        on the detector's side
                   = p(theta + 180, c - t)  on the other side

with t measured from the axis at detector position c. Where both cover
the same t (the overlap next to the axis), they are blended linearly.
The result has the axis at its centre and angles from 0 to 180 degrees,
ready for any parallel-beam reconstruction.
"""

import numpy as np

import tomviz.operators


def opposite_pairs(angles):
    """Index pairs (i, j) with angle j 180 degrees after angle i, for the
    first half of the scan."""
    angles = np.asarray(angles, dtype=np.float64).ravel()
    steps = np.diff(np.sort(angles))
    step = np.median(steps[steps > 0]) if np.any(steps > 0) else 1.0
    tolerance = max(0.25 * step, 1e-3)
    pairs = []
    used = set()
    for i in np.argsort(angles):
        if i in used:
            continue
        gap = np.abs((angles - angles[i]) - 180.0)
        j = int(np.argmin(gap))
        if gap[j] <= tolerance and j not in used:
            pairs.append((int(i), j))
            used.update((int(i), j))
    return pairs


def find_axis(tilt_series, pairs, max_pairs=12):
    """The axis position (pixels from the detector centre) from pairs 180
    degrees apart: the mirrored second projection matches the first over
    their overlap. Correlates over all candidate positions on a 1/4 pixel
    grid and refines with a parabola; median over up to max_pairs."""
    n = tilt_series.shape[1]
    pick = np.unique(np.linspace(0, len(pairs) - 1,
                                 min(max_pairs, len(pairs))).astype(int))
    found = []
    candidates = np.arange(-(n - 1) / 2 + 2, (n - 1) / 2 - 2 + 1e-9, 0.25)
    for k in pick:
        i, j = pairs[k]
        a = np.asarray(tilt_series[:, :, i], dtype=np.float64).mean(axis=0)
        b = np.asarray(tilt_series[:, :, j], dtype=np.float64).mean(axis=0)
        x = np.arange(n)
        scores = []
        for d in candidates:
            c = (n - 1) / 2 + d
            # b mirrored about c, sampled where it lands on a's pixels.
            mirrored = 2 * c - x
            ok = (mirrored >= 0) & (mirrored <= n - 1)
            if ok.sum() < 8:
                scores.append(-np.inf)
                continue
            bm = np.interp(mirrored[ok], x, b)
            aa = a[ok] - a[ok].mean()
            bb = bm - bm.mean()
            norm = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
            scores.append((aa * bb).sum() / norm if norm > 0 else -np.inf)
        scores = np.array(scores)
        m = int(np.argmax(scores))
        frac = 0.0
        near = scores[m - 1:m + 2]
        if 0 < m < len(scores) - 1 and np.all(np.isfinite(near)):
            left, mid, right = scores[m - 1:m + 2]
            denom = left - 2 * mid + right
            frac = 0.5 * (left - right) / denom if denom != 0 else 0.0
        found.append(candidates[m] + frac * 0.25)
    return float(np.median(found))


def stitch(tilt_series, angles, axis_offset):
    """The 180 degree tilt series (rows, 2R + 1, n/2) and its angles, for
    an axis axis_offset pixels from the detector centre."""
    angles = np.asarray(angles, dtype=np.float64).ravel()
    pairs = opposite_pairs(angles)
    if not pairs:
        raise RuntimeError('Stitching needs a 360 degree scan with '
                           'projections 180 degrees apart')
    rows, n, _ = tilt_series.shape
    c = (n - 1) / 2 + axis_offset
    if abs(axis_offset) < 1:
        raise RuntimeError('The axis is at the detector centre: this is not '
                           'a half-acquisition scan')
    # Axis right of centre: the detector sees t from -c to n - 1 - c, the
    # mirrored partner the rest. Left of centre: the other way round.
    right = axis_offset > 0
    reach = c if right else (n - 1 - c)          # far side, own detector
    near = (n - 1 - c) if right else c           # overlap width
    radius = int(np.floor(reach))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    sign = 1.0 if right else -1.0
    x = np.arange(n, dtype=np.float64)
    own_pos = c + t            # where t lies on the first projection
    other_pos = c - t          # where t lies on the mirrored partner
    own_ok = (own_pos >= 0) & (own_pos <= n - 1)
    other_ok = (other_pos >= 0) & (other_pos <= n - 1)
    # Weight of the first projection: 1 on its own side, falling linearly
    # to 0 across the overlap (|t| < near) on the partner's side.
    w = np.where(sign * t <= 0, 1.0,
                 np.clip(1.0 - sign * t / max(near, 1e-9), 0.0, 1.0))
    w = np.where(own_ok & ~other_ok, 1.0, np.where(other_ok & ~own_ok,
                                                   0.0, w))
    out = np.zeros((rows, len(t), len(pairs)), dtype=np.float32, order='F')
    for k, (i, j) in enumerate(pairs):
        p = np.asarray(tilt_series[:, :, i], dtype=np.float64)
        q = np.asarray(tilt_series[:, :, j], dtype=np.float64)
        for r in range(rows):
            own = np.interp(own_pos, x, p[r], left=0.0, right=0.0)
            other = np.interp(other_pos, x, q[r], left=0.0, right=0.0)
            out[r, :, k] = w * own + (1 - w) * other
    return out, angles[[i for i, _ in pairs]]


class StitchHalfAcquisitionOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, axis_offset=0.0):
        """axis_offset: where the rotation axis projects, in pixels from
        the detector centre (positive: towards higher y); 0 finds it."""
        data = dataset.active_scalars
        angles = dataset.tilt_angles
        if data is None or data.ndim != 3:
            raise RuntimeError('Expected a tilt series')
        if angles is None or len(angles) != data.shape[2]:
            raise RuntimeError('Every projection needs a tilt angle')
        self.progress.maximum = 2
        if axis_offset == 0:
            pairs = opposite_pairs(angles)
            if not pairs:
                raise RuntimeError('Stitching needs a 360 degree scan with '
                                   'projections 180 degrees apart')
            self.progress.message = 'Finding the rotation axis'
            axis_offset = find_axis(data, pairs)
            print(f'Rotation axis found {axis_offset:.2f} pixels from the '
                  'detector centre')
        self.progress.value = 1
        self.progress.message = 'Joining opposite projections'
        out, new_angles = stitch(data, angles, axis_offset)
        dataset.active_scalars = out
        dataset.tilt_angles = new_angles
        self.progress.value = 2
