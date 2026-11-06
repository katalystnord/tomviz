"""Ring artefact removal on projections, before reconstruction.

A detector pixel that responds a little differently from its neighbours
puts a stripe into every sinogram it takes part in, and a stripe
reconstructs as a ring around the rotation axis. This follows Vo, Atwood
and Drakopoulos, "Superior techniques for eliminating ring artifacts in
X-ray micro-tomography", Opt. Express 26:28396 (2018), adapted to line
integrals: a gain error multiplies the intensity, so after the logarithm
it adds an offset. Every correction here is therefore additive, which
keeps data near zero (air, the detector margins) safe, where the ratios of
the original intensity-domain formulation are unstable.

Two steps, both per sinogram (one detector row over all angles):

1. Dead pixels: a pixel whose values hardly vary over all angles while
   its neighbours' do carries no signal, and is interpolated across.

2. Stripes (Vo's algorithm 2). Each column is split into a slowly varying
   part (a Gaussian low-pass over angle) and the rest. Stripes live in the
   slow part; it is sorted over angle, median filtered across columns, and
   put back in each column's own order, which keeps the object's signal.

Vo's large-stripe detection (algorithms 4 and 5) is deliberately not
used: on real cone-beam data (CWI Walnut 1, dev/ring_bench.py) it flagged
7 to 20 per cent of the pixels of clean data and left more error than no
correction, in both the intensity and the line-integral domain.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

import numpy as np

from ..projectors.backproject_cpu import default_threads


def _sorted_median(block, size):
    """Sort each column of a (n_angles, rows, n_u) block over angle,
    median filter the sorted values across columns, and return them in
    each column's original angular order."""
    from scipy.ndimage import median_filter
    order = np.argsort(block, axis=0, kind='stable')
    s = np.take_along_axis(block, order, axis=0)
    s = median_filter(s, size=(1, 1, size), mode='reflect')
    out = np.empty_like(block)
    np.put_along_axis(out, order, s, axis=0)
    return out, order


def _dead_pixels(block):
    """Find pixels that do not respond: over all angles their values vary
    far less than their neighbours' do. Returns a (rows, n_u) mask. In air
    every pixel varies only by noise, so nothing is flagged there."""
    from scipy.ndimage import median_filter
    med = np.median(block, axis=0)
    spread = np.median(np.abs(block - med), axis=0)
    local = median_filter(spread, size=(1, 9), mode='reflect')
    return spread < 0.1 * local


def _interpolate_across(block, mask):
    """Replace masked columns of each row by linear interpolation across
    the detector from the nearest good columns, at every angle."""
    out = block.copy()
    for r in np.where(mask.any(axis=1))[0]:
        bad = np.where(mask[r])[0]
        good = np.where(~mask[r])[0]
        if good.size < 2:
            continue
        for k in range(block.shape[0]):
            out[k, r, bad] = np.interp(bad, good, block[k, r, good])
    return out


def _filtering(block, sigma, size):
    """Vo's algorithm 2: sort-and-median the low-pass part over angle."""
    n_a = block.shape[0]
    pad = min(150, int(0.1 * n_a))
    padded = np.pad(block, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
    m = padded.shape[0]
    k = np.fft.rfftfreq(m) * m
    window = np.exp(-0.5 * (k / sigma) ** 2)[:, None, None]
    low = np.fft.irfft(np.fft.rfft(padded, axis=0) * window, n=m,
                       axis=0)[pad:pad + n_a].astype(block.dtype)
    corrected, _ = _sorted_median(low, size)
    return block - low + corrected


def remove_rings(projections, size=None, sigma=3.0, dead=True, threads=None,
                 progress=None, should_stop=None, out=None):
    """Remove ring artefacts from a stack of line-integral projections.

    projections: (n_angles, n_v, n_u) array of line integrals, after
        flat/dark correction and the logarithm. Stripes run along the
        angle axis; the detector u axis is across the rotation axis.
    size: median width across the detector, in pixels. Wider removes
        wider stripes but starts to alter real structure. Default: 1% of
        the detector width, at least 5.
    sigma: width of the Gaussian low-pass over angle, in frequency
        samples; 3 to 10 is typical.
    dead: also repair pixels that do not respond (interpolated across).
    threads: worker threads (rows are independent).
    progress: optional callable(fraction done).
    should_stop: optional callable; when it returns True, returns None.
    out: optional output array of the same shape.

    Returns the corrected float32 array (out, if given).
    """
    proj = np.asarray(projections)
    if proj.ndim != 3:
        raise ValueError('Projections must be a 3D stack of frames')
    n_a, n_v, n_u = proj.shape
    if n_a < 8:
        raise ValueError('Ring removal needs at least 8 projections')
    if size is None:
        size = max(5, int(0.01 * n_u))
    size = int(size)
    if size < 3:
        raise ValueError('The median width must be at least 3 pixels')
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    if out is None:
        out = np.empty(proj.shape, dtype=np.float32)
    threads = threads or default_threads()

    rows = max(1, min(16, n_v // max(1, threads)))
    chunks = [slice(r, min(r + rows, n_v)) for r in range(0, n_v, rows)]
    # progress() and should_stop() are only called from this thread;
    # workers see cancellation through the event.
    stop = threading.Event()

    def work(sl):
        if stop.is_set():
            return
        block = np.asarray(proj[:, sl, :], dtype=np.float32)
        if dead:
            block = _interpolate_across(block, _dead_pixels(block))
        out[:, sl, :] = _filtering(block, sigma, size)

    done = 0
    with ThreadPoolExecutor(threads) as ex:
        pending = {ex.submit(work, sl) for sl in chunks}
        while pending:
            finished, pending = wait(pending, timeout=0.2,
                                     return_when=FIRST_COMPLETED)
            if should_stop is not None and should_stop():
                stop.set()
                return None
            for fut in finished:
                fut.result()
                done += 1
                if progress is not None:
                    progress(done / len(chunks))
    return out
