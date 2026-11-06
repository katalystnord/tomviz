import numpy as np
from scipy.interpolate import interp1d
import tomviz.operators
import time

# Interpolations the WebGPU backprojector computes ('spline' is
# first-order, so it is linear too). Everything else, and every run without
# a WebGPU adapter, uses the slice-by-slice code below: for 2D parallel-beam
# WBP that is faster than the shared numpy backprojector.
FAST_INTERPOLATIONS = ('linear', 'spline')


class ReconWBPOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, Nrecon=None, filter=None, interp=None,
                  Nupdates=None, backend='auto'):
        """
        3D Reconstruct from a tilt series using Weighted Back-projection Method
        """
        self.progress.maximum = 1

        interpolation_methods = ('linear', 'nearest', 'spline', 'cubic')
        filter_methods = ('none', 'ramp', 'shepp-logan',
                          'cosine', 'hamming', 'hann')

        # Get Tilt angles
        tilt_angles = dataset.tilt_angles
        tiltSeries = dataset.active_scalars
        if tiltSeries is None:
            raise RuntimeError("No scalars found!")

        Nslice = tiltSeries.shape[0]

        if (interpolation_methods[interp] in FAST_INTERPOLATIONS and
                backend != 'cpu'):
            from tomviz.projectors.backend import get_backend
            be = get_backend(backend)  # raises if 'gpu' is unavailable
            if be.name == 'gpu':
                return self._fast(dataset, tiltSeries, tilt_angles, Nrecon,
                                  filter_methods[filter], Nupdates, be)

        self.progress.maximum = Nslice
        step = 0

        recon = np.zeros([Nslice, Nrecon, Nrecon], dtype=np.float32, order='F')
        t0 = time.time()
        counter = 1
        etcMessage = 'Estimated time to complete: n/a'

        child = dataset.create_child_dataset() #create child for recon

        for i in range(Nslice):
            if self.canceled:
                return
            self.progress.message = 'Slice No.%d/%d. ' % (
                i + 1, Nslice) + etcMessage

            recon[i, :, :] = wbp2(tiltSeries[i, :, :], tilt_angles, Nrecon,
                                  filter_methods[filter],
                                  interpolation_methods[interp])
            step += 1
            self.progress.value = step
            timeLeft = (time.time() - t0) / counter * (Nslice - counter)
            counter += 1
            timeLeftMin, timeLeftSec = divmod(timeLeft, 60)
            timeLeftHour, timeLeftMin = divmod(timeLeftMin, 60)
            etcMessage = 'Estimated time to complete: %02d:%02d:%02d' % (
                timeLeftHour, timeLeftMin, timeLeftSec)

            # Update only once every so many steps
            if Nupdates != 0 and (i + 1) % (Nslice//Nupdates) == 0:
                child.active_scalars = recon #add recon to child
                # This copies data to the main thread
                self.progress.data = child

        # One last update of the child data.
        child.active_scalars = recon #add recon to child
        self.progress.data = child

        returnValues = {}
        returnValues["reconstruction"] = child
        return returnValues

    def _fast(self, dataset, tiltSeries, tilt_angles, Nrecon, filter_name,
              Nupdates, be):
        """WBP with linear interpolation on the WebGPU backprojector."""
        Nslice = tiltSeries.shape[0]
        self.progress.message = f'Back projection on {be.description}'
        self.progress.maximum = 100
        self.progress.value = 0
        child = dataset.create_child_dataset()
        recon = np.zeros((Nslice, 1, 1), dtype=np.float32)

        # Live updates: reconstruct in that many chunks of slices.
        chunks = max(1, min(Nslice, Nupdates or 1))
        bounds = np.linspace(0, Nslice, chunks + 1).astype(int)
        for c in range(chunks):
            s0, s1 = bounds[c], bounds[c + 1]

            def progress(done, total):
                self.progress.value = int(
                    100 * (s0 + (s1 - s0) * done / total) / Nslice)

            part = wbp3(tiltSeries[s0:s1], tilt_angles, Nrecon, filter_name,
                        be, progress=progress,
                        should_stop=lambda: self.canceled)
            if part is None:
                return None
            if c == 0:
                recon = np.zeros((Nslice,) + part.shape[1:],
                                 dtype=np.float32, order='F')
            recon[s0:s1] = part
            if c < chunks - 1:
                child.active_scalars = recon
                self.progress.data = child

        child.active_scalars = recon
        self.progress.data = child
        return {"reconstruction": child}


def wbp3(tilt_series, angles, N=None, filter="ramp", backend='auto',
         progress=None, should_stop=None):
    """WBP of a whole tilt series (Nslice, Nray, Nproj) with linear
    interpolation, on a tomviz.projectors backend. Gives the same result as
    calling wbp2 on every slice, except within one pixel of the detector
    edges, where it fades to zero instead of cutting off.

    Returns a float32 (Nslice, N, N) array, or None if should_stop()
    returned True.
    """
    from tomviz.projectors import ParallelGeometry, VolumeGeometry
    from tomviz.projectors.backend import get_backend

    Nslice, Nray, Nproj = tilt_series.shape
    angles = np.asarray(angles, dtype=np.float64).ravel()
    if Nproj != angles.size:
        raise ValueError('Tilt series does not match angles!')
    if not N:
        N = int(np.floor(np.sqrt(Nray**2 / 2.0)))
    be = get_backend(backend) if isinstance(backend, str) else backend

    # Filter exactly as wbp2 does, one projection at a time, folding in
    # wbp2's normalisation.
    F = makeFilter(Nray, filter)[:, 0]
    n2 = F.size
    F = F[:n2 // 2 + 1] * (np.pi / 2 / Nproj)
    projections = np.empty((Nproj, Nslice, Nray), dtype=np.float32)
    for j in range(Nproj):
        s = np.fft.rfft(tilt_series[:, :, j], n=n2, axis=1) * F
        projections[j] = np.fft.irfft(s, n=n2, axis=1)[:, :Nray]

    # wbp2 samples projection pixel Nray // 2 + y cos(a) - x sin(a) for the
    # voxel at (x, y) pixels from voxel N // 2, with x along the second
    # array axis and y along the third. ParallelGeometry.circular maps
    # world (x, y) to y cos(b) - x sin(b), with x along the last axis, so
    # b = -(a + 90 degrees) swaps the axes into wbp2's order.
    geom = ParallelGeometry.circular(
        angles + 90.0, 1.0, 1.0, Nray, Nslice,
        offset_u=Nray // 2 - (Nray - 1) / 2, clockwise=True)
    c = (N - 1) / 2 - N // 2
    vol_geom = VolumeGeometry((Nslice, N, N), 1.0, center=(0.0, c, c))
    return be.backproject(projections, geom, vol_geom, progress=progress,
                          should_stop=should_stop)


def wbp2(sinogram, angles, N=None, filter="ramp", interp="linear"):
    if sinogram.ndim != 2:
        raise ValueError('Sinogram must be 2D')
    (Nray, Nproj) = sinogram.shape
    if Nproj != angles.size:
        raise ValueError('Sinogram does not match angles!')

    interpolation_methods = ('linear', 'nearest', 'spline', 'cubic')
    if interp not in interpolation_methods:
        raise ValueError("Unknown interpolation: %s" % interp)
    if not N:  # if output size is not given
        N = int(np.floor(np.sqrt(Nray**2 / 2.0)))

    ang = np.double(angles) * np.pi / 180.0
    # Create Fourier filter
    F = makeFilter(Nray, filter)
    # Pad sinogram for filtering
    s = np.pad(sinogram, ((0, F.size - Nray), (0, 0)),
               'constant', constant_values=(0, 0))
    # Apply Fourier filter
    s = np.fft.fft(s, axis=0) * F
    s = np.real(np.fft.ifft(s, axis=0))
    # Change back to original
    s = s[:Nray, :]

    # Back projection
    recon = np.zeros((N, N), np.float32)
    center_proj = Nray // 2  # Index of center of projection
    [X, Y] = np.mgrid[0:N, 0:N]
    xpr = X - int(N) // 2
    ypr = Y - int(N) // 2

    for j in range(Nproj):
        t = ypr * np.cos(ang[j]) - xpr * np.sin(ang[j])
        x = np.arange(Nray) - center_proj
        if interp == 'linear':
            bp = np.interp(t, x, s[:, j], left=0, right=0)
        elif interp == 'spline':
            interpolant = interp1d(
                x, s[:, j], kind='slinear', bounds_error=False, fill_value=0)
            bp = interpolant(t)
        else:
            interpolant = interp1d(
                x, s[:, j], kind=interp, bounds_error=False, fill_value=0)
            bp = interpolant(t)
        recon = recon + bp

    # Normalize
    recon = recon * np.pi / 2 / Nproj
    return recon

# Filter (1D) projections.


def makeFilter(Nray, filterMethod="ramp"):
    # Calculate next power of 2
    N2 = 2**np.ceil(np.log2(Nray))
    # Make a ramp filter.
    freq = np.fft.fftfreq(int(N2)).reshape(-1, 1)
    omega = 2 * np.pi * freq
    filter = 2 * np.abs(freq)

    if filterMethod == "ramp":
        pass
    elif filterMethod == "shepp-logan":
        filter[1:] = filter[1:] * np.sin(omega[1:]) / omega[1:]
    elif filterMethod == "cosine":
        filter[1:] = filter[1:] * np.cos(filter[1:])
    elif filterMethod == "hamming":
        filter[1:] = filter[1:] * (0.54 + 0.46 * np.cos(omega[1:] / 2))
    elif filterMethod == "hann":
        filter[1:] = filter[1:] * (1 + np.cos(omega[1:] / 2)) / 2
    elif filterMethod == "none":
        filter[:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filterMethod)

    return filter
