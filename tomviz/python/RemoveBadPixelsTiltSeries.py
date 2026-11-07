def transform(dataset, threshold=5.0):
    """Remove bad pixels (zingers, hot and dead pixels) in tilt series.

    A pixel is bad when it differs from the median of its 3 x 3
    neighbourhood by more than threshold times the image's noise level,
    and is then replaced by that median. The noise level is a robust
    standard deviation of those differences over the whole image (1.4826
    x their median absolute value), so a spike cannot raise its own
    yardstick the way the standard deviation of its own window does."""

    import scipy.ndimage
    import numpy as np

    if threshold is None or threshold <= 0:
        raise RuntimeError('The threshold must be a positive number of '
                           'noise standard deviations')

    tiltSeries = dataset.active_scalars.astype(np.float32)

    for i in range(tiltSeries.shape[2]):
        I = tiltSeries[:, :, i]
        medianI = scipy.ndimage.median_filter(I, size=3, mode='nearest')
        residual = I - medianI
        sigma = 1.4826 * np.median(np.abs(residual))
        if sigma == 0:
            # Flat or quantised images: fall back to the plain spread.
            sigma = residual.std()
        if sigma == 0:
            continue

        # Identify and replace bad pixels
        badPixelsMask = np.abs(residual) > threshold * sigma
        I[badPixelsMask] = medianI[badPixelsMask]
        tiltSeries[:, :, i] = I

    # Set the result as the new scalars.
    dataset.active_scalars = tiltSeries
