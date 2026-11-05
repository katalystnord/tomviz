import numpy as np

from tomviz.conebeam.preprocess import flat_dark_correct


def transform(dataset, i0=0.0, air_percentile=99.5, dark_level=0.0,
              min_transmission=1e-6):
    """-ln((I - dark) / (I0 - dark)) with constant or estimated I0."""
    projections = dataset.active_scalars
    if projections is None:
        raise RuntimeError('No scalars found!')
    if projections.ndim != 3:
        raise RuntimeError('Expected a stack of projections')
    frames = np.moveaxis(projections, 2, 0)

    if i0 > 0:
        flat = np.float32(i0)
    else:
        # One open-beam level per projection, from its brightest pixels.
        levels = np.percentile(frames.reshape(len(frames), -1),
                               air_percentile, axis=1)
        flat = levels.astype(np.float32)[:, None, None]
    if np.any(np.asarray(flat) <= dark_level):
        raise RuntimeError('The open-beam intensity must be above the dark '
                           'level')

    out = np.empty(frames.shape, dtype=np.float32)
    if np.ndim(flat) == 0:
        flat_dark_correct(frames, flat=flat, dark=dark_level,
                          min_transmission=min_transmission, out=out)
    else:
        for k in range(len(frames)):
            flat_dark_correct(frames[k:k + 1], flat=flat[k, 0, 0],
                              dark=dark_level,
                              min_transmission=min_transmission,
                              out=out[k:k + 1])
    dataset.active_scalars = np.asfortranarray(np.moveaxis(out, 0, 2))
