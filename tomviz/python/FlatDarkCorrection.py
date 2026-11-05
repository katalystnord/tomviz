import numpy as np

from tomviz.conebeam.preprocess import flat_dark_correct


def _frames(array):
    """Tomviz stores a stack as (x, y, n) with images in the x-y plane. The
    correction works on (n, x, y) frames."""
    a = np.asarray(array, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[2] == 1:
            return a[:, :, 0]
        return np.moveaxis(a, 2, 0)
    raise RuntimeError(f'Expected an image or image stack, got {a.shape}')


def transform(dataset, flat=None, dark=None, line_integrals=True,
              min_transmission=1e-6):
    """Flat-field and dark-field correction, optionally with -log."""
    projections = dataset.active_scalars
    if projections is None:
        raise RuntimeError('No scalars found!')
    if projections.ndim != 3:
        raise RuntimeError('Expected a stack of projections')
    flat_frames = _frames(flat.active_scalars) if flat is not None else None
    dark_frames = _frames(dark.active_scalars) if dark is not None else None

    frames = np.moveaxis(projections, 2, 0)
    try:
        out = flat_dark_correct(frames, flat=flat_frames, dark=dark_frames,
                                log=line_integrals,
                                min_transmission=min_transmission)
    except ValueError as e:
        raise RuntimeError(str(e)) from e
    dataset.active_scalars = np.asfortranarray(np.moveaxis(out, 0, 2))
