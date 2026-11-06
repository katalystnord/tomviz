import numpy as np

from tomviz.conebeam.preprocess import flat_dark_correct


def _frames(array, frame_shape):
    """Flat or dark images as (n, x, y) frames matching the projections'
    (x, y) frame shape. Tomviz may put the image count on any axis (a
    single image loaded from a file can come in as (x, 1, y)), so take the
    axis whose removal leaves a frame of the right shape, transposed if
    need be."""
    a = np.asarray(array, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3:
        for axis in (2, 0, 1):
            rest = tuple(n for i, n in enumerate(a.shape) if i != axis)
            if rest == tuple(frame_shape) or rest[::-1] == tuple(frame_shape):
                a = np.moveaxis(a, axis, 0)
                break
        else:
            raise RuntimeError(
                f'A flat or dark image of shape {a.shape} does not fit '
                f'projections of {frame_shape[0]} x {frame_shape[1]} pixels')
    else:
        raise RuntimeError(f'Expected an image or image stack, got {a.shape}')
    if a.shape[1:] != tuple(frame_shape):
        a = a.transpose(0, 2, 1)
    return a[0] if len(a) == 1 else a


def transform(dataset, flat=None, dark=None, line_integrals=True,
              min_transmission=1e-6):
    """Flat-field and dark-field correction, optionally with -log."""
    projections = dataset.active_scalars
    if projections is None:
        raise RuntimeError('No scalars found!')
    if projections.ndim != 3:
        raise RuntimeError('Expected a stack of projections')
    frame_shape = projections.shape[:2]
    flat_frames = (_frames(flat.active_scalars, frame_shape)
                   if flat is not None else None)
    dark_frames = (_frames(dark.active_scalars, frame_shape)
                   if dark is not None else None)

    frames = np.moveaxis(projections, 2, 0)
    try:
        out = flat_dark_correct(frames, flat=flat_frames, dark=dark_frames,
                                log=line_integrals,
                                min_transmission=min_transmission)
    except ValueError as e:
        raise RuntimeError(str(e)) from e
    dataset.active_scalars = np.asfortranarray(np.moveaxis(out, 0, 2))
