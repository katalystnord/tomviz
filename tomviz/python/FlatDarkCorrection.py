import numpy as np

from tomviz.conebeam.preprocess import (
    flat_dark_correct, flat_dark_correct_interpolated)


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


def _frame_positions(ds, count):
    """Frame numbers in the acquisition that a file reader stored in the
    dataset's metadata ('frame_index'), when there is one per frame."""
    meta = getattr(ds, 'metadata', None) or {}
    index = meta.get('frame_index')
    if index is not None and len(index) == count:
        return np.asarray(index, dtype=np.float64)
    return None


def _interpolation_positions(dataset, flat, n_flat, n_proj, flats_before):
    """Positions of the flat frames and of the projections: from the
    files when both carry frame numbers, otherwise flats_before frames
    taken before the scan and the rest after it."""
    proj_pos = _frame_positions(dataset, n_proj)
    flat_pos = _frame_positions(flat, n_flat)
    if proj_pos is not None and flat_pos is not None:
        return flat_pos, proj_pos
    if not 0 < flats_before < n_flat:
        raise RuntimeError(
            'To interpolate between flats taken before and after the scan, '
            f"set how many of the {n_flat} flat frames were taken before it "
            '(the files do not say).')
    flat_pos = np.where(np.arange(n_flat) < flats_before, -1.0,
                        float(n_proj))
    return flat_pos, np.arange(n_proj, dtype=np.float64)


def transform(dataset, flat=None, dark=None, line_integrals=True,
              min_transmission=1e-6, flat_mode='average', flats_before=0):
    """Flat-field and dark-field correction, optionally with -log."""
    projections = dataset.active_scalars
    if projections is None:
        raise RuntimeError('No scalars found!')
    if projections.ndim != 3:
        raise RuntimeError('Expected a stack of projections')
    if flat is None:
        # Without a flat field there is nothing to normalise by.
        raise RuntimeError('Connect a flat field (open-beam image) to the '
                           "'flat' input. For data that are already "
                           'normalised, use Intensity to Line Integrals.')
    frame_shape = projections.shape[:2]
    flat_frames = (_frames(flat.active_scalars, frame_shape)
                   if flat is not None else None)
    dark_frames = (_frames(dark.active_scalars, frame_shape)
                   if dark is not None else None)

    frames = np.moveaxis(projections, 2, 0)
    try:
        if flat_mode == 'interpolate' and np.ndim(flat_frames) == 3:
            flat_pos, proj_pos = _interpolation_positions(
                dataset, flat, len(flat_frames), len(frames), flats_before)
            out = flat_dark_correct_interpolated(
                frames, flat_frames, flat_pos, proj_pos, dark=dark_frames,
                log=line_integrals, min_transmission=min_transmission)
        else:
            out = flat_dark_correct(frames, flat=flat_frames,
                                    dark=dark_frames, log=line_integrals,
                                    min_transmission=min_transmission)
    except ValueError as e:
        raise RuntimeError(str(e)) from e
    dataset.active_scalars = np.asfortranarray(np.moveaxis(out, 0, 2))
