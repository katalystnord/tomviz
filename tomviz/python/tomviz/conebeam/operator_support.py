###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Shared by the cone-beam operators: turning operator parameters into
geometries, and volumes back into tomviz datasets."""

import numpy as np

from tomviz.projectors.geometry import ConeGeometry, VolumeGeometry


def load_geometry_vectors(path):
    """An (n, 12) array of ASTRA cone_vec rows from a text or .npy file."""
    if not path:
        raise RuntimeError('Choose a geometry vectors file')
    if str(path).lower().endswith('.npy'):
        vecs = np.load(path)
    else:
        vecs = np.loadtxt(path, delimiter=',' if str(path).lower()
                          .endswith('.csv') else None, ndmin=2)
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim != 2 or vecs.shape[1] != 12:
        raise RuntimeError(f'{path}: expected 12 numbers per row, got an '
                           f'array of shape {vecs.shape}')
    return vecs


# The scanner geometry a file reader found, stored in dataset.metadata
# under 'geometry'. Lengths in mm, offsets in detector pixels (of the
# detector_shape given), angles in degrees:
#
#   {'sod': ..., 'sdd': ..., 'pixel_size_u': ..., 'pixel_size_v': ...,
#    'offset_u': 0.0, 'offset_v': 0.0, 'clockwise': False,
#    'detector_shape': [n_v, n_u], 'source': 'Nikon .xtekct'}
#
# or, for a measured per-projection geometry, 'vectors': n rows of the 12
# ASTRA cone_vec numbers (source, detector centre, u step, v step).


def geometry_from_metadata(metadata, n_angles, n_u, n_v, angles_deg):
    """The ConeGeometry a reader stored with the data. The projections
    may have been binned since: the pixel pitch and offsets follow when
    the detector size divides evenly."""
    g = (metadata or {}).get('geometry')
    if not g:
        raise RuntimeError(
            "This dataset carries no scanner geometry: its file format "
            "does not store one. Choose 'Circular Orbit' and enter the "
            "distances, or 'Vectors From File'.")
    shape = g.get('detector_shape') or [n_v, n_u]
    bin_v, bin_u = shape[0] / n_v, shape[1] / n_u
    for b, axis in ((bin_v, 'rows'), (bin_u, 'columns')):
        if abs(b - round(b)) > 1e-9 or round(b) < 1:
            raise RuntimeError(
                f'The projections have {n_v} x {n_u} pixels but the scanner '
                f'geometry is for {shape[0]} x {shape[1]}; the detector '
                f'{axis} were cropped, so the stored geometry no longer '
                "fits. Use 'Circular Orbit' with the right values.")
    bin_v, bin_u = round(bin_v), round(bin_u)
    if 'vectors' in g:
        vecs = np.asarray(g['vectors'], dtype=np.float64)
        if vecs.shape != (n_angles, 12):
            raise RuntimeError(f'The stored geometry has {len(vecs)} '
                               f'projections, the data {n_angles}')
        vecs = vecs.copy()
        vecs[:, 6:9] *= bin_u
        vecs[:, 9:12] *= bin_v
        return ConeGeometry.from_astra_vectors(vecs, n_u, n_v)
    angles = g.get('angles', angles_deg)
    if angles is None or len(angles) != n_angles:
        raise RuntimeError('Every projection needs a tilt angle (degrees)')
    return ConeGeometry.circular(
        angles, float(g['sod']), float(g['sdd']),
        float(g['pixel_size_u']) * bin_u, float(g['pixel_size_v']) * bin_v,
        n_u, n_v, offset_u=float(g.get('offset_u', 0.0)) / bin_u,
        offset_v=float(g.get('offset_v', 0.0)) / bin_v,
        clockwise=bool(g.get('clockwise', False)))


def build_geometry(n_angles, n_u, n_v, angles_deg, geometry_source, sod,
                   sdd, pixel_size, offset_u, offset_v, clockwise,
                   geometry_file, metadata=None):
    if geometry_source == 'metadata':
        return geometry_from_metadata(metadata, n_angles, n_u, n_v,
                                      angles_deg)
    if geometry_source == 'file':
        vecs = load_geometry_vectors(geometry_file)
        if len(vecs) != n_angles:
            raise RuntimeError(
                f'The geometry file has {len(vecs)} rows but there are '
                f'{n_angles} projections')
        return ConeGeometry.from_astra_vectors(vecs, n_u, n_v)
    if angles_deg is None or len(angles_deg) != n_angles:
        raise RuntimeError('Every projection needs a tilt angle (degrees)')
    return ConeGeometry.circular(angles_deg, sod, sdd, pixel_size,
                                 pixel_size, n_u, n_v, offset_u=offset_u,
                                 offset_v=offset_v, clockwise=clockwise)


def build_volume(geom, volume_size_xy=0, volume_size_axial=0,
                 voxel_size=0.0):
    """The volume to reconstruct; zeros mean 'cover the field of view' and
    'detector pitch over magnification'."""
    default = geom.default_volume(
        voxel_size=voxel_size if voxel_size > 0 else None)
    nz, ny, _ = default.shape
    if volume_size_axial > 0:
        nz = volume_size_axial
    if volume_size_xy > 0:
        ny = volume_size_xy
    return VolumeGeometry((nz, ny, ny), default.voxel_size)


def projections_from_tilt_series(tilt_series):
    """Tomviz tilt series (x along the rotation axis, y across it, angle)
    to a C-contiguous float32 (angle, v, u) stack, a new array."""
    if tilt_series is None:
        raise RuntimeError('No scalars found!')
    if tilt_series.ndim != 3:
        raise RuntimeError('Expected a stack of projections')
    # A copy even when moveaxis already gives a C-contiguous view (a single
    # detector row does): the reconstruction filters this array in place.
    return np.array(np.moveaxis(tilt_series, 2, 0), dtype=np.float32,
                    order='C', copy=True)


def set_volume(child, vol, vol_geom):
    """Store a (z along the axis, y, x) volume in a tomviz dataset, which
    wants x along the axis. (z, x, y) keeps the axes right-handed."""
    child.active_scalars = np.asfortranarray(vol.transpose(0, 2, 1))
    dz, dy, dx = vol_geom.voxel_size
    child.spacing = [dz, dx, dy]
