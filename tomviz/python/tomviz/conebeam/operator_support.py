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


def build_geometry(n_angles, n_u, n_v, angles_deg, geometry_source, sod,
                   sdd, pixel_size, offset_u, offset_v, clockwise,
                   geometry_file):
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
