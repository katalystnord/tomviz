###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Cone- and parallel-beam acquisition geometry, and volume geometry.

Coordinates
-----------
World coordinates are in physical units (usually mm). The rotation axis is
the world z axis and passes through the origin, which is also the centre of
the reconstructed volume.

Every projection is described by four vectors, the same layout as ASTRA's
``cone_vec`` geometry, so measured per-projection geometries can be used
as they are:

* ``source``: the X-ray source position.
* ``det_center``: the centre of the detector.
* ``det_u``: the step from one detector column to the next.
* ``det_v``: the step from one detector row to the next.

Projections are stored as ``(n_angles, n_v, n_u)`` arrays: index ``v`` runs
along ``det_v`` (roughly parallel to the rotation axis), index ``u`` runs
along ``det_u``. Detector index ``(n - 1) / 2`` is the detector centre.

Volumes are stored as ``(n_z, n_y, n_x)`` arrays, ``z`` along the rotation
axis, centred on the origin unless a ``center`` is given.

Parallel-beam geometries (``ParallelGeometry``) use the same conventions
with a ray direction in place of the source, like ASTRA's
``parallel3d_vec``. Both kinds provide ``projection_matrices()``, which is
all the backprojectors need.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VolumeGeometry:
    """A box of voxels, centred on the origin unless center is given."""

    shape: tuple  # (n_z, n_y, n_x)
    voxel_size: tuple  # (dz, dy, dx), physical units
    center: tuple = (0.0, 0.0, 0.0)  # (z, y, x) of the box centre

    def __post_init__(self):
        self.shape = tuple(int(n) for n in self.shape)
        if np.isscalar(self.voxel_size):
            self.voxel_size = (float(self.voxel_size),) * 3
        self.voxel_size = tuple(float(s) for s in self.voxel_size)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError(f'Invalid volume shape {self.shape}')
        if len(self.voxel_size) != 3 or min(self.voxel_size) <= 0:
            raise ValueError(f'Invalid voxel size {self.voxel_size}')
        self.center = tuple(float(c) for c in self.center)
        if len(self.center) != 3:
            raise ValueError(f'Invalid volume centre {self.center}')

    def axis_coords(self, axis):
        """World coordinates of voxel centres along one array axis."""
        n = self.shape[axis]
        return ((np.arange(n) - (n - 1) / 2) * self.voxel_size[axis] +
                self.center[axis])

    def z_coords(self):
        return self.axis_coords(0)

    def y_coords(self):
        return self.axis_coords(1)

    def x_coords(self):
        return self.axis_coords(2)

    def slab_corners(self, z0, z1):
        """World coordinates of the 8 corners of the slab of z slices
        [z0, z1), measured at the outer voxel faces."""
        dz, dy, dx = self.voxel_size
        nz, ny, nx = self.shape
        zs = ((z0 - nz / 2) * dz, (z1 - nz / 2) * dz)
        ys = (-ny * dy / 2, ny * dy / 2)
        xs = (-nx * dx / 2, nx * dx / 2)
        cz, cy, cx = self.center
        return np.array([[x + cx, y + cy, z + cz]
                         for z in zs for y in ys for x in xs])


@dataclass
class ConeGeometry:
    """Per-projection cone-beam geometry. See the module docstring."""

    source: np.ndarray  # (n_angles, 3)
    det_center: np.ndarray  # (n_angles, 3)
    det_u: np.ndarray  # (n_angles, 3), one column step
    det_v: np.ndarray  # (n_angles, 3), one row step
    n_u: int
    n_v: int
    # Nominal angles in radians, when the geometry came from a circular
    # orbit. Otherwise they are derived from the source positions.
    nominal_angles: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        for name in ('source', 'det_center', 'det_u', 'det_v'):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f'{name} must have shape (n_angles, 3)')
            setattr(self, name, v)
        n = len(self.source)
        if n < 1:
            raise ValueError('Need at least one projection')
        for name in ('det_center', 'det_u', 'det_v'):
            if len(getattr(self, name)) != n:
                raise ValueError(f'{name} has a different number of '
                                 'projections from source')
        self.n_u = int(self.n_u)
        self.n_v = int(self.n_v)
        if self.n_u < 1 or self.n_v < 1:
            raise ValueError('Detector must have at least one pixel')
        if np.any(np.linalg.norm(np.cross(self.det_u, self.det_v),
                                 axis=1) == 0):
            raise ValueError('det_u and det_v must not be parallel')
        n_dist = np.einsum('ij,ij->i', self._normal_unit(),
                           self.det_center - self.source)
        if np.any(np.abs(n_dist) == 0):
            raise ValueError('The source lies in the detector plane')

    # -- constructors ------------------------------------------------------

    @classmethod
    def circular(cls, angles_deg, sod, sdd, pixel_u, pixel_v, n_u, n_v,
                 offset_u=0.0, offset_v=0.0, clockwise=False):
        """A circular orbit around the z axis.

        angles_deg: projection angles in degrees.
        sod: source to rotation axis distance.
        sdd: source to detector distance.
        pixel_u, pixel_v: detector pixel pitch.
        n_u, n_v: detector columns and rows.
        offset_u, offset_v: where the central ray (the ray through the
            rotation axis, perpendicular to the detector) hits the detector,
            in pixels from the detector centre. Use offset_u to correct a
            rotation axis that does not project onto the centre column.
        clockwise: rotation direction seen from +z. Flipping it mirrors the
            reconstruction.
        """
        if not sod > 0 or not sdd > sod:
            raise ValueError('Need 0 < source-object distance < '
                             f'source-detector distance (got {sod}, {sdd})')
        if not pixel_u > 0 or not pixel_v > 0:
            raise ValueError('Pixel pitch must be positive')
        beta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64).ravel())
        if clockwise:
            beta = -beta
        c, s = np.cos(beta), np.sin(beta)
        zero = np.zeros_like(beta)
        e_s = np.stack([c, s, zero], axis=1)  # axis -> source direction
        e_u = np.stack([-s, c, zero], axis=1)
        e_v = np.tile([0.0, 0.0, 1.0], (len(beta), 1))
        source = sod * e_s
        # The central ray hits the detector at index (n-1)/2 + offset, so
        # the detector centre sits -offset pixels from that point.
        det_center = (-(sdd - sod) * e_s -
                      offset_u * pixel_u * e_u -
                      offset_v * pixel_v * e_v)
        geom = cls(source, det_center, pixel_u * e_u, pixel_v * e_v,
                   n_u, n_v, nominal_angles=beta)
        return geom

    @classmethod
    def from_astra_vectors(cls, vectors, n_u, n_v):
        """From an ASTRA ``cone_vec`` array of shape (n_angles, 12):
        source, detector centre, u step, v step."""
        v = np.asarray(vectors, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 12:
            raise ValueError('ASTRA vectors must have shape (n_angles, 12)')
        return cls(v[:, 0:3], v[:, 3:6], v[:, 6:9], v[:, 9:12], n_u, n_v)

    # -- derived quantities ------------------------------------------------

    @property
    def n_angles(self):
        return len(self.source)

    @property
    def detector_shape(self):
        return (self.n_v, self.n_u)

    def _normal_unit(self):
        n = np.cross(self.det_u, self.det_v)
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def central_axis(self):
        """Unit vector from each source perpendicular to its detector."""
        n = self._normal_unit()
        sign = np.sign(np.einsum('ij,ij->i', n,
                                 self.det_center - self.source))
        return n * sign[:, None]

    def sod(self):
        """Distance from each source to the rotation axis, measured along
        the central axis."""
        return np.einsum('ij,ij->i', -self.source, self.central_axis())

    def sdd(self):
        """Perpendicular distance from each source to its detector."""
        return np.einsum('ij,ij->i', self.det_center - self.source,
                         self.central_axis())

    def angles(self):
        """Source angle around z for every projection, unwrapped, in
        radians, increasing in the direction of rotation."""
        if self.nominal_angles is not None:
            beta = np.asarray(self.nominal_angles, dtype=np.float64)
        else:
            beta = np.unwrap(np.arctan2(self.source[:, 1],
                                        self.source[:, 0]))
        return beta

    def projection_matrices(self):
        """(n_angles, 3, 4) matrices P such that for a world point r,
        ``h = P @ [r, 1]`` gives detector index coordinates
        ``u = h[0] / h[2]``, ``v = h[1] / h[2]``, and ``h[2]`` is the
        point's depth along the central axis divided by the SOD (so it is
        1 on the rotation axis). FDK's distance weight is ``1 / h[2]**2``.
        """
        S, D, U, V = self.source, self.det_center, self.det_u, self.det_v
        n = np.cross(U, V)
        # Dual basis vectors: x . u_star gives the u coordinate of an
        # in-plane vector x, even when U and V are not orthogonal.
        u_star = np.cross(V, n)
        u_star /= np.einsum('ij,ij->i', U, u_star)[:, None]
        v_star = np.cross(n, U)
        v_star /= np.einsum('ij,ij->i', V, v_star)[:, None]
        d = D - S
        dn = np.einsum('ij,ij->i', d, n)
        cu = (self.n_u - 1) / 2
        cv = (self.n_v - 1) / 2

        def row(vec):
            # Linear form x -> vec . (x - S), as a 4-vector.
            return np.concatenate(
                [vec, -np.einsum('ij,ij->i', vec, S)[:, None]], axis=1)

        den = row(n)
        sd_u = np.einsum('ij,ij->i', S - D, u_star)
        sd_v = np.einsum('ij,ij->i', S - D, v_star)
        num_u = sd_u[:, None] * den + dn[:, None] * row(u_star) + cu * den
        num_v = sd_v[:, None] * den + dn[:, None] * row(v_star) + cv * den
        P = np.stack([num_u, num_v, den], axis=1)
        # Scale so that h[2] is 1 at the origin (on the rotation axis).
        P /= P[:, 2, 3][:, None, None]
        return P

    def principal_point(self):
        """Detector index coordinates (u, v) of the foot of the
        perpendicular from each source onto its detector plane."""
        P = self.projection_matrices()
        foot = self.source + self.central_axis() * self.sdd()[:, None]
        h = np.einsum('kij,kj->ki', P,
                      np.concatenate([foot, np.ones((len(foot), 1))],
                                     axis=1))
        return h[:, 0] / h[:, 2], h[:, 1] / h[:, 2]

    def pixel_positions(self, k):
        """World positions of the centres of every pixel of projection k,
        shape (n_v, n_u, 3)."""
        iu = np.arange(self.n_u) - (self.n_u - 1) / 2
        iv = np.arange(self.n_v) - (self.n_v - 1) / 2
        return (self.det_center[k] +
                iu[None, :, None] * self.det_u[k] +
                iv[:, None, None] * self.det_v[k])

    def ray_parameters(self):
        """Rays as origin(iu, iv) + t * direction(iu, iv), both linear in
        the pixel indices: six (n_angles, 3) arrays (o0, o_u, o_v, d0, d_u,
        d_v) with origin = o0 + iu o_u + iv o_v and likewise direction.
        Cone beams start at the source and reach the pixel at t = 1."""
        cu = (self.n_u - 1) / 2
        cv = (self.n_v - 1) / 2
        zero = np.zeros_like(self.source)
        d0 = self.det_center - cu * self.det_u - cv * self.det_v - self.source
        return self.source, zero, zero, d0, self.det_u, self.det_v

    def default_volume(self, shape=None, voxel_size=None):
        """A volume that covers the field of view, with the voxel size
        equal to the detector pitch demagnified to the rotation axis."""
        mag = float(np.median(self.sdd() / self.sod()))
        pu = float(np.median(np.linalg.norm(self.det_u, axis=1)))
        pv = float(np.median(np.linalg.norm(self.det_v, axis=1)))
        if voxel_size is None:
            voxel_size = pu / mag
        if shape is None:
            nxy = max(1, int(round(self.n_u * pu / mag / voxel_size)))
            nz = max(1, int(round(self.n_v * pv / mag / voxel_size)))
            shape = (nz, nxy, nxy)
        return VolumeGeometry(shape, voxel_size)


@dataclass
class ParallelGeometry:
    """Per-projection parallel-beam geometry, ASTRA ``parallel3d_vec``
    layout: ray direction, detector centre, u step, v step."""

    ray: np.ndarray  # (n_angles, 3), direction the X-rays travel
    det_center: np.ndarray  # (n_angles, 3)
    det_u: np.ndarray  # (n_angles, 3), one column step
    det_v: np.ndarray  # (n_angles, 3), one row step
    n_u: int
    n_v: int

    def __post_init__(self):
        for name in ('ray', 'det_center', 'det_u', 'det_v'):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f'{name} must have shape (n_angles, 3)')
            setattr(self, name, v)
        n = len(self.ray)
        if n < 1:
            raise ValueError('Need at least one projection')
        for name in ('det_center', 'det_u', 'det_v'):
            if len(getattr(self, name)) != n:
                raise ValueError(f'{name} has a different number of '
                                 'projections from ray')
        self.n_u = int(self.n_u)
        self.n_v = int(self.n_v)
        if self.n_u < 1 or self.n_v < 1:
            raise ValueError('Detector must have at least one pixel')
        det_n = np.cross(self.det_u, self.det_v)
        if np.any(np.einsum('ij,ij->i', det_n, self.ray) == 0):
            raise ValueError('Rays must not run parallel to the detector, '
                             'and det_u, det_v must not be parallel')

    @classmethod
    def circular(cls, angles_deg, pixel_u, pixel_v, n_u, n_v, offset_u=0.0,
                 offset_v=0.0, clockwise=False):
        """Rotation about the z axis, with the same angle convention as
        ConeGeometry.circular: at angle 0 the rays travel along -x and
        det_u points along +y.

        offset_u, offset_v: where the rotation axis projects onto the
            detector, in pixels from the detector centre.
        """
        if not pixel_u > 0 or not pixel_v > 0:
            raise ValueError('Pixel pitch must be positive')
        beta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64).ravel())
        if clockwise:
            beta = -beta
        c, s = np.cos(beta), np.sin(beta)
        zero = np.zeros_like(beta)
        e_s = np.stack([c, s, zero], axis=1)
        e_u = np.stack([-s, c, zero], axis=1)
        e_v = np.tile([0.0, 0.0, 1.0], (len(beta), 1))
        det_center = -offset_u * pixel_u * e_u - offset_v * pixel_v * e_v
        return cls(-e_s, det_center, pixel_u * e_u, pixel_v * e_v, n_u, n_v)

    @classmethod
    def from_astra_vectors(cls, vectors, n_u, n_v):
        """From an ASTRA ``parallel3d_vec`` array of shape (n_angles, 12):
        ray direction, detector centre, u step, v step."""
        v = np.asarray(vectors, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 12:
            raise ValueError('ASTRA vectors must have shape (n_angles, 12)')
        return cls(v[:, 0:3], v[:, 3:6], v[:, 6:9], v[:, 9:12], n_u, n_v)

    @property
    def n_angles(self):
        return len(self.ray)

    @property
    def detector_shape(self):
        return (self.n_v, self.n_u)

    def pixel_positions(self, k):
        """World positions of the centres of every pixel of projection k,
        shape (n_v, n_u, 3)."""
        iu = np.arange(self.n_u) - (self.n_u - 1) / 2
        iv = np.arange(self.n_v) - (self.n_v - 1) / 2
        return (self.det_center[k] +
                iu[None, :, None] * self.det_u[k] +
                iv[:, None, None] * self.det_v[k])

    def ray_parameters(self):
        """Rays as origin(iu, iv) + t * direction(iu, iv); see
        ConeGeometry.ray_parameters. Parallel rays start at the pixel."""
        cu = (self.n_u - 1) / 2
        cv = (self.n_v - 1) / 2
        zero = np.zeros_like(self.ray)
        o0 = self.det_center - cu * self.det_u - cv * self.det_v
        return o0, self.det_u, self.det_v, self.ray, zero, zero

    def projection_matrices(self):
        """(n_angles, 3, 4) affine matrices P: ``h = P @ [r, 1]`` gives
        detector index coordinates ``u = h[0]``, ``v = h[1]``, and
        ``h[2] = 1``, so the cone-beam backprojectors apply no distance
        weight."""
        U, V, R = self.det_u, self.det_v, self.ray
        # Project along the ray onto the detector plane: x - D = a U + b V
        # + t R, solved with dual vectors orthogonal to the other two.
        u_star = np.cross(V, R)
        u_star /= np.einsum('ij,ij->i', U, u_star)[:, None]
        v_star = np.cross(R, U)
        v_star /= np.einsum('ij,ij->i', V, v_star)[:, None]
        cu = (self.n_u - 1) / 2
        cv = (self.n_v - 1) / 2
        D = self.det_center
        n = self.n_angles
        P = np.zeros((n, 3, 4))
        P[:, 0, :3] = u_star
        P[:, 0, 3] = cu - np.einsum('ij,ij->i', u_star, D)
        P[:, 1, :3] = v_star
        P[:, 1, 3] = cv - np.einsum('ij,ij->i', v_star, D)
        P[:, 2, 3] = 1.0
        return P
