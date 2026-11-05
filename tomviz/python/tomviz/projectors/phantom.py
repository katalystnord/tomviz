###############################################################################
# This source file is part of the Tomviz project, https://tomviz.org/.
# It is released under the 3-Clause BSD License, see "LICENSE".
###############################################################################
"""Ellipsoid phantoms with exact cone-beam projections.

The projections are analytic line integrals through each ellipsoid, so they
are exact and independent of any projector we are trying to test.
"""

import numpy as np

# 3D Shepp-Logan with the higher-contrast ("modified") densities, as in
# Toft (1996) / Kak & Slaney. Columns: density, semi-axes (a, b, c),
# centre (x, y, z), rotation about z in degrees. Lengths are fractions of
# the phantom radius.
SHEPP_LOGAN_3D = np.array([
    [1.0, 0.6900, 0.920, 0.810, 0.000, 0.0000, 0.000, 0.0],
    [-0.8, 0.6624, 0.874, 0.780, 0.000, -0.0184, 0.000, 0.0],
    [-0.2, 0.1100, 0.310, 0.220, 0.220, 0.0000, 0.000, -18.0],
    [-0.2, 0.1600, 0.410, 0.280, -0.220, 0.0000, 0.000, 18.0],
    [0.1, 0.2100, 0.250, 0.410, 0.000, 0.3500, -0.150, 0.0],
    [0.1, 0.0460, 0.046, 0.050, 0.000, 0.1000, 0.250, 0.0],
    [0.1, 0.0460, 0.046, 0.050, 0.000, -0.1000, 0.250, 0.0],
    [0.1, 0.0460, 0.023, 0.050, -0.080, -0.6050, 0.000, 0.0],
    [0.1, 0.0230, 0.023, 0.020, 0.000, -0.6060, 0.000, 0.0],
    [0.1, 0.0230, 0.046, 0.020, 0.060, -0.6050, 0.000, 0.0],
])


class EllipsoidPhantom:
    """A sum of uniform ellipsoids, each rotated about z."""

    def __init__(self, table, radius=1.0):
        table = np.asarray(table, dtype=np.float64)
        self.density = table[:, 0]
        self.axes = table[:, 1:4] * radius
        self.center = table[:, 4:7] * radius
        phi = np.deg2rad(table[:, 7])
        c, s = np.cos(phi), np.sin(phi)
        # Rotation matrices taking ellipsoid-local to world coordinates.
        self.rot = np.zeros((len(table), 3, 3))
        self.rot[:, 0, 0], self.rot[:, 0, 1] = c, -s
        self.rot[:, 1, 0], self.rot[:, 1, 1] = s, c
        self.rot[:, 2, 2] = 1.0

    @classmethod
    def shepp_logan(cls, radius=1.0):
        return cls(SHEPP_LOGAN_3D, radius)

    def _local(self, i, points):
        """World points to the unit-sphere frame of ellipsoid i."""
        return ((points - self.center[i]) @ self.rot[i]) / self.axes[i]

    def voxelize(self, vol_geom, supersample=1):
        """Sample the phantom on a VolumeGeometry. With supersample > 1,
        each voxel is the mean of supersample**3 sub-samples."""
        nz, ny, nx = vol_geom.shape
        dz, dy, dx = vol_geom.voxel_size
        out = np.zeros(vol_geom.shape, dtype=np.float64)
        offs = (np.arange(supersample) + 0.5) / supersample - 0.5
        z0, y0, x0 = (vol_geom.z_coords(), vol_geom.y_coords(),
                      vol_geom.x_coords())
        for oz in offs:
            for oy in offs:
                for ox in offs:
                    Z, Y, X = np.meshgrid(z0 + oz * dz, y0 + oy * dy,
                                          x0 + ox * dx, indexing='ij')
                    pts = np.stack([X, Y, Z], axis=-1)
                    for i in range(len(self.density)):
                        q = self._local(i, pts)
                        inside = np.einsum('...i,...i->...', q, q) <= 1.0
                        out[inside] += self.density[i]
        return out / supersample**3

    def line_integrals(self, origins, directions):
        """Exact integrals along rays origin + t * direction, t >= 0, for
        unit direction vectors. Both arrays are (..., 3)."""
        total = np.zeros(origins.shape[:-1], dtype=np.float64)
        for i in range(len(self.density)):
            p = self._local(i, origins)
            d = (directions @ self.rot[i]) / self.axes[i]
            a = np.einsum('...i,...i->...', d, d)
            b = 2.0 * np.einsum('...i,...i->...', p, d)
            c = np.einsum('...i,...i->...', p, p) - 1.0
            disc = b * b - 4.0 * a * c
            chord = np.where(disc > 0, np.sqrt(np.maximum(disc, 0.0)) / a,
                             0.0)
            total += self.density[i] * chord
        return total

    def project(self, geom, dtype=np.float32):
        """Exact cone-beam projections, shape (n_angles, n_v, n_u)."""
        out = np.empty((geom.n_angles, geom.n_v, geom.n_u), dtype=dtype)
        for k in range(geom.n_angles):
            pix = geom.pixel_positions(k)
            d = pix - geom.source[k]
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            origins = np.broadcast_to(geom.source[k], pix.shape)
            out[k] = self.line_integrals(origins, d)
        return out
