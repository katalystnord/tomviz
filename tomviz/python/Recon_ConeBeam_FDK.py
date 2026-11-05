import numpy as np

import tomviz.operators
from tomviz.projectors import ConeGeometry, VolumeGeometry
from tomviz.conebeam import fdk


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


class ReconConeBeamFDKOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, geometry_source='circular', sod=100.0,
                  sdd=200.0, pixel_size=0.1, offset_u=0.0, offset_v=0.0,
                  clockwise=False, geometry_file=None,
                  filter_name='ram-lak', short_scan='auto',
                  volume_size_xy=0, volume_size_axial=0, voxel_size=0.0,
                  backend='auto'):
        """Cone-beam FDK reconstruction of a tilt series of projections."""
        tilt_series = dataset.active_scalars
        if tilt_series is None:
            raise RuntimeError('No scalars found!')
        if tilt_series.ndim != 3:
            raise RuntimeError('Expected a stack of projections')

        # Tomviz: (x along the rotation axis, y across it, angle).
        # tomviz.conebeam: (angle, v along the axis, u across it).
        n_v, n_u, n_angles = tilt_series.shape
        try:
            geom = build_geometry(
                n_angles, n_u, n_v, dataset.tilt_angles, geometry_source,
                sod, sdd, pixel_size, offset_u, offset_v, clockwise,
                geometry_file)
            default = geom.default_volume(
                voxel_size=voxel_size if voxel_size > 0 else None)
            nz, ny, _ = default.shape
            if volume_size_axial > 0:
                nz = volume_size_axial
            if volume_size_xy > 0:
                ny = volume_size_xy
            vol_geom = VolumeGeometry((nz, ny, ny), default.voxel_size)
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self.progress.maximum = 100
        self.progress.value = 0
        notes = []

        def log(msg):
            notes.append(msg)
            self.progress.message = '; '.join(notes)

        def progress(fraction):
            self.progress.value = int(round(100 * fraction))

        projections = np.ascontiguousarray(
            np.moveaxis(tilt_series, 2, 0), dtype=np.float32)
        short = {'auto': 'auto', 'on': True, 'off': False}[short_scan]
        vol = fdk(projections, geom, vol_geom, filter_name=filter_name,
                  short_scan=short, backend=backend, progress=progress,
                  should_stop=lambda: self.canceled, log=log)
        if vol is None:
            return None

        # tomviz.conebeam volume is (z along the axis, y, x); tomviz wants
        # (x along the axis, ...). (z, x, y) keeps the axes right-handed.
        child = dataset.create_child_dataset()
        child.active_scalars = np.asfortranarray(vol.transpose(0, 2, 1))
        dz, dy, dx = vol_geom.voxel_size
        child.spacing = [dz, dx, dy]
        return {'reconstruction': child}
