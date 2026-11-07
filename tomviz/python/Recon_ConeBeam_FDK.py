import tomviz.operators
from tomviz.conebeam import fdk
from tomviz.conebeam.operator_support import (
    build_geometry, build_volume, projections_from_tilt_series, set_volume)


class ReconConeBeamFDKOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, geometry_source='circular', sod=100.0,
                  sdd=200.0, pixel_size=0.1, offset_u=0.0, offset_v=0.0,
                  clockwise=False, geometry_file=None,
                  filter_name='ram-lak', short_scan='auto',
                  volume_size_xy=0, volume_size_axial=0, voxel_size=0.0,
                  backend='auto'):
        """Cone-beam FDK reconstruction of a tilt series of projections."""
        projections = projections_from_tilt_series(dataset.active_scalars)
        n_angles, n_v, n_u = projections.shape
        try:
            geom = build_geometry(
                n_angles, n_u, n_v, dataset.tilt_angles, geometry_source,
                sod, sdd, pixel_size, offset_u, offset_v, clockwise,
                geometry_file,
                metadata=getattr(dataset, 'metadata', None))
            vol_geom = build_volume(geom, volume_size_xy, volume_size_axial,
                                    voxel_size)
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

        short = {'auto': 'auto', 'on': True, 'off': False}[short_scan]
        vol = fdk(projections, geom, vol_geom, filter_name=filter_name,
                  short_scan=short, backend=backend, progress=progress,
                  should_stop=lambda: self.canceled, log=log,
                  overwrite=True)
        if vol is None:
            return None

        child = dataset.create_child_dataset()
        set_volume(child, vol, vol_geom)
        return {'reconstruction': child}
