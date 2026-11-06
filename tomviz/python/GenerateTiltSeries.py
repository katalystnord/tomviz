import numpy as np
import scipy.ndimage

import tomviz.operators


class GenerateTiltSeriesOperator(tomviz.operators.CancelableOperator):

    def transform(self, dataset, start_angle=-90.0, angle_increment=3.0,
                  num_tilts=60, backend='auto'):
        """Generate Tilt Series from Volume"""
        self.progress.maximum = 1

        self.progress.message = 'Initialization'
        # Generate Tilt Angles.
        angles = np.linspace(start_angle, start_angle +
                             (num_tilts - 1) * angle_increment, num_tilts)

        volume = dataset.active_scalars
        Ny = volume.shape[1]
        Nz = volume.shape[2]
        # calculate the size s.t. it contains the entire volume
        N = np.round(np.sqrt(Ny**2 + Nz**2))
        N = int(np.floor(N / 2.0) * 2 + 1)  # make the size an odd integer

        if backend != 'cpu':
            from tomviz.projectors.backend import get_backend
            be = get_backend(backend)  # raises if 'gpu' is unavailable
            if be.name == 'gpu':
                tiltSeries = self._project(volume, angles, N, be)
                if tiltSeries is None:
                    return
                dataset.active_scalars = tiltSeries
                dataset.tilt_angles = angles
                return

        # pad volume
        pad_y_pre = int(np.ceil((N - Ny) / 2.0))
        pad_y_post = int(np.floor((N - Ny) / 2.0))
        pad_z_pre = int(np.ceil((N - Nz) / 2.0))
        pad_z_post = int(np.floor((N - Nz) / 2.0))
        volume_pad = np.pad(
            volume, ((0, 0), (pad_y_pre, pad_y_post), (pad_z_pre, pad_z_post)),
            'constant')

        Nslice = volume.shape[0]  # Number of slices along rotation axis.
        tiltSeries = np.empty([Nslice, N, num_tilts], dtype=float, order='F')

        self.progress.maximum = num_tilts
        step = 0

        for i in range(num_tilts):
            if self.canceled:
                return
            self.progress.message = 'Generating tilt image No.%d/%d' % (
                i + 1, num_tilts)

            # Rotate volume about x-axis
            rotatedVolume = np.empty_like(volume_pad)
            scipy.ndimage.rotate(
                volume_pad, angles[i], axes=(1, 2), reshape=False, order=1,
                output=rotatedVolume)
            # Calculate projection
            tiltSeries[:, :, i] = np.sum(rotatedVolume, axis=2)

            step += 1
            self.progress.value = step

        # Set the result as the new scalars.
        dataset.active_scalars = tiltSeries

        # Save tilt angles.
        dataset.tilt_angles = angles

    def _project(self, volume, angles, N, be):
        """The same projections on the WebGPU forward projector: rays along
        the volume's third axis, rotating about its first, with the
        rotation centre and detector of the code above. Returns None if
        canceled."""
        from tomviz.projectors import ParallelGeometry, VolumeGeometry

        Nslice, Ny, Nz = volume.shape
        self.progress.message = f'Projecting on {be.description}'
        self.progress.maximum = 100
        self.progress.value = 0
        # scipy.ndimage.rotate turns the padded volume about its centre,
        # (N - 1) / 2, which is this index of the unpadded volume.
        cy = (N - 1) / 2 - int(np.ceil((N - Ny) / 2.0))
        cz = (N - 1) / 2 - int(np.ceil((N - Nz) / 2.0))
        # tomviz.projectors volumes are (z: rotation axis, y, x), so the
        # third axis here is x there, where rays at angle 0 travel.
        vol_geom = VolumeGeometry(volume.shape, 1.0,
                                  center=(0.0, (Ny - 1) / 2 - cy,
                                          (Nz - 1) / 2 - cz))
        geom = ParallelGeometry.circular(angles, 1.0, 1.0, N, Nslice)

        def progress(done, total):
            self.progress.value = int(100 * done / total)

        proj = be.forward_project(np.asarray(volume, dtype=np.float32),
                                  geom, vol_geom, progress=progress,
                                  should_stop=lambda: self.canceled)
        if proj is None:
            return None
        # (angle, slice, ray) -> (slice, ray, angle), float64 as above.
        return np.asfortranarray(proj.transpose(1, 2, 0), dtype=float)
